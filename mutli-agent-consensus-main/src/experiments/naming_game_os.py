"""
OS-only naming game experiment implementation.

This variant is intended for open-source HuggingFace models and uses
tokenizer-safe single-token labels to avoid first-token collisions.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from ..core.state import RunResult
from ..core.prompts import ABLATION_CONDITIONS, PromptBuilder
from ..core.pairing import pair_indices, shared_allowed_for_round, uniform_model_assignment
from ..core.engine import EngineConfig, compute_run_summary
from ..core.utils import ensure_dir, set_global_seed
from ..core.io import (
    init_run_dir,
    dump_resolved_config,
    open_trials_writer,
    write_run_summary,
    hash_prompt,
    save_raw_response,
)
from ..llms.base import ChoiceResult

logger = logging.getLogger(__name__)


HF_LABEL_CANDIDATES = [
    "ziv", "qam", "rol", "teb", "vuk", "paf", "lod", "mur", "zel", "kov",
    "bex", "daj", "fim", "gup", "hix", "jot", "kez", "niv", "pom", "rax",
    "bam", "cal", "dek", "fel", "gar", "hob", "jil", "kep", "lim", "mok",
    "nal", "pel", "ral", "sol", "tal", "vel", "wim", "yel", "zol", "bor",
    "dol", "fol", "gol", "hol", "jol", "kol", "mol", "nol", "pol", "vol",
    "wol", "yol", "bul", "dul", "ful", "gul", "hul", "jul", "kul", "mul",
    "nul", "pul", "rul", "sul", "tul", "vul", "wul", "yul", "zul", "bel",
    "del", "gel", "hel", "jel", "kel", "mel", "nel", "rel", "sel", "tel",
    "wel", "bil", "dil", "fil", "gil", "hil",
]


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass
class NamingGameOSConfig:
    # Agent parameters
    n_agents: int = 24
    n_rounds: int = 200
    n_runs: int = 10
    history_length: int = 3

    # Name vocabulary
    n_names: int = 10
    allowed_path: Optional[str] = None

    # Conditions to run
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_CONDITIONS.keys()))

    # LLM settings
    model_names: List[str] = field(default_factory=list)
    assignment_seed_offset: int = 777
    temperature: float = 0.3
    max_tokens: int = 8
    round1_fullstring_argmax: bool = False

    # Output
    outdir: str = "results/os/naming_game"

    # Reproducibility
    seed: int = 12345

    # Logging
    store_raw_responses: bool = False
    verbosity: int = 1


class NamingGameOSExperiment:
    """
    Naming game runner specialized for open-source HuggingFace models.
    """

    def __init__(self, clients, config: Optional[NamingGameOSConfig] = None):
        self.config = config or NamingGameOSConfig()
        self.clients = self._normalize_clients(clients)
        if not self.config.model_names:
            self.config.model_names = list(self.clients.keys())
        else:
            self.config.model_names = _dedup_preserve_order(self.config.model_names)

        missing = [m for m in self.config.model_names if m not in self.clients]
        if missing:
            raise ValueError(f"Missing clients for model_names: {missing}")

        for model_name in self.config.model_names:
            c = self.clients[model_name]
            if not hasattr(c, "tokenizer"):
                raise ValueError(
                    f"NamingGameOSExperiment requires tokenizer-backed clients; "
                    f"'{model_name}' has no tokenizer."
                )

        self.allowed_names = self._build_tokenizer_safe_labels()
        ensure_dir(self.config.outdir)

    def _normalize_clients(self, clients) -> Dict[str, Any]:
        if isinstance(clients, dict):
            return dict(clients)

        model_name = getattr(clients, "model", "unknown")
        return {model_name: clients}

    def _load_candidates(self) -> List[str]:
        if self.config.allowed_path:
            p = Path(self.config.allowed_path)
            if not p.exists():
                raise FileNotFoundError(f"allowed_path not found: {p}")
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return _dedup_preserve_order(lines)
        return _dedup_preserve_order(HF_LABEL_CANDIDATES)

    def _build_tokenizer_safe_labels(self) -> List[str]:
        tokenizers = [self.clients[m].tokenizer for m in self.config.model_names]
        candidates = self._load_candidates()

        selected: List[str] = []
        used_ids = [set() for _ in tokenizers]
        for label in candidates:
            per_tok_ids = []
            ok = True
            for tok in tokenizers:
                ids = tok.encode(label, add_special_tokens=False)
                if len(ids) != 1:
                    ok = False
                    break
                per_tok_ids.append(int(ids[0]))
            if not ok:
                continue
            if any(per_tok_ids[i] in used_ids[i] for i in range(len(tokenizers))):
                continue
            selected.append(label)
            for i, token_id in enumerate(per_tok_ids):
                used_ids[i].add(token_id)
            if len(selected) >= self.config.n_names:
                break

        if len(selected) < self.config.n_names:
            raise ValueError(
                f"Could not find {self.config.n_names} tokenizer-safe labels; "
                f"found {len(selected)}."
            )

        return selected

    def _chat_formatted_prompt(self, client: Any, user_prompt: str) -> str:
        system = "Answer with exactly one token from the allowed list. No extra words."
        tok = getattr(client, "tokenizer", None)
        if tok is None:
            return user_prompt
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
            return tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return (
                f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n"
                f"[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]\n"
            )

    def _generate_choice(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        allowed_labels: List[str],
        temperature: float,
        seed: int,
        round_idx: int,
    ) -> ChoiceResult:
        if round_idx == 0 and self.config.round1_fullstring_argmax:
            if hasattr(client, "score_full_string"):
                try:
                    prompt_for_scoring = self._chat_formatted_prompt(client, prompt)
                    scores = client.score_full_string(prompt_for_scoring, allowed_labels)
                    if len(scores) == len(allowed_labels):
                        best_idx = max(range(len(scores)), key=lambda i: scores[i])
                        choice = allowed_labels[best_idx]
                        return ChoiceResult(
                            text_raw=choice,
                            choice=choice,
                            valid=True,
                            retries=0,
                            meta={
                                "provider": "hf",
                                "model": model_name,
                                "mode": "round1_fullstring_argmax",
                            },
                        )
                except Exception as exc:
                    if self.config.verbosity >= 1:
                        logger.warning(
                            f"round1_fullstring_argmax failed; falling back to default choice path: {exc}"
                        )

        return client.generate_choice(
            prompt,
            allowed_labels,
            temperature=temperature,
            seed=seed,
        )

    def run_single(
        self,
        condition: str,
        seed: int,
        run_idx: int = 0,
        trials_writer=None,
        run_path: Optional[str] = None,
        run_id: str = "",
    ) -> RunResult:
        import random
        import time

        set_global_seed(seed)
        for client in self.clients.values():
            if hasattr(client, "set_seed"):
                client.set_seed(seed)

        N = self.config.n_agents
        H = self.config.history_length
        W = self.config.n_names
        rng = random.Random(seed)
        prompt_builder = PromptBuilder.from_condition(condition)

        agents = [
            {"history": deque(maxlen=H), "cum_reward": 0.0}
            for _ in range(N)
        ]
        if len(self.config.model_names) == 1:
            agent_types = [self.config.model_names[0]] * N
        else:
            assignment, _ = uniform_model_assignment(
                N,
                len(self.config.model_names),
                seed=seed + self.config.assignment_seed_offset,
            )
            agent_types = [self.config.model_names[idx] for idx in assignment]

        per_round_success = []
        per_round_choices = []
        invalids = 0
        total_retries = 0
        total_trials = 0
        within_type_matches = []
        cross_type_matches = []

        for round_idx in range(self.config.n_rounds):
            pairs = pair_indices(N, rng)
            round_matches = 0
            round_choices = []

            for i, j in pairs:
                agent_i = agents[i]
                agent_j = agents[j]
                allowed_round = shared_allowed_for_round(self.allowed_names, rng)

                prompt_i = prompt_builder.build(
                    allowed_names=allowed_round,
                    history=agent_i["history"],
                    cum_score=agent_i["cum_reward"],
                )
                prompt_j = prompt_builder.build(
                    allowed_names=allowed_round,
                    history=agent_j["history"],
                    cum_score=agent_j["cum_reward"],
                )

                model_i = agent_types[i]
                model_j = agent_types[j]
                client_i = self.clients[model_i]
                client_j = self.clients[model_j]

                t0 = time.time()
                result_i = self._generate_choice(
                    client=client_i,
                    model_name=model_i,
                    prompt=prompt_i,
                    allowed_labels=allowed_round,
                    temperature=self.config.temperature,
                    seed=seed + round_idx * 1000 + i,
                    round_idx=round_idx,
                )
                latency_i = (time.time() - t0) * 1000

                t0 = time.time()
                result_j = self._generate_choice(
                    client=client_j,
                    model_name=model_j,
                    prompt=prompt_j,
                    allowed_labels=allowed_round,
                    temperature=self.config.temperature,
                    seed=seed + round_idx * 1000 + j,
                    round_idx=round_idx,
                )
                latency_j = (time.time() - t0) * 1000

                choice_i = result_i.choice if result_i.valid else rng.choice(allowed_round)
                choice_j = result_j.choice if result_j.valid else rng.choice(allowed_round)

                if not result_i.valid:
                    invalids += 1
                if not result_j.valid:
                    invalids += 1
                total_retries += result_i.retries + result_j.retries

                match = choice_i == choice_j
                if match:
                    round_matches += 1
                same_type = (model_i == model_j)
                if same_type:
                    within_type_matches.append(1 if match else 0)
                else:
                    cross_type_matches.append(1 if match else 0)

                reward_i = 1.0 if match else 0.0
                reward_j = 1.0 if match else 0.0
                agent_i["cum_reward"] += reward_i
                agent_j["cum_reward"] += reward_j

                agent_i["history"].append((choice_j, choice_i, match))
                agent_j["history"].append((choice_i, choice_j, match))

                round_choices.extend([choice_i, choice_j])
                total_trials += 2

                if trials_writer:
                    trials_writer.log({
                        "run_id": run_id,
                        "experiment": "naming_game_os",
                        "variant_id": condition,
                        "seed": seed,
                        "repeat_idx": run_idx,
                        "round": round_idx,
                        "phase": "na",
                        "agent_id": i,
                        "agent_type": model_i,
                        "partner_id": j,
                        "partner_type": model_j,
                        "provider": result_i.meta.get("provider", ""),
                        "model_name": result_i.meta.get("model", model_i),
                        "temperature": self.config.temperature,
                        "H": H,
                        "N": N,
                        "W": W,
                        "prompt_hash": hash_prompt(prompt_i),
                        "allowed_set_id": str(hash(tuple(sorted(self.allowed_names)))),
                        "allowed_order_id": str(hash(tuple(allowed_round))),
                        "prompt_variant": condition,
                        "choice": choice_i,
                        "choice_valid": result_i.valid,
                        "partner_choice": choice_j,
                        "match": match,
                        "reward": reward_i,
                        "cum_reward": agent_i["cum_reward"],
                        "latency_ms": latency_i,
                        "retries": result_i.retries,
                        "raw_response_path": "",
                    })
                    trials_writer.log({
                        "run_id": run_id,
                        "experiment": "naming_game_os",
                        "variant_id": condition,
                        "seed": seed,
                        "repeat_idx": run_idx,
                        "round": round_idx,
                        "phase": "na",
                        "agent_id": j,
                        "agent_type": model_j,
                        "partner_id": i,
                        "partner_type": model_i,
                        "provider": result_j.meta.get("provider", ""),
                        "model_name": result_j.meta.get("model", model_j),
                        "temperature": self.config.temperature,
                        "H": H,
                        "N": N,
                        "W": W,
                        "prompt_hash": hash_prompt(prompt_j),
                        "allowed_set_id": str(hash(tuple(sorted(self.allowed_names)))),
                        "allowed_order_id": str(hash(tuple(allowed_round))),
                        "prompt_variant": condition,
                        "choice": choice_j,
                        "choice_valid": result_j.valid,
                        "partner_choice": choice_i,
                        "match": match,
                        "reward": reward_j,
                        "cum_reward": agent_j["cum_reward"],
                        "latency_ms": latency_j,
                        "retries": result_j.retries,
                        "raw_response_path": "",
                    })

                if self.config.store_raw_responses and run_path:
                    key_i = f"r{round_idx}_a{i}"
                    key_j = f"r{round_idx}_a{j}"
                    save_raw_response(run_path, key_i, {
                        "prompt": prompt_i,
                        "response": result_i.text_raw,
                        "meta": result_i.meta,
                    })
                    save_raw_response(run_path, key_j, {
                        "prompt": prompt_j,
                        "response": result_j.text_raw,
                        "meta": result_j.meta,
                    })

            n_pairs = len(pairs)
            match_rate = round_matches / n_pairs if n_pairs > 0 else 0.0
            per_round_success.append(match_rate)
            per_round_choices.append(round_choices)

            if self.config.verbosity >= 1:
                log_interval = max(1, min(10, self.config.n_rounds // 10))
                if (round_idx + 1) % log_interval == 0 or round_idx == self.config.n_rounds - 1:
                    avg_so_far = sum(per_round_success) / len(per_round_success)
                    recent_avg = sum(per_round_success[-log_interval:]) / min(log_interval, len(per_round_success))
                    pct = (round_idx + 1) / self.config.n_rounds * 100
                    logger.info(
                        f"    Round {round_idx + 1:4d}/{self.config.n_rounds} ({pct:5.1f}%) | "
                        f"this={match_rate:.2f} recent={recent_avg:.2f} avg={avg_so_far:.2f}"
                    )

        invalid_rate = 100 * invalids / total_trials if total_trials > 0 else 0.0
        within_rate = sum(within_type_matches) / len(within_type_matches) if within_type_matches else None
        cross_rate = sum(cross_type_matches) / len(cross_type_matches) if cross_type_matches else None

        return RunResult(
            per_round_success=per_round_success,
            per_round_choices=per_round_choices,
            invalid_rate=invalid_rate,
            total_retries=total_retries,
            within_type_rate=within_rate,
            cross_type_rate=cross_rate,
            config={
                "allowed_names": self.allowed_names,
                "model_names": self.config.model_names,
                "mixed": len(self.config.model_names) > 1,
            },
        )

    def run_condition(self, condition: str) -> List[RunResult]:
        import time

        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"OS CONDITION: {condition.upper()}")
            logger.info(f"{'='*60}")
            logger.info(
                f"  Runs: {self.config.n_runs}, Rounds: {self.config.n_rounds}, "
                f"Agents: {self.config.n_agents}"
            )
            logger.info(f"  Models: {self.config.model_names}")

        if len(self.config.model_names) <= 3:
            model_tag = "+".join(self.config.model_names)
        else:
            model_tag = f"{len(self.config.model_names)}models"
        run_id, run_path = init_run_dir(
            self.config.outdir,
            experiment="naming_game_os",
            model=model_tag,
            variant=condition,
            seed=self.config.seed,
        )

        config_dict = {
            "experiment": "naming_game_os",
            "condition": condition,
            "n_agents": self.config.n_agents,
            "n_rounds": self.config.n_rounds,
            "n_runs": self.config.n_runs,
            "history_length": self.config.history_length,
            "n_names": self.config.n_names,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "models": self.config.model_names,
            "mixed": len(self.config.model_names) > 1,
            "allowed_names": self.allowed_names,
            "round1_fullstring_argmax": self.config.round1_fullstring_argmax,
        }
        config_digest = dump_resolved_config(run_path, config_dict)

        results = []
        with open_trials_writer(run_path) as trials_writer:
            for run_idx in range(self.config.n_runs):
                seed = self.config.seed + run_idx

                if self.config.verbosity >= 1:
                    logger.info("")
                    logger.info(f"  --- Run {run_idx + 1}/{self.config.n_runs} (seed={seed}) ---")

                t0 = time.time()
                result = self.run_single(
                    condition=condition,
                    seed=seed,
                    run_idx=run_idx,
                    trials_writer=trials_writer,
                    run_path=run_path,
                    run_id=run_id,
                )
                elapsed = time.time() - t0

                results.append(result)

                if self.config.verbosity >= 1:
                    avg_success = np.mean(result.per_round_success)
                    final_success = (
                        np.mean(result.per_round_success[-10:])
                        if len(result.per_round_success) >= 10
                        else avg_success
                    )
                    logger.info(
                        f"  Run {run_idx + 1} DONE: {elapsed:.1f}s | "
                        f"avg={avg_success:.3f} final={final_success:.3f}"
                    )

        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            run_id=run_id,
            experiment="naming_game_os",
            variant_id=condition,
        )
        summary = compute_run_summary(results, engine_config, config_digest)
        write_run_summary(run_path, summary)

        if self.config.verbosity >= 1:
            all_avg = [np.mean(r.per_round_success) for r in results]
            all_final = [
                np.mean(r.per_round_success[-10:])
                if len(r.per_round_success) >= 10
                else np.mean(r.per_round_success)
                for r in results
            ]
            logger.info("")
            logger.info(f"  OS CONDITION SUMMARY: {condition}")
            logger.info(f"    Overall:  {np.mean(all_avg):.3f} ± {np.std(all_avg):.3f}")
            logger.info(f"    Final 10: {np.mean(all_final):.3f} ± {np.std(all_final):.3f}")
            logger.info(f"    Saved to: {run_path}")

        return results

    def run_all(self) -> Dict[str, List[RunResult]]:
        all_results = {}
        for condition in self.config.conditions:
            all_results[condition] = self.run_condition(condition)
        return all_results
