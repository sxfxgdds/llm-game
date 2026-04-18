"""
Tipping (committed minority) experiment for naming-game LLMs.

Idea:
- A fraction p of agents are "committed" to a target label w* and always output it.
- The rest are normal LLM agents.
- Measure when the population tips into the target convention.

Uses the shared engine + standardized logging.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from ..core.prompts import ABLATION_CONDITIONS
from ..core.utils import ensure_dir, set_global_seed, make_simple_tokens
from ..core.engine import EngineConfig, run_population_game, compute_run_summary
from ..core.io import init_run_dir, dump_resolved_config, open_trials_writer, write_run_summary

logger = logging.getLogger(__name__)


@dataclass
class TippingConfig:
    # Population
    n_agents: int = 24
    n_rounds: int = 200
    n_runs: int = 10

    # Game prompt/history
    history_length: int = 3
    n_names: int = 10
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_CONDITIONS.keys()))

    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 8

    # Tipping sweep
    fractions: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    target_label: str = "w0"          # committed label
    tip_threshold: float = 0.9        # "success" if final dominant share >= threshold AND dominant label == target

    # Output
    outdir: str = "results/tipping"

    # Reproducibility
    seed: int = 12345

    # Logging
    store_raw_responses: bool = False
    verbosity: int = 1


class CommittedClient:
    """
    Minimal client that always returns a fixed allowed label.
    Designed to be compatible with core.engine's expectation of a client.generate_choice(...)
    returning a ChoiceResult-like object.
    """
    def __init__(self, fixed_label: str, model_name: str = "committed"):
        self.fixed_label = fixed_label
        self.model = model_name

    def set_seed(self, seed: int) -> None:
        # deterministic anyway
        return

    def generate_choice(self, prompt: str, allowed_labels: List[str], temperature: Optional[float] = None, seed: int = None, **kwargs):
        # Lazy import to avoid circular issues; matches your OpenAIClient return schema
        from ..llms.base import ChoiceResult  # your repo already has this

        choice = self.fixed_label if self.fixed_label in allowed_labels else allowed_labels[0]
        return ChoiceResult(
            text_raw=choice,
            choice=choice,
            valid=True,
            retries=0,
            meta={"provider": "committed", "model": self.model, "fixed_label": self.fixed_label},
        )

    @property
    def supports_constrained_generation(self) -> bool:
        return True

    @property
    def supports_token_scores(self) -> bool:
        return False


def _p_tag(p: float) -> str:
    # 0.15 -> "0p15" ; 1.0 -> "1"
    if abs(p - round(p)) < 1e-9:
        return str(int(round(p)))
    return f"{p:.6g}".replace(".", "p")


class TippingExperiment:
    """
    Runs tipping experiment for one base client (LLM) plus committed minority.
    """

    def __init__(self, client, config: Optional[TippingConfig] = None):
        self.client = client
        self.config = config or TippingConfig()
        self.allowed_names = make_simple_tokens(self.config.n_names)
        ensure_dir(self.config.outdir)

    def _agent_types_for_run(self, seed: int, p: float, base_type: str, committed_type: str) -> Tuple[List[str], int]:
        import random
        rng = random.Random(seed + 999)
        n_commit = int(round(p * self.config.n_agents))
        idxs = set(rng.sample(range(self.config.n_agents), k=n_commit)) if n_commit > 0 else set()
        agent_types = [committed_type if i in idxs else base_type for i in range(self.config.n_agents)]
        return agent_types, n_commit

    def run_single(
        self,
        condition: str,
        p: float,
        seed: int,
        run_idx: int,
        trials_writer,
        run_path: str,
        run_id: str,
    ):
        import random

        set_global_seed(seed)

        # Best-effort set seed on base client
        if hasattr(self.client, "set_seed"):
            self.client.set_seed(seed)

        base_type = getattr(self.client, "model", "base_llm")
        committed_type = f"committed::{self.config.target_label}"

        committed_client = CommittedClient(
            fixed_label=self.config.target_label,
            model_name=committed_type,
        )

        clients = {
            base_type: self.client,
            committed_type: committed_client,
        }

        agent_types, n_commit = self._agent_types_for_run(seed, p, base_type, committed_type)

        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            n_names=self.config.n_names,
            history_length=self.config.history_length,
            temperature=self.config.temperature,
            prompt_variant=condition,
            use_nonce_tokens=False,
            store_raw_responses=self.config.store_raw_responses,
            verbosity=self.config.verbosity,
            run_id=run_id,
            experiment="tipping",
            variant_id=f"{condition}_p{_p_tag(p)}",
            seed=seed,
            repeat_idx=run_idx,
        )

        rng = random.Random(seed)
        result = run_population_game(
            agent_types=agent_types,
            clients=clients,
            config=engine_config,
            trials_writer=trials_writer,
            run_path=run_path,
            rng=rng,
        )

        # Attach run-level notes for debugging/analysis
        if hasattr(result, "notes") and isinstance(result.notes, dict):
            result.notes.update({"p": p, "n_committed": n_commit, "target_label": self.config.target_label})
        return result

    def run_fraction_condition(self, condition: str, p: float):
        cfg = self.config

        # directory naming consistent with your style (experiment / model / variant / seed)
        base_model = getattr(self.client, "model", cfg.__dict__.get("model", "unknown"))
        variant = f"{condition}_p{_p_tag(p)}_target{cfg.target_label}"

        run_id, run_path = init_run_dir(
            cfg.outdir,
            experiment="tipping",
            model=base_model,
            variant=variant,
            seed=cfg.seed,
        )

        config_dict = {
            "experiment": "tipping",
            "base_model": base_model,
            "condition": condition,
            "fraction_committed": p,
            "target_label": cfg.target_label,
            "tip_threshold": cfg.tip_threshold,
            "n_agents": cfg.n_agents,
            "n_rounds": cfg.n_rounds,
            "n_runs": cfg.n_runs,
            "history_length": cfg.history_length,
            "n_names": cfg.n_names,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        }
        config_digest = dump_resolved_config(run_path, config_dict)

        results = []
        with open_trials_writer(run_path) as trials_writer:
            for run_idx in range(cfg.n_runs):
                seed = cfg.seed + run_idx
                if cfg.verbosity >= 1:
                    logger.info(f"  Run {run_idx+1}/{cfg.n_runs} | cond={condition} p={p:g} seed={seed}")
                t0 = time.time()
                res = self.run_single(
                    condition=condition,
                    p=p,
                    seed=seed,
                    run_idx=run_idx,
                    trials_writer=trials_writer,
                    run_path=run_path,
                    run_id=run_id,
                )
                results.append(res)
                if cfg.verbosity >= 1:
                    elapsed = time.time() - t0
                    avg = float(np.mean(res.per_round_success))
                    fin = float(np.mean(res.per_round_success[-10:])) if len(res.per_round_success) >= 10 else avg
                    logger.info(f"    DONE {elapsed:.1f}s | avg={avg:.3f} final10={fin:.3f}")

        # summary via shared helper
        engine_config = EngineConfig(
            n_agents=cfg.n_agents,
            n_rounds=cfg.n_rounds,
            run_id=run_id,
            experiment="tipping",
            variant_id=f"{condition}_p{_p_tag(p)}",
        )
        summary = compute_run_summary(results, engine_config, config_digest)

        # Add tipping-specific success metric (based on run_summary info if present)
        # We try to infer dominant label/share from summary; if your compute_run_summary provides different fields,
        # you still have trials.csv to compute it in analysis.
        notes = summary.get("notes", {}) if isinstance(summary.get("notes", {}), dict) else {}
        notes.update({
            "fraction_committed": p,
            "target_label": cfg.target_label,
            "tip_threshold": cfg.tip_threshold,
        })
        summary["notes"] = notes

        write_run_summary(run_path, summary)

        if cfg.verbosity >= 1:
            logger.info(f"  Saved to: {run_path}")

        return summary

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        ensure_dir(cfg.outdir)

        if cfg.verbosity >= 1:
            logger.info("=" * 60)
            logger.info("TIPPING EXPERIMENT")
            logger.info("=" * 60)
            logger.info(f"Base model: {getattr(self.client, 'model', 'unknown')}")
            logger.info(f"Fractions: {cfg.fractions}")
            logger.info(f"Target: {cfg.target_label} | Tip threshold: {cfg.tip_threshold}")
            logger.info(f"Agents={cfg.n_agents} Rounds={cfg.n_rounds} Runs={cfg.n_runs} H={cfg.history_length} T={cfg.temperature}")
            logger.info(f"Conditions: {cfg.conditions}")
            logger.info("-" * 60)

        summaries = []
        for condition in cfg.conditions:
            for p in cfg.fractions:
                if cfg.verbosity >= 1:
                    logger.info("")
                    logger.info(f"[cond={condition}] [p={p:g}]")
                summaries.append(self.run_fraction_condition(condition, p))

        return {
            "experiment": "tipping",
            "n_summaries": len(summaries),
            "summaries": summaries,
        }
