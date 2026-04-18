# src/experiments/mixed_naming_game.py

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from ..core.prompts import ABLATION_CONDITIONS
from ..core.utils import ensure_dir, set_global_seed, make_simple_tokens
from ..core.pairing import uniform_model_assignment
from ..core.engine import EngineConfig, run_population_game, compute_run_summary
from ..core.io import (
    init_run_dir, dump_resolved_config, open_trials_writer, write_run_summary
)

logger = logging.getLogger(__name__)


@dataclass
class MixedNamingGameConfig:
    # Agent parameters
    n_agents: int = 24
    n_rounds: int = 200
    n_runs: int = 10
    history_length: int = 3

    # Name vocabulary
    n_names: int = 10

    # Conditions to run
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_CONDITIONS.keys()))

    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 8

    # Mixed model settings
    model_names: List[str] = field(default_factory=list)  # resolved model IDs
    assignment_seed_offset: int = 777

    # Output
    outdir: str = "results/naming_game_mixed"

    # Reproducibility
    seed: int = 12345

    # Logging
    store_raw_responses: bool = False

    # Verbosity
    verbosity: int = 1


class MixedNamingGameExperiment:
    """
    Mixed-model naming game using shared engine.
    Each agent is assigned a model type; pairings are random per run.
    """

    def __init__(
        self,
        clients: Dict[str, Any],   # model_name -> client
        config: Optional[MixedNamingGameConfig] = None,
    ):
        self.clients = clients
        self.config = config or MixedNamingGameConfig()

        if not self.config.model_names:
            self.config.model_names = list(clients.keys())

        self.allowed_names = make_simple_tokens(self.config.n_names)
        ensure_dir(self.config.outdir)

    def _assign_models(self, seed: int) -> Tuple[List[str], List[int]]:
        """
        Uniform assignment of model indices to agents.
        Returns (agent_types, counts_per_model_index).
        """
        assignment, counts = uniform_model_assignment(
            self.config.n_agents,
            len(self.config.model_names),
            seed=seed + self.config.assignment_seed_offset,
        )
        agent_types = [self.config.model_names[i] for i in assignment]
        return agent_types, counts

    def run_single(
        self,
        condition: str,
        seed: int,
        run_idx: int,
        trials_writer,
        run_path: str,
        run_id: str,
    ):
        import random

        set_global_seed(seed)

        # Update client seeds (best-effort) for reproducibility
        for c in self.clients.values():
            if hasattr(c, "set_seed"):
                c.set_seed(seed)

        agent_types, _ = self._assign_models(seed)

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
            experiment="mixed_naming_game",
            variant_id=condition,
            seed=seed,
            repeat_idx=run_idx,
        )

        rng = random.Random(seed)
        result = run_population_game(
            agent_types=agent_types,
            clients=self.clients,
            config=engine_config,
            trials_writer=trials_writer,
            run_path=run_path,
            rng=rng,
        )
        return result

    def run_condition(self, condition: str):
        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"MIXED CONDITION: {condition.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"  Runs: {self.config.n_runs}, Rounds: {self.config.n_rounds}, Agents: {self.config.n_agents}")
            logger.info(f"  Models: {self.config.model_names}")

        model_str = "+".join(self.config.model_names) if len(self.config.model_names) <= 3 else f"{len(self.config.model_names)}models"

        run_id, run_path = init_run_dir(
            self.config.outdir,
            experiment="mixed_naming_game",
            model=model_str,
            variant=condition,
            seed=self.config.seed,
        )

        config_dict = {
            "experiment": "mixed_naming_game",
            "condition": condition,
            "n_agents": self.config.n_agents,
            "n_rounds": self.config.n_rounds,
            "n_runs": self.config.n_runs,
            "history_length": self.config.history_length,
            "n_names": self.config.n_names,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "models": self.config.model_names,
        }
        config_digest = dump_resolved_config(run_path, config_dict)

        results = []
        with open_trials_writer(run_path) as trials_writer:
            for run_idx in range(self.config.n_runs):
                seed = self.config.seed + run_idx
                if self.config.verbosity >= 1:
                    logger.info("")
                    logger.info(f"  --- Run {run_idx + 1}/{self.config.n_runs} (seed={seed}) ---")

                result = self.run_single(
                    condition=condition,
                    seed=seed,
                    run_idx=run_idx,
                    trials_writer=trials_writer,
                    run_path=run_path,
                    run_id=run_id,
                )
                results.append(result)

                if self.config.verbosity >= 1:
                    avg_success = float(np.mean(result.per_round_success))
                    final_success = float(np.mean(result.per_round_success[-10:])) if len(result.per_round_success) >= 10 else avg_success
                    logger.info(f"  Run {run_idx + 1} DONE | avg={avg_success:.3f} final={final_success:.3f}")

        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            run_id=run_id,
            experiment="mixed_naming_game",
            variant_id=condition,
        )
        summary = compute_run_summary(results, engine_config, config_digest)
        write_run_summary(run_path, summary)

        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"  MIXED CONDITION SUMMARY: {condition}")
            logger.info(f"    Saved to: {run_path}")

        return results

    def run_all(self):
        all_results = {}
        for condition in self.config.conditions:
            all_results[condition] = self.run_condition(condition)
        return all_results
