"""
Shock experiment for testing LLM adaptability.

Tests agent behavior when reward structure flips mid-run:
- Pre-shock: Match = reward
- Post-shock: Mismatch = reward (if flip_goal)

Uses the shared engine (core.engine) for consistency.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

from ..core.state import RunResult
from ..core.prompts import ABLATION_CONDITIONS
from ..core.metrics import time_to_recoord
from ..core.utils import ensure_dir, set_global_seed, make_simple_tokens
from ..core.engine import EngineConfig, run_population_game, compute_run_summary
from ..core.io import (
    init_run_dir, dump_resolved_config, open_trials_writer,
    write_run_summary, finalize_run
)

logger = logging.getLogger(__name__)


@dataclass
class ShockConfig:
    """Configuration for shock experiment."""
    n_agents: int = 24
    n_rounds: int = 200
    n_runs: int = 10
    history_length: int = 3
    
    # Shock timing (fraction in (0,1) or absolute round >=1)
    shock_at: float = 0.5
    
    # Goal flip
    flip_goal: bool = True
    
    # Vocabulary
    n_names: int = 10
    
    # LLM settings
    temperature: float = 1.0
    
    # Conditions
    conditions: List[str] = field(default_factory=lambda: ["scored", "structure_only"])
    
    # Re-coordination metrics
    recoord_threshold: float = 0.8
    recoord_streak: int = 3
    
    # Output
    outdir: str = "results/shock"
    
    # Logging
    store_raw_responses: bool = False
    
    # Reproducibility
    seed: int = 12345
    
    # Verbosity
    verbosity: int = 1


class ShockExperiment:
    """
    Shock experiment runner.
    
    Tests LLM agent adaptability when reward structure flips.
    Uses the shared engine for consistency with other experiments.
    """
    
    def __init__(
        self,
        client,
        config: Optional[ShockConfig] = None,
    ):
        self.client = client
        self.config = config or ShockConfig()
        self.allowed = make_simple_tokens(self.config.n_names)
        
        # Resolve shock round
        if self.config.shock_at < 1:
            self.shock_round = int(self.config.n_rounds * self.config.shock_at)
        else:
            self.shock_round = int(self.config.shock_at)
        self.shock_round = max(1, min(self.config.n_rounds - 1, self.shock_round))
        
        ensure_dir(self.config.outdir)
    
    def run_single(
        self,
        condition: str,
        seed: int,
        run_idx: int = 0,
        trials_writer=None,
        run_path: str = None,
        run_id: str = "",
    ) -> RunResult:
        """
        Run a single shock experiment trial using the shared engine.
        """
        import random
        
        set_global_seed(seed)
        
        # Update client seed for reproducibility
        if hasattr(self.client, 'set_seed'):
            self.client.set_seed(seed)
        
        # Build engine config with shock parameters
        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            n_names=self.config.n_names,
            history_length=self.config.history_length,
            temperature=self.config.temperature,
            prompt_variant=condition,
            shock_round=self.shock_round,
            flip_goal=self.config.flip_goal,
            store_raw_responses=self.config.store_raw_responses,
            verbosity=self.config.verbosity,
            run_id=run_id,
            experiment="shock",
            variant_id=condition,
            seed=seed,
            repeat_idx=run_idx,
        )
        
        # Homogeneous: all agents use the same client
        model_name = getattr(self.client, 'model', 'unknown')
        agent_types = [model_name] * self.config.n_agents
        clients = {model_name: self.client}
        
        # Run the shared engine
        rng = random.Random(seed)
        result = run_population_game(
            agent_types=agent_types,
            clients=clients,
            config=engine_config,
            trials_writer=trials_writer,
            run_path=run_path,
            rng=rng,
        )
        
        return result
    
    def run_condition(self, condition: str) -> List[RunResult]:
        """Run all trials for a condition."""
        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"SHOCK EXPERIMENT: {condition.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"  Shock at round {self.shock_round}/{self.config.n_rounds}")
            logger.info(f"  Runs: {self.config.n_runs}, Agents: {self.config.n_agents}")
        
        # Initialize run directory with descriptive naming
        model_name = getattr(self.client, 'model', 'unknown')
        run_id, run_path = init_run_dir(
            self.config.outdir,
            experiment="shock",
            model=model_name,
            variant=condition,
            seed=self.config.seed,
        )
        
        # Dump config
        config_dict = {
            "experiment": "shock",
            "condition": condition,
            "n_agents": self.config.n_agents,
            "n_rounds": self.config.n_rounds,
            "n_runs": self.config.n_runs,
            "history_length": self.config.history_length,
            "n_names": self.config.n_names,
            "temperature": self.config.temperature,
            "shock_round": self.shock_round,
            "flip_goal": self.config.flip_goal,
            "seed": self.config.seed,
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
                    pre_shock_avg = np.mean(result.per_round_success[:self.shock_round]) if self.shock_round > 0 else 0
                    post_shock_avg = np.mean(result.per_round_success[self.shock_round:]) if self.shock_round < len(result.per_round_success) else 0
                    logger.info(f"  Run {run_idx + 1} DONE: {elapsed:.1f}s | pre={pre_shock_avg:.3f} post={post_shock_avg:.3f}")
        
        # Compute recoordination time
        all_success = [r.per_round_success for r in results]
        mean_curve = np.mean(all_success, axis=0)
        
        recoord_time = time_to_recoord(
            mean_curve,
            self.shock_round,
            threshold=self.config.recoord_threshold,
            streak=self.config.recoord_streak,
            bin_size=self.config.n_agents,
        )
        
        # Compute summary with shock-specific stats
        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            run_id=run_id,
            experiment="shock",
            variant_id=condition,
        )
        summary = compute_run_summary(results, engine_config, config_digest)
        summary["shock_round"] = self.shock_round
        summary["flip_goal"] = self.config.flip_goal
        summary["recoord_time"] = recoord_time
        summary["recoord_threshold"] = self.config.recoord_threshold
        
        write_run_summary(run_path, summary)
        
        if self.config.verbosity >= 1:
            logger.info(f"  Re-coordination time: {recoord_time} rounds")
            logger.info(f"  Saved to: {run_path}")
        
        return results
    
    def run_all(self) -> Dict[str, List[RunResult]]:
        """Run shock experiment for all conditions."""
        all_results = {}
        
        for condition in self.config.conditions:
            results = self.run_condition(condition)
            all_results[condition] = results
        
        return all_results


def run_shock_experiment(
    client,
    n_agents: int = 24,
    n_rounds: int = 200,
    n_runs: int = 10,
    shock_at: float = 0.5,
    flip_goal: bool = True,
    conditions: Optional[List[str]] = None,
    outdir: str = "results/shock",
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Convenience function for shock experiment.
    
    Args:
        client: LLM client instance
        n_agents: Number of agents
        n_rounds: Rounds per run
        n_runs: Number of runs
        shock_at: Shock timing (fraction or round)
        flip_goal: Whether to flip goal text post-shock
        conditions: List of conditions
        outdir: Output directory
        verbosity: Logging verbosity
        
    Returns:
        Dictionary of results by condition
    """
    config = ShockConfig(
        n_agents=n_agents,
        n_rounds=n_rounds,
        n_runs=n_runs,
        shock_at=shock_at,
        flip_goal=flip_goal,
        conditions=conditions or ["scored", "structure_only"],
        outdir=outdir,
        verbosity=verbosity,
    )
    
    experiment = ShockExperiment(client, config)
    return experiment.run_all()
