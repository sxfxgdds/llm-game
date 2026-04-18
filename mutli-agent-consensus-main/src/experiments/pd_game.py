"""
LLM Naming Game experiment implementation.

Runs multi-agent naming game with various ablation conditions.
Uses the shared engine for consistency with other experiments.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

from ..core.state import RunResult
from ..core.prompts import ABLATION_CONDITIONS
from ..core.utils import ensure_dir, set_global_seed, make_simple_tokens
from ..core.engine import EngineConfig, run_population_game, compute_run_summary
from ..core.io import (
    init_run_dir, dump_resolved_config, open_trials_writer,
    write_run_summary, finalize_run
)

logger = logging.getLogger(__name__)


@dataclass
class NamingGameConfig:
    """Configuration for naming game experiment."""
    # Agent parameters
    n_agents: int = 24
    n_rounds: int = 200
    n_runs: int = 10
    history_length: int = 3
    
    # Name vocabulary
    n_names: int = 10
    use_nonce_names: bool = False
    nonce_seed: int = 0
    
    # Conditions to run
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_CONDITIONS.keys()))
    
    # LLM settings (if not using client directly)
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 8
    
    # Output
    outdir: str = "results/naming_game"
    
    # Reproducibility
    seed: int = 12345
    
    # Logging
    store_raw_responses: bool = False
    
    # Verbosity (0=quiet, 1=normal, 2=detailed, 3=debug)
    verbosity: int = 1


class NamingGameExperiment:
    """
    LLM Naming Game experiment runner.
    
    Runs ablation experiments testing the effect of different
    prompt components on coordination success.
    
    Uses the shared engine (core.engine) to prevent code drift
    with other experiment types.
    """
    
    def __init__(
        self,
        client,
        config: Optional[NamingGameConfig] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            client: LLM client instance (from llms module)
            config: Experiment configuration
        """
        self.client = client
        self.config = config or NamingGameConfig()
        
        # Generate allowed names
        if self.config.use_nonce_names:
            from ..core.utils import make_nonce_tokens
            self.allowed_names = make_nonce_tokens(
                self.config.n_names, 
                seed=self.config.nonce_seed
            )
        else:
            self.allowed_names = make_simple_tokens(self.config.n_names)
        
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
        Run a single experiment trial using the shared engine.
        
        Args:
            condition: Ablation condition name
            seed: Random seed for this run
            run_idx: Index of this run
            trials_writer: Optional TrialsWriter for logging
            run_path: Path to run directory
            run_id: Run identifier
            
        Returns:
            RunResult with per-round metrics
        """
        import random
        
        set_global_seed(seed)
        
        # Update client seed for reproducibility
        if hasattr(self.client, 'set_seed'):
            self.client.set_seed(seed)
        
        # Build engine config
        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            n_names=self.config.n_names,
            history_length=self.config.history_length,
            temperature=self.config.temperature,
            prompt_variant=condition,
            use_nonce_tokens=self.config.use_nonce_names,
            store_raw_responses=self.config.store_raw_responses,
            verbosity=self.config.verbosity,
            run_id=run_id,
            experiment="naming_game",
            variant_id=condition,
            seed=seed,
            repeat_idx=run_idx,
        )
        
        # Homogeneous: all agents use the same client
        # Get model name from client
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
        """
        Run all trials for a single condition.
        
        Args:
            condition: Ablation condition name
            
        Returns:
            List of RunResult objects
        """
        import time
        
        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"CONDITION: {condition.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"  Runs: {self.config.n_runs}, Rounds: {self.config.n_rounds}, Agents: {self.config.n_agents}")
        
        # Initialize run directory with descriptive naming
        run_id, run_path = init_run_dir(
            self.config.outdir,
            experiment="naming_game",
            model=self.config.model,
            variant=condition,
            seed=self.config.seed,
        )
        
        # Dump config
        config_dict = {
            "experiment": "naming_game",
            "condition": condition,
            "n_agents": self.config.n_agents,
            "n_rounds": self.config.n_rounds,
            "n_runs": self.config.n_runs,
            "history_length": self.config.history_length,
            "n_names": self.config.n_names,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "model": self.config.model,
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
                    final_success = np.mean(result.per_round_success[-10:]) if len(result.per_round_success) >= 10 else avg_success
                    logger.info(f"  Run {run_idx + 1} DONE: {elapsed:.1f}s | avg={avg_success:.3f} final={final_success:.3f}")
        
        # Compute and write summary
        engine_config = EngineConfig(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            run_id=run_id,
            experiment="naming_game",
            variant_id=condition,
        )
        summary = compute_run_summary(results, engine_config, config_digest)
        write_run_summary(run_path, summary)
        
        if self.config.verbosity >= 1:
            # Condition summary
            all_avg = [np.mean(r.per_round_success) for r in results]
            all_final = [np.mean(r.per_round_success[-10:]) if len(r.per_round_success) >= 10 else np.mean(r.per_round_success) for r in results]
            logger.info("")
            logger.info(f"  CONDITION SUMMARY: {condition}")
            logger.info(f"    Overall:  {np.mean(all_avg):.3f} ± {np.std(all_avg):.3f}")
            logger.info(f"    Final 10: {np.mean(all_final):.3f} ± {np.std(all_final):.3f}")
            logger.info(f"    Saved to: {run_path}")
        
        return results
    
    def run_all(self) -> Dict[str, List[RunResult]]:
        """
        Run all configured conditions.
        
        Returns:
            Dictionary mapping condition names to lists of RunResult
        """
        all_results = {}
        
        for condition in self.config.conditions:
            results = self.run_condition(condition)
            all_results[condition] = results
        
        return all_results


def run_naming_game(
    client,
    n_agents: int = 24,
    n_rounds: int = 200,
    n_runs: int = 10,
    conditions: Optional[List[str]] = None,
    outdir: str = "results/naming_game",
    seed: int = 12345,
    verbosity: int = 1,
) -> Dict[str, List[RunResult]]:
    """
    Convenience function to run naming game experiment.
    
    Args:
        client: LLM client instance
        n_agents: Number of agents
        n_rounds: Rounds per run
        n_runs: Number of runs per condition
        conditions: List of conditions (default: all)
        outdir: Output directory
        seed: Random seed
        verbosity: Logging verbosity
        
    Returns:
        Dictionary of results by condition
    """
    config = NamingGameConfig(
        n_agents=n_agents,
        n_rounds=n_rounds,
        n_runs=n_runs,
        conditions=conditions or list(ABLATION_CONDITIONS.keys()),
        outdir=outdir,
        seed=seed,
        verbosity=verbosity,
    )
    
    experiment = NamingGameExperiment(client, config)
    return experiment.run_all()
