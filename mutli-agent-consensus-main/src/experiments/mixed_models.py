"""
Mixed-model (heterogeneous cohort) naming game experiments.

This module runs naming game experiments with agents using different
LLM models, enabling analysis of within-type vs cross-type coordination.
"""

import os
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np

from ..core.state import RunResult
from ..core.pairing import uniform_model_assignment
from ..core.utils import set_global_seed, ensure_dir, get_short_model_name
from ..core.io import (
    init_run_dir, dump_resolved_config, open_trials_writer,
    write_run_summary, finalize_run
)
from ..core.engine import EngineConfig, run_population_game, compute_run_summary
from ..llms.presets import create_client, resolve_preset

logger = logging.getLogger(__name__)


@dataclass
class CohortComposition:
    """Specification for a model in the cohort."""
    preset: str
    fraction: float
    
    def __post_init__(self):
        if not 0 < self.fraction <= 1:
            raise ValueError(f"Fraction must be in (0, 1], got {self.fraction}")


@dataclass
class MixedModelsConfig:
    """Configuration for mixed-model experiments."""
    
    # Cohort composition
    composition: List[CohortComposition] = field(default_factory=list)
    
    # Game parameters
    n_agents: int = 24
    n_rounds: int = 200
    n_names: int = 10
    history_length: int = 6
    temperature: float = 1.0
    
    # Prompt configuration
    prompt_variant: str = "scored"
    use_nonce_tokens: bool = False
    
    # Run configuration
    n_runs: int = 10
    seeds: List[int] = field(default_factory=lambda: [12345])
    repeats_per_seed: int = 1
    
    # Output
    outdir: str = "results/mixed_models"
    run_name: Optional[str] = None
    store_raw_responses: bool = False
    verbosity: int = 1
    
    def __post_init__(self):
        # Validate fractions sum to 1
        if self.composition:
            total = sum(c.fraction for c in self.composition)
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Cohort fractions must sum to 1.0, got {total}")
    
    def get_presets(self) -> List[str]:
        """Get list of preset names used in composition."""
        return [c.preset for c in self.composition]


class MixedModelsExperiment:
    """
    Run heterogeneous cohort naming game experiments.
    
    Supports:
    - Multiple model types in same population
    - Within-type vs cross-type match analysis
    - MI(model_type ; chosen_token) tracking
    
    Example:
        config = MixedModelsConfig(
            composition=[
                CohortComposition(preset="gpt4o-mini", fraction=0.5),
                CohortComposition(preset="gemini-flash", fraction=0.5),
            ],
            n_agents=24,
            n_rounds=200,
        )
        experiment = MixedModelsExperiment(config)
        results = experiment.run()
    """
    
    def __init__(self, config: MixedModelsConfig):
        self.config = config
        self.clients: Dict[str, Any] = {}
        self.run_path: Optional[str] = None
        self.run_id: Optional[str] = None
    
    def _load_clients(self) -> None:
        """Initialize LLM clients for each preset in composition."""
        logger.info("Loading LLM clients...")
        
        for comp in self.config.composition:
            if comp.preset not in self.clients:
                logger.info(f"  Loading {comp.preset}...")
                t0 = time.time()
                self.clients[comp.preset] = create_client(
                    preset=comp.preset,
                    temperature=self.config.temperature,
                )
                logger.info(f"    Loaded in {time.time() - t0:.1f}s")
    
    def _assign_agents(self, seed: int) -> List[str]:
        """Assign model types to agents based on composition fractions."""
        rng = random.Random(seed)
        
        # Calculate target counts per type
        n = self.config.n_agents
        presets = [c.preset for c in self.config.composition]
        fractions = [c.fraction for c in self.config.composition]
        
        # Distribute agents
        counts = []
        remaining = n
        for i, frac in enumerate(fractions[:-1]):
            cnt = round(n * frac)
            counts.append(min(cnt, remaining))
            remaining -= counts[-1]
        counts.append(remaining)  # Last type gets remainder
        
        # Build assignment list
        assignment = []
        for preset, count in zip(presets, counts):
            assignment.extend([preset] * count)
        
        # Shuffle to randomize positions
        rng.shuffle(assignment)
        
        return assignment
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete mixed-model experiment.
        
        Returns:
            Summary dictionary with results and statistics
        """
        # Initialize
        self._load_clients()
        
        # Create run directory with descriptive naming
        # Build model string from cohort composition (e.g., "gpt4o-mini+gemini-flash")
        model_str = "+".join(c.preset for c in self.config.composition)
        
        self.run_id, self.run_path = init_run_dir(
            self.config.outdir,
            experiment="mixed_models",
            run_name=self.config.run_name,
            model=model_str,
            variant=self.config.prompt_variant,
            seed=self.config.seed,
        )
        
        # Save config
        config_dict = {
            "experiment": "mixed_models",
            "cohort": {
                "composition": [
                    {"preset": c.preset, "fraction": c.fraction}
                    for c in self.config.composition
                ]
            },
            "game": {
                "n_agents": self.config.n_agents,
                "n_rounds": self.config.n_rounds,
                "n_names": self.config.n_names,
                "history_length": self.config.history_length,
                "temperature": self.config.temperature,
                "prompt_variant": self.config.prompt_variant,
            },
            "run": {
                "seeds": self.config.seeds,
                "repeats_per_seed": self.config.repeats_per_seed,
            }
        }
        config_digest = dump_resolved_config(self.run_path, config_dict)
        
        if self.config.verbosity >= 1:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"MIXED MODELS EXPERIMENT")
            logger.info(f"{'='*60}")
            logger.info(f"  Run ID: {self.run_id}")
            logger.info(f"  Output: {self.run_path}")
            logger.info(f"  Composition: {[(c.preset, f'{c.fraction:.0%}') for c in self.config.composition]}")
            logger.info(f"  Agents: {self.config.n_agents}, Rounds: {self.config.n_rounds}")
        
        # Open trials writer
        trials_writer = open_trials_writer(self.run_path)
        
        # Run experiments
        all_results: List[RunResult] = []
        total_runs = len(self.config.seeds) * self.config.repeats_per_seed
        run_idx = 0
        
        for seed in self.config.seeds:
            for repeat_idx in range(self.config.repeats_per_seed):
                run_idx += 1
                if self.config.verbosity >= 1:
                    logger.info("")
                    logger.info(f"  --- Run {run_idx}/{total_runs} (seed={seed}, repeat={repeat_idx}) ---")
                
                # Set global seed
                set_global_seed(seed + repeat_idx * 1000)
                
                # Assign agents to models
                agent_types = self._assign_agents(seed + repeat_idx)
                
                # Log assignment summary
                if self.config.verbosity >= 2:
                    from collections import Counter
                    type_counts = Counter(agent_types)
                    logger.info(f"    Agent assignment: {dict(type_counts)}")
                
                # Update client seeds
                for client in self.clients.values():
                    if hasattr(client, 'set_seed'):
                        client.set_seed(seed + repeat_idx)
                
                # Create engine config
                engine_config = EngineConfig(
                    n_agents=self.config.n_agents,
                    n_rounds=self.config.n_rounds,
                    n_names=self.config.n_names,
                    history_length=self.config.history_length,
                    temperature=self.config.temperature,
                    prompt_variant=self.config.prompt_variant,
                    use_nonce_tokens=self.config.use_nonce_tokens,
                    store_raw_responses=self.config.store_raw_responses,
                    verbosity=self.config.verbosity,
                    run_id=self.run_id,
                    experiment="mixed_models",
                    variant_id=f"seed{seed}_rep{repeat_idx}",
                    seed=seed,
                    repeat_idx=repeat_idx,
                )
                
                # Run game
                t0 = time.time()
                result = run_population_game(
                    agent_types=agent_types,
                    clients=self.clients,
                    config=engine_config,
                    trials_writer=trials_writer,
                    run_path=self.run_path,
                )
                elapsed = time.time() - t0
                
                all_results.append(result)
                
                # Log progress
                if self.config.verbosity >= 1:
                    avg_rate = np.mean(result.per_round_success) if result.per_round_success else 0
                    final_rate = np.mean(result.per_round_success[-10:]) if len(result.per_round_success) >= 10 else avg_rate
                    within_str = f"{result.within_type_rate:.3f}" if result.within_type_rate else "N/A"
                    cross_str = f"{result.cross_type_rate:.3f}" if result.cross_type_rate else "N/A"
                    logger.info(f"  Run {run_idx} DONE: {elapsed:.1f}s | avg={avg_rate:.3f} final={final_rate:.3f} within={within_str} cross={cross_str}")
        
        # Compute summary
        summary = compute_run_summary(all_results, engine_config, config_digest)
        
        # Add cohort-specific stats
        summary["cohort_composition"] = [
            {"preset": c.preset, "fraction": c.fraction}
            for c in self.config.composition
        ]
        
        # Predicted ceiling from fractions (Σf²)
        predicted_ceiling = sum(c.fraction ** 2 for c in self.config.composition)
        summary["notes"]["predicted_random_match"] = predicted_ceiling
        
        # Finalize
        finalize_run(self.run_path, summary, trials_writer)
        
        logger.info(f"Experiment complete. Results saved to: {self.run_path}")
        
        return summary


def run_mixed_models(
    composition: List[Dict[str, Any]],
    n_agents: int = 24,
    n_rounds: int = 200,
    n_runs: int = 10,
    temperature: float = 1.0,
    outdir: str = "results/mixed_models",
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Convenience function for running mixed-model experiments.
    
    Args:
        composition: List of {"preset": str, "fraction": float} dicts
        n_agents: Number of agents
        n_rounds: Rounds per run
        n_runs: Number of runs
        temperature: Sampling temperature
        outdir: Output directory
        verbosity: Logging level
        
    Returns:
        Summary dictionary
    """
    config = MixedModelsConfig(
        composition=[
            CohortComposition(preset=c["preset"], fraction=c["fraction"])
            for c in composition
        ],
        n_agents=n_agents,
        n_rounds=n_rounds,
        temperature=temperature,
        seeds=list(range(n_runs)),
        outdir=outdir,
        verbosity=verbosity,
    )
    
    experiment = MixedModelsExperiment(config)
    return experiment.run()
