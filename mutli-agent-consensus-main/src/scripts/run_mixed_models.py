#!/usr/bin/env python3
"""
Run mixed-model (heterogeneous cohort) naming game experiments.

Examples:
    # Run with config file
    python -m src.scripts.run_mixed_models --config src/config/mixed_models.yaml
    
    # Quick CLI run with two models
    python -m src.scripts.run_mixed_models \\
        --presets gpt4o-mini,gemini-flash \\
        --fractions 0.5,0.5 \\
        --agents 24 --rounds 100 --runs 5
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def parse_composition(presets_str: str, fractions_str: str):
    """Parse preset and fraction strings into composition list."""
    presets = [p.strip() for p in presets_str.split(",")]
    fractions = [float(f.strip()) for f in fractions_str.split(",")]
    
    if len(presets) != len(fractions):
        raise ValueError(f"Number of presets ({len(presets)}) must match fractions ({len(fractions)})")
    
    total = sum(fractions)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")
    
    return [{"preset": p, "fraction": f} for p, f in zip(presets, fractions)]


def main():
    parser = argparse.ArgumentParser(
        description="Run mixed-model naming game experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # CLI composition options
    parser.add_argument("--presets", type=str, 
                       help="Comma-separated preset names (e.g., gpt4o-mini,gemini-flash)")
    parser.add_argument("--fractions", type=str,
                       help="Comma-separated fractions (e.g., 0.5,0.5)")
    
    # Game parameters
    parser.add_argument("--agents", type=int, default=24, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=200, help="Rounds per run")
    parser.add_argument("--names", type=int, default=10, help="Number of name tokens")
    parser.add_argument("--history", type=int, default=6, help="History length H")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--condition", type=str, default="scored",
                       help="Prompt condition (scored, structure_only, no_score, etc.)")
    
    # Run configuration
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="results/mixed_models",
                       help="Output directory")
    
    # Logging
    parser.add_argument("-v", "--verbose", action="count", default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MIXED-MODEL NAMING GAME EXPERIMENT")
    logger.info("=" * 60)
    
    from src.experiments.mixed_models import (
        MixedModelsExperiment, MixedModelsConfig, CohortComposition
    )
    
    # Load config from file or CLI
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            
            # Extract composition
            composition = [
                CohortComposition(preset=c["preset"], fraction=c["fraction"])
                for c in cfg["cohort"]["composition"]
            ]
            
            config = MixedModelsConfig(
                composition=composition,
                n_agents=cfg["game"].get("N", 24),
                n_rounds=cfg["game"].get("rounds", 200),
                n_names=cfg["game"].get("W", 10),
                history_length=cfg["game"].get("H", 6),
                temperature=cfg["game"].get("temperature", 1.0),
                prompt_variant=cfg["game"].get("prompt_variant", "scored"),
                seeds=cfg["run"].get("seeds", list(range(10))),
                repeats_per_seed=cfg["run"].get("repeats_per_seed", 1),
                outdir=cfg["run"].get("out_dir", "results/mixed_models"),
                store_raw_responses=cfg.get("logging", {}).get("store_raw_responses", False),
                verbosity=args.verbose,
            )
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            sys.exit(1)
    
    elif args.presets and args.fractions:
        logger.info("Using CLI composition...")
        composition_list = parse_composition(args.presets, args.fractions)
        composition = [
            CohortComposition(preset=c["preset"], fraction=c["fraction"])
            for c in composition_list
        ]
        
        config = MixedModelsConfig(
            composition=composition,
            n_agents=args.agents,
            n_rounds=args.rounds,
            n_names=args.names,
            history_length=args.history,
            temperature=args.temperature,
            prompt_variant=args.condition,
            seeds=list(range(args.seed, args.seed + args.runs)),
            outdir=args.outdir,
            verbosity=args.verbose,
        )
    
    else:
        parser.error("Must provide --config OR both --presets and --fractions")
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Composition: {[(c.preset, c.fraction) for c in config.composition]}")
    logger.info(f"  Agents: {config.n_agents}, Rounds: {config.n_rounds}")
    logger.info(f"  Temperature: {config.temperature}, Condition: {config.prompt_variant}")
    logger.info(f"  Seeds: {config.seeds}")
    logger.info(f"  Output: {config.outdir}")
    logger.info("-" * 60)
    
    # Run experiment
    total_start = time.time()
    experiment = MixedModelsExperiment(config)
    summary = experiment.run()
    total_elapsed = time.time() - total_start
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Final match rate: {summary['final_match_rate']:.3f} ± {summary.get('final_match_rate_std', 0):.3f}")
    
    if summary.get('within_type_match_final') is not None:
        logger.info(f"Within-type match: {summary['within_type_match_final']:.3f}")
        logger.info(f"Cross-type match: {summary['cross_type_match_final']:.3f}")
    
    logger.info(f"Results saved to: {experiment.run_path}")


if __name__ == "__main__":
    main()
