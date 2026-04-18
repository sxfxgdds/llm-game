#!/usr/bin/env python3
"""
Run LLM Naming Game ablation experiments.

Examples:
    # Run with OpenAI
    python -m src.scripts.run_naming_game --preset gpt4o-mini --runs 10 --rounds 200

    # Run with HuggingFace model
    python -m src.scripts.run_naming_game --preset qwen7b --runs 5 --rounds 100

    # Run specific conditions
    python -m src.scripts.run_naming_game --preset gpt4o-mini --conditions scored structure_only
    
    # Run with detailed per-interaction logging (shows shuffled order + choices)
    python -m src.scripts.run_naming_game --preset gpt4o-mini -vv --rounds 5
"""

import argparse
import os
import sys
import time
import logging

# Add parent to path for imports
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


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Naming Game ablation experiments"
    )
    
    # Model selection
    parser.add_argument(
        "--preset", 
        type=str, 
        default=None,
        help="Model preset (e.g., gpt4o-mini, qwen7b)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "hf", "gemini", "gguf"],
        default=None,
        help="LLM backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path"
    )
    
    # Experiment parameters
    parser.add_argument("--agents", "-N", type=int, default=24, help="Number of agents")
    parser.add_argument("--rounds", "-R", type=int, default=200, help="Rounds per run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per condition")
    parser.add_argument("--history", "-H", type=int, default=3, help="History length")
    parser.add_argument("--names", "-W", type=int, default=10, help="Number of allowed names")
    
    # LLM parameters
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens")
    
    # Conditions
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["scored", "structure_only", "no_score_in_history", "no_score_no_goal"],
        help="Ablation conditions to run"
    )
    
    # Output
    parser.add_argument("--outdir", type=str, default="results/naming_game", help="Output directory")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "-v", "--verbose", 
        action="count", 
        default=1, 
        help="Verbosity level: -v for progress, -vv for per-interaction details (shuffled order + choices)"
    )
    
    args = parser.parse_args()
    
    # Validate model selection
    if not args.preset and not (args.backend and args.model):
        parser.error("Must provide --preset OR both --backend and --model")
    
    logger.info("=" * 60)
    logger.info("LLM NAMING GAME EXPERIMENT")
    logger.info("=" * 60)
    
    # Import after argument parsing
    from src.llms.presets import create_client
    from src.experiments.naming_game import NamingGameExperiment, NamingGameConfig
    
    # Create client
    logger.info(f"Loading model: {args.preset or args.model}...")
    start_time = time.time()
    
    if args.preset:
        client = create_client(
            preset=args.preset,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    else:
        client = create_client(
            backend=args.backend,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    
    logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
    
    # Create config
    config = NamingGameConfig(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_runs=args.runs,
        history_length=args.history,
        n_names=args.names,
        conditions=args.conditions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        outdir=args.outdir,
        seed=args.seed,
        verbosity=args.verbose,
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Agents: {args.agents}, Rounds: {args.rounds}, Runs: {args.runs}")
    logger.info(f"  Conditions: {args.conditions}")
    logger.info(f"  Temperature: {args.temperature}, History: {args.history}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {args.outdir}")
    logger.info("-" * 60)
    
    # Run experiment
    total_start = time.time()
    experiment = NamingGameExperiment(client, config)
    results = experiment.run_all()
    total_elapsed = time.time() - total_start
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {args.outdir}")
    logger.info("")
    logger.info("Summary:")
    
    # Print summary
    for condition, runs in results.items():
        import numpy as np
        avg_success = np.mean([np.mean(r.per_round_success) for r in runs])
        std_success = np.std([np.mean(r.per_round_success) for r in runs])
        logger.info(f"  {condition}: {avg_success:.3f} ± {std_success:.3f}")


if __name__ == "__main__":
    main()
