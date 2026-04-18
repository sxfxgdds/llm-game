#!/usr/bin/env python3
"""
Run shock experiments.

Tests LLM adaptability when reward structure flips mid-run.

Examples:
    python -m src.scripts.run_shock --preset gpt4o-mini --shock-at 0.5
    
    python -m src.scripts.run_shock --preset qwen7b --flip-goal --runs 10
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


def main():
    parser = argparse.ArgumentParser(
        description="Run shock experiments"
    )
    
    # Model
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    
    # Experiment
    parser.add_argument("--agents", "-N", type=int, default=24)
    parser.add_argument("--rounds", "-R", type=int, default=200)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--history", "-H", type=int, default=3)
    parser.add_argument("--names", "-W", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    # Shock settings
    parser.add_argument("--shock-at", type=float, default=0.5, 
                       help="Shock timing (fraction 0-1 or absolute round)")
    parser.add_argument("--flip-goal", action="store_true",
                       help="Flip goal text post-shock")
    
    # Conditions
    parser.add_argument("--conditions", nargs="+", 
                       default=["scored", "structure_only"])
    
    # Re-coordination metrics
    parser.add_argument("--thresh", type=float, default=0.8)
    parser.add_argument("--streak", type=int, default=3)
    
    # Output
    parser.add_argument("--outdir", type=str, default="results/shock")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    
    args = parser.parse_args()
    
    if args.names < 2:
        parser.error("Must have at least 2 names")
    
    if not args.preset and not (args.backend and args.model):
        parser.error("Must provide --preset OR both --backend and --model")
    
    logger.info("=" * 60)
    logger.info("SHOCK EXPERIMENT")
    logger.info("=" * 60)
    
    from src.llms.presets import create_client
    from src.experiments.shock import ShockExperiment, ShockConfig
    
    logger.info(f"Loading model: {args.preset or args.model}...")
    start_time = time.time()
    
    if args.preset:
        client = create_client(preset=args.preset, seed=args.seed)
    else:
        client = create_client(backend=args.backend, model=args.model, seed=args.seed)
    
    logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
    
    # Calculate shock round for display / path
    if args.shock_at < 1:
        shock_round = int(args.rounds * args.shock_at)
    else:
        shock_round = int(args.shock_at)
    
    # Build descriptive outdir that encodes key parameters:
    #   {base}/H{history}/T{temp}/S{shock_round}_{flip}/thr{thresh}_k{streak}
    def _ftag(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}".replace(".", "p")
    
    preset_tag = args.preset or args.model
    flip_tag = "flip" if args.flip_goal else "noflip"
    outdir = os.path.join(
        args.outdir,
        preset_tag,
        f"H{args.history}",
        f"T{_ftag(args.temperature)}",
        f"S{shock_round}_{flip_tag}",
        f"thr{_ftag(args.thresh)}_k{args.streak}",
    )
    
    config = ShockConfig(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_runs=args.runs,
        history_length=args.history,
        n_names=args.names,
        temperature=args.temperature,
        shock_at=args.shock_at,
        flip_goal=args.flip_goal,
        conditions=args.conditions,
        recoord_threshold=args.thresh,
        recoord_streak=args.streak,
        outdir=outdir,
        seed=args.seed,
        verbosity=args.verbose,
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Agents: {args.agents}, Rounds: {args.rounds}, Runs: {args.runs}")
    logger.info(f"  History: {args.history}, Temperature: {args.temperature}")
    logger.info(f"  Shock at round: {shock_round} (flip_goal={args.flip_goal})")
    logger.info(f"  Recoord threshold: {args.thresh}, streak: {args.streak}")
    logger.info(f"  Conditions: {args.conditions}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {outdir}")
    logger.info("-" * 60)
    
    total_start = time.time()
    experiment = ShockExperiment(client, config)
    results = experiment.run_all()
    total_elapsed = time.time() - total_start
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
