#!/usr/bin/env python3
"""
Run logprob-based beta calibration experiments.

Measures LLM score sensitivity via logprob margins instead of sampled
choices, avoiding the saturation problem of the choice-based approach.

Hard-codes:
    beta_method  = "alt_mode"
    logprobs     = True
    top_logprobs = 5
    max_tokens   = 1

Examples:
    python -m src.scripts.run_beta_cal_logprobs \\
        --preset gpt41-mini \\
        --temps 0.1,0.2,0.3,0.4,0.5,0.7,1.0 \\
        --balanced --reps-per-score 30 \\
        --score-low 0 --score-high 10 \\
        --seed 12345 \\
        --outdir results/beta_calibration_logprobs/openai

    python -m src.scripts.run_beta_cal_logprobs \\
        --preset gpt41-nano \\
        --temps 0.1,0.5,1.0 \\
        --balanced --reps-per-score 20 \\
        --seed 42
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP-level logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_temps(s: str):
    """Parse comma-separated temperature list."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run logprob-based beta calibration experiments",
    )

    # Model selection
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)

    # Temperature sweep
    parser.add_argument(
        "--temps",
        type=str,
        default="0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated temperatures",
    )

    # Trial design
    parser.add_argument(
        "--balanced", action="store_true", help="Use balanced factorial design",
    )
    parser.add_argument(
        "--reps-per-score", type=int, default=30,
        help="Reps per score level (balanced mode)",
    )
    parser.add_argument("--score-low", type=int, default=0, help="Minimum score")
    parser.add_argument("--score-high", type=int, default=10, help="Maximum score")
    parser.add_argument(
        "--names", "-W", type=int, default=10,
        help="Vocabulary pool size (single-letter labels A..Z)",
    )

    # Output & reproducibility
    parser.add_argument(
        "--outdir", type=str, default="results/beta_calibration_logprobs",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("-v", "--verbose", action="count", default=1)

    args = parser.parse_args()

    if not args.preset and not (args.backend and args.model):
        parser.error("Must provide --preset OR both --backend and --model")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("LOGPROB-BASED BETA CALIBRATION")
    logger.info("=" * 60)

    from src.llms.presets import create_client
    from src.experiments.beta_calibration_logprobs import (
        BetaCalLogprobsExperiment,
        BetaCalLogprobsConfig,
    )

    temps = parse_temps(args.temps)

    logger.info(f"Loading model: {args.preset or args.model} …")
    start_time = time.time()

    # Create client with logprobs ENABLED, single-token output
    client_kwargs = dict(
        temperature=temps[0],
        seed=args.seed,
        logprobs=True,
        top_logprobs=5,
        max_tokens=1,
    )
    if args.preset:
        client = create_client(preset=args.preset, **client_kwargs)
    else:
        client = create_client(
            backend=args.backend, model=args.model, **client_kwargs,
        )

    logger.info(f"Model loaded in {time.time() - start_time:.1f}s")

    config = BetaCalLogprobsConfig(
        temperatures=temps,
        balanced=args.balanced,
        reps_per_score=args.reps_per_score,
        score_low=args.score_low,
        score_high=args.score_high,
        n_names=args.names,
        outdir=args.outdir,
        seed=args.seed,
        verbosity=args.verbose,
    )

    logger.info("Configuration:")
    logger.info(f"  Method:       alt_mode + logprob margin")
    logger.info(f"  Temperatures: {temps}")
    logger.info(f"  Design:       {'balanced' if args.balanced else 'random'}")
    logger.info(f"  Score range:  [{args.score_low}, {args.score_high}]")
    logger.info(f"  Vocab pool:   {args.names} tokens")
    logger.info(f"  Logprobs:     True  (top_logprobs=5, max_tokens=1)")
    logger.info(f"  Seed:         {args.seed}")
    logger.info(f"  Output:       {args.outdir}")
    logger.info("-" * 60)

    total_start = time.time()
    experiment = BetaCalLogprobsExperiment(client, config)
    summary = experiment.run()
    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed / 60:.1f} minutes")
    logger.info(f"Results saved to: {args.outdir}")

    notes = summary.get("notes", {})
    mean_tb = notes.get("mean_tau_b", 0)
    cv_tb = notes.get("cv_tau_b", 0)
    cov = notes.get("overall_coverage", 0)

    logger.info(
        f"Invariance: mean(τ·b_τ) = {mean_tb:.4f},  CV = {cv_tb:.3f}"
    )
    logger.info(f"Coverage:   {cov:.3f}")


if __name__ == "__main__":
    main()
