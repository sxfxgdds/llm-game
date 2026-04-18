#!/usr/bin/env python3
"""
Run tipping (committed minority) experiments for the naming game.

Example:
python -m src.scripts.run_tipping \
  --preset gpt4o-mini \
  --agents 24 --rounds 200 --runs 10 \
  --history 6 --names 10 \
  --temperature 0.7 \
  --target-label w3 \
  --fractions 0,0.02,0.05,0.08,0.1,0.12,0.15,0.2 \
  --conditions scored structure_only no_score_in_history no_score_no_goal \
  --seed 12345 \
  --outdir results/tipping/gpt4o-mini
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _parse_fractions(s: str):
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def main():
    ap = argparse.ArgumentParser(description="Run tipping experiment (committed minority)")

    ap.add_argument("--preset", type=str, required=True)

    ap.add_argument("--agents", "-N", type=int, default=24)
    ap.add_argument("--rounds", "-R", type=int, default=200)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--history", "-H", type=int, default=3)
    ap.add_argument("--names", "-W", type=int, default=10)

    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=8)

    ap.add_argument("--target-label", type=str, default="w0")
    ap.add_argument("--fractions", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--tip-threshold", type=float, default=0.9)

    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["scored", "structure_only", "no_score_in_history", "no_score_no_goal"],
    )

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, default="results/tipping")
    ap.add_argument("-v", "--verbose", action="count", default=1)

    args = ap.parse_args()

    from src.llms.presets import create_client, resolve_preset
    from src.experiments.tipping import TippingExperiment, TippingConfig

    _, model_id = resolve_preset(args.preset)

    # Build descriptive outdir:
    #   {base}/{preset}/H{history}/T{temp}
    def _ftag(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}".replace(".", "p")

    outdir = os.path.join(
        args.outdir,
        args.preset,
        f"H{args.history}",
        f"T{_ftag(args.temperature)}",
    )

    logger.info("=" * 60)
    logger.info("TIPPING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Preset: {args.preset} | Model: {model_id}")
    logger.info(f"Agents={args.agents} Rounds={args.rounds} Runs={args.runs} H={args.history} W={args.names}")
    logger.info(f"T={args.temperature} | Target={args.target_label} | Tip={args.tip_threshold}")
    logger.info(f"Fractions={args.fractions}")
    logger.info(f"Conditions={args.conditions}")
    logger.info(f"Seed={args.seed}")
    logger.info(f"Out={outdir}")
    logger.info("-" * 60)

    client = create_client(
        preset=args.preset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    cfg = TippingConfig(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_runs=args.runs,
        history_length=args.history,
        n_names=args.names,
        conditions=args.conditions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        fractions=_parse_fractions(args.fractions),
        target_label=args.target_label,
        tip_threshold=args.tip_threshold,
        outdir=outdir,
        seed=args.seed,
        verbosity=args.verbose,
    )

    t0 = time.time()
    exp = TippingExperiment(client, cfg)
    out = exp.run()
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
