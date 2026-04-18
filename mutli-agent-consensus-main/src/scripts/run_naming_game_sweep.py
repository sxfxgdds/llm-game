#!/usr/bin/env python3
"""
Run naming-game sweeps over temperatures x histories, for:
  (A) each model individually (homogeneous)
  (B) optionally one mixed-model cohort at the end

Examples:

Single model sweep:
python -m src.scripts.run_naming_game_sweep \
  --preset gemini-2-flash \
  --conditions scored structure_only no_score_in_history no_score_no_goal \
  --agents 24 --rounds 50 --runs 10 \
  --histories 0,1,3,10 \
  --temperatures 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --names 10 \
  --seed 12345 \
  --outdir results/naming_game_sweeps

Many models sequentially + mixed after:
python -m src.scripts.run_naming_game_sweep \
  --presets gemini-flash gemini-pro gemini-2-flash gpt4o-mini gpt4o gpt41-nano gpt41-mini \
  --conditions scored structure_only no_score_in_history no_score_no_goal \
  --agents 24 --rounds 50 --runs 10 \
  --histories 0,1,3,10 \
  --temperatures 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --names 10 \
  --seed 12345 \
  --outdir results/naming_game_sweeps \
  --also-mixed
"""

import argparse
import os
import sys
import time
import logging
from typing import List

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


def _parse_floats_csv(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_ints_csv(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _safe_tag_float(x: float) -> str:
    # T0p3 instead of T0.3 for filesystem safety
    s = f"{x:.6g}"
    return s.replace(".", "p")


def main():
    ap = argparse.ArgumentParser(description="Run naming-game sweeps over temps x histories")

    # Model selection
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--presets", nargs="+", default=None, help="Run each preset sequentially")
    ap.add_argument("--also-mixed", action="store_true", help="After individual models, run one mixed-model sweep")
    ap.add_argument("--mixed-only", action="store_true", help="Run only the mixed-model sweep (requires --presets)")

    # Experiment
    ap.add_argument("--agents", "-N", type=int, default=24)
    ap.add_argument("--rounds", "-R", type=int, default=200)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--names", "-W", type=int, default=10)

    # Sweeps
    ap.add_argument("--histories", type=str, default="3", help="Comma-separated history lengths (e.g., 0,1,3,10)")
    ap.add_argument("--temperatures", type=str, default="0.3", help="Comma-separated temperatures (e.g., 0,0.2,0.5,1.0)")

    # LLM
    ap.add_argument("--max-tokens", type=int, default=8)

    # Conditions
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["scored", "structure_only", "no_score_in_history", "no_score_no_goal"],
    )

    # Output
    ap.add_argument("--outdir", type=str, default="results/naming_game_sweeps")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("-v", "--verbose", action="count", default=1)

    args = ap.parse_args()

    if args.mixed_only and not args.presets:
        ap.error("--mixed-only requires --presets")

    if not args.preset and not args.presets:
        ap.error("Provide --preset for one model or --presets for many models")

    histories = _parse_ints_csv(args.histories)
    temps = _parse_floats_csv(args.temperatures)

    from src.llms.presets import create_client, resolve_preset
    from src.experiments.naming_game import NamingGameExperiment, NamingGameConfig
    from src.experiments.mixed_naming_game import MixedNamingGameExperiment, MixedNamingGameConfig

    ensure_base = args.outdir
    os.makedirs(ensure_base, exist_ok=True)

    # Build preset list
    preset_list = []
    if args.preset:
        preset_list = [args.preset]
    if args.presets:
        preset_list = args.presets

    # Resolve model IDs for mixed
    resolved_model_ids = []
    for p in preset_list:
        _, mid = resolve_preset(p)
        resolved_model_ids.append(mid)

    def run_one_homogeneous(preset: str):
        backend, model_id = resolve_preset(preset)

        logger.info("=" * 70)
        logger.info(f"SWEEP (HOMOGENEOUS) | preset={preset} | model_id={model_id}")
        logger.info("=" * 70)

        # Create client ONCE per model (HF load is expensive)
        client = create_client(
            preset=preset,
            temperature=temps[0] if temps else 0.3,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        for H in histories:
            for T in temps:
                # Use a hierarchical outdir to keep runs separated and easy to aggregate
                combo_out = os.path.join(
                    ensure_base,
                    "single",
                    preset,
                    f"H{H}",
                    f"T{_safe_tag_float(T)}",
                )
                os.makedirs(combo_out, exist_ok=True)

                cfg = NamingGameConfig(
                    n_agents=args.agents,
                    n_rounds=args.rounds,
                    n_runs=args.runs,
                    history_length=H,
                    n_names=args.names,
                    conditions=args.conditions,
                    temperature=T,
                    max_tokens=args.max_tokens,
                    outdir=combo_out,
                    seed=args.seed,
                    verbosity=args.verbose,
                )
                # Important: this string is used in init_run_dir naming inside NamingGameExperiment
                cfg.model = model_id

                logger.info(f"[run] preset={preset} | H={H} | T={T} | out={combo_out}")
                exp = NamingGameExperiment(client, cfg)
                exp.run_all()

    def run_one_mixed(presets: List[str]):
        logger.info("=" * 70)
        logger.info(f"SWEEP (MIXED) | presets={presets}")
        logger.info("=" * 70)

        # Create one client per model_id
        clients = {}
        model_names = []
        for p in presets:
            _, mid = resolve_preset(p)
            model_names.append(mid)
            clients[mid] = create_client(
                preset=p,
                temperature=temps[0] if temps else 0.3,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )

        for H in histories:
            for T in temps:
                combo_out = os.path.join(
                    ensure_base,
                    "mixed",
                    "+".join(presets),
                    f"H{H}",
                    f"T{_safe_tag_float(T)}",
                )
                os.makedirs(combo_out, exist_ok=True)

                cfg = MixedNamingGameConfig(
                    n_agents=args.agents,
                    n_rounds=args.rounds,
                    n_runs=args.runs,
                    history_length=H,
                    n_names=args.names,
                    conditions=args.conditions,
                    temperature=T,
                    max_tokens=args.max_tokens,
                    outdir=combo_out,
                    seed=args.seed,
                    verbosity=args.verbose,
                    model_names=model_names,
                )

                logger.info(f"[run] MIXED presets={presets} | H={H} | T={T} | out={combo_out}")
                exp = MixedNamingGameExperiment(clients, cfg)
                exp.run_all()

    t0 = time.time()

    if not args.mixed_only:
        for p in preset_list:
            run_one_homogeneous(p)

    if args.mixed_only or args.also_mixed:
        # Mixed is only meaningful for 2+ presets
        if len(preset_list) >= 2:
            run_one_mixed(preset_list)
        else:
            logger.warning("Skipping mixed sweep (need at least 2 presets).")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results under: {ensure_base}")


if __name__ == "__main__":
    main()
