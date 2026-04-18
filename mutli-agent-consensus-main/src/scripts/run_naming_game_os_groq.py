#!/usr/bin/env python3
"""
Run OS naming game ablation experiments with Groq-capable backend options.

This runner supports Groq API models (and optionally HF), and uses
`NamingGameOSExperiment` from `naming_game_os_groq`.
"""

import argparse
import logging
import os
import sys
import time

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


def _split_csv_items(values):
    out = []
    for v in values:
        out.extend([x.strip() for x in str(v).split(",") if x.strip()])
    return out


def _dedup_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run OS-only naming game ablation experiments")

    parser.add_argument("--preset", type=str, default=None, help="Preset (HF or Groq)")
    parser.add_argument("--presets", nargs="+", default=None, help="One or more presets (space or comma separated)")
    parser.add_argument("--model", type=str, default=None, help="Model id/path for --backend")
    parser.add_argument("--models", nargs="+", default=None, help="One or more model ids/paths for --backend")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "groq"],
        default="hf",
        help="Backend used with --model/--models (ignored when using presets).",
    )

    parser.add_argument("--agents", "-N", type=int, default=24, help="Number of agents")
    parser.add_argument("--rounds", "-R", type=int, default=200, help="Rounds per run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per condition")
    parser.add_argument("--history", "-H", type=int, default=3, help="History length")
    parser.add_argument("--names", "-W", type=int, default=10, help="Number of allowed names")

    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens")
    parser.add_argument("--allowed-path", type=str, default=None, help="Optional custom label list path")
    parser.add_argument(
        "--round1-fullstring-argmax",
        action="store_true",
        help="Use full-string continuation scoring with greedy argmax on round 1 only.",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["scored", "structure_only", "no_score_in_history", "no_score_no_goal"],
        help="Ablation conditions to run",
    )

    parser.add_argument("--outdir", type=str, default="results/os/naming_game", help="Output directory")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Verbosity level")

    args = parser.parse_args()

    preset_inputs = []
    if args.preset:
        preset_inputs.append(args.preset)
    if args.presets:
        preset_inputs.extend(_split_csv_items(args.presets))
    preset_inputs = _dedup_preserve_order(preset_inputs)

    model_inputs = []
    if args.model:
        model_inputs.append(args.model)
    if args.models:
        model_inputs.extend(_split_csv_items(args.models))
    model_inputs = _dedup_preserve_order(model_inputs)

    if not preset_inputs and not model_inputs:
        parser.error("Must provide preset(s) or model(s)")
    if preset_inputs and model_inputs:
        parser.error("Use either preset(s) OR model(s), not both")

    from src.llms.presets import create_client, resolve_preset
    from src.experiments.naming_game_os_groq import NamingGameOSExperiment, NamingGameOSConfig

    logger.info("=" * 60)
    logger.info("OS NAMING GAME EXPERIMENT")
    logger.info("=" * 60)

    clients = {}
    model_names = []
    mode = "single"
    t0 = time.time()

    if preset_inputs:
        logger.info(f"Loading preset(s): {preset_inputs}")
        for preset in preset_inputs:
            backend, model_name = resolve_preset(preset)
            if backend not in ("hf", "groq"):
                parser.error(
                    f"Preset '{preset}' resolves to backend '{backend}', expected one of: hf, groq"
                )
            if model_name not in clients:
                clients[model_name] = create_client(
                    preset=preset,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
            model_names.append(model_name)
        model_names = _dedup_preserve_order(model_names)
    else:
        logger.info(f"Loading model(s): {model_inputs}")
        for model_name in model_inputs:
            if model_name not in clients:
                clients[model_name] = create_client(
                    backend=args.backend,
                    model=model_name,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
            model_names.append(model_name)
        model_names = _dedup_preserve_order(model_names)

    if len(model_names) > 1:
        mode = "mixed"

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    config = NamingGameOSConfig(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_runs=args.runs,
        history_length=args.history,
        n_names=args.names,
        allowed_path=args.allowed_path,
        conditions=args.conditions,
        model_names=model_names,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        round1_fullstring_argmax=args.round1_fullstring_argmax,
        outdir=args.outdir,
        seed=args.seed,
        verbosity=args.verbose,
    )

    logger.info("Configuration:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Agents: {args.agents}, Rounds: {args.rounds}, Runs: {args.runs}")
    logger.info(f"  Conditions: {args.conditions}")
    logger.info(f"  Temperature: {args.temperature}, History: {args.history}")
    logger.info(f"  round1_fullstring_argmax: {args.round1_fullstring_argmax}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {args.outdir}")
    logger.info("-" * 60)

    total_start = time.time()
    experiment = NamingGameOSExperiment(clients, config)
    logger.info(f"Tokenizer-safe allowed names: {experiment.allowed_names}")
    results = experiment.run_all()
    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info("OS EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {args.outdir}")
    logger.info("")
    logger.info("Summary:")

    for condition, runs in results.items():
        import numpy as np
        avg_success = np.mean([np.mean(r.per_round_success) for r in runs])
        std_success = np.std([np.mean(r.per_round_success) for r in runs])
        logger.info(f"  {condition}: {avg_success:.3f} ± {std_success:.3f}")


if __name__ == "__main__":
    main()
