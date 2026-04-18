#!/usr/bin/env python3
"""
Run prior probe experiments across all OS backends.

Supports mixed runs with:
- local HuggingFace (`hf`)
- Hugging Face Router API (`hf_api`)
- Groq API (`groq`)
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


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
    parser = argparse.ArgumentParser(
        description="Run prior probe experiments across hf / hf_api / groq backends"
    )

    # Model selection
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--presets", nargs="+", default=None, help="Multiple presets for mixed")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "hf_api", "groq"],
        default="hf",
        help="Backend for --model/--models (ignored when using presets).",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models")

    # Experiment
    parser.add_argument("--agents", "-N", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--names", "-W", type=int, default=10)
    parser.add_argument("--mixed", action="store_true", help="Mix models across agents")

    # Probe settings
    parser.add_argument("--show-goal", action="store_true")
    parser.add_argument("--score-mode", choices=["full_string", "first_token"], default="full_string")
    parser.add_argument(
        "--probe-method",
        choices=["generate_choice", "score_argmax"],
        default="generate_choice",
        help="Probe path: default T=0 constrained generation or explicit score-based argmax.",
    )
    parser.add_argument(
        "--full-string-argmax",
        action="store_true",
        help="Shortcut for --probe-method score_argmax --score-mode full_string.",
    )

    # Output
    parser.add_argument("--outdir", type=str, default="results/prior_probe_all_os")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="count", default=1)

    args = parser.parse_args()

    if args.full_string_argmax:
        args.probe_method = "score_argmax"
        args.score_mode = "full_string"

    from src.llms.presets import create_client, resolve_preset
    from src.experiments.prior_probe_groq import PriorProbeGroqExperiment, PriorProbeGroqConfig

    model_names = []
    clients = {}

    if args.presets:
        for preset in args.presets:
            backend, model_id = resolve_preset(preset)
            if backend not in ("hf", "hf_api", "groq"):
                parser.error(
                    f"Preset '{preset}' resolves to backend '{backend}', expected one of: hf, hf_api, groq"
                )
            model_names.append(model_id)
            if model_id not in clients:
                clients[model_id] = create_client(preset=preset, seed=args.seed)
        args.mixed = True
    elif args.preset:
        backend, model_id = resolve_preset(args.preset)
        if backend not in ("hf", "hf_api", "groq"):
            parser.error(
                f"Preset '{args.preset}' resolves to backend '{backend}', expected one of: hf, hf_api, groq"
            )
        model_names.append(model_id)
        clients[model_id] = create_client(preset=args.preset, seed=args.seed)
    elif args.models:
        for m in args.models.split(","):
            m = m.strip()
            if not m:
                continue
            model_names.append(m)
            if m not in clients:
                clients[m] = create_client(backend=args.backend, model=m, seed=args.seed)
        if len(model_names) > 1:
            args.mixed = True
    elif args.model:
        model_names.append(args.model)
        clients[args.model] = create_client(backend=args.backend, model=args.model, seed=args.seed)
    else:
        parser.error("Must provide --preset, --presets, --model, or --models")

    model_names = _dedup_preserve_order(model_names)

    logger.info("=" * 60)
    logger.info("PRIOR PROBE ALL OS EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Models: {model_names}")
    logger.info(f"Agents: {args.agents}, Runs: {args.runs}")
    logger.info(
        f"Mixed: {args.mixed}, Probe method: {args.probe_method}, "
        f"Score mode: {args.score_mode}"
    )
    logger.info(f"Seed: {args.seed}")
    logger.info("-" * 60)

    config = PriorProbeGroqConfig(
        n_agents=args.agents,
        n_runs=args.runs,
        n_names=args.names,
        show_goal=args.show_goal,
        score_mode=args.score_mode,
        probe_method=args.probe_method,
        mixed=args.mixed,
        outdir=args.outdir,
        experiment_name="prior_probe_all_os",
        seed=args.seed,
        verbosity=args.verbose,
    )

    total_start = time.time()
    experiment = PriorProbeGroqExperiment(clients, model_names, config)
    summary = experiment.run()
    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    notes = summary.get("notes", {})
    logger.info(f"Overall match rate: {summary.get('final_match_rate', 0):.3f} ± {notes.get('overall_match_rate_std', 0):.3f}")
    if args.mixed:
        logger.info(f"Same-model: {notes.get('same_model_match_rate_mean', float('nan')):.3f}")
        logger.info(f"Cross-model: {notes.get('cross_model_match_rate_mean', float('nan')):.3f}")


if __name__ == "__main__":
    main()

