#!/usr/bin/env python3
"""
Run argmax-stability diagnostics to support Lemma: argmax stability under bounded perturbations.

Example:
  python -m src.scripts.run_argmax_stability \
      --preset yi6b \
      --contexts 200 \
      --perturb 20 \
      --history 6 \
      --names 10 \
      --temperature 1.0 \
      --seed 12345 \
      --outdir results/argmax_stability

Notes:
- Designed primarily for HF backends where we can compute logits exactly.
- Requires HuggingFaceClient to expose `self.model` and `self.tokenizer`.
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Run argmax-stability diagnostics (lemma support)"
    )

    # Model selection
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)

    # Experiment design
    parser.add_argument("--contexts", type=int, default=200, help="Number of base contexts")
    parser.add_argument("--perturb", type=int, default=20, help="Perturbations per context")
    parser.add_argument("--history", "-H", type=int, default=6, help="History length")
    parser.add_argument("--names", "-W", type=int, default=10, help="Number of allowed names")
    parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens (use 1 for this diagnostic)")
    parser.add_argument("--logprobs", action="store_true", help="Request token logprobs (OpenAI only)")
    parser.add_argument("--top-logprobs", type=int, default=20, help="Top-k tokens to return with logprobs (OpenAI max=20)")


    # LLM params
    parser.add_argument("--temperature", type=float, default=1.0)

    # Output & misc
    parser.add_argument("--outdir", type=str, default="results/argmax_stability")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--store-prompts", action="store_true", help="Store a few prompt samples")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N contexts (0 disables).")

    args = parser.parse_args()

    if not args.preset and not (args.backend and args.model):
        parser.error("Must provide --preset OR both --backend and --model")

    logger.info("=" * 60)
    logger.info("ARGMAX STABILITY DIAGNOSTIC (LEMMA SUPPORT)")
    logger.info("=" * 60)

    from src.llms.presets import create_client
    from src.experiments.argmax_stability import ArgmaxStabilityExperiment, ArgmaxStabilityConfig

    logger.info(f"Loading model: {args.preset or args.model} …")
    t0 = time.time()

    # NOTE: we keep max_tokens small, but scoring uses forward-pass logits (HF)
    client_kwargs = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    if args.logprobs:
        client_kwargs["logprobs"] = True
        client_kwargs["top_logprobs"] = args.top_logprobs

    if args.preset:
        client = create_client(preset=args.preset, **client_kwargs)
    else:
        client = create_client(backend=args.backend, model=args.model, **client_kwargs)

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    cfg = ArgmaxStabilityConfig(
        n_contexts=args.contexts,
        n_perturb=args.perturb,
        history_length=args.history,
        n_names=args.names,
        temperature=args.temperature,
        seed=args.seed,
        outdir=args.outdir,
        store_prompts=args.store_prompts,
        verbosity=args.verbose,
        progress_every=args.progress_every,
    )

    logger.info("Configuration:")
    logger.info(f"  Contexts:     {cfg.n_contexts}")
    logger.info(f"  Perturb/ctx:  {cfg.n_perturb}")
    logger.info(f"  H, W:         {cfg.history_length}, {cfg.n_names}")
    logger.info(f"  Temperature:  {cfg.temperature}")
    logger.info(f"  Seed:         {cfg.seed}")
    logger.info(f"  Output:       {cfg.outdir}")
    logger.info("-" * 60)

    exp = ArgmaxStabilityExperiment(client, cfg)
    summary = exp.run()

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 60)
    logger.info(f"flip_rate            = {summary['metrics']['flip_rate']:.4f}")
    logger.info(f"certified_coverage    = {summary['metrics']['certified_coverage']:.4f}")
    logger.info("Wrote: trials.csv and run_summary.json")


if __name__ == "__main__":
    main()
