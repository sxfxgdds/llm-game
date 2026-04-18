#!/usr/bin/env python3
"""
Run ONE naming-game job (one condition, one run_idx) for either:
- single preset (homogeneous)
- mixed presets (heterogeneous cohort)

This is meant to be called by the parallel sweep launcher.
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


def _safe_tag_float(x: float) -> str:
    # Match your existing directory style: 0.5 -> "0p5", 1.0 -> "1", 1.5 -> "1p5"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.6g}".replace(".", "p")
    return s


def _parse_presets_list(xs: List[str]) -> str:
    # Used only as a directory tag
    return "+".join(xs)


def main():
    ap = argparse.ArgumentParser(description="Run one naming-game job")

    # Mode
    ap.add_argument("--mode", choices=["single", "mixed"], required=True)

    # Single
    ap.add_argument("--preset", type=str, default=None)

    # Mixed
    ap.add_argument("--presets", nargs="+", default=None)

    # Job parameters
    ap.add_argument("--condition", required=True)
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--run-idx", type=int, required=True)

    # Shared experiment settings
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--names", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=8)

    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("-v", "--verbose", action="count", default=1)

    args = ap.parse_args()

    from src.llms.presets import create_client, resolve_preset
    from src.core.utils import set_global_seed
    from src.core.pairing import uniform_model_assignment
    from src.core.engine import EngineConfig, run_population_game, compute_run_summary
    from src.core.io import init_run_dir, dump_resolved_config, open_trials_writer, write_run_summary

    # Deterministic seed for this job
    seed_for_run = args.seed + args.run_idx
    set_global_seed(seed_for_run)

    H = args.history
    T = args.temperature
    condition = args.condition

    # Leaf directory matches your existing sweep allocation style
    if args.mode == "single":
        if not args.preset:
            raise SystemExit("--preset is required for mode=single")
        leaf = os.path.join(
            args.outdir, "single", args.preset, f"H{H}", f"T{_safe_tag_float(T)}"
        )
    else:
        if not args.presets or len(args.presets) < 2:
            raise SystemExit("--presets (>=2) is required for mode=mixed")
        leaf = os.path.join(
            args.outdir, "mixed", _parse_presets_list(args.presets), f"H{H}", f"T{_safe_tag_float(T)}"
        )

    os.makedirs(leaf, exist_ok=True)

    # Build run directory (keeps the naming_game/<run_id> pattern you already have)
    if args.mode == "single":
        _, model_id = resolve_preset(args.preset)
        model_tag_for_runid = model_id
    else:
        # Shorten the mixed run_id model string to keep paths reasonable
        model_tag_for_runid = f"mixed_{len(args.presets)}models"

    run_id, run_path = init_run_dir(
        leaf,
        experiment="naming_game",
        model=model_tag_for_runid,
        variant=condition,
        seed=args.seed,  # keep s12345 like your existing runs
    )

    # Dump config (include run_idx + seed_for_run so you can aggregate later)
    config_dict = {
        "experiment": "naming_game",
        "mode": args.mode,
        "condition": condition,
        "n_agents": args.agents,
        "n_rounds": args.rounds,
        "n_runs": 1,
        "run_idx": args.run_idx,
        "seed_base": args.seed,
        "seed_for_run": seed_for_run,
        "history_length": H,
        "n_names": args.names,
        "temperature": T,
        "max_tokens": args.max_tokens,
        "presets": [args.preset] if args.mode == "single" else list(args.presets),
    }
    config_digest = dump_resolved_config(run_path, config_dict)

    # Build clients + agent_types and run the shared engine
    with open_trials_writer(run_path) as trials_writer:
        if args.mode == "single":
            client = create_client(
                preset=args.preset,
                temperature=T,
                max_tokens=args.max_tokens,
                seed=seed_for_run,
            )
            if hasattr(client, "set_seed"):
                client.set_seed(seed_for_run)

            # Homogeneous cohort
            model_name = getattr(client, "model", model_id)
            agent_types = [model_name] * args.agents
            clients = {model_name: client}

        else:
            # Create one client per model_id
            clients = {}
            model_ids = []
            for p in args.presets:
                _, mid = resolve_preset(p)
                model_ids.append(mid)
                c = create_client(
                    preset=p,
                    temperature=T,
                    max_tokens=args.max_tokens,
                    seed=seed_for_run,
                )
                if hasattr(c, "set_seed"):
                    c.set_seed(seed_for_run)
                clients[mid] = c

            # Uniform assignment of model types to agents (per run_idx seed)
            assignment, _ = uniform_model_assignment(
                args.agents,
                len(model_ids),
                seed=seed_for_run + 777,
            )
            agent_types = [model_ids[i] for i in assignment]

        engine_config = EngineConfig(
            n_agents=args.agents,
            n_rounds=args.rounds,
            n_names=args.names,
            history_length=H,
            temperature=T,
            prompt_variant=condition,
            use_nonce_tokens=False,
            store_raw_responses=False,
            verbosity=args.verbose,
            run_id=run_id,
            experiment="naming_game",
            variant_id=condition,
            seed=seed_for_run,
            repeat_idx=args.run_idx,
        )

        result = run_population_game(
            agent_types=agent_types,
            clients=clients,
            config=engine_config,
            trials_writer=trials_writer,
            run_path=run_path,
            rng=None,  # engine can use seed inside config; or leave None
        )

    # Write run_summary.json for this single run
    summary = compute_run_summary([result], engine_config, config_digest)

    # Add a tiny note so you can later filter/aggregate
    notes = summary.get("notes", {}) if isinstance(summary.get("notes", {}), dict) else {}
    notes.update({
        "mode": args.mode,
        "run_idx": args.run_idx,
        "seed_for_run": seed_for_run,
        "presets": [args.preset] if args.mode == "single" else list(args.presets),
        "history_length": H,
        "temperature": T,
    })
    summary["notes"] = notes

    write_run_summary(run_path, summary)

    logger.info(f"[done] {args.mode} | H={H} T={T} cond={condition} run_idx={args.run_idx} -> {run_path}")


if __name__ == "__main__":
    main()
