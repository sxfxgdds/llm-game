#!/usr/bin/env python3
"""
Parallel sweep launcher for naming game.

- Temperatures: sequential
- For each temperature: launch all histories x presets x conditions x runs in parallel
- Optionally include mixed cohort jobs
- Skips already-completed (mode, preset_tag, H, T, condition) by scanning run_summary.json
"""

import argparse
import asyncio
import os
import sys
import json
import logging
from typing import List, Set, Tuple

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


def _parse_ints_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _safe_tag_float(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}".replace(".", "p")


def _join_presets(presets: List[str]) -> str:
    return "+".join(presets)


def _scan_done_keys(root_outdir: str) -> Set[Tuple[str, str, int, str, str]]:
    """
    Returns a set of done keys:
      (mode, preset_tag, H, Ttag, condition)
    where:
      mode in {"single","mixed"}
      preset_tag is <preset> (single) or "+".join(presets) (mixed)
      H integer
      Ttag like "0p5", "1", "1p5"
      condition string

    We treat presence of ANY run_summary.json under that key as "done"
    to respect "no need to run these again."
    """
    done = set()

    for dirpath, dirnames, filenames in os.walk(root_outdir):
        if "run_summary.json" not in filenames:
            continue

        # Parse path components
        parts = dirpath.replace("\\", "/").split("/")

        try:
            # Expect .../single/<preset>/H#/T#/naming_game/<run_id>
            if "single" in parts:
                mode = "single"
                i = parts.index("single")
                preset_tag = parts[i + 1]
            elif "mixed" in parts:
                mode = "mixed"
                i = parts.index("mixed")
                preset_tag = parts[i + 1]
            else:
                continue

            H_part = next(p for p in parts if p.startswith("H") and p[1:].isdigit())
            T_part = next(p for p in parts if p.startswith("T"))
            H = int(H_part[1:])
            Ttag = T_part[1:]
        except Exception:
            continue

        # Condition from run_summary.json (variant_id)
        try:
            with open(os.path.join(dirpath, "run_summary.json"), "r", encoding="utf-8") as f:
                js = json.load(f)
            condition = js.get("variant_id", js.get("variant", None))
            if condition is None:
                continue
        except Exception:
            continue

        done.add((mode, preset_tag, H, Ttag, condition))

    return done


async def _run_subprocess(cmd: List[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        logger.error("FAILED: %s", " ".join(cmd))
        if out:
            logger.error("stdout:\n%s", out.decode("utf-8", errors="ignore"))
        if err:
            logger.error("stderr:\n%s", err.decode("utf-8", errors="ignore"))
    return proc.returncode


async def main_async():
    ap = argparse.ArgumentParser(description="Parallel naming-game sweep launcher")

    ap.add_argument("--presets", nargs="+", required=True)
    ap.add_argument("--also-mixed", action="store_true")
    ap.add_argument("--mixed-only", action="store_true", help="Run only the mixed cohort (skip all single-model jobs).")

    ap.add_argument("--conditions", nargs="+", required=True)

    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--names", type=int, default=10)

    ap.add_argument("--histories", type=str, default="0,1,3,10")
    ap.add_argument("--temperatures", type=str, default="0.1,0.5,0.7,1.0,1.5,2.0")

    ap.add_argument("--max-tokens", type=int, default=8)

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, required=True)

    # Optional global cap (default unlimited, per your request)
    ap.add_argument("--max-parallel", type=int, default=0, help="0 = unlimited; otherwise cap concurrent jobs.")
    ap.add_argument("-v", "--verbose", action="count", default=1)

    args = ap.parse_args()

    if args.mixed_only:
        args.also_mixed = True

    histories = _parse_ints_csv(args.histories)
    temps = _parse_floats_csv(args.temperatures)

    os.makedirs(args.outdir, exist_ok=True)

    done = _scan_done_keys(args.outdir)
    logger.info(f"[skip-index] found {len(done)} completed keys under {args.outdir}")

    sem = asyncio.Semaphore(args.max_parallel) if args.max_parallel and args.max_parallel > 0 else None

    async def run_job(cmd: List[str]) -> int:
        if sem is None:
            return await _run_subprocess(cmd)
        async with sem:
            return await _run_subprocess(cmd)

    # Temperatures sequential
    for T in temps:
        Ttag = _safe_tag_float(T)
        logger.info("=" * 70)
        logger.info(f"TEMPERATURE BATCH: T={T} (Ttag={Ttag})")
        logger.info("=" * 70)

        tasks = []
        scheduled = 0
        skipped = 0

        # Single-model jobs
        if not args.mixed_only:
            for preset in args.presets:
                for H in histories:
                    for condition in args.conditions:
                        key = ("single", preset, H, Ttag, condition)
                        if key in done:
                            skipped += args.runs
                            continue
                        for run_idx in range(args.runs):
                            cmd = [
                                sys.executable, "-m", "src.scripts.run_naming_game_job",
                                "--mode", "single",
                                "--preset", preset,
                                "--condition", condition,
                                "--history", str(H),
                                "--temperature", str(T),
                                "--run-idx", str(run_idx),
                                "--agents", str(args.agents),
                                "--rounds", str(args.rounds),
                                "--names", str(args.names),
                                "--max-tokens", str(args.max_tokens),
                                "--seed", str(args.seed),
                                "--outdir", args.outdir,
                                "-v" * max(args.verbose, 1),
                            ]
                            tasks.append(asyncio.create_task(run_job(cmd)))
                            scheduled += 1

        # Mixed jobs
        if args.also_mixed and len(args.presets) >= 2:
            mixed_tag = _join_presets(args.presets)
            for H in histories:
                for condition in args.conditions:
                    key = ("mixed", mixed_tag, H, Ttag, condition)
                    if key in done:
                        skipped += args.runs
                        continue
                    for run_idx in range(args.runs):
                        cmd = [
                            sys.executable, "-m", "src.scripts.run_naming_game_job",
                            "--mode", "mixed",
                            "--presets", *args.presets,
                            "--condition", condition,
                            "--history", str(H),
                            "--temperature", str(T),
                            "--run-idx", str(run_idx),
                            "--agents", str(args.agents),
                            "--rounds", str(args.rounds),
                            "--names", str(args.names),
                            "--max-tokens", str(args.max_tokens),
                            "--seed", str(args.seed),
                            "--outdir", args.outdir,
                            "-v" * max(args.verbose, 1),
                        ]
                        tasks.append(asyncio.create_task(run_job(cmd)))
                        scheduled += 1

        logger.info(f"[batch] scheduled={scheduled} jobs | skipped~={skipped} (runs) | max_parallel={args.max_parallel or 'unlimited'}")

        if tasks:
            results = await asyncio.gather(*tasks)
            n_fail = sum(1 for rc in results if rc != 0)
            logger.info(f"[batch done] T={T} finished | failures={n_fail}/{len(results)}")
        else:
            logger.info(f"[batch done] T={T} nothing to do (all keys already done)")

    logger.info("=" * 70)
    logger.info("PARALLEL SWEEP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results under: {args.outdir}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
