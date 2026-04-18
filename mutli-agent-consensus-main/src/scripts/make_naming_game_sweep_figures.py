#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_runs(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "run_summary.json" in filenames:
            out.append(os.path.join(dirpath, "run_summary.json"))
    return out


def _parse_combo_from_path(path: str) -> Tuple[str, int, float, str]:
    """
    Best-effort parse from .../single/<preset>/H{H}/T{T}/.../run_summary.json
                        .../mixed/<presets...>/H{H}/T{T}/.../run_summary.json
    Returns: (mode, H, T, model_tag)
    """
    parts = path.replace("\\", "/").split("/")
    mode = "unknown"
    H = None
    T = None
    model_tag = "unknown"

    # Find 'single' or 'mixed'
    if "single" in parts:
        mode = "single"
        i = parts.index("single")
        if i + 1 < len(parts):
            model_tag = parts[i + 1]
    elif "mixed" in parts:
        mode = "mixed"
        i = parts.index("mixed")
        if i + 1 < len(parts):
            model_tag = parts[i + 1]

    for p in parts:
        if p.startswith("H") and p[1:].isdigit():
            H = int(p[1:])
        if p.startswith("T"):
            # T0p3 -> 0.3
            ts = p[1:].replace("p", ".")
            try:
                T = float(ts)
            except:
                pass

    return mode, H, T, model_tag


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser(description="Make sweep figures from naming_game_sweeps directory")
    ap.add_argument("--input", required=True, help="Root sweep directory (e.g., results/naming_game_sweeps)")
    ap.add_argument("--output", required=True, help="Output dir for figures + aggregated.csv")
    ap.add_argument("--metric", default="final_match_rate", help="Metric from run_summary.json (default: final_match_rate)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summaries = _find_runs(args.input)
    if not summaries:
        raise RuntimeError(f"No run_summary.json found under {args.input}")

    rows = []
    for s in summaries:
        js = _read_json(s)
        mode, H, T, model_tag = _parse_combo_from_path(s)

        metric_val = js.get(args.metric, None)
        if metric_val is None:
            # fallback for some summaries: sometimes metric might be inside notes
            metric_val = _safe_get(js, "notes", args.metric, default=None)

        rows.append({
            "path": s,
            "mode": mode,
            "model_tag": model_tag,   # preset folder tag (single) or presets-join (mixed)
            "condition": js.get("variant_id", js.get("variant", "unknown")),
            "H": H,
            "T": T,
            "metric": metric_val,
            "n_trials_total": js.get("n_trials_total", None),
            "within_type_match_final": js.get("within_type_match_final", None),
            "cross_type_match_final": js.get("cross_type_match_final", None),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["H", "T", "metric"])
    df["H"] = df["H"].astype(int)
    df["T"] = df["T"].astype(float)
    df["metric"] = df["metric"].astype(float)

    # Save aggregated CSV for tables
    agg_path = os.path.join(args.output, "aggregated_sweep_metrics.csv")
    df.to_csv(agg_path, index=False)

    # --------- PLOTS ----------
    # 1) Heatmap per (mode, model_tag, condition)
    for (mode, model_tag, condition), g in df.groupby(["mode", "model_tag", "condition"], sort=True):
        # pivot with rows=H, cols=T
        piv = g.pivot_table(index="H", columns="T", values="metric", aggfunc="mean")
        piv = piv.sort_index(axis=0).sort_index(axis=1)

        plt.figure()
        im = plt.imshow(piv.values, aspect="auto")
        plt.colorbar(im)
        plt.xticks(range(len(piv.columns)), [str(x) for x in piv.columns], rotation=20, ha="right")
        plt.yticks(range(len(piv.index)), [str(h) for h in piv.index])
        plt.xlabel("temperature")
        plt.ylabel("history length H")
        plt.title(f"{args.metric} | {mode} | {model_tag} | {condition}")
        plt.tight_layout()

        fname = f"heatmap_{args.metric}__{mode}__{model_tag}__{condition}.png".replace("/", "_")
        plt.savefig(os.path.join(args.output, fname), dpi=200)
        plt.close()

    # 2) Line plots: metric vs T, separate curves for H (small-multiples by condition)
    for (mode, model_tag, condition), g in df.groupby(["mode", "model_tag", "condition"], sort=True):
        plt.figure()
        for H, gg in g.groupby("H", sort=True):
            gg = gg.sort_values("T")
            plt.plot(gg["T"].values, gg["metric"].values, marker="o", label=f"H={H}")
        plt.ylim(-0.02, 1.02)
        plt.xlabel("temperature")
        plt.ylabel(args.metric)
        plt.title(f"{args.metric} vs T | {mode} | {model_tag} | {condition}")
        plt.legend()
        plt.tight_layout()

        fname = f"lines_{args.metric}__{mode}__{model_tag}__{condition}.png".replace("/", "_")
        plt.savefig(os.path.join(args.output, fname), dpi=200)
        plt.close()

    print("Wrote:", agg_path)
    print("Wrote figures to:", args.output)


if __name__ == "__main__":
    main()
