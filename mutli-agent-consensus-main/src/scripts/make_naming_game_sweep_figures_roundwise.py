#!/usr/bin/env python3
"""
Make round-wise sweep figures for naming_game sweeps.

Reads trials.csv from results/naming_game_sweeps/{single|mixed}/.../naming_game/<run_id>/trials.csv
and produces:
  1) Per-round match rate curves: mean +/- std across runs, for each model/mode/condition
     - histories are separate panels
     - temperatures are separate colored curves
  2) A single "creative" summary bubble chart:
     - y = condition, x = model
     - color = best final-10 mean match rate over all (H,T)
     - size = std at that best point
     - annotate with (H,T) of the best point

Also writes:
  - per_round_aggregates.csv
  - final10_summary.csv
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Parsing helpers
# -----------------------

def _safe_tag_to_float(ttag: str) -> Optional[float]:
    # ttag like "0p5" or "1" or "1p5"
    try:
        return float(ttag.replace("p", "."))
    except Exception:
        return None


def _extract_sweep_meta_from_path(path: str) -> Tuple[str, str, Optional[int], Optional[float]]:
    """
    Parse mode/model_tag/H/T from a path that contains:
      .../single/<preset>/H{H}/T{Ttag}/naming_game/<run_id>/trials.csv
      .../mixed/<presets+...>/H{H}/T{Ttag}/naming_game/<run_id>/trials.csv

    Returns: (mode, model_tag, H, T)
    """
    parts = path.replace("\\", "/").split("/")
    mode = None
    model_tag = None
    H = None
    T = None

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
            ttag = p[1:]
            T = _safe_tag_to_float(ttag)

    return mode or "unknown", model_tag or "unknown", H, T


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_trials(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "trials.csv" in filenames:
            # only naming_game trials
            if "/naming_game/" in dirpath.replace("\\", "/") or dirpath.replace("\\", "/").endswith("/naming_game"):
                out.append(os.path.join(dirpath, "trials.csv"))
    return out


def _condition_from_run_summary(run_dir: str) -> Optional[str]:
    js = _read_json_if_exists(os.path.join(run_dir, "run_summary.json"))
    if not js:
        return None
    # naming_game uses variant_id as condition in your code
    cond = js.get("variant_id", None) or js.get("variant", None)
    return cond


def _pretty_model_label(mode: str, model_tag: str) -> str:
    if mode == "mixed":
        # shorten long joined preset strings
        if "+" in model_tag:
            n = model_tag.count("+") + 1
            return f"mixed_{n}models"
        return f"mixed_{model_tag}"
    return model_tag


# -----------------------
# Stats computation
# -----------------------

def _per_round_match_series(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-round match rate for a single replicate (one run_id + repeat_idx).
    trials.csv has one row per agent interaction; we just average 'match' within each round.
    """
    d = df.copy()
    # ensure numeric
    if d["match"].dtype != np.float64 and d["match"].dtype != np.int64:
        d["match"] = d["match"].astype(float)
    return d.groupby("round")["match"].mean().sort_index()


def _final10_metric(per_round: np.ndarray, k: int = 10) -> float:
    if len(per_round) == 0:
        return float("nan")
    if len(per_round) < k:
        return float(np.nanmean(per_round))
    return float(np.nanmean(per_round[-k:]))


@dataclass
class CurveAgg:
    mean: np.ndarray
    std: np.ndarray
    n: int
    rounds: np.ndarray


# -----------------------
# Plot helpers
# -----------------------

def _get_temp_colors(temps: List[float]):
    # Map temperatures to a colormap (colorful but consistent)
    cmap = plt.get_cmap("viridis")
    if len(temps) <= 1:
        return {temps[0]: cmap(0.6)} if temps else {}
    temps_sorted = sorted(temps)
    return {t: cmap(i / (len(temps_sorted) - 1)) for i, t in enumerate(temps_sorted)}


def _plot_condition_figure(
    outpath: str,
    title: str,
    histories: List[int],
    temps: List[float],
    curves: Dict[Tuple[int, float], CurveAgg],
    n_rounds: int,
):
    """
    histories -> panels (columns)
    temps -> lines
    curves indexed by (H, T)
    """
    temps_sorted = sorted(temps)
    histories_sorted = sorted(histories)
    temp_colors = _get_temp_colors(temps_sorted)

    ncols = len(histories_sorted)
    fig_w = max(8, 4 * ncols)
    fig_h = 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    for idx, H in enumerate(histories_sorted):
        ax = fig.add_subplot(1, ncols, idx + 1)
        ax.set_title(f"H={H}")
        ax.set_xlim(0, n_rounds - 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("round")
        if idx == 0:
            ax.set_ylabel("match success rate")

        for T in temps_sorted:
            key = (H, T)
            if key not in curves:
                continue
            agg = curves[key]
            x = agg.rounds

            m = agg.mean
            s = agg.std
            lo = np.clip(m - s, 0.0, 1.0)
            hi = np.clip(m + s, 0.0, 1.0)

            ax.plot(x, m, label=f"T={T:g}", color=temp_colors[T])
            ax.fill_between(x, lo, hi, alpha=0.15, color=temp_colors[T])

        if idx == ncols - 1:
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _bubble_summary_plot(
    outpath: str,
    df_final: pd.DataFrame,
    title: str,
):
    """
    Bubble chart:
      x = model_label
      y = condition
      bubble color = best_mean_final10
      bubble size  = best_std_final10
      text = (H,T)
    """
    # Ensure ordering
    conds = sorted(df_final["condition"].unique().tolist())
    models = df_final["model_label"].unique().tolist()

    # Keep single models first, mixed last if present
    models_sorted = sorted([m for m in models if not m.startswith("mixed_")])
    models_sorted += sorted([m for m in models if m.startswith("mixed_")])

    x_map = {m: i for i, m in enumerate(models_sorted)}
    y_map = {c: i for i, c in enumerate(conds)}

    xs = df_final["model_label"].map(x_map).values
    ys = df_final["condition"].map(y_map).values

    means = df_final["best_mean_final10"].values.astype(float)
    stds = df_final["best_std_final10"].values.astype(float)

    # bubble sizes: scale std to something visible
    # (std=0 -> tiny, std large -> bigger)
    sizes = 50 + 800 * np.clip(stds, 0.0, 0.5)  # clip for sanity

    plt.figure(figsize=(max(10, 1.4 * len(models_sorted)), 5.5))
    sc = plt.scatter(xs, ys, s=sizes, c=means, alpha=0.85)

    plt.xticks(range(len(models_sorted)), models_sorted, rotation=20, ha="right")
    plt.yticks(range(len(conds)), conds)
    plt.xlabel("model (single) and mixed cohort")
    plt.ylabel("condition")
    plt.title(title)
    plt.colorbar(sc, label="best mean final-10 match rate")

    # annotate with (H,T)
    for _, r in df_final.iterrows():
        x = x_map[r["model_label"]]
        y = y_map[r["condition"]]
        ht = f"H{int(r['best_H'])},T{r['best_T']:g}"
        plt.text(x, y, ht, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Make round-wise sweep figures for naming_game")
    ap.add_argument("--input", required=True, help="Root sweep directory (e.g., results/naming_game_sweeps)")
    ap.add_argument("--output", required=True, help="Output directory for figures and CSVs")
    ap.add_argument("--final-k", type=int, default=10, help="Final-k rounds used for final metric (default 10)")
    ap.add_argument("--min-reps", type=int, default=2, help="Min repeats required to draw std bands (default 2)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    trials_paths = _find_trials(args.input)
    if not trials_paths:
        raise RuntimeError(f"No trials.csv found under: {args.input}")

    records = []  # per replicate per round
    meta_rows = []  # one per replicate

    for tp in trials_paths:
        run_dir = os.path.dirname(tp)
        mode, model_tag, H, T = _extract_sweep_meta_from_path(tp)
        if H is None or T is None:
            continue

        condition = _condition_from_run_summary(run_dir)
        if condition is None:
            # fallback: try infer from directory name
            base = os.path.basename(run_dir)
            # naming_game_<model>_<condition>_<timestamp>...
            m = re.search(r"_([a-zA-Z0-9_]+)_\d{8}_\d{6}_s", base)
            condition = m.group(1) if m else "unknown"

        df = pd.read_csv(tp)

        # Identify replicates (run_id + repeat_idx) inside this trials.csv
        # (handles both sequential sweeps and parallel job sweeps)
        if "run_id" not in df.columns or "repeat_idx" not in df.columns:
            continue

        for (run_id, rep), g in df.groupby(["run_id", "repeat_idx"], sort=False):
            s = _per_round_match_series(g)
            n_rounds = int(g["round"].max() + 1) if len(g) else 0

            # store per-round points
            for rnd, val in s.items():
                records.append({
                    "mode": mode,
                    "model_tag": model_tag,
                    "model_label": _pretty_model_label(mode, model_tag),
                    "condition": condition,
                    "H": int(H),
                    "T": float(T),
                    "run_id": run_id,
                    "repeat_idx": int(rep),
                    "round": int(rnd),
                    "match_rate": float(val),
                })

            # store replicate-level summary
            per_round = s.reindex(range(n_rounds)).to_numpy(dtype=float)
            meta_rows.append({
                "mode": mode,
                "model_tag": model_tag,
                "model_label": _pretty_model_label(mode, model_tag),
                "condition": condition,
                "H": int(H),
                "T": float(T),
                "run_id": run_id,
                "repeat_idx": int(rep),
                "n_rounds": n_rounds,
                "final_k": args.final_k,
                "final_k_mean": _final10_metric(per_round, k=args.final_k),
                "avg_mean": float(np.nanmean(per_round)) if len(per_round) else float("nan"),
            })

    df_round = pd.DataFrame(records)
    df_rep = pd.DataFrame(meta_rows)

    # Save raw aggregates for tables
    df_round_path = os.path.join(args.output, "per_round_points.csv")
    df_rep_path = os.path.join(args.output, "per_run_replicates.csv")
    df_round.to_csv(df_round_path, index=False)
    df_rep.to_csv(df_rep_path, index=False)

    # Aggregate per-round curves across replicates for each combo
    curve_rows = []
    curves_by_group: Dict[Tuple[str, str, str], Dict[Tuple[int, float], CurveAgg]] = {}

    # Group by (mode, model_label, condition, H, T)
    group_cols = ["mode", "model_label", "condition", "H", "T"]
    for (mode, model_label, condition, H, T), g in df_round.groupby(group_cols, sort=True):
        # reconstruct replicate series matrix by (run_id, repeat_idx)
        reps = []
        for (rid, rep), gg in g.groupby(["run_id", "repeat_idx"], sort=False):
            s = gg.set_index("round")["match_rate"].sort_index()
            n_rounds = int(s.index.max() + 1)
            arr = s.reindex(range(n_rounds)).to_numpy(dtype=float)
            reps.append(arr)

        if not reps:
            continue

        # pad to same length (max over reps)
        max_len = max(len(a) for a in reps)
        mat = np.full((len(reps), max_len), np.nan, dtype=float)
        for i, a in enumerate(reps):
            mat[i, :len(a)] = a

        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        rounds = np.arange(max_len)

        agg = CurveAgg(mean=mean, std=std, n=len(reps), rounds=rounds)

        key_top = (mode, model_label, condition)
        if key_top not in curves_by_group:
            curves_by_group[key_top] = {}
        curves_by_group[key_top][(int(H), float(T))] = agg

        # write a row per round for table-ready output
        for r in range(max_len):
            curve_rows.append({
                "mode": mode,
                "model_label": model_label,
                "condition": condition,
                "H": int(H),
                "T": float(T),
                "round": int(r),
                "mean": float(mean[r]) if np.isfinite(mean[r]) else np.nan,
                "std": float(std[r]) if np.isfinite(std[r]) else np.nan,
                "n_reps": int(len(reps)),
            })

    df_curve = pd.DataFrame(curve_rows)
    df_curve_path = os.path.join(args.output, "per_round_aggregates.csv")
    df_curve.to_csv(df_curve_path, index=False)

    # Produce per-model figures (one per condition)
    # We do: for each (mode, model_label), and for each condition -> one figure with panels=histories, curves=temps
    for (mode, model_label, condition), curves in curves_by_group.items():
        # infer histories/temps present
        keys = list(curves.keys())
        histories = sorted({h for (h, t) in keys})
        temps = sorted({t for (h, t) in keys})

        # infer n_rounds from max curve length
        n_rounds = max(len(curves[k].mean) for k in keys)

        outname = f"roundcurves__{mode}__{model_label}__{condition}.png"
        outpath = os.path.join(fig_dir, outname)

        title = f"{mode} | {model_label} | {condition} | mean±std over runs"
        _plot_condition_figure(
            outpath=outpath,
            title=title,
            histories=histories,
            temps=temps,
            curves=curves,
            n_rounds=n_rounds,
        )

    # Build bubble summary: best final-k across (H,T) for each (mode, model_label, condition)
    final_rows = []
    for (mode, model_label, condition), g in df_rep.groupby(["mode", "model_label", "condition"], sort=True):
        # For each (H,T) compute mean/std of final_k_mean
        best = None
        for (H, T), gg in g.groupby(["H", "T"], sort=True):
            vals = gg["final_k_mean"].to_numpy(dtype=float)
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            if best is None or m > best["best_mean_final10"]:
                best = {
                    "mode": mode,
                    "model_label": model_label,
                    "condition": condition,
                    "best_H": int(H),
                    "best_T": float(T),
                    "best_mean_final10": m,
                    "best_std_final10": s,
                    "n_reps": int(len(vals)),
                }
        if best:
            final_rows.append(best)

    df_final = pd.DataFrame(final_rows)
    final_csv = os.path.join(args.output, "final10_summary.csv")
    df_final.to_csv(final_csv, index=False)

    # Bubble chart
    bubble_path = os.path.join(fig_dir, "summary_bubbles_best_final10.png")
    _bubble_summary_plot(
        outpath=bubble_path,
        df_final=df_final,
        title=f"Best final-{args.final_k} match rate over (H,T) | color=mean size=std",
    )

    print("Wrote:")
    print(" -", df_round_path)
    print(" -", df_rep_path)
    print(" -", df_curve_path)
    print(" -", final_csv)
    print("Figures in:", fig_dir)
    print(" - roundcurves__...png (one per mode×model×condition)")
    print(" - summary_bubbles_best_final10.png")


if __name__ == "__main__":
    main()
