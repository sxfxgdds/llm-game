#!/usr/bin/env python3
"""
Identify pre- vs post-temperature behavior from first-token top_logprobs
without using logit_bias.

Method:
- Query the same prompts across a temperature grid.
- From each response, keep logprobs for allowed one-token labels.
- For each (prompt, replicate, token-pair), compute gap series:
      g_ij(tau) = log P(i) - log P(j)
- Fit competing models per series:
      pre-temp:  g(tau) = c
      post-temp: g(tau) = c / tau
- Aggregate evidence across all fitted series.
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _init_openai_sdk_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for this script. Install with: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or .env.")
    return OpenAI(api_key=api_key)


def _extract_first_token_top_logprobs(resp: Any) -> List[Dict[str, Any]]:
    try:
        lp_data = getattr(resp.choices[0], "logprobs", None)
        if not lp_data or not getattr(lp_data, "content", None):
            return []
        first_tok = lp_data.content[0]
        top = getattr(first_tok, "top_logprobs", None) or []
        return [{"token": tp.token, "logprob": tp.logprob} for tp in top]
    except (AttributeError, IndexError, TypeError):
        return []


def _request_top_logprobs(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    top_logprobs: int,
    retries: int,
    retry_sleep_s: float,
) -> List[Dict[str, Any]]:
    system = "Answer with exactly one token from the allowed list. Do not add punctuation or extra words."
    attempts = max(int(retries), 1)
    last_err = None
    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                seed=seed,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            return _extract_first_token_top_logprobs(resp)
        except Exception as exc:
            last_err = exc
            if i + 1 < attempts and retry_sleep_s > 0:
                time.sleep(retry_sleep_s)
    raise RuntimeError(f"OpenAI request failed after {attempts} attempts: {last_err}") from last_err


def _normalize_allowed_logprobs(
    top: Sequence[Dict[str, Any]],
    allowed: Sequence[str],
) -> Dict[str, float]:
    allowed_set = {x.strip().upper() for x in allowed}
    out: Dict[str, float] = {}
    for item in top or []:
        tok = str(item.get("token", "")).strip().upper()
        if tok not in allowed_set:
            continue
        lp = float(item.get("logprob", float("-inf")))
        prev = out.get(tok)
        if prev is None or lp > prev:
            out[tok] = lp
    return out


def _fit_pre_constant(gs: List[float]) -> Tuple[float, float]:
    if not gs:
        return float("nan"), float("nan")
    c = statistics.mean(gs)
    sse = sum((g - c) ** 2 for g in gs)
    return c, sse


def _fit_post_inverse_tau(taus: List[float], gs: List[float]) -> Tuple[float, float]:
    if not gs:
        return float("nan"), float("nan")
    xs = [1.0 / t for t in taus]
    denom = sum(x * x for x in xs)
    if denom <= 0:
        return float("nan"), float("nan")
    c = sum(g * x for g, x in zip(gs, xs)) / denom
    sse = sum((g - c / t) ** 2 for g, t in zip(gs, taus))
    return c, sse


def _aic(n: int, sse: float, k: int = 1) -> float:
    if n <= 0 or not math.isfinite(sse):
        return float("nan")
    sse = max(sse, 1e-12)
    return n * math.log(sse / n) + 2 * k


def _bic(n: int, sse: float, k: int = 1) -> float:
    if n <= 0 or not math.isfinite(sse):
        return float("nan")
    sse = max(sse, 1e-12)
    return n * math.log(sse / n) + k * math.log(max(n, 1))


def _safe_cv(xs: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs)
    return sd / (abs(mu) + 1e-12)


def _parse_tokens(tokens_arg: str, n_tokens: int) -> List[str]:
    if tokens_arg.strip():
        toks = [x.strip().upper() for x in tokens_arg.split(",") if x.strip()]
        toks = list(dict.fromkeys(toks))
        if len(toks) < 2:
            raise RuntimeError("Need at least 2 tokens.")
        return toks
    if n_tokens < 2 or n_tokens > 26:
        raise RuntimeError("--n-tokens must be in [2, 26] when --tokens is not set.")
    return [chr(ord("A") + i) for i in range(n_tokens)]


def _build_prompt_variants(tokens: Sequence[str], n_prompts: int) -> List[str]:
    tok_line = ", ".join(tokens)
    templates = [
        (
            "You must choose exactly ONE token from the allowed list.\n"
            "Return only the token.\n"
            f"Allowed tokens: {tok_line}"
        ),
        (
            "Select one label only.\n"
            "Do not explain.\n"
            f"Allowed labels: {tok_line}"
        ),
        (
            "Output exactly one symbol from the set below.\n"
            "No extra text.\n"
            f"Set: {tok_line}"
        ),
        (
            "Pick one option.\n"
            "Reply with one token and nothing else.\n"
            f"Choices: {tok_line}"
        ),
        (
            "Choose one entry from the allowed inventory.\n"
            "Strict format: one token only.\n"
            f"Inventory: {tok_line}"
        ),
        (
            "Single-token response required.\n"
            "Any added text is invalid.\n"
            f"Allowed tokens: {tok_line}"
        ),
        (
            "Return exactly one of the listed labels.\n"
            "No punctuation.\n"
            f"Labels: {tok_line}"
        ),
        (
            "Output one token from the candidate list.\n"
            "Do not add words.\n"
            f"Candidate list: {tok_line}"
        ),
    ]
    n = max(1, min(n_prompts, len(templates)))
    return templates[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--taus", default="0.2,0.35,0.5,0.75,1.0,1.5")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tokens", default="", help="Comma-separated tokens (overrides --n-tokens).")
    ap.add_argument("--n-tokens", type=int, default=10)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--top-logprobs", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep-s", type=float, default=0.5)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--min-valid-taus", type=int, default=4)
    ap.add_argument(
        "--max-pairs-per-unit",
        type=int,
        default=20,
        help="Cap token pairs per (prompt,rep) for speed; <=0 means all pairs.",
    )
    ap.add_argument("--outdir", default="results/logprobs_gap_scaling")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    if len(taus) < 3:
        raise RuntimeError("Need at least 3 temperatures.")
    if any(t <= 0 for t in taus):
        raise RuntimeError("All temperatures must be > 0.")

    min_valid_taus = max(3, min(args.min_valid_taus, len(taus)))
    tokens = _parse_tokens(args.tokens, args.n_tokens)
    prompts = _build_prompt_variants(tokens, args.n_prompts)
    client = _init_openai_sdk_client()

    total_calls = len(prompts) * args.reps * len(taus)
    print(
        f"[progress] start model={args.model} prompts={len(prompts)} reps={args.reps} "
        f"taus={len(taus)} api_calls~{total_calls}",
        flush=True,
    )

    call_rows: List[Dict[str, Any]] = []
    obs: Dict[Tuple[int, int, float], Dict[str, float]] = {}
    started = time.time()
    n_done = 0
    n_ok = 0
    cov_acc = 0.0

    for p_idx, prompt in enumerate(prompts):
        print(f"[progress] prompt {p_idx + 1}/{len(prompts)} begin", flush=True)
        for rep in range(args.reps):
            unit_seed = args.seed + p_idx * 1_000_000 + rep
            for tau in taus:
                n_done += 1
                try:
                    top = _request_top_logprobs(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=tau,
                        seed=unit_seed,
                        max_tokens=args.max_tokens,
                        top_logprobs=args.top_logprobs,
                        retries=args.retries,
                        retry_sleep_s=args.retry_sleep_s,
                    )
                    token_lp = _normalize_allowed_logprobs(top, tokens)
                    req_ok = 1
                    n_ok += 1
                except RuntimeError as exc:
                    top = []
                    token_lp = {}
                    req_ok = 0
                    err_str = str(exc)
                else:
                    err_str = ""

                cov = len(token_lp) / max(len(tokens), 1)
                cov_acc += cov
                obs[(p_idx, rep, tau)] = token_lp
                call_rows.append(
                    {
                        "prompt_id": p_idx,
                        "rep": rep,
                        "tau": tau,
                        "seed": unit_seed,
                        "request_ok": req_ok,
                        "n_allowed_present": len(token_lp),
                        "allowed_coverage": cov,
                        "error": err_str,
                    }
                )

                if args.progress_every > 0 and (n_done % args.progress_every == 0 or n_done == total_calls):
                    elapsed = time.time() - started
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (total_calls - n_done) / max(rate, 1e-9)
                    mean_cov = cov_acc / max(n_done, 1)
                    ok_rate = n_ok / max(n_done, 1)
                    print(
                        f"[progress] calls={n_done}/{total_calls} ok_rate={ok_rate:.3f} "
                        f"mean_allowed_cov={mean_cov:.3f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
        print(f"[progress] prompt {p_idx + 1}/{len(prompts)} done", flush=True)

    tau_stats: List[Dict[str, Any]] = []
    for tau in taus:
        rows = [r for r in call_rows if r["tau"] == tau]
        n_tau = len(rows)
        ok_rate = (sum(r["request_ok"] for r in rows) / max(n_tau, 1))
        mean_cov = statistics.mean([r["allowed_coverage"] for r in rows]) if rows else float("nan")
        mean_n = statistics.mean([r["n_allowed_present"] for r in rows]) if rows else float("nan")
        tau_stats.append(
            {
                "tau": tau,
                "n_calls": n_tau,
                "request_ok_rate": ok_rate,
                "mean_allowed_coverage": mean_cov,
                "mean_n_allowed_present": mean_n,
            }
        )

    series_rows: List[Dict[str, Any]] = []
    total_units = len(prompts) * args.reps
    for unit_idx, (p_idx, rep) in enumerate(
        itertools.product(range(len(prompts)), range(args.reps)),
        start=1,
    ):
        tau_maps = {tau: obs.get((p_idx, rep, tau), {}) for tau in taus}
        pair_list = list(itertools.combinations(tokens, 2))
        if args.max_pairs_per_unit > 0 and len(pair_list) > args.max_pairs_per_unit:
            rng = random.Random(args.seed + p_idx * 1_000_000 + rep + 777)
            pair_list = rng.sample(pair_list, args.max_pairs_per_unit)

        for t_i, t_j in pair_list:
            used_taus: List[float] = []
            gaps: List[float] = []
            for tau in taus:
                mp = tau_maps[tau]
                if t_i in mp and t_j in mp:
                    used_taus.append(tau)
                    gaps.append(mp[t_i] - mp[t_j])
            n_tau = len(used_taus)
            if n_tau < min_valid_taus:
                continue

            c_pre, sse_pre = _fit_pre_constant(gaps)
            c_post, sse_post = _fit_post_inverse_tau(used_taus, gaps)
            aic_pre = _aic(n_tau, sse_pre, k=1)
            aic_post = _aic(n_tau, sse_post, k=1)
            bic_pre = _bic(n_tau, sse_pre, k=1)
            bic_post = _bic(n_tau, sse_post, k=1)
            delta_aic = aic_post - aic_pre
            delta_bic = bic_post - bic_pre

            tau_gaps = [t * g for t, g in zip(used_taus, gaps)]
            cv_g = _safe_cv(gaps)
            cv_tau_g = _safe_cv(tau_gaps)
            mean_g = statistics.mean(gaps)
            mean_tau_g = statistics.mean(tau_gaps)

            series_rows.append(
                {
                    "prompt_id": p_idx,
                    "rep": rep,
                    "token_i": t_i,
                    "token_j": t_j,
                    "n_tau": n_tau,
                    "taus_used": ",".join(str(t) for t in used_taus),
                    "mean_gap": mean_g,
                    "mean_tau_gap": mean_tau_g,
                    "cv_gap": cv_g,
                    "cv_tau_gap": cv_tau_g,
                    "c_pre": c_pre,
                    "sse_pre": sse_pre,
                    "aic_pre": aic_pre,
                    "bic_pre": bic_pre,
                    "c_post": c_post,
                    "sse_post": sse_post,
                    "aic_post": aic_post,
                    "bic_post": bic_post,
                    "delta_aic_post_minus_pre": delta_aic,
                    "delta_bic_post_minus_pre": delta_bic,
                }
            )

        if args.progress_every > 0 and (
            unit_idx % max(1, args.progress_every // max(len(taus), 1)) == 0 or unit_idx == total_units
        ):
            print(
                f"[progress] fit_units={unit_idx}/{total_units} fitted_series={len(series_rows)}",
                flush=True,
            )

    deltas_aic = [r["delta_aic_post_minus_pre"] for r in series_rows if math.isfinite(r["delta_aic_post_minus_pre"])]
    deltas_bic = [r["delta_bic_post_minus_pre"] for r in series_rows if math.isfinite(r["delta_bic_post_minus_pre"])]
    cv_pairs = [
        (r["cv_gap"], r["cv_tau_gap"])
        for r in series_rows
        if math.isfinite(r["cv_gap"]) and math.isfinite(r["cv_tau_gap"])
    ]

    n_series = len(deltas_aic)
    if n_series == 0:
        verdict = "inconclusive_no_fittable_series"
        agg = {
            "n_series": 0,
            "mean_delta_aic_post_minus_pre": float("nan"),
            "median_delta_aic_post_minus_pre": float("nan"),
            "mean_delta_bic_post_minus_pre": float("nan"),
            "median_delta_bic_post_minus_pre": float("nan"),
            "frac_post_favored_delta_aic_lt_0": float("nan"),
            "frac_strong_post_delta_aic_le_-10": float("nan"),
            "frac_strong_pre_delta_aic_ge_10": float("nan"),
            "frac_cv_support_post": float("nan"),
        }
    else:
        frac_post = sum(d < 0 for d in deltas_aic) / n_series
        frac_strong_post = sum(d <= -10 for d in deltas_aic) / n_series
        frac_strong_pre = sum(d >= 10 for d in deltas_aic) / n_series
        frac_cv_post = (
            sum(cv_tau < cv for cv, cv_tau in cv_pairs) / len(cv_pairs)
            if cv_pairs
            else float("nan")
        )
        mean_daic = statistics.mean(deltas_aic)
        med_daic = statistics.median(deltas_aic)
        mean_dbic = statistics.mean(deltas_bic) if deltas_bic else float("nan")
        med_dbic = statistics.median(deltas_bic) if deltas_bic else float("nan")

        if frac_strong_post >= 0.67 and med_daic <= -10:
            verdict = "strong_evidence_post_temperature_logprobs"
        elif frac_strong_pre >= 0.67 and med_daic >= 10:
            verdict = "strong_evidence_pre_temperature_logprobs"
        elif frac_post >= 0.60 and med_daic < 0:
            verdict = "weak_to_moderate_evidence_post_temperature_logprobs"
        elif frac_post <= 0.40 and med_daic > 0:
            verdict = "weak_to_moderate_evidence_pre_temperature_logprobs"
        else:
            verdict = "inconclusive"

        agg = {
            "n_series": n_series,
            "mean_delta_aic_post_minus_pre": mean_daic,
            "median_delta_aic_post_minus_pre": med_daic,
            "mean_delta_bic_post_minus_pre": mean_dbic,
            "median_delta_bic_post_minus_pre": med_dbic,
            "frac_post_favored_delta_aic_lt_0": frac_post,
            "frac_strong_post_delta_aic_le_-10": frac_strong_post,
            "frac_strong_pre_delta_aic_ge_10": frac_strong_pre,
            "frac_cv_support_post": frac_cv_post,
        }

    print("\n=== LOGPROBS GAP-SCALING TEST (NO BIAS) ===")
    print(
        f"model={args.model} prompts={len(prompts)} reps={args.reps} "
        f"tokens={len(tokens)} top_logprobs={args.top_logprobs}"
    )
    print(f"taus={taus}")
    print("")
    print("tau\tcalls\tok_rate\tmean_allowed_cov\tmean_n_allowed")
    for s in tau_stats:
        print(
            f"{s['tau']:.3f}\t{s['n_calls']}\t{s['request_ok_rate']:.3f}\t"
            f"{s['mean_allowed_coverage']:.3f}\t{s['mean_n_allowed_present']:.2f}"
        )
    print("")
    print(f"fitted_series={agg['n_series']}")
    print(f"mean_delta_aic(post-pre)={agg['mean_delta_aic_post_minus_pre']:.3f}")
    print(f"median_delta_aic(post-pre)={agg['median_delta_aic_post_minus_pre']:.3f}")
    print(f"frac(delta_aic<0)={agg['frac_post_favored_delta_aic_lt_0']:.3f}")
    print(f"frac(delta_aic<=-10)={agg['frac_strong_post_delta_aic_le_-10']:.3f}")
    print(f"frac(delta_aic>=10)={agg['frac_strong_pre_delta_aic_ge_10']:.3f}")
    print(f"frac(cv_tau_gap<cv_gap)={agg['frac_cv_support_post']:.3f}")
    print(f"verdict={verdict}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_id = f"logprobs_gap_scaling_{stamp}_s{args.seed}{tag}"
    run_dir = os.path.join(args.outdir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "taus": taus,
        "tokens": tokens,
        "n_prompts": len(prompts),
        "reps": args.reps,
        "top_logprobs": args.top_logprobs,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "retry_sleep_s": args.retry_sleep_s,
        "min_valid_taus": min_valid_taus,
        "max_pairs_per_unit": args.max_pairs_per_unit,
        "prompt_variants": prompts,
        "tau_stats": tau_stats,
        "aggregate": agg,
        "verdict": verdict,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "calls.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "rep",
                "tau",
                "seed",
                "request_ok",
                "n_allowed_present",
                "allowed_coverage",
                "error",
            ],
        )
        w.writeheader()
        for row in call_rows:
            w.writerow(row)

    with open(os.path.join(run_dir, "series_fits.csv"), "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "prompt_id",
            "rep",
            "token_i",
            "token_j",
            "n_tau",
            "taus_used",
            "mean_gap",
            "mean_tau_gap",
            "cv_gap",
            "cv_tau_gap",
            "c_pre",
            "sse_pre",
            "aic_pre",
            "bic_pre",
            "c_post",
            "sse_post",
            "aic_post",
            "bic_post",
            "delta_aic_post_minus_pre",
            "delta_bic_post_minus_pre",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in series_rows:
            w.writerow(row)

    print(f"\nSaved: {run_dir}")


if __name__ == "__main__":
    main()

