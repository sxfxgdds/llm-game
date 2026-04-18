#!/usr/bin/env python3
import argparse, math, statistics
from typing import Optional, List, Dict, Any

def find_lp(top: List[Dict[str, Any]], target: str) -> Optional[float]:
    """Match by stripping whitespace; handles tokens like ' A' vs 'A'."""
    t = target.strip()
    for d in top or []:
        tok = (d.get("token") or "").strip()
        if tok == t:
            return float(d.get("logprob"))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--taus", default="0.2,0.5,0.75,1.0")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--top_logprobs", type=int, default=5)
    ap.add_argument("--tokenA", default="A")
    ap.add_argument("--tokenB", default="H")
    ap.add_argument("--tau_ref", type=float, default=1.0, help="reference tau for m(ref)/tau prediction")

    args = ap.parse_args()
    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]

    # Use your OpenAIClient (must have the logprobs patch you already pasted)
    from src.llms.openai_client import OpenAIClient

    client = OpenAIClient(
        model=args.model,
        temperature=1.0,          # overridden per call
        max_tokens=1,             # IMPORTANT: only first token matters
        top_p=1.0,
        seed=args.seed,
        logprobs=True,
        top_logprobs=args.top_logprobs,
    )

    # Fixed 2-option prompt; no naming-game semantics; just "choose one token"
    # Keep it constant across taus.
    tokA, tokB = args.tokenA.strip(), args.tokenB.strip()
    prompt = "\n".join([
        "You must choose exactly ONE token from the allowed list.",
        "Reply with exactly one token and nothing else.",
        f"Allowed tokens: {tokA}, {tokB}",
    ])

    def sample_margin_at_tau(tau: float) -> List[float]:
        margins = []
        misses = 0
        for r in range(args.reps):
            res = client.generate_choice(prompt, [tokA, tokB], temperature=tau, seed=args.seed + r)
            top = res.meta.get("first_token_top_logprobs", [])
            lpA = find_lp(top, tokA)
            lpB = find_lp(top, tokB)
            if lpA is None or lpB is None:
                misses += 1
                continue
            margins.append(lpA - lpB)  # log(P(A)/P(B))
        return margins, misses

    # Collect margins
    out = {}
    for tau in taus:
        margins, misses = sample_margin_at_tau(tau)
        cov = (len(margins) / max(args.reps, 1))
        mbar = statistics.mean(margins) if margins else float("nan")
        sbar = statistics.pstdev(margins) if len(margins) > 1 else 0.0
        out[tau] = dict(mean=mbar, std=sbar, cov=cov, n=len(margins), misses=misses)

    # Optional: compute a reference margin at tau_ref (if included)
    # and predict scaling m(tau) ≈ m(tau_ref) * (tau_ref / tau)
    ref_tau = args.tau_ref
    if ref_tau not in out or math.isnan(out[ref_tau]["mean"]):
        # try to compute it even if not in taus
        ref_margins, _ = sample_margin_at_tau(ref_tau)
        ref_mean = statistics.mean(ref_margins) if ref_margins else float("nan")
    else:
        ref_mean = out[ref_tau]["mean"]

    print("\n=== LOG-ODDS TEMPERATURE TEST ===")
    print(f"model={args.model}  reps={args.reps}  top_logprobs={args.top_logprobs}")
    print(f"tokens: A='{tokA}'  B='{tokB}'")
    print(f"reference tau_ref={ref_tau}  ref_mean_margin={ref_mean:.6f}")
    print("")
    print("tau\tcov\tmean_margin\tstd\t(tau*mean)\tpred_if_1/tau")
    for tau in taus:
        m = out[tau]["mean"]
        cov = out[tau]["cov"]
        std = out[tau]["std"]
        tau_m = (tau * m) if not math.isnan(m) else float("nan")
        pred = (ref_mean * (ref_tau / tau)) if (not math.isnan(ref_mean) and not math.isnan(m)) else float("nan")
        print(f"{tau:.3f}\t{cov:.2f}\t{m: .6f}\t{std: .6f}\t{tau_m: .6f}\t{pred: .6f}")

    # Quick heuristic conclusion
    # Compare variability of mean_margin vs variability of tau*mean_margin
    means = [out[t]["mean"] for t in taus if not math.isnan(out[t]["mean"])]
    scaled = [t * out[t]["mean"] for t in taus if not math.isnan(out[t]["mean"])]
    if len(means) >= 2:
        cv_means = (statistics.pstdev(means) / (abs(statistics.mean(means)) + 1e-9))
        cv_scaled = (statistics.pstdev(scaled) / (abs(statistics.mean(scaled)) + 1e-9))
        print("\nHeuristic check:")
        print(f"CV(mean_margin)     = {cv_means:.3f}")
        print(f"CV(tau*mean_margin) = {cv_scaled:.3f}")
        if cv_scaled < cv_means:
            print("=> Looks more consistent with POST-temperature logprobs (1/tau scaling).")
        else:
            print("=> Looks more consistent with PRE-temperature logprobs (no 1/tau scaling), or coverage/selection bias.")
    else:
        print("\nNot enough valid points to compare CVs (try increasing top_logprobs or changing tokens).")

if __name__ == "__main__":
    main()
