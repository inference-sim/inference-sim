#!/usr/bin/env python3
"""Iter 4 Approach A: Journey β + Per-Model δ Search + α Search

Three phases:
  Phase 1: Use Iter 2 per-model journey β (validated, 3.8-18.2% TTFT analytical)
           Grid-search per-model δ∈[0,20000] using BLIS replay
  Phase 2: Search (α₀, α₂) via Nelder-Mead with BLIS replay objective
  Phase 3: Full validation on train/validate/test

Key insight: Step-only β + measured δ fails because the analytical decomposition
(GPU + scheduling) doesn't match how BLIS composes step times through its
scheduler. Instead, use the validated journey β and let BLIS replay find each
model's δ — the delta that makes the simulator match reality.

Usage:
    python3 training/iter4_joint.py                    # all phases
    python3 training/iter4_joint.py --phase 1          # δ search only
    python3 training/iter4_joint.py --phase 2          # + α search
    python3 training/iter4_joint.py --include-test     # + test set eval
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from split import EXPERIMENTS, Split

TD = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TD)
REPLAY_BINARY = os.path.join(REPO_ROOT, "replay")
REPLAY_DATA_DIR = os.path.join(TD, "replay_data")

# Iter 3 global α (warm-start for Phase 2 search)
ITER3_ALPHA = [13732.0, 0.0, 860.6]

# δ search grid (µs) — covers 0 to 20ms in 500µs steps
DELTA_GRID = list(range(0, 20001, 500))

# Fast subset: 1 experiment per model (for search objectives)
FAST_SUBSET = {
    "llama-2-7b":    "20260217-231439-llama-2-7b-tp1-general",
    "llama-2-70b":   "20260217-202857-llama-2-70b-tp4-general",
    "mixtral-8x7b":  "20260218-130541-mixtral-8x7b-v0-1-tp2-general",
    "codellama-34b": "20260218-150304-codellama-34b-tp2-general",
}


# ─── Replay Execution ────────────────────────────────────

def run_replay(exp_dir_name, beta, alpha, delta_us=None, seed=42,
               backend="blackbox"):
    """Run the BLIS replay binary and parse JSON output."""
    input_path = os.path.join(REPLAY_DATA_DIR, f"{exp_dir_name}.json")
    cmd = [
        REPLAY_BINARY,
        "--input", input_path,
        "--seed", str(seed),
        "--backend", backend,
        "--beta", ",".join(f"{v:.6f}" for v in beta),
        "--alpha", ",".join(f"{v:.6f}" for v in alpha),
    ]
    if delta_us is not None and delta_us > 0:
        cmd.extend(["--delta", f"{delta_us:.6f}"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Replay failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


def load_ground_truth(exp_dir_name):
    """Load ground truth metrics from the JSON file."""
    with open(os.path.join(REPLAY_DATA_DIR, f"{exp_dir_name}_ground_truth.json")) as f:
        return json.load(f)


def relative_error(sim_val, real_val):
    """Signed relative error: (sim - real) / real."""
    if real_val == 0:
        return 0.0 if sim_val == 0 else float("inf")
    return (sim_val - real_val) / real_val


def evaluate_experiments(experiments, per_model_beta, alpha,
                         per_model_delta=None, label=""):
    """Run replay on experiments using per-model β and δ."""
    if label:
        print(f"\n{'=' * 110}")
        print(f"  {label}")
        print(f"{'=' * 110}")

    print(f"\n  {'Experiment':<45} {'TTFT mean':>10} {'TTFT p99':>10} "
          f"{'E2E mean':>10} {'E2E p99':>10} {'Tput':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    ttft_res, e2e_res, tput_res = [], [], []
    comparisons = []

    for exp in experiments:
        beta = per_model_beta.get(exp.model_short)
        if beta is None:
            print(f"  {exp.model_short}: no β coefficients", file=sys.stderr)
            continue
        delta = per_model_delta.get(exp.model_short, 0) if per_model_delta else None
        try:
            out = run_replay(exp.dir_name, beta, alpha, delta)
            gt = load_ground_truth(exp.dir_name)
            s = out["summary"]

            ttft_re = relative_error(s["ttft_mean_ms"], gt["ttft"]["mean_ms"])
            ttft_p99_re = relative_error(s["ttft_p99_ms"], gt["ttft"]["p99_ms"])
            e2e_re = relative_error(s["e2e_mean_ms"], gt["e2e"]["mean_ms"])
            e2e_p99_re = relative_error(s["e2e_p99_ms"], gt["e2e"]["p99_ms"])
            tput_re = relative_error(
                s["responses_per_sec"], gt["throughput"]["requests_per_sec"]
            )

            label_str = f"{exp.model_short}/{exp.profile} ({exp.split.value})"
            print(f"  {label_str:<45} {ttft_re:>+9.1%} {ttft_p99_re:>+9.1%} "
                  f"{e2e_re:>+9.1%} {e2e_p99_re:>+9.1%} {tput_re:>+7.1%}")

            ttft_res.append(abs(ttft_re))
            e2e_res.append(abs(e2e_re))
            tput_res.append(abs(tput_re))
            comparisons.append({
                "experiment": exp.dir_name,
                "model_short": exp.model_short,
                "profile": exp.profile,
                "split": exp.split.value,
                "ttft_mean_re": ttft_re,
                "ttft_p99_re": ttft_p99_re,
                "e2e_mean_re": e2e_re,
                "e2e_p99_re": e2e_p99_re,
                "throughput_re": tput_re,
                "delta_used": delta,
            })
        except Exception as e:
            print(f"  {exp.dir_name}: ERROR {e}", file=sys.stderr)

    if ttft_res:
        print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        print(f"  {'Mean |RE|':<45} {statistics.mean(ttft_res):>9.1%} {'':>10} "
              f"{statistics.mean(e2e_res):>9.1%} {'':>10} "
              f"{statistics.mean(tput_res):>7.1%}")

    return ttft_res, e2e_res, tput_res, comparisons


# ─── Phase 1: Per-Model δ Grid Search ────────────────────

def search_delta_per_model(per_model_beta, alpha):
    """Grid-search δ per model using BLIS replay on the fast subset.

    For each model independently, try δ ∈ [0, 20000] µs and pick the value
    that minimizes 0.5·|TTFT_RE| + 0.3·|E2E_RE| + 0.2·|Tput_RE|.
    """
    print(f"\n{'=' * 110}")
    print(f"  PHASE 1: PER-MODEL δ GRID SEARCH (0-20ms, 500µs steps)")
    print(f"{'=' * 110}")

    best_deltas = {}

    for model, exp_dir in sorted(FAST_SUBSET.items()):
        beta = per_model_beta[model]
        gt = load_ground_truth(exp_dir)

        print(f"\n  {model} ({exp_dir}):")
        print(f"    β = [{beta[0]:.1f}, {beta[1]:.3f}, {beta[2]:.3f}]")
        print(f"    Real: TTFT={gt['ttft']['mean_ms']:.1f}ms E2E={gt['e2e']['mean_ms']:.1f}ms")

        best_delta = 0
        best_loss = float("inf")
        results = []

        for dv in DELTA_GRID:
            try:
                out = run_replay(exp_dir, beta, alpha, dv if dv > 0 else None)
                s = out["summary"]
                ttft_re = relative_error(s["ttft_mean_ms"], gt["ttft"]["mean_ms"])
                e2e_re = relative_error(s["e2e_mean_ms"], gt["e2e"]["mean_ms"])
                tput_re = relative_error(
                    s["responses_per_sec"], gt["throughput"]["requests_per_sec"]
                )
                loss = 0.5 * abs(ttft_re) + 0.3 * abs(e2e_re) + 0.2 * abs(tput_re)
                results.append((dv, ttft_re, e2e_re, tput_re, loss))
                if loss < best_loss:
                    best_loss = loss
                    best_delta = dv
            except Exception as e:
                print(f"    δ={dv}: ERROR {e}", file=sys.stderr)

        # Print top 5 results
        results.sort(key=lambda r: r[4])
        print(f"    {'δ (µs)':>8} {'TTFT RE':>10} {'E2E RE':>10} {'Tput RE':>10} {'Loss':>8}")
        for dv, tre, ere, tpre, lo in results[:5]:
            marker = " <--" if dv == best_delta else ""
            print(f"    {dv:>8} {tre:>+9.1%} {ere:>+9.1%} {tpre:>+9.1%} {lo:>7.4f}{marker}")

        best_deltas[model] = best_delta
        print(f"    Best δ = {best_delta} µs ({best_delta/1000:.1f} ms), loss = {best_loss:.4f}")

    print(f"\n  Per-model δ search results:")
    for model, delta in sorted(best_deltas.items()):
        print(f"    {model:<16}: {delta:>6} µs ({delta / 1000:.1f} ms)")

    return best_deltas


# ─── Phase 2: α Search ────────────────────────────────

def search_alpha(per_model_beta, alpha_init, per_model_delta, max_evals=50):
    """Search α₀, α₂ via Nelder-Mead using BLIS replay objective."""
    print(f"\n{'=' * 110}")
    print(f"  PHASE 2: ALPHA SEARCH (Nelder-Mead, max {max_evals} evals)")
    print(f"  Loss = 0.5·TTFT + 0.3·E2E + 0.2·Tput on fast subset")
    print(f"{'=' * 110}")

    fast_exps = [(m, d) for m, d in sorted(FAST_SUBSET.items())]
    eval_count = [0]
    best_loss = [float("inf")]

    def objective(x):
        a0, a2 = x
        if a0 < 0 or a2 < 0:
            return 10.0
        alpha = [a0, 0.0, a2]

        ttft_abs, e2e_abs, tput_abs = [], [], []
        for model, exp_dir in fast_exps:
            beta = per_model_beta[model]
            delta = per_model_delta.get(model, 0)
            try:
                out = run_replay(exp_dir, beta, alpha, delta)
                gt = load_ground_truth(exp_dir)
                s = out["summary"]
                ttft_abs.append(abs(relative_error(
                    s["ttft_mean_ms"], gt["ttft"]["mean_ms"])))
                e2e_abs.append(abs(relative_error(
                    s["e2e_mean_ms"], gt["e2e"]["mean_ms"])))
                tput_abs.append(abs(relative_error(
                    s["responses_per_sec"], gt["throughput"]["requests_per_sec"])))
            except Exception:
                return 10.0

        ttft_mape = statistics.mean(ttft_abs)
        e2e_mape = statistics.mean(e2e_abs)
        tput_mre = statistics.mean(tput_abs)
        loss = 0.5 * ttft_mape + 0.3 * e2e_mape + 0.2 * tput_mre

        eval_count[0] += 1
        marker = ""
        if loss < best_loss[0]:
            best_loss[0] = loss
            marker = " *"
        print(f"    eval {eval_count[0]:>3}: α=[{a0:>8.0f}, 0, {a2:>7.1f}] "
              f"=> TTFT={ttft_mape:.1%} E2E={e2e_mape:.1%} Tput={tput_mre:.1%} "
              f"Loss={loss:.4f}{marker}")
        sys.stdout.flush()
        return loss

    x0 = [alpha_init[0], alpha_init[2]]
    print(f"\n  Starting from α₀={x0[0]:.0f}, α₂={x0[1]:.1f}")

    result = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={
            "maxfev": max_evals,
            "xatol": 50,
            "fatol": 0.001,
            "adaptive": True,
        },
    )

    best_alpha = [result.x[0], 0.0, result.x[1]]
    print(f"\n  Converged: {result.success} ({result.message})")
    print(f"  Best α = [{result.x[0]:.1f}, 0.0, {result.x[1]:.1f}]")
    print(f"  Best loss = {result.fun:.4f} ({eval_count[0]} evals)")

    return best_alpha


# ─── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iter 4 Approach A: Journey β + Per-Model δ Search + α Search"
    )
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--max-evals", type=int, default=50,
                        help="Max Nelder-Mead evaluations for Phase 2")
    parser.add_argument("--include-test", action="store_true",
                        help="Include test set in final validation")
    parser.add_argument("--delta-preset", action="store_true",
                        help="Skip Phase 1 δ search; use Phase 1 results directly")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 110)
    print("  ITER 4 APPROACH A: Journey β + Per-Model δ Search + α Search")
    print("=" * 110)

    if not os.path.exists(REPLAY_BINARY):
        print(f"ERROR: replay binary not found at {REPLAY_BINARY}", file=sys.stderr)
        print("Build: go build -o replay training/cmd/replay/main.go", file=sys.stderr)
        sys.exit(1)

    # Load Iter 2 per-model journey β
    with open(os.path.join(TD, "iter3_physics_results.json")) as f:
        iter3 = json.load(f)
    per_model_beta_raw = iter3.get("per_model_beta_baseline", {})
    per_model_beta = {m: list(v) for m, v in per_model_beta_raw.items()}

    print(f"\n  Per-model journey β (from Iter 2, journey-constrained NNLS):")
    print(f"  {'Model':<16} {'β₀':>10} {'β₁_pf':>10} {'β₂_dc':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for m in sorted(per_model_beta.keys()):
        b = per_model_beta[m]
        print(f"  {m:<16} {b[0]:>10.1f} {b[1]:>10.3f} {b[2]:>10.3f}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: Per-Model δ Grid Search
    # ═══════════════════════════════════════════════════════
    alpha_init = list(ITER3_ALPHA)

    # Phase 1 results (from grid search on fast subset)
    PHASE1_DELTAS = {
        "codellama-34b": 4000,
        "llama-2-70b":   500,
        "llama-2-7b":    5000,
        "mixtral-8x7b":  2000,
    }

    if args.delta_preset:
        per_model_delta = PHASE1_DELTAS
        print(f"\n  Using preset δ values from Phase 1:")
        for m, d in sorted(per_model_delta.items()):
            print(f"    {m:<16}: {d:>6} µs ({d / 1000:.1f} ms)")
    else:
        per_model_delta = search_delta_per_model(per_model_beta, alpha_init)

    if args.phase == "1":
        # Still run full training eval for visibility
        train_exps = [e for e in EXPERIMENTS if e.split == Split.TRAIN]
        evaluate_experiments(
            train_exps, per_model_beta, alpha_init, per_model_delta,
            "PHASE 1 RESULT: Journey β + Searched δ + Iter 3 α (all training)")
        print(f"\n  Phase 1 done in {time.time() - t0:.1f}s")
        return

    # ═══════════════════════════════════════════════════════
    #  PHASE 2: α₀, α₂ Search via Nelder-Mead
    # ═══════════════════════════════════════════════════════
    best_alpha = search_alpha(
        per_model_beta, alpha_init, per_model_delta, args.max_evals
    )

    # Validate on all 10 training experiments
    train_exps = [e for e in EXPERIMENTS if e.split == Split.TRAIN]
    ttft2, e2e2, _, train_comps = evaluate_experiments(
        train_exps, per_model_beta, best_alpha, per_model_delta,
        "PHASE 2: Journey β + Searched δ + Optimized α (all training)")

    # ═══════════════════════════════════════════════════════
    #  VALIDATION + TEST
    # ═══════════════════════════════════════════════════════
    val_exps = [e for e in EXPERIMENTS if e.split == Split.VALIDATE]
    _, _, _, val_comps = evaluate_experiments(
        val_exps, per_model_beta, best_alpha, per_model_delta,
        "VALIDATION SET (3 experiments)")

    test_comps = []
    if args.include_test:
        test_exps = [e for e in EXPERIMENTS if e.split == Split.TEST]
        _, _, _, test_comps = evaluate_experiments(
            test_exps, per_model_beta, best_alpha, per_model_delta,
            "TEST SET (3 experiments)")

    # ═══════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ═══════════════════════════════════════════════════════
    results = {
        "iteration": "4a-journey-beta-delta-search",
        "description": "Journey β + per-model δ (searched via BLIS replay) + optimized α",
        "coefficients": {
            "per_model_beta": per_model_beta,
            "beta_form": "step_time = β₀ + β₁·cacheMissTokens + β₂·decodeTokens",
            "alpha": list(best_alpha),
            "alpha_form": "QueueingTime = α₀ + α₁·input_tokens, OutputProcessing = α₂·output_tokens",
            "per_model_delta": per_model_delta,
        },
        "comparison": {
            "iter3_alpha": ITER3_ALPHA,
        },
        "training_results": train_comps,
        "validation_results": val_comps,
        "test_results": test_comps,
    }

    out_path = os.path.join(TD, "iter4a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # ═══════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 110}")
    print(f"  Per-model β (journey-constrained, blackbox form):")
    for m in sorted(per_model_beta.keys()):
        b = per_model_beta[m]
        print(f"    {m:<16}: [{b[0]:.1f}, {b[1]:.3f}, {b[2]:.3f}]")
    print(f"  Per-model δ (searched via BLIS replay):")
    for m, d in sorted(per_model_delta.items()):
        print(f"    {m:<16}: {d} µs ({d / 1000:.1f} ms)")
    print(f"  alpha (optimized) = [{', '.join(f'{v:.1f}' for v in best_alpha)}]")

    if ttft2:
        ttft_mean = statistics.mean(ttft2)
        e2e_mean = statistics.mean(e2e2)
        passes = ttft_mean < 0.25 and e2e_mean < 0.20
        print(f"\n  Training TTFT mean |RE|: {ttft_mean:.1%} (target: <25%)")
        print(f"  Training E2E  mean |RE|: {e2e_mean:.1%} (target: <20%)")
        print(f"  Gate: {'PASS' if passes else 'FAIL'}")

    print(f"\n  Total time: {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
