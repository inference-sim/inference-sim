#!/usr/bin/env python3.11
"""
Iter31: Sequential golden section search for kernel-lookup model.

Starting from warm-start (γ=1, α₀=8000µs, α₂=200µs) calibrated to match
aiconfigurator predictions at batch=1 zero-load. Searches:
  γ₁ (GEMM+logits) → γ₂ (ctx-attn) → γ₃ (gen-attn) → α₀ → γ₁ (round 2)

γ < 1 expected: corrects for CUDA-graph speedup vs isolated kernel measurements.
α₀ expected ~5000-15000 µs: API/scheduling overhead per request.

Run from project root:
    python3.11 training/iter31_golden_section.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
BLIS_BINARY  = str(PROJECT_ROOT / "blis")
DATA_DIR     = str(TRAINING_DIR / "trainval_data")
PROFILES_DIR = str(TRAINING_DIR / "kernel_profiles")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter31"
LOG_CSV      = str(ITER_DIR / "iter31_search_log.csv")
RESULTS_JSON = str(ITER_DIR / "inner_loop_results.json")

# Warm-start: α calibrated from AIC comparison at batch=1 zero-load,
# γ=1 because kernel profiles already reflect measured silicon data.
ALPHA_WS = [8000.0, 0.0, 200.0]   # [α₀, α₁, α₂] in µs
BETA_WS  = [
    1.0,   # γ₁  index 0  GEMM + logits (includes vocab projection)  ← SEARCH 1,5
    1.0,   # γ₂  index 1  context attention (FlashAttention)         ← SEARCH 2
    1.0,   # γ₃  index 2  generation attention (PagedAttention)      ← SEARCH 3
    1.0,   # γ₄  index 3  unused (kept for alignment)
    0.0,   # γ₅  index 4  AllReduce (CUDA graph amortized; starts at 0)
    1.0,   # γ₆  index 5  MoE expert compute
    0.0,   # γ₇  index 6  µs/layer overhead
   40.0,   # γ₈  index 7  µs/request overhead
    3.0,   # γ₉  index 8  µs/step overhead
  100.0,   # γ₁₀ index 9  reserved
]

# Sequential search plan: (name, param, index, lo, hi, tol)
# "param" is "beta" or "alpha"
SEARCH_PLAN = [
    ("γ₁ gemm+logits",    "beta",  0,  0.05, 1.5,    0.005),
    ("γ₂ ctx_attn",       "beta",  1,  0.05, 1.5,    0.005),
    ("γ₃ gen_attn",       "beta",  2,  0.05, 1.5,    0.005),
    ("α₀ queueing_us",    "alpha", 0,  0.0,  30000.0, 200.0),
    ("γ₁ gemm+logits r2", "beta",  0,  0.02, 1.2,    0.003),
    ("γ₂ ctx_attn r2",    "beta",  1,  0.02, 1.2,    0.003),
    ("γ₃ gen_attn r2",    "beta",  2,  0.02, 1.2,    0.003),
]


def coeff_str(vals: list) -> str:
    return ",".join(f"{x:.10e}" for x in vals)


def eval_coeffs(alpha: list, beta: list) -> float:
    """Evaluate loss for the given alpha+beta. Returns overall_loss."""
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--latency-model", "kernel-lookup",
         "--alpha-coeffs", coeff_str(alpha),
         "--beta-coeffs",  coeff_str(beta),
         "--blis-binary",  BLIS_BINARY,
         "--data-dir",     DATA_DIR,
         "--kernel-profiles-dir", PROFILES_DIR,
         "--max-workers",  "15"],
        capture_output=True, text=True, timeout=600,
        cwd=str(TRAINING_DIR),
    )
    if r.returncode != 0:
        raise RuntimeError(f"BLIS failed: {r.stderr[:300]}")
    return json.loads(r.stdout)["overall_loss"]


def golden_section(name: str, alpha: list, beta: list,
                   param: str, idx: int,
                   lo: float, hi: float, tol: float,
                   writer: csv.writer, f_log) -> tuple[float, float]:
    """Golden section search over alpha[idx] or beta[idx] in [lo, hi]."""
    phi = (1 + 5 ** 0.5) / 2
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    cache = {}
    eval_count = [0]

    def cached(x: float) -> float:
        x = round(x, 8)
        if x not in cache:
            a_candidate = alpha[:]
            b_candidate = beta[:]
            if param == "alpha":
                a_candidate[idx] = x
            else:
                b_candidate[idx] = x
            t0 = time.time()
            loss = eval_coeffs(a_candidate, b_candidate)
            elapsed = time.time() - t0
            eval_count[0] += 1
            cache[x] = loss
            print(f"    eval {eval_count[0]:3d}: {name}={x:.6f} → loss={loss:.4f}%  ({elapsed:.1f}s)",
                  flush=True)
            writer.writerow([name, x, f"{loss:.6f}", f"{elapsed:.1f}"])
            f_log.flush()
        return cache[x]

    while abs(b - a) > tol:
        if cached(c) < cached(d):
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    best_val = round((a + b) / 2, 8)
    best_loss = cached(best_val)
    return best_val, best_loss


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(BLIS_BINARY):
        print(f"ERROR: blis binary not found at {BLIS_BINARY}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter31: kernel-lookup sequential golden section")
    print(f"  γ₁ → γ₂ → γ₃ → α₀ → γ₁(r2) → γ₂(r2) → γ₃(r2)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Warm-start: alpha={ALPHA_WS}  beta={BETA_WS[:6]}...")
    print(f"Log: {LOG_CSV}")
    print("=" * 70)

    alpha = ALPHA_WS[:]
    beta  = BETA_WS[:]
    current_loss = None
    phase_results = []

    with open(LOG_CSV, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["coeff", "value", "loss", "elapsed_s"])
        f_log.flush()

        for phase_num, (name, param, idx, lo, hi, tol) in enumerate(SEARCH_PLAN, 1):
            cur_val = alpha[idx] if param == "alpha" else beta[idx]
            print(f"\n{'─' * 70}")
            print(f"Phase {phase_num}/{len(SEARCH_PLAN)}: {name}  ∈ [{lo}, {hi}]  tol={tol}")
            print(f"  Current value: {cur_val:.6f}  |  Current loss: "
                  f"{f'{current_loss:.4f}%' if current_loss else 'unknown'}")
            print()

            t_phase = time.time()
            best_val, best_loss = golden_section(
                name, alpha, beta, param, idx, lo, hi, tol, writer, f_log
            )
            elapsed = time.time() - t_phase

            if param == "alpha":
                alpha[idx] = best_val
            else:
                beta[idx] = best_val
            current_loss = best_loss

            result = {
                "phase": phase_num, "coeff": name, "param": param, "index": idx,
                "best_value": best_val, "best_loss": best_loss,
                "elapsed_s": round(elapsed, 1),
            }
            phase_results.append(result)
            print(f"\n  ✓ Phase {phase_num} done: {name}={best_val:.6f}  loss={best_loss:.4f}%  ({elapsed:.0f}s)")

    print(f"\n{'=' * 70}")
    print(f"FINAL COEFFICIENTS")
    print(f"  alpha = {alpha}")
    print(f"  beta  = {beta}")
    print(f"  loss  = {current_loss:.4f}%")

    output = {
        "iteration": 31,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "optimization": {
            "method": "sequential_golden_section",
            "search_order": [p["coeff"] for p in phase_results],
        },
        "best_params": {"alpha": alpha, "beta": beta},
        "loss": {"overall_loss": current_loss},
        "phase_results": phase_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
