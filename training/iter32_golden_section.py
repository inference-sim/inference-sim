#!/usr/bin/env python3.11
"""
Iter32: Extended golden section search for kernel-lookup model.

Starts from iter31 best, searches the remaining overhead coefficients
(γ₇ per-layer, γ₈ per-request, γ₉ per-step, α₂ per-token) plus
additional rounds of γ₁/γ₂/γ₃ at tighter tolerances.

Run from project root:
    python3.11 training/iter32_golden_section.py
"""

import csv, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
BLIS_BINARY  = str(PROJECT_ROOT / "blis")
DATA_DIR     = str(TRAINING_DIR / "trainval_data")
PROFILES_DIR = str(TRAINING_DIR / "kernel_profiles")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter32"
LOG_CSV      = str(ITER_DIR / "iter32_search_log.csv")
RESULTS_JSON = str(ITER_DIR / "inner_loop_results.json")

# Iter31 best as warm-start
ALPHA_WS = [28345.95091932, 0.0, 200.0]
BETA_WS  = [
    0.10241521,   # γ₁  index 0  GEMM + logits
    0.49923495,   # γ₂  index 1  context attention
    0.75267741,   # γ₃  index 2  generation attention
    1.0,          # γ₄  index 3  unused
    0.0,          # γ₅  index 4  AllReduce
    1.0,          # γ₆  index 5  MoE
    0.0,          # γ₇  index 6  µs/layer overhead     ← NEW SEARCH
   40.0,          # γ₈  index 7  µs/request overhead   ← NEW SEARCH
    3.0,          # γ₉  index 8  µs/step overhead      ← NEW SEARCH
  100.0,          # γ₁₀ index 9  reserved
]

SEARCH_PLAN = [
    # Overhead terms — free parameters that can absorb systematic biases
    ("γ₇ per_layer_us",    "beta",  6,    0.0,   200.0,   1.0),
    ("γ₈ per_req_us",      "beta",  7,    0.0,  2000.0,  10.0),
    ("γ₉ per_step_us",     "beta",  8,    0.0,  1000.0,   5.0),
    ("α₂ per_token_us",    "alpha", 2,    0.0,  2000.0,  10.0),
    # AllReduce — small contribution but may matter for TP>1 models
    ("γ₅ allreduce",       "beta",  4,    0.0,     0.5,  0.002),
    # Round 3: re-tune the core gammas at tight tolerance with full context
    ("γ₁ gemm r3",         "beta",  0,    0.05,    0.3,  0.001),
    ("γ₂ ctx_attn r3",     "beta",  1,    0.2,     1.0,  0.002),
    ("γ₃ gen_attn r3",     "beta",  2,    0.3,     1.2,  0.002),
    ("α₀ queueing r2",     "alpha", 0, 15000.0, 45000.0, 100.0),
    ("γ₁ gemm r4",         "beta",  0,    0.05,    0.25, 0.001),
    ("γ₂ ctx_attn r4",     "beta",  1,    0.2,     0.9,  0.001),
    ("γ₃ gen_attn r4",     "beta",  2,    0.3,     1.1,  0.001),
]


def coeff_str(vals):
    return ",".join(f"{x:.10e}" for x in vals)


def eval_coeffs(alpha, beta):
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


def golden_section(name, alpha, beta, param, idx, lo, hi, tol, writer, f_log):
    phi = (1 + 5**0.5) / 2
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    cache = {}
    n = [0]

    def cached(x):
        x = round(x, 8)
        if x not in cache:
            ac, bc = alpha[:], beta[:]
            if param == "alpha": ac[idx] = x
            else: bc[idx] = x
            t0 = time.time()
            loss = eval_coeffs(ac, bc)
            elapsed = time.time() - t0
            n[0] += 1
            cache[x] = loss
            print(f"    eval {n[0]:3d}: {name}={x:.6f} → loss={loss:.4f}%  ({elapsed:.1f}s)", flush=True)
            writer.writerow([name, x, f"{loss:.6f}", f"{elapsed:.1f}"])
            f_log.flush()
        return cache[x]

    while abs(b - a) > tol:
        if cached(c) < cached(d): b = d
        else: a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    best = round((a + b) / 2, 8)
    return best, cached(best)


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS_BINARY):
        print(f"ERROR: blis not found", file=sys.stderr); sys.exit(1)

    print("=" * 70)
    print("Iter32: kernel-lookup extended search (overhead terms + round 3/4)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iter31 loss: 118.89%  |  Target: beat iter29 (34.57%)")
    print("=" * 70)

    alpha, beta = ALPHA_WS[:], BETA_WS[:]
    current_loss = None
    phase_results = []

    with open(LOG_CSV, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["coeff", "value", "loss", "elapsed_s"])
        f_log.flush()

        for i, (name, param, idx, lo, hi, tol) in enumerate(SEARCH_PLAN, 1):
            cur = alpha[idx] if param == "alpha" else beta[idx]
            print(f"\n{'─'*70}")
            print(f"Phase {i}/{len(SEARCH_PLAN)}: {name}  ∈ [{lo}, {hi}]  tol={tol}")
            print(f"  Current value: {cur:.6f}  |  Loss: {f'{current_loss:.4f}%' if current_loss else 'unknown'}")
            print()

            t0 = time.time()
            best_val, best_loss = golden_section(name, alpha, beta, param, idx, lo, hi, tol, writer, f_log)
            elapsed = time.time() - t0

            if param == "alpha": alpha[idx] = best_val
            else: beta[idx] = best_val
            current_loss = best_loss

            phase_results.append({"phase": i, "coeff": name, "value": best_val,
                                   "loss": best_loss, "elapsed_s": round(elapsed, 1)})
            print(f"\n  ✓ Phase {i}: {name}={best_val:.6f}  loss={best_loss:.4f}%  ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print(f"FINAL COEFFICIENTS")
    print(f"  alpha = {alpha}")
    print(f"  beta  = {beta}")
    print(f"  loss  = {current_loss:.4f}%")

    result = {
        "iteration": 32, "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha, "beta": beta},
        "loss": {"overall_loss": current_loss},
        "phase_results": phase_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
