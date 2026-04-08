#!/usr/bin/env python3.11
"""
Iter33: Joint 2D Nelder-Mead search for split overhead coefficients.

Root cause from iter32 analysis: gamma[6] = 187µs/layer was calibrated as a
constant per-step overhead but was absorbed entirely as decode overhead. At
high concurrency (8 decode requests × 28 layers), this 5.2ms constant overhead
dominates step time, inflating TTFT predictions at high load.

Fix (Option A): Split gamma[6] into two phase-specific terms:
  - gamma[6] = γ₇_pf: µs/layer per prefill sequence in batch (scales with #sequences)
  - gamma[9] = γ₇_dc: µs/layer decode overhead base, amortized as base/√decodeReqs

Search strategy: Joint 2D Nelder-Mead over (gamma[6], gamma[9]).
All other coefficients held at iter32 best values.

Run from project root:
    python3.11 training/iter33_joint_search.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
BLIS_BINARY  = str(PROJECT_ROOT / "blis")
DATA_DIR     = str(TRAINING_DIR / "trainval_data")
PROFILES_DIR = str(TRAINING_DIR / "kernel_profiles")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter33"
LOG_CSV      = str(ITER_DIR / "iter33_search_log.csv")
RESULTS_JSON = str(ITER_DIR / "inner_loop_results.json")

# Iter32 best — hold all fixed except gamma[6] and gamma[9]
ALPHA_FIXED = [17658.51401226, 0.0, 3.10562002]
BETA_FIXED  = [
    0.08599466,   # gamma[0] = γ₁  GEMM + logits       (FIXED)
    0.30222664,   # gamma[1] = γ₂  context attention    (FIXED)
    0.93858709,   # gamma[2] = γ₃  generation attention (FIXED)
    1.0,          # gamma[3] = γ₄  unused               (FIXED)
    0.00890702,   # gamma[4] = γ₅  AllReduce            (FIXED)
    1.0,          # gamma[5] = γ₆  MoE                  (FIXED)
    0.0,          # gamma[6] = γ₇_pf  ← SEARCH (was 187.15 constant, now per-prefill-seq)
    85.87808244,  # gamma[7] = γ₈  per-request overhead (FIXED)
    214.41517253, # gamma[8] = γ₉  per-step overhead    (FIXED)
   187.15494451,  # gamma[9] = γ₇_dc ← SEARCH (new: amortized decode overhead base)
]

# NM warm start: (gamma[6], gamma[9])
# gamma[6] starts at 0 — old 187 was absorbing decode overhead, not prefill overhead.
# gamma[9] starts at 187.15 — same magnitude as old gamma[6], now properly amortized.
X0 = np.array([0.0, 187.15494451])

# Search bounds (used for initial simplex construction, not enforced as hard constraints)
BOUNDS_LO = np.array([0.0,   0.0])
BOUNDS_HI = np.array([300.0, 500.0])

# Initial simplex: x0, x0+step_6, x0+step_9
NM_INITIAL_SIMPLEX = np.array([
    X0,
    X0 + np.array([30.0,   0.0]),  # explore γ₇_pf axis
    X0 + np.array([ 0.0, 50.0]),   # explore γ₇_dc axis
])

# Parallelism: each NM function evaluation runs all experiments in parallel.
# Set as high as CPU count allows since experiments are I/O + subprocess bound.
MAX_EVAL_WORKERS = 15

class _State:
    eval_count: int = 0
    log_writer: Optional[Any] = None
    log_file: Optional[Any] = None
    start_time: float = 0.0

_state = _State()


def coeff_str(vals):
    return ",".join(f"{x:.10e}" for x in vals)


def eval_at(gamma6, gamma9):
    """Evaluate loss with gamma[6]=gamma6, gamma[9]=gamma9, all others fixed."""
    beta = BETA_FIXED[:]
    beta[6] = gamma6
    beta[9] = gamma9
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--latency-model", "kernel-lookup",
         "--alpha-coeffs", coeff_str(ALPHA_FIXED),
         "--beta-coeffs",  coeff_str(beta),
         "--blis-binary",  BLIS_BINARY,
         "--data-dir",     DATA_DIR,
         "--kernel-profiles-dir", PROFILES_DIR,
         "--max-workers",  str(MAX_EVAL_WORKERS)],
        capture_output=True, text=True, timeout=600,
        cwd=str(TRAINING_DIR),
    )
    if r.returncode != 0:
        raise RuntimeError(f"BLIS failed: {r.stderr[:300]}")
    return json.loads(r.stdout)["overall_loss"]


def objective(x):
    """Nelder-Mead objective: clamp to [0, ∞) then evaluate."""
    gamma6 = float(np.clip(x[0], BOUNDS_LO[0], BOUNDS_HI[0]))
    gamma9 = float(np.clip(x[1], BOUNDS_LO[1], BOUNDS_HI[1]))

    t0 = time.time()
    loss = eval_at(gamma6, gamma9)
    elapsed = time.time() - t0

    _state.eval_count += 1
    n = _state.eval_count
    wall = time.time() - _state.start_time

    print(
        f"  eval {n:3d}: γ₇_pf={gamma6:8.4f}  γ₇_dc={gamma9:8.4f}"
        f"  → loss={loss:.4f}%  ({elapsed:.1f}s, +{wall:.0f}s total)",
        flush=True,
    )
    if _state.log_writer is not None and _state.log_file is not None:
        _state.log_writer.writerow([n, gamma6, gamma9, f"{loss:.6f}", f"{elapsed:.1f}"])
        _state.log_file.flush()

    return loss


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS_BINARY):
        print(f"ERROR: blis not found at {BLIS_BINARY}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter33: Joint 2D Nelder-Mead for (γ₇_pf, γ₇_dc)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iter32 loss: 56.49%  |  Target: beat iter29 (34.57%)")
    print(f"All other coeffs held at iter32 best.")
    print(f"Warm start: γ₇_pf={X0[0]:.4f}  γ₇_dc={X0[1]:.4f}")
    print("=" * 70)

    _state.start_time = time.time()

    with open(LOG_CSV, "w", newline="") as f_log:
        _state.log_file = f_log
        writer = csv.writer(f_log)
        writer.writerow(["eval", "gamma6_pf", "gamma9_dc", "loss", "elapsed_s"])
        f_log.flush()
        _state.log_writer = writer

        # Evaluate warm-start point first for reference
        print(f"\nWarm-start evaluation (iter32 gamma[6]=0, gamma[9]=187.15):")
        ws_loss = objective(X0)
        print(f"  Warm-start loss: {ws_loss:.4f}%\n")

        print(f"Starting Nelder-Mead with initial simplex:")
        for i, v in enumerate(NM_INITIAL_SIMPLEX):
            print(f"  vertex {i}: γ₇_pf={v[0]:.2f}  γ₇_dc={v[1]:.2f}")
        print()

        result = minimize(
            objective,
            X0,
            method="Nelder-Mead",
            options={
                "initial_simplex": NM_INITIAL_SIMPLEX,
                "xatol": 0.5,       # convergence tolerance on coefficient values
                "fatol": 0.05,      # convergence tolerance on loss (% units)
                "maxiter": 200,
                "disp": True,
                "adaptive": True,   # adaptive NM: better for non-smooth objectives
            },
        )

    total_elapsed = time.time() - _state.start_time

    best_gamma6 = float(np.clip(result.x[0], BOUNDS_LO[0], BOUNDS_HI[0]))
    best_gamma9 = float(np.clip(result.x[1], BOUNDS_LO[1], BOUNDS_HI[1]))
    best_loss   = result.fun

    print(f"\n{'='*70}")
    print(f"NELDER-MEAD CONVERGED ({result.message})")
    print(f"  γ₇_pf (gamma[6]) = {best_gamma6:.6f}")
    print(f"  γ₇_dc (gamma[9]) = {best_gamma9:.6f}")
    print(f"  loss             = {best_loss:.4f}%")
    print(f"  evaluations      = {_state.eval_count}")
    print(f"  total time       = {total_elapsed:.0f}s")

    beta_final = BETA_FIXED[:]
    beta_final[6] = best_gamma6
    beta_final[9] = best_gamma9

    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {ALPHA_FIXED}")
    print(f"  beta  = {beta_final}")

    output = {
        "iteration": 33,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": ALPHA_FIXED, "beta": beta_final},
        "loss": {"overall_loss": best_loss},
        "warm_start_loss": ws_loss,
        "nm_result": {
            "success": result.success,
            "message": result.message,
            "nit": result.nit,
            "nfev": result.nfev,
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
