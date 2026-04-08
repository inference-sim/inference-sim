#!/usr/bin/env python3.11
"""
Iter33 CMA-ES: Joint optimization of 9 coefficients with corrected basis functions.

Changes in iter33 vs iter32:
  1. gamma[6] = per-layer per-prefill-seq overhead (gamma7_pf)
     gamma[9] = per-layer decode overhead amortized by sqrt(batch) (gamma7_dc)
  2. FlashAttention basis function fixed for KV prefix cache hits:
     full_s = len(InputTokens) for first-step requests (ProgressIndex=0).

Free parameters (9 total):
  alpha[0] = queueing overhead (us)
  gamma[0] = GEMM + logits correction
  gamma[1] = context attention correction
  gamma[2] = generation attention correction
  gamma[6] = per-layer per-prefill-seq overhead (us)
  gamma[7] = per-request overhead (us)
  gamma[8] = per-step overhead (us)
  gamma[9] = per-layer decode overhead base (us), amortized by sqrt(batch)
  alpha[2] = per-output-token overhead (us)

Fixed: gamma[3]=1.0, gamma[4]=0.009, gamma[5]=1.0, alpha[1]=0.0.

Run from project root:
    python3.11 training/iter33_cmaes.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cma
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
BLIS_BINARY  = str(PROJECT_ROOT / "blis")
DATA_DIR     = str(TRAINING_DIR / "trainval_data")
PROFILES_DIR = str(TRAINING_DIR / "kernel_profiles")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter33"
LOG_CSV      = str(ITER_DIR / "iter33_cmaes_log.csv")
RESULTS_JSON = str(ITER_DIR / "iter33_cmaes_results.json")

# Fixed coefficients (held at iter32 best; NM updated gamma[6] and gamma[9])
ALPHA_FIXED = [17658.51401226, 0.0, 3.10562002]
BETA_FIXED  = [
    0.08599466,   # gamma[0] γ₁  GEMM
    0.30222664,   # gamma[1] γ₂  ctx attn
    0.93858709,   # gamma[2] γ₃  gen attn
    1.0,          # gamma[3] γ₄  unused (FIXED)
    0.00890702,   # gamma[4] γ₅  AllReduce (FIXED)
    1.0,          # gamma[5] γ₆  MoE (FIXED)
    0.0,          # gamma[6] γ₇_pf (NM result)
    85.87808244,  # gamma[7] γ₈  per-req
    214.41517253, # gamma[8] γ₉  per-step
    484.02994451, # gamma[9] γ₇_dc (NM result)
]

# Search vector layout (9 elements)
# x[0]  alpha[0]   α₀    queueing
# x[1]  gamma[0]   γ₁    GEMM
# x[2]  gamma[1]   γ₂    ctx attn
# x[3]  gamma[2]   γ₃    gen attn
# x[4]  gamma[6]   γ₇_pf
# x[5]  gamma[7]   γ₈    per-request
# x[6]  gamma[8]   γ₉    per-step
# x[7]  gamma[9]   γ₇_dc
# x[8]  alpha[2]   α₂    per-token

X0 = np.array([
    17658.51,   # α₀
    0.08599,    # γ₁
    0.30223,    # γ₂
    0.93859,    # γ₃
    0.0,        # γ₇_pf
    85.878,     # γ₈
    214.415,    # γ₉
    484.030,    # γ₇_dc
    3.106,      # α₂
])

BOUNDS_LO = np.array([5000.0,  0.03, 0.1,  0.2,   0.0,   0.0,   0.0,    0.0,  0.0])
BOUNDS_HI = np.array([50000.0, 0.30, 1.2,  1.8, 300.0, 500.0, 1000.0, 1000.0, 20.0])

# Parallelism: PARALLEL_JOBS candidates computed at once, each using
# BLIS_WORKERS internal workers across the 17 training experiments.
PARALLEL_JOBS = 4
BLIS_WORKERS  = 4   # 4 × 4 = 16 total BLIS processes


def x_to_coeffs(x: np.ndarray):
    alpha = ALPHA_FIXED[:]
    beta  = BETA_FIXED[:]
    alpha[0] = float(x[0])
    beta[0]  = float(x[1])
    beta[1]  = float(x[2])
    beta[2]  = float(x[3])
    beta[6]  = float(x[4])
    beta[7]  = float(x[5])
    beta[8]  = float(x[6])
    beta[9]  = float(x[7])
    alpha[2] = float(x[8])
    return alpha, beta


def coeff_str(vals):
    return ",".join(f"{x:.10e}" for x in vals)


def run_loss(x: np.ndarray) -> float:
    """Compute loss for one candidate vector. Returns inf on subprocess error."""
    xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
    alpha, beta = x_to_coeffs(xc)
    try:
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--latency-model", "kernel-lookup",
             "--alpha-coeffs", coeff_str(alpha),
             "--beta-coeffs",  coeff_str(beta),
             "--blis-binary",  BLIS_BINARY,
             "--data-dir",     DATA_DIR,
             "--kernel-profiles-dir", PROFILES_DIR,
             "--max-workers",  str(BLIS_WORKERS)],
            capture_output=True, text=True, timeout=600,
            cwd=str(TRAINING_DIR),
        )
        if r.returncode != 0:
            return float("inf")
        return json.loads(r.stdout)["overall_loss"]
    except Exception:
        return float("inf")


def run_generation(solutions: list) -> list:
    """Compute loss for an entire CMA-ES generation in parallel."""
    results = [None] * len(solutions)
    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        futures = {pool.submit(run_loss, x): i for i, x in enumerate(solutions)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS_BINARY):
        print(f"ERROR: blis not found at {BLIS_BINARY}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter33 CMA-ES: 9-parameter joint optimization")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Basis fix: FlashAttn context = len(InputTokens) for first-step prefill")
    print(f"Target: beat iter29 (34.57%)")
    print(f"Parallelism: {PARALLEL_JOBS} concurrent jobs x {BLIS_WORKERS} BLIS workers each")
    print("=" * 70)

    start_time = time.time()
    job_count  = 0
    best_loss  = float("inf")
    best_x     = X0.copy()

    with open(LOG_CSV, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["gen", "job", "a0", "g1", "g2", "g3", "g7pf",
                          "g8", "g9", "g7dc", "a2", "loss", "wall_s"])
        f_log.flush()

        opts = cma.CMAOptions()
        opts["bounds"]    = [BOUNDS_LO.tolist(), BOUNDS_HI.tolist()]
        opts["tolx"]      = 1e-4
        opts["tolfun"]    = 0.02
        opts["maxiter"]   = 300
        opts["verbose"]   = -9
        opts["CMA_stds"]  = (BOUNDS_HI - BOUNDS_LO).tolist()

        es = cma.CMAEvolutionStrategy(X0.tolist(), SIGMA0 if True else 0.25, opts)

        while not es.stop():
            solutions = es.ask()
            gen = es.result.iterations + 1

            t0 = time.time()
            losses = run_generation(solutions)
            gen_elapsed = time.time() - t0

            es.tell(solutions, losses)

            for x, loss in zip(solutions, losses):
                job_count += 1
                xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
                wall = time.time() - start_time
                if loss < best_loss:
                    best_loss = loss
                    best_x    = xc.copy()
                writer.writerow([gen, job_count,
                                  f"{xc[0]:.2f}", f"{xc[1]:.5f}", f"{xc[2]:.5f}",
                                  f"{xc[3]:.5f}", f"{xc[4]:.3f}",
                                  f"{xc[5]:.3f}", f"{xc[6]:.3f}", f"{xc[7]:.3f}",
                                  f"{xc[8]:.4f}", f"{loss:.6f}", f"{wall:.0f}"])
            f_log.flush()

            gen_best = min(losses)
            print(
                f"Gen {gen:3d}:  pop={len(solutions):2d}  gen_best={gen_best:.4f}%"
                f"  overall_best={best_loss:.4f}%  ({gen_elapsed:.0f}s)",
                flush=True,
            )

    total_elapsed = time.time() - start_time
    alpha_final, beta_final = x_to_coeffs(best_x)

    print(f"\n{'='*70}")
    print(f"CMA-ES DONE  ({es.result.stop})")
    print(f"  best loss   = {best_loss:.4f}%")
    print(f"  total jobs  = {job_count}")
    print(f"  total time  = {total_elapsed:.0f}s")
    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {alpha_final}")
    print(f"  beta  = {beta_final}")

    output = {
        "iteration": "33-cmaes",
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha_final, "beta": beta_final},
        "loss": {"overall_loss": best_loss},
        "cmaes_stop": str(es.result.stop),
        "total_jobs": job_count,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {RESULTS_JSON}")


SIGMA0 = 0.25

if __name__ == "__main__":
    main()
