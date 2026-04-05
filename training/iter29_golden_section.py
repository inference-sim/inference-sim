#!/usr/bin/env python3.11
"""
Iter29: Sequential golden section search over 5 coefficients.

Search order: β₃ → β₆ → β₅ → β₈ → β₂ᵦ
Each search starts from the best value found by the previous search.
All 15 experiments are parallelized within each evaluation (--max-workers 15).

Run from project root:
    python3.11 training/iter29_golden_section.py
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
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter29"
LOG_CSV      = str(ITER_DIR / "iter29_search_log.csv")
RESULTS_JSON = str(ITER_DIR / "inner_loop_results.json")

# iter27 best — starting point for all searches
ALPHA = "15563.199579,777.3455,45.907545"
BETA_ITER27 = [
    0.152128,    # β₁ₐ  index 0  prefill compute
    0.0,         # β₂ₐ  index 1  frozen=0
    1.363621,    # β₃   index 2  weight loading       ← SEARCH 1
    0.752037,    # β₄   index 3  TP All-Reduce
    32.394131,   # β₅   index 4  per-layer            ← SEARCH 3
    2.805128,    # β₆   index 5  per-request          ← SEARCH 2
    126.024825,  # β₇   index 6  per-step constant
    505.508488,  # β₈   index 7  MoE overhead         ← SEARCH 4
    0.0,         # β₁ᵦ  index 8  frozen=0
    1.922366,    # β₂ᵦ  index 9  decode memory        ← SEARCH 5
]

# Sequential search plan: (display_name, beta_index, lo, hi, tol)
SEARCH_PLAN = [
    ("β₃ weight_loading",  2,  0.5,   3.0,  0.015),
    ("β₆ per_request",     5,  0.5,   8.0,  0.05),
    ("β₅ per_layer",       4,  5.0,  80.0,  0.5),
    ("β₈ moe_overhead",    7, 200.0, 900.0,  5.0),
    ("β₂ᵦ decode_memory",  9,  0.5,   5.0,  0.02),
]


def beta_str(beta: list) -> str:
    return ",".join(f"{x:.10e}" for x in beta)


def eval_beta(beta: list) -> float:
    """Evaluate loss for a full beta vector. Returns overall_loss."""
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--latency-model", "evolved",
         "--alpha-coeffs", ALPHA,
         "--beta-coeffs", beta_str(beta),
         "--blis-binary", BLIS_BINARY,
         "--data-dir", DATA_DIR,
         "--max-workers", "15"],
        capture_output=True, text=True, timeout=300,
        cwd=str(TRAINING_DIR),
    )
    if r.returncode != 0:
        raise RuntimeError(f"BLIS failed: {r.stderr[:300]}")
    return json.loads(r.stdout)["overall_loss"]


def golden_section(name: str, beta: list, idx: int, lo: float, hi: float, tol: float,
                   writer: csv.writer, f_log) -> tuple[float, float]:
    """
    Golden section search over beta[idx] in [lo, hi].
    All other beta values are held fixed.
    Returns (best_value, best_loss).
    """
    phi = (1 + 5 ** 0.5) / 2
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    cache = {}
    eval_count = [0]

    def cached(x: float) -> float:
        x = round(x, 8)
        if x not in cache:
            candidate = beta[:]
            candidate[idx] = x
            t0 = time.time()
            loss = eval_beta(candidate)
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
    # Final eval at midpoint (may be cached)
    best_loss = cached(best_val)
    return best_val, best_loss


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Iter29: Sequential golden section — β₃ → β₆ → β₅ → β₈ → β₂ᵦ")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iter27 baseline: 34.6564%")
    print(f"Log: {LOG_CSV}")
    print("=" * 65)

    # Verify blis binary
    if not os.path.exists(BLIS_BINARY):
        print(f"ERROR: blis binary not found at {BLIS_BINARY}", file=sys.stderr)
        sys.exit(1)

    beta = BETA_ITER27[:]
    current_loss = None
    phase_results = []

    with open(LOG_CSV, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["coeff", "value", "loss", "elapsed_s"])
        f_log.flush()

        for phase_num, (name, idx, lo, hi, tol) in enumerate(SEARCH_PLAN, 1):
            print(f"\n{'─' * 65}")
            print(f"Phase {phase_num}/5: {name}  ∈ [{lo}, {hi}]  tol={tol}")
            print(f"  Starting value: {beta[idx]:.6f}")
            if current_loss is not None:
                print(f"  Current loss: {current_loss:.4f}%")
            print()

            t_phase = time.time()
            best_val, best_loss = golden_section(
                name, beta, idx, lo, hi, tol, writer, f_log
            )
            elapsed_phase = time.time() - t_phase

            improvement = (current_loss - best_loss) if current_loss is not None else 0.0
            print(f"\n  ✓ Phase {phase_num} done: {name}={best_val:.6f}  "
                  f"loss={best_loss:.4f}%  ({elapsed_phase:.0f}s)")
            if current_loss is not None:
                delta = current_loss - best_loss
                sign = "↓" if delta > 0 else ("↑" if delta < 0 else "—")
                print(f"    Δ vs previous: {sign}{abs(delta):.4f}")

            phase_results.append({
                "phase": phase_num,
                "coeff": name,
                "beta_index": idx,
                "old_value": BETA_ITER27[idx],
                "best_value": best_val,
                "best_loss": best_loss,
                "elapsed_s": round(elapsed_phase, 1),
            })

            beta[idx] = best_val
            current_loss = best_loss

    # Save results JSON
    output = {
        "iteration": 29,
        "backend_name": "evolved",
        "timestamp": datetime.now().isoformat(),
        "optimization": {
            "method": "sequential_golden_section",
            "search_order": [p["coeff"] for p in phase_results],
        },
        "best_params": {
            "alpha": [float(x) for x in ALPHA.split(",")],
            "beta": beta,
        },
        "loss": {
            "overall_loss": current_loss,
            "iter27_baseline": 34.6564,
            "improvement": round(34.6564 - current_loss, 4) if current_loss else None,
        },
        "phase_results": phase_results,
    }
    with open(RESULTS_JSON, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n" + "=" * 65)
    print("ITER29 COMPLETE")
    print("=" * 65)
    print(f"Final loss:      {current_loss:.4f}%")
    print(f"Iter27 baseline: 34.6564%")
    delta = 34.6564 - current_loss
    print(f"Improvement:     {delta:+.4f} points")
    print(f"\nFinal beta vector:")
    names = ["β₁ₐ", "β₂ₐ", "β₃", "β₄", "β₅", "β₆", "β₇", "β₈", "β₁ᵦ", "β₂ᵦ"]
    for i, (n, v_old, v_new) in enumerate(zip(names, BETA_ITER27, beta)):
        changed = " ←" if abs(v_old - v_new) > 1e-9 else ""
        print(f"  {n:6s} [idx {i}]: {v_old:.6f} → {v_new:.6f}{changed}")
    print(f"\nResults: {RESULTS_JSON}")
    print(f"Log:     {LOG_CSV}")


if __name__ == "__main__":
    main()
