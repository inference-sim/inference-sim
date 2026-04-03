#!/usr/bin/env python3.11
"""
Iter26 golden section search for β₄ (TP All-Reduce correction).
Writes each evaluation to stdout immediately (flush=True) and to
training/iterations/iter26/iter26_search_log.csv for progress tracking.

Run from: /Users/sri/Documents/Projects/inference-sim
  python3.11 training/iter26_golden_section.py
"""

import subprocess, json, sys, csv, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
BLIS_BINARY  = str(PROJECT_ROOT / "blis")
DATA_DIR     = str(TRAINING_DIR / "trainval_data")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
LOG_CSV      = str(PROJECT_ROOT / "training/iterations/iter26/iter26_search_log.csv")

ALPHA = "15561.959717498621,776.243476414174,45.910232684500556"
# β₁ₐ, β₂ₐ, β₃, β₄(pivot), β₅, β₆, β₇, β₈, β₁ᵦ, β₂ᵦ
BETA_TEMPLATE = "0.138541,0.0,1.363060401466404,{b4},62.28932987355146,2.7976795228174027,169.36568163371626,427.3,0.0,1.2632"

os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

def eval_coeff(name, value, extra_betas=""):
    """Evaluate loss for a single coefficient value. Returns loss."""
    if name == "b4":
        beta = BETA_TEMPLATE.format(b4=value)
    elif name == "b5":
        beta = f"0.138541,0.0,1.363060401466404,{extra_betas},{value},2.7976795228174027,169.36568163371626,427.3,0.0,1.2632"
    else:
        raise ValueError(f"Unknown coeff: {name}")

    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--latency-model", "evolved",
         "--alpha-coeffs", ALPHA,
         "--beta-coeffs", beta,
         "--blis-binary", BLIS_BINARY,
         "--data-dir", DATA_DIR,
         "--max-workers", "15"],  # 15 experiments run in parallel within each trial
        capture_output=True, text=True, timeout=300,
        cwd=str(TRAINING_DIR)
    )
    if r.returncode != 0:
        raise RuntimeError(f"BLIS failed: {r.stderr[:200]}")
    return json.loads(r.stdout)["overall_loss"]


def golden_section(coeff_name, lo, hi, tol, extra=""):
    """Run golden section search. Prints and logs each evaluation."""
    phi = (1 + 5**0.5) / 2
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    cache = {}

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        def cached(x):
            x = round(x, 6)
            if x not in cache:
                t0 = time.time()
                loss = eval_coeff(coeff_name, x, extra)
                elapsed = time.time() - t0
                cache[x] = loss
                print(f"  {coeff_name}={x:.6f} → loss={loss:.4f}  ({elapsed:.0f}s)", flush=True)
                writer.writerow([coeff_name, x, loss, elapsed])
                f.flush()
            return cache[x]

        while abs(b - a) > tol:
            if cached(c) < cached(d):
                b = d
            else:
                a = c
            c = b - (b - a) / phi
            d = a + (b - a) / phi

    best = (a + b) / 2
    best_loss = eval_coeff(coeff_name, round(best, 6), extra)
    return round(best, 6), best_loss


def main():
    # Write CSV header
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["coeff", "value", "loss", "elapsed_s"])

    print("=" * 60)
    print("Iter26: Golden section search")
    print(f"Log: {LOG_CSV}")
    print("=" * 60)

    # --- Phase 1: β₄ (TP All-Reduce) ---
    print(f"\nPhase 1: β₄ ∈ [0.0, 0.5], tol=0.003")
    print(f"Baseline (iter25, β₄=0): 39.18%")
    b4_best, b4_loss = golden_section("b4", 0.0, 0.5, 0.003)
    improvement = 39.1797 - b4_loss
    print(f"\nPhase 1 result: β₄={b4_best}, loss={b4_loss:.4f}")
    print(f"Improvement over iter25: {improvement:.4f}")

    # --- Phase 2: β₅ (per-layer, optional) ---
    if improvement > 0.1:
        print(f"\nPhase 2: β₅ ∈ [40.0, 90.0], tol=0.5 (β₄={b4_best} fixed)")
        b5_best, b5_loss = golden_section("b5", 40.0, 90.0, 0.5, extra=str(b4_best))
        improvement2 = b4_loss - b5_loss
        print(f"\nPhase 2 result: β₅={b5_best}, loss={b5_loss:.4f}")
        print(f"Additional improvement: {improvement2:.4f}")
        final_loss = b5_loss
        final_b5 = b5_best
    else:
        print(f"\nPhase 2 skipped (improvement {improvement:.4f} ≤ 0.1)")
        final_loss = b4_loss
        final_b5 = 62.28932987355146  # unchanged

    print("\n" + "=" * 60)
    print("FINAL ITER26 COEFFICIENTS")
    print("=" * 60)
    print(f"  β₄ = {b4_best}")
    print(f"  β₅ = {final_b5}")
    print(f"  loss = {final_loss:.4f}%")
    print(f"  iter25 baseline = 39.18%")
    print(f"  improvement = {39.1797 - final_loss:.4f}")
    print(f"\nLog saved to: {LOG_CSV}")


if __name__ == "__main__":
    main()
