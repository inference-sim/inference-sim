#!/usr/bin/env -S python3.11 -u
"""
Iter36: Golden section search on trained-physics model.

Goal: replicate iter29 (34.57%) with post-bugfix blis binary.
Bugfixes since iter29: session ClientID (#974), SLOClass (#965), workload fixes.
Iter29 coefficients give 35.67% with current binary — need to re-calibrate.

Same process as iter29 but:
  - All paths derived from this file's location (worktree) — no path ambiguity
  - All 9 free parameters searched (iter29 only searched 5)
  - 3 rounds with shrinking windows
  - 15 BLIS workers/eval for max parallelism (~16s/eval)

Greedy order (based on iter29 sensitivity: β₆ was +57%, β₃ +1%, others small):
  β₆ per_request, β₃ weight_loading, β₅ per_layer, β₈ moe_overhead,
  β₂ᵦ decode_memory, α₀ queueing, β₄ tp_allreduce, β₁ₐ prefill_compute,
  β₇ per_step, α₁ post_decode, α₂ per_token

Run from worktree root:
    python3.11 training/iter36_golden.py
"""

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── All paths from this file's location ──────────────────────────────────────
WORKTREE     = Path(__file__).parent.parent.resolve()
TRAINING_DIR = WORKTREE / "training"
BLIS         = str(WORKTREE / "blis")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
DATA_DIR     = "/Users/sri/Documents/Projects/inference-sim/training/trainval_data"
PROFILES_DIR = "/Users/sri/Documents/Projects/inference-sim/training/kernel_profiles"
ITER_DIR     = TRAINING_DIR / "iterations" / "iter36"

# ── Warm-start: iter29 best coefficients ─────────────────────────────────────
# These give 35.67% with current binary (vs 34.57% with iter29 binary).
# Post-bugfix delta is ~1.1pp — golden section should recover this.
WS_ALPHA = [15563.199579, 777.3455, 45.907545]
WS_BETA  = [
    0.152128,     # β₁ₐ  index 0  prefill compute
    0.0,          # β₂ₐ  index 1  FROZEN = 0
    1.36252915,   # β₃   index 2  weight loading
    0.752037,     # β₄   index 3  TP AllReduce
    32.09546717,  # β₅   index 4  per-layer overhead
    4.41684444,   # β₆   index 5  per-request overhead
    126.024825,   # β₇   index 6  per-step constant
    481.8613888,  # β₈   index 7  MoE overhead
    0.0,          # β₁ᵦ  index 8  FROZEN = 0
    1.94710771,   # β₂ᵦ  index 9  decode memory
]

# ── Search specs: (name, param_type, index, lo, hi, tol_r1, tol_r2, tol_r3) ─
# Greedy order: highest sensitivity first (based on iter29 findings).
# param_type: "alpha" or "beta"
SEARCH = [
    ("β₆ per_req",    "beta",  5,  0.5,    10.0,  0.05,  0.02, 0.005),
    ("β₃ wt_load",    "beta",  2,  0.3,     4.0,  0.015, 0.006, 0.002),
    ("β₅ per_layer",  "beta",  4,  5.0,    80.0,  0.5,   0.2,   0.05),
    ("β₈ moe",        "beta",  7, 100.0,  900.0,  5.0,   2.0,   0.5),
    ("β₂ᵦ dc_mem",    "beta",  9,  0.3,     5.0,  0.02,  0.008, 0.002),
    ("α₀ queueing",   "alpha", 0, 5000.,30000.,  200.,   80.,   20.),
    ("β₄ tp_ar",      "beta",  3,  0.1,     3.0,  0.015, 0.006, 0.002),
    ("β₁ₐ pf_cmp",   "beta",  0,  0.05,    0.5,  0.005, 0.002, 0.0005),
    ("β₇ per_step",   "beta",  6, 20.0,   400.0,  2.0,   0.8,   0.2),
    ("α₁ post_dc",    "alpha", 1,  0.,   3000.,  20.,    8.,    2.),
    ("α₂ per_tok",    "alpha", 2,  0.,    200.,   2.,    0.8,   0.2),
]

BLIS_WORKERS = 15
PHI = (1 + math.sqrt(5)) / 2


def run_loss(alpha: list, beta: list) -> float:
    cs = lambda v: ",".join(f"{x:.10e}" for x in v)
    try:
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--latency-model",       "trained-physics",
             "--alpha-coeffs",        cs(alpha),
             "--beta-coeffs",         cs(beta),
             "--blis-binary",         BLIS,
             "--data-dir",            DATA_DIR,
             "--kernel-profiles-dir", PROFILES_DIR,
             "--max-workers",         str(BLIS_WORKERS)],
            capture_output=True, text=True, timeout=90,
            cwd=str(TRAINING_DIR),
        )
        if r.returncode != 0:
            return float("inf")
        return float(json.loads(r.stdout)["overall_loss"])
    except Exception:
        return float("inf")


def golden_section(
    name: str, ptype: str, idx: int,
    alpha: list, beta: list,
    lo: float, hi: float, tol: float,
    state: dict, writer, logfile, rnd: int,
) -> tuple:
    cache = {}

    def cached(v):
        v = round(v, 8)
        if v not in cache:
            a, b = list(alpha), list(beta)
            if ptype == "alpha":
                a[idx] = v
            else:
                b[idx] = v
            loss = run_loss(a, b)
            state["evals"] += 1
            cache[v] = loss
            wall = time.time() - state["t0"]
            m, s = divmod(int(wall), 60)
            print(f"    {name}={v:.6f} → {loss:.4f}%  [+{m}m{s:02d}s  #{state['evals']}]",
                  flush=True)
            writer.writerow([rnd, name, f"{v:.8f}", f"{loss:.6f}", f"{wall:.0f}"])
            logfile.flush()
        return cache[v]

    a_, b_ = lo, hi
    c = b_ - (b_ - a_) / PHI
    d = a_ + (b_ - a_) / PHI
    while abs(b_ - a_) > tol:
        if cached(c) < cached(d):
            b_ = d
        else:
            a_ = c
        c = b_ - (b_ - a_) / PHI
        d = a_ + (b_ - a_) / PHI

    best_v = round((a_ + b_) / 2, 8)
    return best_v, cached(best_v)


def monitor(state, interval_s=60):
    while not state["done"]:
        time.sleep(interval_s)
        if state["done"]:
            break
        elapsed = int(time.time() - state["t0"])
        m, s = divmod(elapsed, 60)
        print(f"  [+{m}m{s:02d}s] best={state['best']:.4f}%  "
              f"evals={state['evals']}  {state['phase']}", flush=True)


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS):
        print(f"ERROR: blis not found at {BLIS}", file=sys.stderr); sys.exit(1)

    print("=" * 70)
    print("Iter36: trained-physics golden section")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Worktree:   {WORKTREE}")
    print(f"BLIS:       {BLIS}")
    print(f"Target:     ≤ iter29 (34.57%)")
    print(f"Warm-start: iter29 best (35.67% with current binary)")
    print(f"Workers:    {BLIS_WORKERS}/eval")
    print("=" * 70)

    print("\nEvaluating warm-start...", flush=True)
    ws_loss = run_loss(WS_ALPHA, WS_BETA)
    print(f"Warm-start: {ws_loss:.4f}%", flush=True)

    state = {"done": False, "t0": time.time(),
             "best": ws_loss, "evals": 1, "phase": "init"}
    threading.Thread(target=monitor, args=(state,), daemon=True).start()

    alpha = list(WS_ALPHA)
    beta  = list(WS_BETA)
    best_loss = ws_loss

    tols = [
        {s[0]: s[5] for s in SEARCH},   # tol_r1
        {s[0]: s[6] for s in SEARCH},   # tol_r2
        {s[0]: s[7] for s in SEARCH},   # tol_r3
    ]

    log_path = ITER_DIR / "iter36_golden_log.csv"
    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["round", "param", "value", "loss", "wall_s"])
        flog.flush()

        for rnd in range(3):
            print(f"\n{'─'*70}")
            print(f"Round {rnd+1}/3  (current best: {best_loss:.4f}%)", flush=True)
            state["phase"] = f"round{rnd+1}"
            round_improved = False

            for name, ptype, idx, lo, hi, *_ in SEARCH:
                tol = tols[rnd][name]
                cur = alpha[idx] if ptype == "alpha" else beta[idx]

                if rnd == 0:
                    lo_s, hi_s = lo, hi
                else:
                    shrink = 0.30 ** rnd
                    half   = (hi - lo) * shrink / 2
                    lo_s   = max(lo, cur - half)
                    hi_s   = min(hi, cur + half)

                print(f"\n  [{rnd+1}] {name}  ∈ [{lo_s:.4f}, {hi_s:.4f}]"
                      f"  tol={tol}  cur={cur:.6f}", flush=True)

                best_v, loss = golden_section(
                    name, ptype, idx, alpha, beta,
                    lo_s, hi_s, tol, state, wr, flog, rnd + 1,
                )

                if loss < best_loss:
                    best_loss = loss
                    if ptype == "alpha":
                        alpha[idx] = best_v
                    else:
                        beta[idx]  = best_v
                    state["best"] = best_loss
                    round_improved = True
                    print(f"  ✓ {name}={best_v:.6f}  loss={best_loss:.4f}%", flush=True)
                else:
                    print(f"  ✗ {name}={best_v:.6f}  loss={loss:.4f}%", flush=True)

            print(f"\nRound {rnd+1}: best={best_loss:.4f}%", flush=True)
            if not round_improved and rnd > 0:
                print("Converged.", flush=True)
                break

    state["done"] = True
    total_wall = time.time() - state["t0"]
    print(f"\n{'='*70}")
    print(f"DONE  best={best_loss:.4f}%  evals={state['evals']}"
          f"  wall={total_wall:.0f}s ({total_wall/3600:.1f}h)")
    print(f"alpha = {alpha}")
    print(f"beta  = {beta}")

    result = {
        "iteration": 36, "backend_name": "trained-physics",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha, "beta": beta},
        "loss": {"overall_loss": best_loss},
        "warm_start_loss": ws_loss,
        "total_evals": state["evals"],
        "total_wall_s": round(total_wall),
    }
    out = ITER_DIR / "iter36_golden_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
