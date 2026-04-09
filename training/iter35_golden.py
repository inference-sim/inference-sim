#!/usr/bin/env -S python3.11 -u
"""
Iter35 golden section: sequential 1D coordinate search, 3 rounds.

Formula: iter35 kernel-lookup with fixed γ₇_dc × L (constant, no batch scaling).
Warm-start: iter33 best + γ₇_pf=0, γ₇_dc=187 → 55.75% (beats iter32=56.49%).

Everything runs inside the worktree to avoid path ambiguity:
  - BLIS = <worktree>/blis  (iter35 formula, built from worktree source)
  - runner = <worktree>/training/run_blis_and_compute_loss.py
  - runner cwd = <worktree>/training  (Python imports work)
  - blis cwd = <worktree>  (finds defaults.yaml, model_configs/)
  - data / profiles = absolute paths to main project (gitignored, not in worktree)

Coefficient layout (9 params):
  idx 0  alpha[0]  α₀   queueing overhead (µs)
  idx 1  gamma[0]  γ₁   GEMM + logits correction
  idx 2  gamma[1]  γ₂   FlashAttention correction
  idx 3  gamma[2]  γ₃   PagedAttention correction
  idx 4  gamma[6]  γ₇pf per-layer per-prefill-seq (µs/layer)
  idx 5  gamma[7]  γ₈   per-request overhead (µs)
  idx 6  gamma[8]  γ₉   per-step constant (µs)
  idx 7  gamma[9]  γ₇dc per-layer constant overhead (µs/layer)  [iter35 fix]
  idx 8  alpha[2]  α₂   per-output-token overhead (µs)

Sweep order (greedy: highest-sensitivity first):
  Round 1: γ₇dc, α₀, γ₁, γ₂, γ₃, γ₉, γ₈, γ₇pf, α₂
  Round 2: same order, tighter window (30% of range around current best)
  Round 3: same order, tightest window (9% of range around current best)
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

# ── All paths derived from this file's location (worktree) ────────────────────
WORKTREE     = Path(__file__).parent.parent.resolve()
TRAINING_DIR = WORKTREE / "training"
BLIS         = str(WORKTREE / "blis")
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
# Data lives in main project (gitignored — not in worktree)
DATA_DIR     = "/Users/sri/Documents/Projects/inference-sim/training/trainval_data"
PROFILES_DIR = "/Users/sri/Documents/Projects/inference-sim/training/kernel_profiles"
ITER_DIR     = TRAINING_DIR / "iterations" / "iter35"

# ── Fixed coefficients (not searched) ─────────────────────────────────────────
GAMMA_FIXED = {3: 0.0, 4: 0.00890702, 5: 1.0}   # gamma[3]=0, AllReduce, MoE
ALPHA1_FIXED = 0.0                                  # alpha[1] post-decode

# ── Warm-start (54.93% from 1D sweep; better than 55.75% X0) ─────────────────
# γ₇_pf=0 eliminates per-seq scaling; γ₇_dc=187 provides constant per-layer overhead
# matching iter32's structure but physically correct (constant, not 1/√batch).
WS = {
    "a0":   17658.51,   # α₀  queueing
    "g1":    0.08599,   # γ₁  GEMM
    "g2":    0.30223,   # γ₂  FlashAttn
    "g3":    0.93859,   # γ₃  PagedAttn
    "g7pf":      0.0,   # γ₇_pf per-prefill-seq
    "g8":    85.878,    # γ₈  per-request
    "g9":   214.415,    # γ₉  per-step
    "g7dc":   187.0,    # γ₇_dc per-layer constant
    "a2":     3.106,    # α₂  per-token
}

# ── Search specs: (name, ws_key, lo, hi, tol_r1, tol_r2, tol_r3) ─────────────
# Order: most sensitive first (greedy).  tols: absolute value tolerances.
SEARCH = [
    # Highest sensitivity: new parameter + established drivers
    ("γ₇dc",  "g7dc",   50.0,  300.0,   1.0,   0.5,   0.1),
    ("α₀",    "a0",   5000., 40000., 200.,   50.,   10.),
    ("γ₁",    "g1",     0.03,    0.20,  0.002, 0.001, 0.0002),
    ("γ₂",    "g2",     0.05,    1.20,  0.005, 0.002, 0.0005),
    ("γ₃",    "g3",     0.20,    1.80,  0.005, 0.002, 0.0005),
    # Moderate sensitivity: overhead terms
    ("γ₉",    "g9",     0.,   500.,    2.,    1.,    0.5),
    ("γ₈",    "g8",     0.,   500.,    2.,    1.,    0.5),
    ("γ₇pf",  "g7pf",   0.,    80.,    0.5,   0.2,   0.05),
    # Low sensitivity: per-token overhead
    ("α₂",    "a2",     0.,    20.,    0.1,   0.05,  0.01),
]

BLIS_WORKERS = 15   # All 15 workers → 17 experiments in ~2 rounds → ~16s/eval
PHI = (1 + math.sqrt(5)) / 2


def build_coeffs(ws: dict) -> tuple[list, list]:
    """Build (alpha, beta) from current warm-start dict."""
    alpha = [ws["a0"], ALPHA1_FIXED, ws["a2"]]
    beta  = [
        ws["g1"],            # gamma[0] γ₁
        ws["g2"],            # gamma[1] γ₂
        ws["g3"],            # gamma[2] γ₃
        GAMMA_FIXED[3],      # gamma[3] reserved=0
        GAMMA_FIXED[4],      # gamma[4] AllReduce
        GAMMA_FIXED[5],      # gamma[5] MoE
        ws["g7pf"],          # gamma[6] γ₇_pf
        ws["g8"],            # gamma[7] γ₈
        ws["g9"],            # gamma[8] γ₉
        ws["g7dc"],          # gamma[9] γ₇_dc
    ]
    return alpha, beta


def run_loss(ws: dict) -> float:
    """Evaluate loss for a coefficient dict. Max parallelism: 15 workers."""
    alpha, beta = build_coeffs(ws)
    cs = lambda v: ",".join(f"{x:.10e}" for x in v)
    try:
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--latency-model",       "kernel-lookup",
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
    name: str,
    key: str,
    ws: dict,
    lo: float,
    hi: float,
    tol: float,
    state: dict,
    writer,
    logfile,
    rnd: int,
) -> tuple[float, float]:
    """1D golden section on ws[key] ∈ [lo, hi]. Returns (best_val, best_loss)."""
    cache: dict = {}

    def cached(v: float) -> float:
        v = round(v, 8)
        if v not in cache:
            trial = dict(ws)
            trial[key] = v
            loss = run_loss(trial)
            state["evals"] += 1
            cache[v] = loss
            wall = time.time() - state["t0"]
            m, s = divmod(int(wall), 60)
            print(
                f"    {name}={v:.6f} → {loss:.4f}%"
                f"  [+{m}m{s:02d}s  #{state['evals']}]",
                flush=True,
            )
            writer.writerow([rnd, name, f"{v:.8f}", f"{loss:.6f}", f"{wall:.0f}"])
            logfile.flush()
        return cache[v]

    a, b = lo, hi
    c = b - (b - a) / PHI
    d = a + (b - a) / PHI
    while abs(b - a) > tol:
        if cached(c) < cached(d):
            b = d
        else:
            a = c
        c = b - (b - a) / PHI
        d = a + (b - a) / PHI

    best_v = round((a + b) / 2, 8)
    return best_v, cached(best_v)


def monitor(state: dict, interval_s: int = 60) -> None:
    while not state["done"]:
        time.sleep(interval_s)
        if state["done"]:
            break
        elapsed = int(time.time() - state["t0"])
        m, s = divmod(elapsed, 60)
        print(
            f"  [+{m}m{s:02d}s] best={state['best']:.4f}%"
            f"  evals={state['evals']}  phase={state['phase']}",
            flush=True,
        )


def main() -> None:
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(BLIS):
        print(f"ERROR: blis not found at {BLIS}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter35 golden section — sequential 1D coordinate search")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Worktree:   {WORKTREE}")
    print(f"BLIS:       {BLIS}")
    print(f"Target:     beat iter29 (34.57%)")
    print(f"Warm-start: γ₇_pf=0, γ₇_dc=187 (from 1D sweep, ~54.93%)")
    print(f"Workers:    {BLIS_WORKERS}/eval (~16s per evaluation)")
    print("=" * 70)

    # Evaluate warm-start
    print("\nEvaluating warm-start...", flush=True)
    t0 = time.time()
    ws_loss = run_loss(WS)
    print(f"Warm-start loss: {ws_loss:.4f}%  ({time.time()-t0:.1f}s)", flush=True)

    state = {
        "done": False, "t0": time.time(),
        "best": ws_loss, "evals": 1, "phase": "init",
    }
    mon = threading.Thread(target=monitor, args=(state,), daemon=True)
    mon.start()

    best_ws   = dict(WS)
    best_loss = ws_loss

    log_path = ITER_DIR / "iter35_golden_log.csv"
    tols_by_round = [
        {s[0]: s[4] for s in SEARCH},
        {s[0]: s[5] for s in SEARCH},
        {s[0]: s[6] for s in SEARCH},
    ]

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["round", "param", "value", "loss", "wall_s"])
        flog.flush()

        for rnd in range(3):
            print(f"\n{'─'*70}")
            print(f"Round {rnd+1}/3  (current best: {best_loss:.4f}%)", flush=True)
            state["phase"] = f"round{rnd+1}"
            round_improved = False

            for name, key, lo, hi, *_ in SEARCH:
                tol = tols_by_round[rnd][name]
                cur = best_ws[key]

                # Narrow window around current best after round 1
                if rnd == 0:
                    lo_s, hi_s = lo, hi
                else:
                    shrink = 0.30 ** rnd   # round 2: 30%, round 3: 9%
                    half   = (hi - lo) * shrink / 2
                    lo_s   = max(lo, cur - half)
                    hi_s   = min(hi, cur + half)

                print(
                    f"\n  [{rnd+1}] {name}  ∈ [{lo_s:.4f}, {hi_s:.4f}]"
                    f"  tol={tol}  cur={cur:.6f}",
                    flush=True,
                )

                best_v, loss = golden_section(
                    name, key, best_ws, lo_s, hi_s, tol,
                    state, wr, flog, rnd + 1,
                )

                if loss < best_loss:
                    best_loss = loss
                    best_ws[key] = best_v
                    state["best"] = best_loss
                    round_improved = True
                    print(
                        f"  ✓ {name}={best_v:.6f}  loss={best_loss:.4f}%  (improved)",
                        flush=True,
                    )
                else:
                    print(
                        f"  ✗ {name}={best_v:.6f}  loss={loss:.4f}%  (no improvement)",
                        flush=True,
                    )

            print(f"\nRound {rnd+1} complete: best={best_loss:.4f}%", flush=True)
            if not round_improved and rnd > 0:
                print("Converged — no improvement this round.", flush=True)
                break

    state["done"] = True
    total_wall = time.time() - state["t0"]
    alpha_f, beta_f = build_coeffs(best_ws)

    print(f"\n{'='*70}")
    print(f"GOLDEN SECTION COMPLETE")
    print(f"  Warm-start: {ws_loss:.4f}%")
    print(f"  Best found: {best_loss:.4f}%")
    print(f"  Total evals: {state['evals']}")
    print(f"  Wall time:   {total_wall:.0f}s ({total_wall/3600:.1f}h)")
    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {alpha_f}")
    print(f"  beta  = {beta_f}")
    print(f"\nPer-parameter best values:")
    for name, key, *_ in SEARCH:
        print(f"  {name:8s} = {best_ws[key]:.6f}")

    result = {
        "iteration": 35,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha_f, "beta": beta_f},
        "loss": {"overall_loss": best_loss},
        "warm_start_loss": ws_loss,
        "total_evals": state["evals"],
        "total_wall_s": round(total_wall),
    }
    out_path = ITER_DIR / "iter35_golden_results.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
