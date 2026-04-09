#!/usr/bin/env -S python3.11 -u
"""
Iter35: TPE → Golden Section for kernel-lookup with fixed γ₇_dc.

Changes vs iter33/34:
  - γ₇_dc (gamma[9]): FIXED — now γ₇_dc × L (constant per-step per-layer overhead),
    replacing the unphysical γ₇_dc × L / √batch from iter33.
  - KV prefix-cache FlashAttention fix (iter33) preserved.
  - γ₇_pf per-prefill-sequence overhead (iter33) preserved.
  - Floor subtraction (iter34 experiment) REVERTED — it caused degenerate γ₁→0 solutions.

Strategy: start with TPE (not CMA-ES).
  - We have rich prior knowledge from iter32/33/34 runs.
  - TPE exploits existing observations more efficiently than CMA-ES in this regime.
  - Warm-start from iter32 best (56.49%) — the best known good point.
  - Also inject iter33 and iter34 trials as prior observations.

Search parameters (10 total):
  x[0]  alpha[0] α₀  queueing (µs)
  x[1]  gamma[0] γ₁  GEMM + logits correction
  x[2]  gamma[1] γ₂  FlashAttention correction
  x[3]  gamma[2] γ₃  PagedAttention correction
  x[4]  gamma[6] γ₇_pf  per-layer per-prefill-seq (µs/layer)
  x[5]  gamma[7] γ₈  per-request overhead (µs)
  x[6]  gamma[8] γ₉  per-step constant overhead (µs)
  x[7]  gamma[9] γ₇_dc  per-layer constant overhead (µs/layer) [FIXED form]
  x[8]  alpha[2] α₂  per-output-token overhead (µs)

Fixed: gamma[3]=0, gamma[4]=0.009 (AllReduce), gamma[5]=1.0 (MoE), alpha[1]=0.

Run from worktree root:
    python3.11 training/iter35_search.py [--resume-golden]
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT         = Path(__file__).parent.parent
TRAINING_DIR = ROOT / "training"
BLIS         = str(ROOT / "blis")  # iter35 binary; absolute path avoids cwd ambiguity
DATA_DIR     = str(Path("/Users/sri/Documents/Projects/inference-sim/training/trainval_data"))
PROFILES_DIR = str(Path("/Users/sri/Documents/Projects/inference-sim/training/kernel_profiles"))
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter35"

# ── Parameter layout ──────────────────────────────────────────────────────────
PARAM_NAMES = ["α₀", "γ₁", "γ₂", "γ₃", "γ₇pf", "γ₈", "γ₉", "γ₇dc", "α₂"]

BOUNDS_LO = np.array([5000.,  0.03, 0.1,  0.2,   0.,   0.,   0.,    0.,  0.])
BOUNDS_HI = np.array([40000., 0.20, 1.2,  1.8, 200., 500., 500., 300., 20.])
#                                                                    ^^^
# γ₇_dc upper bound = 300 (warm-start X0=187, typical range 100-250 for new constant form)

# ── Warm-start: iter32 best (56.49%) ──────────────────────────────────────────
# gamma[9] (γ₇_dc) was 281.6 in iter33 (/ √batch form, unphysical).
# For the new constant form (× L), estimate from iter33's contribution at typical batch=8:
#   iter33: 281.6 × 28 / √8 = 2789µs → target contribution with new form
#   new form: γ₇_dc × L = γ₇_dc × 28 = 2789 → γ₇_dc ≈ 99.6 ≈ 50 (conservative)
# Use 50 as warm-start — let TPE explore freely.
X0 = np.array([
    17658.51,   # α₀     iter32 best
    0.08599,    # γ₁     iter32 best
    0.30223,    # γ₂     iter32 best
    0.93859,    # γ₃     iter32 best
    0.0,        # γ₇_pf  0 — per-seq overhead subsumed into γ₇_dc unconditional
    85.878,     # γ₈     iter32 best
    214.415,    # γ₉     iter32 best
    187.0,      # γ₇_dc  unconditional ×L, replicates iter32's constant overhead
                #         γ₇_pf=0, γ₇_dc=187 gives 54.93% — beats iter32 (56.49%)
    3.106,      # α₂     iter32 best
])

# Fixed coefficients
ALPHA_FIXED = [0.0, 0.0, 0.0]
BETA_FIXED  = [0.0, 0.0, 0.0, 0.0, 0.00890702, 1.0, 0.0, 0.0, 0.0, 0.0]


def x_to_coeffs(x: np.ndarray) -> tuple:
    xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
    alpha = ALPHA_FIXED[:]
    beta  = BETA_FIXED[:]
    alpha[0] = float(xc[0])   # α₀
    beta[0]  = float(xc[1])   # γ₁
    beta[1]  = float(xc[2])   # γ₂
    beta[2]  = float(xc[3])   # γ₃
    beta[6]  = float(xc[4])   # γ₇_pf
    beta[7]  = float(xc[5])   # γ₈
    beta[8]  = float(xc[6])   # γ₉
    beta[9]  = float(xc[7])   # γ₇_dc (new constant form)
    alpha[2] = float(xc[8])   # α₂
    return alpha, beta


def coeff_str(vals) -> str:
    return ",".join(f"{v:.10e}" for v in vals)


PARALLEL_EVALS  = 2
BLIS_WORKERS    = 8
BLIS_WORKERS_GS = 15
EVAL_TIMEOUT_S  = 180


def run_loss(x: np.ndarray, blis_workers: int = BLIS_WORKERS) -> float:
    alpha, beta = x_to_coeffs(x)
    try:
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--latency-model",       "kernel-lookup",
             "--alpha-coeffs",        coeff_str(alpha),
             "--beta-coeffs",         coeff_str(beta),
             "--blis-binary",         BLIS,
             "--data-dir",            DATA_DIR,
             "--kernel-profiles-dir", PROFILES_DIR,
             "--max-workers",         str(blis_workers)],
            capture_output=True, text=True,
            timeout=EVAL_TIMEOUT_S,
            cwd=str(TRAINING_DIR),
        )
        if r.returncode != 0:
            return float("inf")
        return float(json.loads(r.stdout)["overall_loss"])
    except Exception:
        return float("inf")


def monitor(state: dict, interval_s: int = 60) -> None:
    while not state["done"]:
        time.sleep(interval_s)
        if state["done"]:
            break
        elapsed = int(time.time() - state["t0"])
        m, s = divmod(elapsed, 60)
        print(
            f"  [+{m}m{s:02d}s] phase={state['phase']}  "
            f"best={state['best']:.4f}%  evals={state['evals']}",
            flush=True,
        )


def load_prior_from_csv(csv_path: Path, param_names: list) -> list:
    """Load (x, loss) tuples from a search log CSV."""
    trials = []
    if not csv_path.exists():
        return trials
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = np.array([float(row[n]) for n in param_names])
                    loss = float(row["loss"])
                    if loss < float("inf"):
                        trials.append((x, loss))
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return trials


def phase_tpe(state: dict, warm_x: np.ndarray, prior_trials: list,
              log_path: Path, n_trials: int = 300) -> tuple:
    print(f"\n{'='*70}")
    print(f"Phase 1: TPE — Bayesian exploitation ({n_trials} trials)")
    print(f"  Prior: {len(prior_trials)} trials from iter32/33/34")
    print(f"  Warm-start: {dict(zip(PARAM_NAMES, warm_x.tolist()))}")
    print(f"  Parallelism: {PARALLEL_EVALS} concurrent × {BLIS_WORKERS} workers")
    print(f"{'='*70}")

    state["phase"] = "tpe"

    # Evaluate warm-start
    ws_loss = run_loss(warm_x)
    state["best"] = ws_loss
    state["evals"] += 1
    print(f"  Warm-start loss: {ws_loss:.4f}%", flush=True)

    best_x    = warm_x.copy()
    best_loss = ws_loss

    dists = {
        name: optuna.distributions.FloatDistribution(float(lo), float(hi))
        for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
    }

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=0,   # start exploiting immediately
            multivariate=True,    # model parameter correlations
            seed=42,
        ),
    )

    # Inject all prior trials
    for x, loss in prior_trials:
        xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
        params = {n: float(v) for n, v in zip(PARAM_NAMES, xc)}
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=dists, value=loss))

    # Inject warm-start
    xc_ws = np.clip(warm_x, BOUNDS_LO, BOUNDS_HI)
    study.add_trial(optuna.trial.create_trial(
        params={n: float(v) for n, v in zip(PARAM_NAMES, xc_ws)},
        distributions=dists, value=ws_loss))

    print(f"  Injected {len(study.trials)} prior trials into TPE", flush=True)

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["trial"] + PARAM_NAMES + ["loss", "wall_s"])
        flog.flush()

        n_new = 0
        while n_new < n_trials:
            batch = min(PARALLEL_EVALS, n_trials - n_new)
            t_batch = [study.ask() for _ in range(batch)]
            xs = []
            for trial in t_batch:
                x = np.array([
                    trial.suggest_float(n, float(lo), float(hi))
                    for n, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
                ])
                xs.append(x)

            losses = [float("inf")] * len(xs)
            with ThreadPoolExecutor(max_workers=PARALLEL_EVALS) as pool:
                fmap = {pool.submit(run_loss, x): i for i, x in enumerate(xs)}
                for fut in as_completed(fmap):
                    losses[fmap[fut]] = fut.result()

            wall = time.time() - state["t0"]
            for trial, x, loss in zip(t_batch, xs, losses):
                study.tell(trial, loss)
                n_new += 1
                state["evals"] += 1
                xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
                if loss < best_loss:
                    best_loss = loss
                    best_x    = xc.copy()
                    state["best"] = best_loss
                wr.writerow([n_new] + [f"{v:.6f}" for v in xc]
                            + [f"{loss:.6f}", f"{wall:.0f}"])
            flog.flush()

            gen_best = min(losses)
            print(
                f"  Trial {n_new:3d}/{n_trials}: "
                f"batch_best={gen_best:.4f}%  overall={best_loss:.4f}%",
                flush=True,
            )

    return best_x, best_loss


PHI = (1 + math.sqrt(5)) / 2


def golden_section_1d(name, x, idx, lo, hi, tol, state, writer, logfile, rnd):
    cache = {}

    def cached(v):
        v = round(v, 8)
        if v not in cache:
            xp = x.copy()
            xp[idx] = v
            loss = run_loss(xp, blis_workers=BLIS_WORKERS_GS)
            state["evals"] += 1
            cache[v] = loss
            wall = time.time() - state["t0"]
            m, s = divmod(int(wall), 60)
            print(f"    {name}={v:.6f} → {loss:.4f}%  [+{m}m{s:02d}s]", flush=True)
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


def phase_golden(state: dict, warm_x: np.ndarray, log_path: Path,
                 n_rounds: int = 3) -> tuple:
    print(f"\n{'='*70}")
    print(f"Phase 2: Golden Section — 1D polish ({n_rounds} rounds, {BLIS_WORKERS_GS} workers/eval)")
    print(f"{'='*70}")

    state["phase"] = "golden"
    best_x    = warm_x.copy()
    best_loss = run_loss(best_x, blis_workers=BLIS_WORKERS_GS)
    state["best"] = best_loss
    state["evals"] += 1
    print(f"  Warm-start: {best_loss:.4f}%", flush=True)

    # Search order: highest-impact params first
    search = [
        # (name,  idx, lo,    hi,     tol_r1, tol_r2, tol_r3)
        ("α₀",    0,   5000., 40000., 200.,   50.,    10.),
        ("γ₁",    1,   0.03,  0.20,   0.002,  0.001,  0.0002),
        ("γ₂",    2,   0.1,   1.2,    0.005,  0.002,  0.0005),
        ("γ₃",    3,   0.2,   1.8,    0.005,  0.002,  0.0005),
        ("γ₇dc",  7,   0.,    100.,   0.5,    0.2,    0.05),
        ("γ₈",    5,   0.,    500.,   2.,     1.,     0.5),
        ("γ₉",    6,   0.,    500.,   2.,     1.,     0.5),
        ("γ₇pf",  4,   0.,    200.,   1.,     0.5,    0.1),
        ("α₂",    8,   0.,    20.,    0.1,    0.05,   0.01),
    ]
    tols = [{s[0]: s[4] for s in search},
            {s[0]: s[5] for s in search},
            {s[0]: s[6] for s in search}]

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["round", "param", "value", "loss", "wall_s"])
        flog.flush()

        for rnd in range(n_rounds):
            print(f"\n  Round {rnd+1}/{n_rounds}:", flush=True)
            improved = False
            for name, idx, lo, hi, *_ in search:
                tol = tols[rnd][name]
                cur = best_x[idx]
                if rnd > 0:
                    span = (hi - lo) * (0.3 ** rnd)
                    lo_s = max(lo, cur - span / 2)
                    hi_s = min(hi, cur + span / 2)
                else:
                    lo_s, hi_s = lo, hi
                print(f"    [{rnd+1}] {name} ∈ [{lo_s:.4f}, {hi_s:.4f}]  cur={cur:.6f}", flush=True)
                best_v, loss = golden_section_1d(
                    name, best_x, idx, lo_s, hi_s, tol, state, wr, flog, rnd + 1)
                if loss < best_loss:
                    best_loss = loss
                    best_x[idx] = best_v
                    state["best"] = best_loss
                    improved = True
                    print(f"    ✓ {name}={best_v:.6f}  loss={loss:.4f}%", flush=True)
                else:
                    print(f"    ✗ {name}={best_v:.6f}  loss={loss:.4f}%", flush=True)
            print(f"  Round {rnd+1}: best={best_loss:.4f}%", flush=True)
            if not improved and rnd > 0:
                print("  Converged.", flush=True)
                break

    return best_x, best_loss


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-golden", action="store_true",
                        help="skip TPE; load iter35_tpe_results.json")
    args = parser.parse_args()

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS):
        print(f"ERROR: blis not found at {BLIS}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter35: TPE → Golden Section (fixed γ₇_dc = constant × L)")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target:     beat iter29 (34.57%)")
    print(f"Warm-start: iter32 best (56.49%) with γ₇_dc=50 (new constant form)")
    print(f"Prior:      loading iter32/33/34 trials for TPE warm-start")
    print("=" * 70)

    # ── Load prior trials from previous iterations ────────────────────────────
    # iter32: search log has same 9 parameter names minus γ₇_dc
    # iter33: similar structure; map to current param layout
    # iter34: 9D CMA-ES log (closest layout)
    prior_trials = []

    # iter34 9D CMA-ES log has the right column structure
    prior_34 = load_prior_from_csv(
        TRAINING_DIR / "iterations" / "iter34" / "iter34_cmaes_log.csv",
        PARAM_NAMES,
    )
    if prior_34:
        print(f"  Loaded {len(prior_34)} prior trials from iter34_cmaes_log.csv", flush=True)
        prior_trials.extend(prior_34)

    # Also evaluate warm-start as a sanity check
    state = {
        "done": False, "t0": time.time(),
        "phase": "init", "best": float("inf"), "evals": 0,
    }
    mon = threading.Thread(target=monitor, args=(state,), daemon=True)
    mon.start()

    tpe_x, tpe_loss = X0.copy(), float("inf")

    if not args.resume_golden:
        tpe_x, tpe_loss = phase_tpe(
            state, X0, prior_trials,
            ITER_DIR / "iter35_tpe_log.csv",
            n_trials=300,
        )
        alpha_f, beta_f = x_to_coeffs(tpe_x)
        _save(ITER_DIR / "iter35_tpe_results.json", {
            "phase": "tpe",
            "best_loss": tpe_loss,
            "best_params": {"alpha": alpha_f, "beta": beta_f},
            "prior_trials": len(prior_trials),
        })
        print(f"\nTPE complete: {tpe_loss:.4f}%", flush=True)
    else:
        data = json.loads((ITER_DIR / "iter35_tpe_results.json").read_text())
        tpe_x = np.array([
            data["best_params"]["alpha"][0],
            data["best_params"]["beta"][0],
            data["best_params"]["beta"][1],
            data["best_params"]["beta"][2],
            data["best_params"]["beta"][6],
            data["best_params"]["beta"][7],
            data["best_params"]["beta"][8],
            data["best_params"]["beta"][9],
            data["best_params"]["alpha"][2],
        ])
        tpe_loss = data["best_loss"]
        print(f"Loaded TPE: {tpe_loss:.4f}%", flush=True)

    final_x, final_loss = phase_golden(
        state, tpe_x,
        ITER_DIR / "iter35_golden_log.csv",
        n_rounds=3,
    )

    state["done"] = True
    total_wall = time.time() - state["t0"]
    alpha_f, beta_f = x_to_coeffs(final_x)

    print(f"\n{'='*70}")
    print(f"ITER35 COMPLETE")
    print(f"  TPE:            {tpe_loss:.4f}%")
    print(f"  Golden section: {final_loss:.4f}%")
    print(f"  Total evals: {state['evals']}  Wall: {total_wall:.0f}s ({total_wall/3600:.1f}h)")
    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {alpha_f}")
    print(f"  beta  = {beta_f}")

    _save(ITER_DIR / "iter35_final_results.json", {
        "iteration": 35,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha_f, "beta": beta_f},
        "loss": {"overall_loss": final_loss},
        "phase_losses": {"tpe": tpe_loss, "golden": final_loss},
        "total_evals": state["evals"],
        "total_wall_s": round(total_wall),
    })
    print(f"Saved: {ITER_DIR / 'iter35_final_results.json'}")


if __name__ == "__main__":
    main()
