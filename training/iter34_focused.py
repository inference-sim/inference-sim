#!/usr/bin/env -S python3.11 -u
"""
Iter34 focused search: 4D CMA-ES on the dimensions that haven't converged.

After 26 gens of 9D CMA-ES (best=65.99%), the following converged:
  γ₁=0.012, γ_wt=0.198, γ₇_pf=38.3, α₀=13737, α₂=14.1

High-variance across top-10 candidates (still being optimized):
  γ₂ (FlashAttn): 0.49–0.74  → x[0]
  γ₃ (PagedAttn): 1.25–1.39  → x[1]
  γ₈ (per-req):   315–401    → x[2]
  γ₉ (per-step):  138–268    → x[3]

Strategy:
  Phase 1: 4D CMA-ES (fast convergence, d²=16 vs 81 in 9D)
  Phase 2: TPE with all 9D + 4D prior injected
  Phase 3: Golden section over all 9 dims

Run from worktree root:
    python3.11 training/iter34_focused.py [--resume-tpe | --resume-golden]
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

import cma
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT         = Path(__file__).parent.parent
TRAINING_DIR = ROOT / "training"
BLIS         = str(ROOT / "blis")
DATA_DIR     = str(Path("/Users/sri/Documents/Projects/inference-sim/training/trainval_data"))
PROFILES_DIR = str(Path("/Users/sri/Documents/Projects/inference-sim/training/kernel_profiles"))
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter34"

# ── Fixed coefficients (from 9D CMA-ES best at gen 17) ───────────────────────
FIXED = {
    "alpha0": 13736.85,
    "gamma1": 0.01205,
    "gamma_wt": 0.19762,
    "gamma7pf": 38.295,
    "alpha2": 14.097,
}

# ── 4D search space: γ₂, γ₃, γ₈, γ₉ ────────────────────────────────────────
# x[0] = γ₂  FlashAttention correction
# x[1] = γ₃  PagedAttention correction
# x[2] = γ₈  per-request overhead (µs)
# x[3] = γ₉  per-step overhead (µs)

PARAM_NAMES_4D = ["γ₂", "γ₃", "γ₈", "γ₉"]

# Warm-start from 9D best (gen 17)
X0_4D = np.array([0.49060, 1.25673, 314.935, 267.839])

BOUNDS_LO_4D = np.array([0.1,   0.5,    0.,    0.])
BOUNDS_HI_4D = np.array([1.5,   1.8,  500.,  500.])

# ── Full 9D param names (for TPE phase) ──────────────────────────────────────
PARAM_NAMES_9D = ["α₀", "γ₁", "γ₂", "γ₃", "γ_wt", "γ₇pf", "γ₈", "γ₉", "α₂"]
BOUNDS_LO_9D = np.array([5000.,  0.01,  0.1,  0.5,   0.0,   0.0,   0.,    0.,  0.])
BOUNDS_HI_9D = np.array([40000., 0.25,  1.5,  1.8,   0.2,  50.0, 500.,  500., 20.])

PARALLEL_EVALS  = 2
BLIS_WORKERS    = 8
BLIS_WORKERS_GS = 15
EVAL_TIMEOUT_S  = 180


def x4d_to_coeffs(x4: np.ndarray) -> tuple:
    """Unpack 4D search vector to full (alpha, beta) with fixed dims."""
    xc = np.clip(x4, BOUNDS_LO_4D, BOUNDS_HI_4D)
    alpha = [FIXED["alpha0"], 0.0, FIXED["alpha2"]]
    beta  = [FIXED["gamma1"], xc[0], xc[1],
             0.4,  # gamma[3] = γ_wt
             0.00890702, 1.0,  # fixed AllReduce, MoE
             FIXED["gamma7pf"], xc[2], xc[3], 0.0]  # γ₇pf, γ₈, γ₉, retired
    # overwrite gamma[3] with actual γ_wt
    beta[3] = FIXED["gamma_wt"]
    return alpha, beta


def x9d_to_coeffs(x9: np.ndarray) -> tuple:
    """Unpack full 9D vector to (alpha, beta)."""
    xc = np.clip(x9, BOUNDS_LO_9D, BOUNDS_HI_9D)
    alpha = [float(xc[0]), 0.0, float(xc[8])]
    beta  = [float(xc[1]), float(xc[2]), float(xc[3]),
             float(xc[4]),
             0.00890702, 1.0,
             float(xc[5]), float(xc[6]), float(xc[7]), 0.0]
    return alpha, beta


def coeff_str(vals) -> str:
    return ",".join(f"{v:.10e}" for v in vals)


def run_loss_4d(x4: np.ndarray, blis_workers: int = BLIS_WORKERS) -> float:
    alpha, beta = x4d_to_coeffs(x4)
    return _call_blis(alpha, beta, blis_workers)


def run_loss_9d(x9: np.ndarray, blis_workers: int = BLIS_WORKERS) -> float:
    alpha, beta = x9d_to_coeffs(x9)
    return _call_blis(alpha, beta, blis_workers)


def _call_blis(alpha, beta, blis_workers: int) -> float:
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


def run_parallel_4d(solutions: list) -> list:
    results = [float("inf")] * len(solutions)
    with ThreadPoolExecutor(max_workers=PARALLEL_EVALS) as pool:
        fmap = {pool.submit(run_loss_4d, x): i for i, x in enumerate(solutions)}
        for fut in as_completed(fmap):
            results[fmap[fut]] = fut.result()
    return results


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


def phase_cmaes_4d(state: dict, log_path: Path) -> tuple:
    print(f"\n{'='*70}")
    print("Phase 1: 4D CMA-ES — focused search on γ₂, γ₃, γ₈, γ₉")
    print(f"  Fixed: {FIXED}")
    print(f"  Warm-start: {dict(zip(PARAM_NAMES_4D, X0_4D.tolist()))}")
    print(f"  Parallelism: {PARALLEL_EVALS} evals × {BLIS_WORKERS} workers")
    print(f"{'='*70}")

    state["phase"] = "cmaes4d"
    best_x4d  = X0_4D.copy()
    best_loss = float("inf")
    trials    = []

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["gen", "eval"] + PARAM_NAMES_4D + ["loss", "wall_s"])
        flog.flush()

        opts = cma.CMAOptions()
        opts.set("bounds",   [BOUNDS_LO_4D.tolist(), BOUNDS_HI_4D.tolist()])
        opts.set("tolx",     0.1)
        opts.set("tolfun",   0.01)
        opts.set("maxiter",  100)
        opts.set("verbose",  -9)
        opts.set("CMA_stds", (BOUNDS_HI_4D - BOUNDS_LO_4D).tolist())

        # sigma0=0.15: tight, exploitative — we know the good region from 9D run
        es = cma.CMAEvolutionStrategy(X0_4D.tolist(), 0.15, opts)
        n_evals = 0

        while not es.stop():
            solutions = es.ask()
            gen = es.result.iterations + 1

            t_gen = time.time()
            losses = run_parallel_4d(solutions)
            gen_elapsed = time.time() - t_gen

            es.tell(solutions, losses)

            wall = time.time() - state["t0"]
            gen_best = min(losses)
            for x, loss in zip(solutions, losses):
                n_evals += 1
                xc = np.clip(x, BOUNDS_LO_4D, BOUNDS_HI_4D)
                # Build full 9D vector for TPE injection
                x9 = np.array([FIXED["alpha0"], FIXED["gamma1"],
                                xc[0], xc[1], FIXED["gamma_wt"],
                                FIXED["gamma7pf"], xc[2], xc[3], FIXED["alpha2"]])
                trials.append((x9, loss))
                if loss < best_loss:
                    best_loss = loss
                    best_x4d  = xc.copy()
                wr.writerow([gen, n_evals]
                            + [f"{v:.5f}" for v in xc]
                            + [f"{loss:.6f}", f"{wall:.0f}"])
            flog.flush()

            state["best"]  = best_loss
            state["evals"] = n_evals
            valid = sum(1 for l in losses if l < float("inf"))
            print(
                f"  Gen {gen:3d}: pop={len(solutions)}  "
                f"gen_best={gen_best:.4f}%  overall={best_loss:.4f}%  "
                f"valid={valid}/{len(solutions)}  σ={es.sigma:.4f}  ({gen_elapsed:.0f}s)",
                flush=True,
            )

    return best_x4d, best_loss, trials


PHI = (1 + math.sqrt(5)) / 2


def golden_section_1d(name, x9, idx, lo, hi, tol, state, writer, logfile, rnd):
    cache = {}

    def cached(v):
        v = round(v, 8)
        if v not in cache:
            xp = x9.copy()
            xp[idx] = v
            loss = run_loss_9d(xp, blis_workers=BLIS_WORKERS_GS)
            state["evals"] += 1
            cache[v] = loss
            wall = time.time() - state["t0"]
            m, s = divmod(int(wall), 60)
            print(f"    {name}={v:.6f} → {loss:.4f}%  [+{m}m{s:02d}s  eval={state['evals']}]", flush=True)
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


def phase_tpe(state, warm_x9, prior_9d, prior_4d, log_path, n_trials=200):
    all_prior = prior_9d + prior_4d
    print(f"\n{'='*70}")
    print(f"Phase 2: TPE — Bayesian refinement ({n_trials} new trials)")
    print(f"  Prior: {len(all_prior)} trials (9D CMA-ES + 4D CMA-ES)")
    print(f"  Parallelism: {PARALLEL_EVALS} concurrent × {BLIS_WORKERS} workers")
    print(f"{'='*70}")

    state["phase"] = "tpe"
    best_x9   = warm_x9.copy()
    best_loss = run_loss_9d(warm_x9)
    state["best"] = best_loss
    state["evals"] += 1
    print(f"  TPE warm-start: {best_loss:.4f}%", flush=True)

    dists = {
        name: optuna.distributions.FloatDistribution(float(lo), float(hi))
        for name, lo, hi in zip(PARAM_NAMES_9D, BOUNDS_LO_9D, BOUNDS_HI_9D)
    }

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=0, seed=42),
    )

    for x9, loss in all_prior:
        if loss == float("inf"):
            continue
        xc = np.clip(x9, BOUNDS_LO_9D, BOUNDS_HI_9D)
        params = {n: float(v) for n, v in zip(PARAM_NAMES_9D, xc)}
        study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=loss))

    # Also add warm-start point
    xc_ws = np.clip(warm_x9, BOUNDS_LO_9D, BOUNDS_HI_9D)
    study.add_trial(optuna.trial.create_trial(
        params={n: float(v) for n, v in zip(PARAM_NAMES_9D, xc_ws)},
        distributions=dists, value=best_loss))

    print(f"  Injected {len(study.trials)} prior trials", flush=True)

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["trial"] + PARAM_NAMES_9D + ["loss", "wall_s"])
        flog.flush()

        n_new = 0
        while n_new < n_trials:
            batch = min(PARALLEL_EVALS, n_trials - n_new)
            trials_batch = [study.ask() for _ in range(batch)]
            xs = []
            for trial in trials_batch:
                x = np.array([
                    trial.suggest_float(n, float(lo), float(hi))
                    for n, lo, hi in zip(PARAM_NAMES_9D, BOUNDS_LO_9D, BOUNDS_HI_9D)
                ])
                xs.append(x)

            losses_batch = []
            with ThreadPoolExecutor(max_workers=PARALLEL_EVALS) as pool:
                fmap = {pool.submit(run_loss_9d, x): i for i, x in enumerate(xs)}
                losses_batch = [float("inf")] * len(xs)
                for fut in as_completed(fmap):
                    losses_batch[fmap[fut]] = fut.result()

            wall = time.time() - state["t0"]
            for trial, x, loss in zip(trials_batch, xs, losses_batch):
                study.tell(trial, loss)
                n_new += 1
                state["evals"] += 1
                xc = np.clip(x, BOUNDS_LO_9D, BOUNDS_HI_9D)
                if loss < best_loss:
                    best_loss = loss
                    best_x9   = xc.copy()
                wr.writerow([n_new] + [f"{v:.5f}" for v in xc]
                            + [f"{loss:.6f}", f"{wall:.0f}"])
            flog.flush()
            state["best"] = best_loss

            print(
                f"  TPE trial {n_new:3d}/{n_trials}: "
                f"batch_best={min(losses_batch):.4f}%  overall={best_loss:.4f}%",
                flush=True,
            )

    return best_x9, best_loss


def phase_golden(state, warm_x9, log_path, n_rounds=3):
    print(f"\n{'='*70}")
    print(f"Phase 3: Golden Section — 1D polish ({n_rounds} rounds, {BLIS_WORKERS_GS} workers/eval)")
    print(f"{'='*70}")

    state["phase"] = "golden"
    best_x9   = warm_x9.copy()
    best_loss = run_loss_9d(best_x9, blis_workers=BLIS_WORKERS_GS)
    state["best"] = best_loss
    state["evals"] += 1
    print(f"  Golden warm-start: {best_loss:.4f}%", flush=True)

    search = [
        # (name,  idx, lo,    hi,     tol_r1, tol_r2, tol_r3)
        ("α₀",    0,   5000., 40000., 200.,   50.,    10.),
        ("γ₂",    2,   0.1,   1.5,    0.005,  0.002,  0.0005),
        ("γ₃",    3,   0.5,   1.8,    0.005,  0.002,  0.0005),
        ("γ_wt",  4,   0.0,   0.2,    0.003,  0.001,  0.0002),
        ("γ₁",    1,   0.01,  0.25,   0.002,  0.001,  0.0002),
        ("γ₈",    6,   0.,    500.,   2.,     1.,     0.5),
        ("γ₉",    7,   0.,    500.,   2.,     1.,     0.5),
        ("γ₇pf",  5,   0.,    50.,    0.5,    0.2,    0.05),
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
                cur = best_x9[idx]
                if rnd > 0:
                    span = (hi - lo) * (0.3 ** rnd)
                    lo_s = max(lo, cur - span / 2)
                    hi_s = min(hi, cur + span / 2)
                else:
                    lo_s, hi_s = lo, hi
                print(f"    [{rnd+1}] {name} ∈ [{lo_s:.4f}, {hi_s:.4f}]  cur={cur:.6f}", flush=True)
                best_v, loss = golden_section_1d(name, best_x9, idx, lo_s, hi_s, tol, state, wr, flog, rnd + 1)
                if loss < best_loss:
                    best_loss = loss
                    best_x9[idx] = best_v
                    state["best"] = best_loss
                    improved = True
                    print(f"    ✓ {name}={best_v:.6f}  loss={loss:.4f}%  (improved)", flush=True)
                else:
                    print(f"    ✗ {name}={best_v:.6f}  loss={loss:.4f}%", flush=True)
            print(f"  Round {rnd+1}: best={best_loss:.4f}%", flush=True)
            if not improved and rnd > 0:
                print("  Converged — stopping early.", flush=True)
                break

    return best_x9, best_loss


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-tpe",    action="store_true")
    parser.add_argument("--resume-golden", action="store_true")
    args = parser.parse_args()

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS):
        print(f"ERROR: blis not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter34 focused: 4D CMA-ES → TPE (9D) → Golden Section (9D)")
    print(f"Start:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target:  beat iter29 (34.57%)")
    print(f"9D seed: 65.99% (gen 17, 260 prior evals)")
    print(f"Focus:   γ₂, γ₃, γ₈, γ₉  (high variance dims from 9D run)")
    print("=" * 70)

    # Load 9D prior trials from previous run
    prior_9d = []
    log_9d = ITER_DIR / "iter34_cmaes_log.csv"
    if log_9d.exists():
        with open(log_9d) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x9 = np.array([float(row[n]) for n in PARAM_NAMES_9D])
                    loss = float(row["loss"])
                    prior_9d.append((x9, loss))
                except (KeyError, ValueError):
                    pass
        print(f"Loaded {len(prior_9d)} prior 9D trials from iter34_cmaes_log.csv", flush=True)

    state = {"done": False, "t0": time.time(),
             "phase": "init", "best": float("inf"), "evals": 0}
    mon = threading.Thread(target=monitor, args=(state,), daemon=True)
    mon.start()

    # Best 9D vector from 9D CMA-ES run
    best_9d_x = np.array([13736.85, 0.01205, 0.49060, 1.25673, 0.19762,
                           38.295, 314.935, 267.839, 14.097])
    cmaes4d_x, cmaes4d_loss, prior_4d = best_9d_x.copy(), 65.996, []

    if not args.resume_tpe and not args.resume_golden:
        x4d_best, cmaes4d_loss, prior_4d = phase_cmaes_4d(
            state, ITER_DIR / "iter34_cmaes4d_log.csv")
        # Build full 9D from 4D result + fixed dims
        cmaes4d_x = np.array([FIXED["alpha0"], FIXED["gamma1"],
                               x4d_best[0], x4d_best[1], FIXED["gamma_wt"],
                               FIXED["gamma7pf"], x4d_best[2], x4d_best[3],
                               FIXED["alpha2"]])
        alpha_f, beta_f = x9d_to_coeffs(cmaes4d_x)
        _save(ITER_DIR / "iter34_cmaes4d_results.json", {
            "phase": "cmaes4d", "best_loss": cmaes4d_loss,
            "best_params": {"alpha": alpha_f, "beta": beta_f},
        })
        print(f"\n4D CMA-ES complete: {cmaes4d_loss:.4f}%", flush=True)
    else:
        p = ITER_DIR / "iter34_cmaes4d_results.json"
        if p.exists():
            data = json.loads(p.read_text())
            cmaes4d_x = np.array([
                data["best_params"]["alpha"][0],
                data["best_params"]["beta"][0],
                data["best_params"]["beta"][1],
                data["best_params"]["beta"][2],
                data["best_params"]["beta"][3],
                data["best_params"]["beta"][6],
                data["best_params"]["beta"][7],
                data["best_params"]["beta"][8],
                data["best_params"]["alpha"][2],
            ])
            cmaes4d_loss = data["best_loss"]
        print(f"Loaded 4D CMA-ES: {cmaes4d_loss:.4f}%", flush=True)

    # ── Phase 2: TPE ──────────────────────────────────────────────────────────
    tpe_x, tpe_loss = cmaes4d_x.copy(), cmaes4d_loss

    if not args.resume_golden:
        tpe_x, tpe_loss = phase_tpe(
            state, cmaes4d_x,
            prior_9d, prior_4d,
            ITER_DIR / "iter34_tpe_log.csv",
            n_trials=200,
        )
        alpha_f, beta_f = x9d_to_coeffs(tpe_x)
        _save(ITER_DIR / "iter34_tpe_results.json", {
            "phase": "tpe", "best_loss": tpe_loss,
            "best_params": {"alpha": alpha_f, "beta": beta_f},
            "seed_loss": cmaes4d_loss,
        })
        print(f"\nTPE complete: {tpe_loss:.4f}%", flush=True)
    else:
        data = json.loads((ITER_DIR / "iter34_tpe_results.json").read_text())
        tpe_x = np.array([
            data["best_params"]["alpha"][0],
            data["best_params"]["beta"][0],
            data["best_params"]["beta"][1],
            data["best_params"]["beta"][2],
            data["best_params"]["beta"][3],
            data["best_params"]["beta"][6],
            data["best_params"]["beta"][7],
            data["best_params"]["beta"][8],
            data["best_params"]["alpha"][2],
        ])
        tpe_loss = data["best_loss"]
        print(f"Loaded TPE: {tpe_loss:.4f}%", flush=True)

    # ── Phase 3: Golden Section ───────────────────────────────────────────────
    final_x, final_loss = phase_golden(
        state, tpe_x, ITER_DIR / "iter34_golden_log.csv", n_rounds=3)

    state["done"] = True
    total_wall = time.time() - state["t0"]
    alpha_f, beta_f = x9d_to_coeffs(final_x)

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"  9D CMA-ES seed: 65.99%")
    print(f"  4D CMA-ES:      {cmaes4d_loss:.4f}%")
    print(f"  TPE:            {tpe_loss:.4f}%")
    print(f"  Golden section: {final_loss:.4f}%")
    print(f"  Total evals: {state['evals']}  Wall: {total_wall:.0f}s ({total_wall/3600:.1f}h)")
    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {alpha_f}")
    print(f"  beta  = {beta_f}")

    _save(ITER_DIR / "iter34_final_results.json", {
        "iteration": 34,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha_f, "beta": beta_f},
        "loss": {"overall_loss": final_loss},
        "phase_losses": {"cmaes4d": cmaes4d_loss, "tpe": tpe_loss, "golden": final_loss},
        "total_evals": state["evals"],
        "total_wall_s": round(total_wall),
    })
    print(f"Saved: {ITER_DIR / 'iter34_final_results.json'}")


if __name__ == "__main__":
    main()
