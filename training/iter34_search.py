#!/usr/bin/env -S python3.11 -u
"""
Iter34: Three-phase search for kernel-lookup latency model with floor subtraction.

Physical changes vs iter33:
  - GEMM and logits floor subtracted (same pattern as AllReduce overhead)
  - gamma[3] = γ_wt: weight-read floor amortization factor (new)
  - gamma[9] retired (γ₇_dc was unphysical: total overhead decreased with decode batch)

Search phases:
  Phase 1 — CMA-ES    : global joint search, warm-start from physical priors
  Phase 2 — TPE       : Bayesian refinement, warm-started from all CMA-ES trials
  Phase 3 — Golden Section: 1D polish per coefficient, 3 rounds until convergence

Parallelism:
  - CMA-ES/TPE: PARALLEL_EVALS concurrent evaluations × BLIS_WORKERS each
  - Golden section: BLIS_WORKERS=15 per eval (sequential, max inner parallelism)
  - Per-minute monitor thread throughout

Run from worktree root:
    python3.11 training/iter34_search.py [--resume-tpe | --resume-golden]

Flags:
  --resume-tpe    skip Phase 1, load CMA-ES results and start at Phase 2
  --resume-golden skip Phases 1+2, load TPE results and start at Phase 3
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
TRAINING_DIR = ROOT / "training"
BLIS         = str(ROOT / "blis")
DATA_DIR     = str(Path("/Users/sri/Documents/Projects/inference-sim/training/trainval_data"))
PROFILES_DIR = str(Path("/Users/sri/Documents/Projects/inference-sim/training/kernel_profiles"))
SCRIPT       = str(TRAINING_DIR / "run_blis_and_compute_loss.py")
ITER_DIR     = TRAINING_DIR / "iterations" / "iter34"

# ── Coefficient layout (10 beta + 3 alpha) ────────────────────────────────────
# x[0]  alpha[0]  α₀  queueing overhead (µs)
# x[1]  gamma[0]  γ₁  net GEMM + logits correction     warm ~0.9
# x[2]  gamma[1]  γ₂  FlashAttention correction         warm ~0.9
# x[3]  gamma[2]  γ₃  PagedAttention correction         warm ~0.9
# x[4]  gamma[3]  γ_wt weight-read floor factor         warm ~0.4
# x[5]  gamma[6]  γ₇_pf per-layer per-prefill-seq (µs) warm ~5.0
# x[6]  gamma[7]  γ₈  per-request overhead (µs)        warm ~150
# x[7]  gamma[8]  γ₉  per-step overhead (µs)           warm ~100
# x[8]  alpha[2]  α₂  per-output-token overhead (µs)   warm ~10

# Warm-start derived from iter33 best + recalibration for floor subtraction.
#
# γ₁ (x[1]): floor subtraction does NOT change the pipelining correction magnitude.
#   vLLM pipelines ~22 layers simultaneously, so γ₁ ≈ 1/22 ≈ 0.046 still holds.
#   The floor subtraction shifts WHERE overhead sits (into γ_wt), not HOW MUCH.
#
# γ_wt (x[4]): calibrated to match iter33's decode contribution at m=4 tokens:
#   iter33: γ₁×T_gemm(4)×L = 0.046×167.8×28 ≈ 216µs  (Qwen-7B profile)
#   iter34: γ_wt×T_floor = γ_wt×(167.1×28 + 373) = γ_wt×5052µs
#   → γ_wt = 216/5052 ≈ 0.043
#   This ensures the MODEL-SIZE BENEFIT: 70B floor (154×80=12320µs) is 2.6×
#   larger than 7B floor (167×28=4679µs), correctly scaling with depth.
#
# γ₂, γ₃: unchanged from iter33 (T_pf_attn/T_dc_attn formulas untouched).
# γ₈, γ₉: reduced because γ_wt now absorbs per-step fixed costs.
X0 = np.array([14692.0, 0.046, 0.302, 0.607, 0.043, 5.0, 150.0, 52.0, 10.0])

BOUNDS_LO = np.array([5000.,  0.01,  0.05, 0.1,   0.0,   0.0,   0.,    0.,  0.])
BOUNDS_HI = np.array([40000., 0.25,  1.5,  1.8,   0.2,  50.0, 500.,  500., 20.])

PARAM_NAMES = ["α₀", "γ₁", "γ₂", "γ₃", "γ_wt", "γ₇pf", "γ₈", "γ₉", "α₂"]

# Fixed coefficients (not searched)
ALPHA_FIXED = [0.0, 0.0, 0.0]   # placeholders; alpha[0] and alpha[2] from x
BETA_FIXED  = [0.0, 0.0, 0.0, 0.0, 0.00890702, 1.0, 0.0, 0.0, 0.0, 0.0]
# gamma[4]=γ₅ AllReduce=0.009, gamma[5]=γ₆ MoE=1.0, gamma[9]=retired=0


def x_to_coeffs(x: np.ndarray) -> tuple:
    xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
    alpha = ALPHA_FIXED[:]
    beta  = BETA_FIXED[:]
    alpha[0] = float(xc[0])   # α₀
    beta[0]  = float(xc[1])   # γ₁
    beta[1]  = float(xc[2])   # γ₂
    beta[2]  = float(xc[3])   # γ₃
    beta[3]  = float(xc[4])   # γ_wt
    beta[6]  = float(xc[5])   # γ₇_pf
    beta[7]  = float(xc[6])   # γ₈
    beta[8]  = float(xc[7])   # γ₉
    alpha[2] = float(xc[8])   # α₂
    return alpha, beta


def coeffs_to_x(alpha: list, beta: list) -> np.ndarray:
    return np.array([
        alpha[0], beta[0], beta[1], beta[2],
        beta[3], beta[6], beta[7], beta[8], alpha[2],
    ])


def coeff_str(vals: list) -> str:
    return ",".join(f"{v:.10e}" for v in vals)


# ── Parallelism config ────────────────────────────────────────────────────────
# CMA-ES / TPE: 2 concurrent evaluations × 8 BLIS workers = 16 total processes
# Golden section: 1 evaluation × 15 BLIS workers (max inner parallelism)
PARALLEL_EVALS_SEARCH  = 2
BLIS_WORKERS_SEARCH    = 8
BLIS_WORKERS_GOLDEN    = 15
EVAL_TIMEOUT_S         = 180


def run_loss(x: np.ndarray, blis_workers: int = BLIS_WORKERS_SEARCH) -> float:
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


def run_parallel(solutions: list, blis_workers: int = BLIS_WORKERS_SEARCH) -> list:
    results = [float("inf")] * len(solutions)
    with ThreadPoolExecutor(max_workers=PARALLEL_EVALS_SEARCH) as pool:
        fmap = {pool.submit(run_loss, x, blis_workers): i
                for i, x in enumerate(solutions)}
        for fut in as_completed(fmap):
            results[fmap[fut]] = fut.result()
    return results


# ── Monitor thread ────────────────────────────────────────────────────────────
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


# ── Phase 1: CMA-ES ───────────────────────────────────────────────────────────
def phase_cmaes(state: dict, log_path: Path) -> tuple:
    """Returns (best_x, best_loss, all_trials) where all_trials is list of (x, loss)."""
    print(f"\n{'='*70}")
    print("Phase 1: CMA-ES — global joint search")
    print(f"  Warm-start: {dict(zip(PARAM_NAMES, X0.tolist()))}")
    print(f"  Parallelism: {PARALLEL_EVALS_SEARCH} evals × {BLIS_WORKERS_SEARCH} workers")
    print(f"{'='*70}")

    state["phase"] = "cmaes"
    best_x    = X0.copy()
    best_loss = float("inf")
    all_trials = []

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["gen", "eval"] + PARAM_NAMES + ["loss", "wall_s"])
        flog.flush()

        opts = cma.CMAOptions()
        opts.set("bounds",   [BOUNDS_LO.tolist(), BOUNDS_HI.tolist()])
        opts.set("tolx",     0.5)
        opts.set("tolfun",   0.02)
        opts.set("maxiter",  150)
        opts.set("verbose",  -9)
        opts.set("CMA_stds", (BOUNDS_HI - BOUNDS_LO).tolist())

        es = cma.CMAEvolutionStrategy(X0.tolist(), 0.2, opts)
        n_evals = 0

        while not es.stop():
            solutions = es.ask()
            gen = es.result.iterations + 1

            t_gen = time.time()
            losses = run_parallel(solutions)
            gen_elapsed = time.time() - t_gen

            es.tell(solutions, losses)

            wall = time.time() - state["t0"]
            gen_best = min(losses)
            for x, loss in zip(solutions, losses):
                n_evals += 1
                xc = np.clip(x, BOUNDS_LO, BOUNDS_HI)
                all_trials.append((xc.copy(), loss))
                if loss < best_loss:
                    best_loss = loss
                    best_x    = xc.copy()
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

    return best_x, best_loss, all_trials


# ── Phase 2: TPE ──────────────────────────────────────────────────────────────
def phase_tpe(
    state: dict,
    warm_x: np.ndarray,
    prior_trials: list,
    log_path: Path,
    n_trials: int = 200,
) -> tuple:
    """Returns (best_x, best_loss).

    Warm-starts the optuna study with all prior CMA-ES trials so TPE has a rich
    prior before it begins exploring. Each new trial uses PARALLEL_EVALS_SEARCH
    concurrent evaluations via a custom sampler wrapper.
    """
    print(f"\n{'='*70}")
    print(f"Phase 2: TPE — Bayesian refinement ({n_trials} new trials)")
    print(f"  Warm-start: {len(prior_trials)} prior CMA-ES trials injected")
    print(f"  Parallelism: {PARALLEL_EVALS_SEARCH} concurrent trials × {BLIS_WORKERS_SEARCH} workers")
    print(f"{'='*70}")

    state["phase"] = "tpe"

    best_x    = warm_x.copy()
    best_loss = float("inf")

    # Evaluate warm-start point if not already in prior (get a clean loss reference)
    ws_loss = run_loss(warm_x)
    print(f"  TPE warm-start loss: {ws_loss:.4f}%", flush=True)
    if ws_loss < best_loss:
        best_loss = ws_loss
        best_x    = warm_x.copy()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=0,   # use prior immediately
            seed=42,
        ),
    )

    # Inject all prior CMA-ES trials as completed observations
    for xc, loss in prior_trials:
        if loss == float("inf"):
            continue
        params = {name: float(val) for name, val in zip(PARAM_NAMES, xc)}
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                name: optuna.distributions.FloatDistribution(float(lo), float(hi))
                for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
            },
            value=loss,
        )
        study.add_trial(trial)
    print(f"  Injected {len(study.trials)} prior trials into TPE study", flush=True)

    # Also inject the warm-start point
    params_ws = {name: float(val) for name, val in zip(PARAM_NAMES, warm_x)}
    trial_ws = optuna.trial.create_trial(
        params=params_ws,
        distributions={
            name: optuna.distributions.FloatDistribution(float(lo), float(hi))
            for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
        },
        value=ws_loss,
    )
    study.add_trial(trial_ws)

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["trial"] + PARAM_NAMES + ["loss", "wall_s"])
        flog.flush()

        n_new = 0
        while n_new < n_trials:
            # Ask for PARALLEL_EVALS_SEARCH candidates at once
            batch_size = min(PARALLEL_EVALS_SEARCH, n_trials - n_new)
            trials = [study.ask() for _ in range(batch_size)]

            xs = []
            for trial in trials:
                x = np.array([
                    trial.suggest_float(name, float(lo), float(hi))
                    for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
                ])
                xs.append(x)

            losses = run_parallel(xs)

            wall = time.time() - state["t0"]
            for trial, x, loss in zip(trials, xs, losses):
                study.tell(trial, loss)
                n_new += 1
                state["evals"] += 1
                if loss < best_loss:
                    best_loss = loss
                    best_x    = x.copy()
                wr.writerow([n_new] + [f"{v:.5f}" for v in x]
                            + [f"{loss:.6f}", f"{wall:.0f}"])
            flog.flush()

            state["best"] = best_loss
            gen_best = min(losses)
            print(
                f"  TPE trial {n_new:3d}/{n_trials}: "
                f"batch_best={gen_best:.4f}%  overall={best_loss:.4f}%",
                flush=True,
            )

    return best_x, best_loss


# ── Phase 3: Golden Section ───────────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2

def golden_section_1d(
    name: str,
    x: np.ndarray,
    idx: int,
    lo: float,
    hi: float,
    tol: float,
    state: dict,
    writer,
    logfile,
    round_n: int,
) -> tuple:
    """1D golden section search over x[idx] in [lo, hi]."""
    cache: dict = {}

    def cached(v: float) -> float:
        v = round(v, 8)
        if v not in cache:
            xp = x.copy()
            xp[idx] = v
            loss = run_loss(xp, blis_workers=BLIS_WORKERS_GOLDEN)
            state["evals"] += 1
            cache[v] = loss
            wall = time.time() - state["t0"]
            m, s = divmod(int(wall), 60)
            print(
                f"    {name}={v:.6f} → {loss:.4f}%  "
                f"[+{m}m{s:02d}s  eval={state['evals']}]",
                flush=True,
            )
            writer.writerow([round_n, name, f"{v:.8f}", f"{loss:.6f}", f"{wall:.0f}"])
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


def phase_golden(state: dict, warm_x: np.ndarray, log_path: Path, n_rounds: int = 3) -> tuple:
    """Sequential 1D golden section over all 9 parameters, n_rounds passes."""
    print(f"\n{'='*70}")
    print(f"Phase 3: Golden Section — 1D polish ({n_rounds} rounds)")
    print(f"  Warm-start: {dict(zip(PARAM_NAMES, warm_x.tolist()))}")
    print(f"  Workers per eval: {BLIS_WORKERS_GOLDEN} (max inner parallelism)")
    print(f"{'='*70}")

    state["phase"] = "golden"

    best_x    = warm_x.copy()
    best_loss = run_loss(best_x, blis_workers=BLIS_WORKERS_GOLDEN)
    state["best"] = best_loss
    state["evals"] += 1
    print(f"  Golden section warm-start: {best_loss:.4f}%", flush=True)

    # Search order: parameters with widest ranges first (most impact potential)
    # then tighter refinement in subsequent rounds
    search_specs = [
        # (name,      idx, lo,    hi,    tol_r1,  tol_r2,  tol_r3)
        ("α₀",        0,   5000., 40000., 200.0,   50.0,    10.0),
        ("γ₁",        1,   0.3,   2.0,    0.005,   0.002,   0.0005),
        ("γ_wt",      4,   0.0,   1.0,    0.005,   0.002,   0.0005),
        ("γ₂",        2,   0.3,   2.0,    0.005,   0.002,   0.0005),
        ("γ₃",        3,   0.3,   2.0,    0.005,   0.002,   0.0005),
        ("γ₈",        6,   0.,    400.,   2.0,     1.0,     0.5),
        ("γ₉",        7,   0.,    500.,   2.0,     1.0,     0.5),
        ("γ₇pf",      5,   0.,    50.,    0.5,     0.2,     0.05),
        ("α₂",        8,   0.,    20.,    0.1,     0.05,    0.01),
    ]
    tol_by_round = [
        {s[0]: s[4] for s in search_specs},
        {s[0]: s[5] for s in search_specs},
        {s[0]: s[6] for s in search_specs},
    ]

    with open(log_path, "w", newline="") as flog:
        wr = csv.writer(flog)
        wr.writerow(["round", "param", "value", "loss", "wall_s"])
        flog.flush()

        for rnd in range(n_rounds):
            print(f"\n  Round {rnd+1}/{n_rounds}:", flush=True)
            round_improved = False
            for name, idx, lo, hi, *_ in search_specs:
                tol = tol_by_round[rnd][name]
                # Narrow search window around current best after first round
                cur = best_x[idx]
                if rnd > 0:
                    span = (hi - lo) * (0.3 ** rnd)
                    lo_s = max(lo, cur - span / 2)
                    hi_s = min(hi, cur + span / 2)
                else:
                    lo_s, hi_s = lo, hi

                print(
                    f"    [{rnd+1}] {name}  ∈ [{lo_s:.4f}, {hi_s:.4f}]  tol={tol}  cur={cur:.6f}",
                    flush=True,
                )
                best_v, loss = golden_section_1d(
                    name, best_x, idx, lo_s, hi_s, tol, state, wr, flog, rnd + 1,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_x[idx] = best_v
                    state["best"] = best_loss
                    round_improved = True
                    print(f"    ✓ {name}={best_v:.6f}  loss={loss:.4f}%  (improved)", flush=True)
                else:
                    print(f"    ✗ {name}={best_v:.6f}  loss={loss:.4f}%  (no improvement)", flush=True)

            print(f"  Round {rnd+1} complete: best={best_loss:.4f}%", flush=True)
            if not round_improved and rnd > 0:
                print("  Converged — no improvement in this round, stopping early.", flush=True)
                break

    return best_x, best_loss


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-tpe",    action="store_true",
                        help="skip Phase 1; load iter34_cmaes_results.json")
    parser.add_argument("--resume-golden", action="store_true",
                        help="skip Phases 1+2; load iter34_tpe_results.json")
    args = parser.parse_args()

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(BLIS):
        print(f"ERROR: blis not found at {BLIS}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Iter34: CMA-ES → TPE → Golden Section")
    print(f"Start:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target:   beat iter29 (34.57%)")
    print(f"Basis:    floor-subtracted kernel-lookup (gemmFloor, logitsFloor)")
    print(f"Changes:  γ_wt (new), γ₇_dc retired, γ₁/γ₂/γ₃ warm-start ~1.0")
    print("=" * 70)

    state = {
        "done": False, "t0": time.time(),
        "phase": "init", "best": float("inf"), "evals": 0,
    }
    mon = threading.Thread(target=monitor, args=(state,), daemon=True)
    mon.start()

    # ── Phase 1: CMA-ES ───────────────────────────────────────────────────────
    cmaes_x, cmaes_loss, all_trials = None, float("inf"), []

    if not args.resume_tpe and not args.resume_golden:
        cmaes_x, cmaes_loss, all_trials = phase_cmaes(
            state,
            ITER_DIR / "iter34_cmaes_log.csv",
        )
        alpha_f, beta_f = x_to_coeffs(cmaes_x)
        _save(ITER_DIR / "iter34_cmaes_results.json", {
            "phase": "cmaes",
            "best_loss": cmaes_loss,
            "best_params": {"alpha": alpha_f, "beta": beta_f},
            "n_trials": len(all_trials),
        })
        print(f"\nPhase 1 complete: {cmaes_loss:.4f}%", flush=True)
    else:
        data = json.loads((ITER_DIR / "iter34_cmaes_results.json").read_text())
        cmaes_x    = coeffs_to_x(data["best_params"]["alpha"], data["best_params"]["beta"])
        cmaes_loss = data["best_loss"]
        # Reconstruct trial list from CSV for TPE warm-start
        csv_path = ITER_DIR / "iter34_cmaes_log.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = np.array([float(row[n]) for n in PARAM_NAMES])
                        loss = float(row["loss"])
                        all_trials.append((x, loss))
                    except (KeyError, ValueError):
                        pass
        print(f"  Loaded CMA-ES results: {cmaes_loss:.4f}%  ({len(all_trials)} trials)", flush=True)

    # ── Phase 2: TPE ──────────────────────────────────────────────────────────
    tpe_x, tpe_loss = cmaes_x.copy(), cmaes_loss

    if not args.resume_golden:
        tpe_x, tpe_loss = phase_tpe(
            state,
            cmaes_x,
            all_trials,
            ITER_DIR / "iter34_tpe_log.csv",
            n_trials=200,
        )
        alpha_f, beta_f = x_to_coeffs(tpe_x)
        _save(ITER_DIR / "iter34_tpe_results.json", {
            "phase": "tpe",
            "best_loss": tpe_loss,
            "best_params": {"alpha": alpha_f, "beta": beta_f},
            "cmaes_seed_loss": cmaes_loss,
        })
        print(f"\nPhase 2 complete: {tpe_loss:.4f}%  (vs CMA-ES {cmaes_loss:.4f}%)", flush=True)
    else:
        data = json.loads((ITER_DIR / "iter34_tpe_results.json").read_text())
        tpe_x    = coeffs_to_x(data["best_params"]["alpha"], data["best_params"]["beta"])
        tpe_loss = data["best_loss"]
        print(f"  Loaded TPE results: {tpe_loss:.4f}%", flush=True)

    # ── Phase 3: Golden Section ───────────────────────────────────────────────
    final_x, final_loss = phase_golden(
        state,
        tpe_x,
        ITER_DIR / "iter34_golden_log.csv",
        n_rounds=3,
    )

    state["done"] = True
    total_wall = time.time() - state["t0"]
    alpha_f, beta_f = x_to_coeffs(final_x)

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"  Phase 1 CMA-ES: {cmaes_loss:.4f}%")
    print(f"  Phase 2 TPE:    {tpe_loss:.4f}%")
    print(f"  Phase 3 Golden: {final_loss:.4f}%")
    print(f"  Total evals: {state['evals']}  Wall time: {total_wall:.0f}s ({total_wall/3600:.1f}h)")
    print(f"\nFINAL COEFFICIENTS")
    print(f"  alpha = {alpha_f}")
    print(f"  beta  = {beta_f}")

    _save(ITER_DIR / "iter34_final_results.json", {
        "iteration": 34,
        "backend_name": "kernel-lookup",
        "timestamp": datetime.now().isoformat(),
        "best_params": {"alpha": alpha_f, "beta": beta_f},
        "loss": {"overall_loss": final_loss},
        "phase_losses": {
            "cmaes": cmaes_loss,
            "tpe":   tpe_loss,
            "golden": final_loss,
        },
        "total_evals": state["evals"],
        "total_wall_s": round(total_wall),
    })
    print(f"Saved: {ITER_DIR / 'iter34_final_results.json'}")


def _save(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
