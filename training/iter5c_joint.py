#!/usr/bin/env python3
"""
Iter 5c — Joint (β, α₀, γ₁, δ₀) Optimization, All Non-Negative

The corrected β from Phase A was too small (KV bandwidth 2.2x under Iter 3).
Instead of fixing β then searching α/γ/δ, jointly optimize all 7 params.

7 params with non-negativity: β₀, β₁, β₂, β₃, α₀, γ₁, δ₀
Warm-start: midpoint between corrected β and Iter 3 β (hedging the bets).
Loss: 0.5 × TTFT_MAPE + 0.5 × E2E_MAPE on 4 general experiments.

Uses scipy.optimize.differential_evolution (global optimizer, handles bounds
natively, no gradient needed, robust to 7D).

Usage:
    python3 training/iter5c_joint.py
"""

from __future__ import annotations

import json
import glob
import os
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution

REPLAY_BIN = '/Users/sri/Documents/Projects/inference-sim/.worktrees/iter5-decomposed/replay'
DATA_DIR = '/Users/sri/Documents/Projects/inference-sim/training'

ITER3_BETA = np.array([116.110, 1226.868, 19.943, 9445.157])
CORRECTED_BETA = np.array([158.365, 563.253, 9.657, 9705.323])

# Midpoint warm-start
WARM_BETA = (ITER3_BETA + CORRECTED_BETA) / 2

# 4 roleplay experiments (fast: 7200 requests, ~5s each)
OPT_EXPS = [
    '20260217-162547-llama-2-7b-tp1-roleplay',
    '20260218-084319-llama-2-70b-tp4-roleplay',
    '20260218-141024-mixtral-8x7b-v0-1-tp2-roleplay',
    '20260218-155500-codellama-34b-tp2-roleplay',
]


def run_replay(input_path, beta, alpha, delta=None):
    cmd = [REPLAY_BIN, '--input', input_path,
           '--beta', ','.join(f'{b:.6f}' for b in beta),
           '--alpha', ','.join(f'{a:.6f}' for a in alpha)]
    if delta:
        cmd.extend(['--delta', ','.join(f'{d:.6f}' for d in delta)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def replay_loss(experiments, beta, alpha, delta, ground_truth):
    ttft_errs, e2e_errs = [], []
    for exp in experiments:
        path = os.path.join(DATA_DIR, 'replay_data', f'{exp}.json')
        gt = ground_truth.get(exp)
        if not gt or not os.path.exists(path):
            continue
        out = run_replay(path, beta, alpha, delta)
        if out is None:
            return 1e6
        gt_ttft = gt['ttft']['mean_ms']
        gt_e2e = gt['e2e']['mean_ms']
        sim_ttft = out['summary']['ttft_mean_ms']
        sim_e2e = out['summary']['e2e_mean_ms']
        if gt_ttft > 0:
            ttft_errs.append(abs(sim_ttft - gt_ttft) / gt_ttft)
        if gt_e2e > 0:
            e2e_errs.append(abs(sim_e2e - gt_e2e) / gt_e2e)
    if not ttft_errs or not e2e_errs:
        return 1e6
    return (0.5 * np.mean(ttft_errs) + 0.5 * np.mean(e2e_errs)) * 100


def main():
    print("=" * 100)
    print("Iter 5c — Joint 7-Param Optimization, All Non-Negative")
    print("=" * 100)

    gt = {}
    for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        gt[d['experiment']] = d

    print(f"\nWarm-start β: [{', '.join(f'{b:.1f}' for b in WARM_BETA)}]")
    print(f"Iter 3 β:     [{', '.join(f'{b:.1f}' for b in ITER3_BETA)}]")
    print(f"Corrected β:  [{', '.join(f'{b:.1f}' for b in CORRECTED_BETA)}]")

    eval_count = [0]
    best_loss = [1e6]

    def objective(params):
        b0, b1, b2, b3, a0, g1, d0 = params
        # Non-negativity penalty
        if any(p < 0 for p in params):
            return 1e6
        beta = [b0, b1, b2, b3]
        alpha = [a0, 0.0, g1]
        delta = [d0] if d0 > 0 else None
        loss = replay_loss(OPT_EXPS, beta, alpha, delta, gt)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            sys.stderr.write(f'\r  eval {eval_count[0]:>4d}: β=[{b0:.0f},{b1:.0f},{b2:.0f},{b3:.0f}] α₀={a0:.0f} γ₁={g1:.0f} δ₀={d0:.0f} loss={loss:.2f}%')
            sys.stderr.flush()
        return loss

    # Bounds: each param ≥ 0, upper bound based on physical limits
    bounds = [
        (50, 400),      # β₀: per-layer, 50-400 µs (32-80 layers → 1.6-32ms)
        (100, 2000),    # β₁: KV bandwidth, 100-2000 µs/kv_unit
        (0, 100),       # β₂: MoE routing, 0-100 µs/token
        (0, 20000),     # β₃: TP sync, 0-20ms per step
        (0, 50000),     # α₀: request overhead, 0-50ms
        (0, 3000),      # γ₁: per-output-token, 0-3ms
        (0, 15000),     # δ₀: per-step overhead, 0-15ms
    ]

    print(f"\nOptimizing 7 params via Nelder-Mead (with non-negativity penalty)...")
    print(f"  Loss: 0.5×TTFT_MAPE + 0.5×E2E_MAPE on {len(OPT_EXPS)} roleplay experiments")

    # Warm start from Iter 3 β with α₀/γ₁ from iter5b insights
    x0 = list(ITER3_BETA) + [20000, 800, 1000]
    # Build initial simplex: perturb each dimension by ±20%
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = list(x0)
        simplex[i + 1][i] *= 1.3  # +30% perturbation

    t0 = time.time()
    from scipy.optimize import minimize as sp_minimize
    result = sp_minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxfev': 300, 'xatol': 50, 'fatol': 0.05,
                 'initial_simplex': simplex},
    )
    elapsed = time.time() - t0
    sys.stderr.write('\n')

    b0, b1, b2, b3, a0, g1, d0 = result.x
    print(f"\n  RESULT:")
    print(f"    β = [{b0:.1f}, {b1:.1f}, {b2:.1f}, {b3:.1f}]")
    print(f"    α₀ = {a0:.0f}µs, γ₁ = {g1:.1f}µs/tok, δ₀ = {d0:.0f}µs")
    print(f"    Loss: {result.fun:.2f}% (combined TTFT+E2E)")
    print(f"    Evaluations: {eval_count[0]}, Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"    All non-negative: {all(x >= 0 for x in result.x)}")

    # Full evaluation on ALL experiments
    print(f"\n[2] Full evaluation...")
    replay_files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*.json'))
                          if '_ground_truth' not in f)

    arms = {
        'Iter 3 (baseline)': {
            'beta': ITER3_BETA.tolist(), 'alpha': [13732, 0, 860.6], 'delta': None,
        },
        f'Iter 5c (joint, β=[{b0:.0f},{b1:.0f},{b2:.0f},{b3:.0f}])': {
            'beta': [b0, b1, b2, b3], 'alpha': [a0, 0, g1], 'delta': [d0] if d0 > 0 else None,
        },
    }

    for arm_name, cfg in arms.items():
        train_ttft, train_e2e, val_ttft, val_e2e = [], [], [], []
        print(f"\n  {arm_name}:")
        print(f"  β={[round(x,1) for x in cfg['beta']]}  α={[round(x,1) for x in cfg['alpha']]}  δ={cfg['delta']}")
        for rf in replay_files:
            exp = os.path.basename(rf).replace('.json', '')
            g = gt.get(exp)
            if not g:
                continue
            out = run_replay(rf, cfg['beta'], cfg['alpha'], cfg.get('delta'))
            if out is None:
                sys.stderr.write(f'    {exp[:50]:50s} FAIL\n')
                continue
            sim_ttft = out['summary']['ttft_mean_ms']
            sim_e2e = out['summary']['e2e_mean_ms']
            gt_ttft = g['ttft']['mean_ms']
            gt_e2e = g['e2e']['mean_ms']
            te = (sim_ttft - gt_ttft) / gt_ttft * 100
            ee = (sim_e2e - gt_e2e) / gt_e2e * 100
            is_reason = g['profile'] == 'reasoning'
            split = g['split']
            tag = f"{'*' if is_reason else ' '}{split}"
            print(f"    {exp:<58} {tag:<10} TTFT={sim_ttft:>8.1f}ms ({te:>+6.1f}%)  E2E={sim_e2e:>8.0f}ms ({ee:>+6.1f}%)")
            if not is_reason:
                if split == 'train':
                    train_ttft.append(abs(te))
                    train_e2e.append(abs(ee))
                elif split == 'validate':
                    val_ttft.append(abs(te))
                    val_e2e.append(abs(ee))

        if train_ttft:
            print(f"  → Train: TTFT MAE={np.mean(train_ttft):.1f}%, E2E MAE={np.mean(train_e2e):.1f}%")
        if val_ttft:
            print(f"  → Val:   TTFT MAE={np.mean(val_ttft):.1f}%, E2E MAE={np.mean(val_e2e):.1f}%")

    # Save
    coeffs = {
        'iter5c': {
            'beta': [float(b0), float(b1), float(b2), float(b3)],
            'alpha': [float(a0), 0.0, float(g1)],
            'delta': [float(d0)] if d0 > 0 else None,
            'loss': float(result.fun),
            'evals': eval_count[0],
        }
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iter5c_coefficients.json')
    with open(out_path, 'w') as f:
        json.dump(coeffs, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
