#!/usr/bin/env python3
"""
Iter 5b — Phase-Aware δ via γ₁, All Coefficients Non-Negative

The H-ablation winner had negative δ (physically nonsensical). This run uses:
- Corrected β (smaller, closer to true GPU compute)
- α₀ ≥ 0: per-request pipeline overhead (adds to TTFT, one-time)
- δ₀ ≥ 0: per-step overhead (adds to both TTFT and E2E, per step)
- γ₁ ≥ 0: per-output-token overhead (adds primarily to E2E, NOT TTFT)

γ₁ is the "phase-aware" mechanism: at TTFT measurement, output_tokens=0 so
γ₁ has zero effect. During decode, γ₁ × output_tokens adds to E2E. This
independently tunes E2E without touching TTFT.

The alpha parameter in the replay binary is [α₀, α₁, γ₁]:
- α₀ = QueueingTime (pre-scheduling overhead)
- α₁ = per-input-token overhead (we set to 0)
- γ₁ = OutputTokenProcessingTime per output token

Multi-objective loss: 0.5 × TTFT_MAPE + 0.5 × E2E_MAPE across 4 general
experiments (one per model, highest load diversity).

Usage:
    python3 training/iter5b_optimize.py
"""

from __future__ import annotations

import csv
import json
import glob
import os
import subprocess
import sys
import time
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize, nnls

REPLAY_BIN = '/Users/sri/Documents/Projects/inference-sim/.worktrees/iter5-decomposed/replay'
DATA_DIR = '/Users/sri/Documents/Projects/inference-sim/training'

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}

ITER3_BETA = [116.110, 1226.868, 19.943, 9445.157]

# 4 general experiments (one per model, highest load diversity)
OPTIMIZER_EXPERIMENTS = [
    '20260217-231439-llama-2-7b-tp1-general',
    '20260217-202857-llama-2-70b-tp4-general',
    '20260218-130541-mixtral-8x7b-v0-1-tp2-general',
    '20260218-150304-codellama-34b-tp2-general',
]


def beta_features(pf, dc, model):
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0
    return np.array([L, dc * L * kv_dim * 1e-6, (pf + dc) * is_moe, is_tp])


def fit_corrected_beta():
    """Fit β from overhead-corrected step wall clock (same as iter5_optimize Phase A)."""
    by_exp = defaultdict(list)
    with open(os.path.join(DATA_DIR, 'iter0_steps.csv')) as f:
        for row in csv.DictReader(f):
            by_exp[row['experiment']].append(row)

    pairs = []
    for exp, rows in by_exp.items():
        rows.sort(key=lambda r: int(r['step_id']))
        for i in range(len(rows) - 1):
            curr, nxt = rows[i], rows[i + 1]
            if int(nxt['step_id']) - int(curr['step_id']) != 1:
                continue
            if curr['split'] != 'train':
                continue
            t_wall = (int(nxt['ts_start_ns']) - int(curr['ts_start_ns'])) / 1000
            pairs.append({
                'model': curr['model_short'],
                'pf': int(curr['prefill_tokens']),
                'dc': int(curr['decode_tokens']),
                't_wall': t_wall,
            })

    iter3_beta = np.array(ITER3_BETA)
    overheads = {}
    for model in MODEL_CONFIGS:
        mp = [p for p in pairs if p['model'] == model]
        if not mp:
            continue
        resids = [p['t_wall'] - float(np.dot(iter3_beta, beta_features(p['pf'], p['dc'], model))) for p in mp]
        overheads[model] = float(np.median(resids))

    X, y = [], []
    for p in pairs:
        feat = beta_features(p['pf'], p['dc'], p['model'])
        t_gpu = p['t_wall'] - overheads.get(p['model'], 0)
        if t_gpu > 0:
            X.append(feat)
            y.append(t_gpu)

    beta, _ = nnls(np.array(X), np.array(y))
    return beta, overheads


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


def replay_loss(experiments, beta, alpha, delta, ground_truth, w_ttft=0.5, w_e2e=0.5):
    """Multi-objective loss: weighted TTFT_MAPE + E2E_MAPE."""
    ttft_errors, e2e_errors = [], []
    for exp in experiments:
        input_path = os.path.join(DATA_DIR, 'replay_data', f'{exp}.json')
        gt = ground_truth.get(exp)
        if not gt or not os.path.exists(input_path):
            continue

        output = run_replay(input_path, beta, alpha, delta)
        if output is None:
            return 1e6

        sim_ttft = output['summary']['ttft_mean_ms']
        sim_e2e = output['summary']['e2e_mean_ms']
        gt_ttft = gt['ttft']['mean_ms']
        gt_e2e = gt['e2e']['mean_ms']

        if gt_ttft > 0:
            ttft_errors.append(abs(sim_ttft - gt_ttft) / gt_ttft)
        if gt_e2e > 0:
            e2e_errors.append(abs(sim_e2e - gt_e2e) / gt_e2e)

    if not ttft_errors or not e2e_errors:
        return 1e6
    return (w_ttft * np.mean(ttft_errors) + w_e2e * np.mean(e2e_errors)) * 100


def optimize_nonneg(beta, ground_truth, max_evals=200):
    """Optimize (α₀, γ₁, δ₀) ≥ 0 with corrected β, multi-objective loss."""
    eval_count = [0]
    best_loss = [1e6]
    best_params = [None]

    def objective(params):
        alpha_0, gamma_1, delta_0 = params
        # Non-negativity via penalty
        if alpha_0 < 0 or gamma_1 < 0 or delta_0 < 0:
            return 1e6

        alpha = [alpha_0, 0.0, gamma_1]
        delta = [delta_0]
        loss = replay_loss(OPTIMIZER_EXPERIMENTS, beta, alpha, delta, ground_truth)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params[0] = params.copy()
            sys.stderr.write(f'\r  eval {eval_count[0]:>3d}: α₀={alpha_0:>8.0f} γ₁={gamma_1:>8.1f} δ₀={delta_0:>8.0f} loss={loss:>6.2f}%')
            sys.stderr.flush()
        return loss

    # Warm start: α₀≈19000 (from H-main), γ₁≈500 (moderate E2E adjustment), δ₀≈500
    x0 = [19000.0, 500.0, 500.0]
    simplex = np.array([
        x0,
        [25000.0, 500.0, 500.0],
        [19000.0, 1500.0, 500.0],
        [19000.0, 500.0, 3000.0],
    ])

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxfev': max_evals, 'xatol': 50, 'fatol': 0.05,
                               'initial_simplex': simplex})
    sys.stderr.write('\n')

    return {
        'alpha_0': max(0, result.x[0]),
        'gamma_1': max(0, result.x[1]),
        'delta_0': max(0, result.x[2]),
        'loss': result.fun,
        'evals': eval_count[0],
        'success': result.success,
    }


def full_evaluate(arms, ground_truth):
    """Evaluate all arms on all experiments."""
    replay_files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*.json'))
                          if '_ground_truth' not in f)

    results = {}
    for arm_name, cfg in arms.items():
        arm_results = {}
        sys.stderr.write(f'\nEvaluating {arm_name}...\n')
        for rf in replay_files:
            exp = os.path.basename(rf).replace('.json', '')
            gt = ground_truth.get(exp)
            if not gt:
                continue

            sys.stderr.write(f'  {exp[:50]:50s}')
            sys.stderr.flush()
            output = run_replay(rf, cfg['beta'], cfg['alpha'], cfg.get('delta'))
            if output is None:
                sys.stderr.write(' FAIL\n')
                continue

            sim_ttft = output['summary']['ttft_mean_ms']
            sim_e2e = output['summary']['e2e_mean_ms']
            gt_ttft = gt['ttft']['mean_ms']
            gt_e2e = gt['e2e']['mean_ms']

            arm_results[exp] = {
                'model': gt['model_short'], 'profile': gt['profile'], 'split': gt['split'],
                'gt_ttft': gt_ttft, 'sim_ttft': sim_ttft,
                'ttft_err': (sim_ttft - gt_ttft) / gt_ttft * 100,
                'gt_e2e': gt_e2e, 'sim_e2e': sim_e2e,
                'e2e_err': (sim_e2e - gt_e2e) / gt_e2e * 100,
            }
            sys.stderr.write(f' TTFT={sim_ttft:.1f}ms ({arm_results[exp]["ttft_err"]:+.1f}%) E2E err={arm_results[exp]["e2e_err"]:+.1f}%\n')

        results[arm_name] = arm_results
    return results


def print_results(results, arms):
    print("\n" + "=" * 130)
    print("ITER 5b — FULL EVALUATION (non-negative coefficients)")
    print("=" * 130)

    for arm_name, cfg in arms.items():
        ar = results.get(arm_name, {})
        if not ar:
            continue
        print(f"\n{'─'*130}")
        print(f"  {arm_name}")
        b = cfg['beta']
        a = cfg['alpha']
        d = cfg.get('delta')
        print(f"  β=[{', '.join(f'{x:.1f}' for x in b)}]  α=[{', '.join(f'{x:.0f}' for x in a)}]  δ={d}")
        print(f"{'─'*130}")

        for label, filt in [('TRAIN (non-reasoning)', lambda v: v['split'] == 'train' and v['profile'] != 'reasoning'),
                             ('VALIDATE', lambda v: v['split'] == 'validate' and v['profile'] != 'reasoning'),
                             ('TEST (reasoning)', lambda v: v['profile'] == 'reasoning')]:
            subset = {k: v for k, v in ar.items() if filt(v)}
            if not subset:
                continue
            ttft_mae = np.mean([abs(v['ttft_err']) for v in subset.values()])
            e2e_mae = np.mean([abs(v['e2e_err']) for v in subset.values()])
            ttft_bias = np.mean([v['ttft_err'] for v in subset.values()])
            e2e_bias = np.mean([v['e2e_err'] for v in subset.values()])

            print(f"\n  {label}: TTFT MAE={ttft_mae:.1f}% (bias={ttft_bias:+.1f}%), E2E MAE={e2e_mae:.1f}% (bias={e2e_bias:+.1f}%)")
            print(f"  {'Experiment':<58} {'GT TTFT':>8} {'Sim':>9} {'Err':>7} | {'GT E2E':>8} {'Sim':>9} {'Err':>7}")
            for exp in sorted(subset):
                v = subset[exp]
                print(f"  {exp:<58} {v['gt_ttft']:>7.1f}ms {v['sim_ttft']:>8.1f}ms {v['ttft_err']:>+6.1f}% | "
                      f"{v['gt_e2e']:>7.0f}ms {v['sim_e2e']:>8.0f}ms {v['e2e_err']:>+6.1f}%")

    # Summary
    print(f"\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    print(f"\n{'Arm':<55} {'Train TTFT':>11} {'Train E2E':>10} {'Val TTFT':>10} {'Val E2E':>9}")
    print("-" * 98)
    for arm_name in arms:
        ar = results.get(arm_name, {})
        train = [v for v in ar.values() if v['split'] == 'train' and v['profile'] != 'reasoning']
        val = [v for v in ar.values() if v['split'] == 'validate' and v['profile'] != 'reasoning']
        t_t = f"{np.mean([abs(v['ttft_err']) for v in train]):.1f}%" if train else "—"
        t_e = f"{np.mean([abs(v['e2e_err']) for v in train]):.1f}%" if train else "—"
        v_t = f"{np.mean([abs(v['ttft_err']) for v in val]):.1f}%" if val else "—"
        v_e = f"{np.mean([abs(v['e2e_err']) for v in val]):.1f}%" if val else "—"
        print(f"{arm_name:<55} {t_t:>11} {t_e:>10} {v_t:>10} {v_e:>9}")

    # Per-model
    models = ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']
    print(f"\nPer-Model Train TTFT MAE / E2E MAE:")
    print(f"{'Arm':<55}", end="")
    for m in models:
        print(f" {m:>20}", end="")
    print()
    print("-" * (55 + 21 * 4))
    for arm_name in arms:
        ar = results.get(arm_name, {})
        print(f"{arm_name:<55}", end="")
        for m in models:
            train_m = [v for v in ar.values() if v['model'] == m and v['split'] == 'train' and v['profile'] != 'reasoning']
            if train_m:
                tt = np.mean([abs(v['ttft_err']) for v in train_m])
                te = np.mean([abs(v['e2e_err']) for v in train_m])
                print(f" {tt:>8.1f}% / {te:>5.1f}%", end="")
            else:
                print(f" {'—':>20}", end="")
        print()


def main():
    print("=" * 130)
    print("Iter 5b — Phase-Aware δ via γ₁, All Non-Negative")
    print("=" * 130)

    # Load ground truth
    ground_truth = {}
    for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        ground_truth[d['experiment']] = d

    # Phase A: corrected β
    print("\n[1] Fitting corrected β...")
    beta_corrected, overheads = fit_corrected_beta()
    print(f"    β_corrected = [{', '.join(f'{b:.1f}' for b in beta_corrected)}]")
    print(f"    Iter 3 β    = [{', '.join(f'{b:.1f}' for b in ITER3_BETA)}]")

    # Optimize (α₀, γ₁, δ₀) ≥ 0
    print("\n[2] Optimizing (α₀, γ₁, δ₀) ≥ 0 with corrected β...")
    print("    Loss = 0.5 × TTFT_MAPE + 0.5 × E2E_MAPE (4 general experiments)")
    t0 = time.time()
    opt = optimize_nonneg(beta_corrected.tolist(), ground_truth, max_evals=200)
    elapsed = time.time() - t0
    print(f"\n    Result: α₀={opt['alpha_0']:.0f}µs, γ₁={opt['gamma_1']:.1f}µs/tok, δ₀={opt['delta_0']:.0f}µs")
    print(f"    Loss: {opt['loss']:.2f}% (combined TTFT+E2E MAPE)")
    print(f"    Evaluations: {opt['evals']}, Time: {elapsed:.0f}s")

    # Build arms for comparison
    # Load H-ablation from Iter 5
    iter5_coeffs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iter5_coefficients.json')
    if os.path.exists(iter5_coeffs_path):
        with open(iter5_coeffs_path) as f:
            iter5_coeffs = json.load(f)
        ha = iter5_coeffs['iter5_hablation']
    else:
        ha = {'beta': ITER3_BETA, 'alpha': [24171, 0, 0], 'delta': [-3542, -252]}

    arms = {
        'H-control (Iter 3, δ=0)': {
            'beta': ITER3_BETA,
            'alpha': [13732.0, 0.0, 860.6],
            'delta': None,
        },
        'Iter5 H-ablation (neg δ, for comparison)': {
            'beta': ha['beta'],
            'alpha': ha['alpha'],
            'delta': ha['delta'],
        },
        f'Iter5b (corrected β, α₀={opt["alpha_0"]:.0f}, γ₁={opt["gamma_1"]:.0f}, δ₀={opt["delta_0"]:.0f})': {
            'beta': beta_corrected.tolist(),
            'alpha': [opt['alpha_0'], 0.0, opt['gamma_1']],
            'delta': [opt['delta_0']],
        },
    }

    # Full evaluation
    print("\n[3] Full evaluation on all experiments...")
    results = full_evaluate(arms, ground_truth)
    print_results(results, arms)

    # Save coefficients
    output = {
        'iter5b': {
            'beta': beta_corrected.tolist(),
            'alpha': [opt['alpha_0'], 0.0, opt['gamma_1']],
            'delta': [opt['delta_0']],
            'all_nonneg': bool(all(b >= 0 for b in beta_corrected) and opt['alpha_0'] >= 0 and opt['gamma_1'] >= 0 and opt['delta_0'] >= 0),
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iter5b_coefficients.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nCoefficients saved to {out_path}")


if __name__ == '__main__':
    main()
