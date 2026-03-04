#!/usr/bin/env python3
"""
Iter 5 — Replay-Calibrated Joint Coefficient Optimization

Strategy Evolution Phase 3: Implement and Execute.

Runs three hypothesis arms:
  H-control:  Iter 3 baseline (δ=0)
  H-main:     Corrected β (Phase A) + Nelder-Mead (α₀, δ₀) via replay (Phase B)
  H-ablation: Iter 3 β + Nelder-Mead (α₀, δ₀, δ₁) via replay

Optimizer uses 4 roleplay experiments (1 per model, ~5s each) for fast iteration.
Final evaluation uses all 12 non-reasoning + 3 reasoning experiments.

Usage:
    python3 training/iter5_optimize.py [--max-evals 200] [--skip-optimize]
"""

from __future__ import annotations

import argparse
import csv
import json
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, nnls


# ─── Constants ───────────────────────────────────────────

REPLAY_BIN = '/Users/sri/Documents/Projects/inference-sim/.worktrees/iter5-decomposed/replay'
DATA_DIR = '/Users/sri/Documents/Projects/inference-sim/training'

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}

ITER3_BETA = [116.110, 1226.868, 19.943, 9445.157]
ITER3_ALPHA = [13732.0, 0.0, 860.6]

# Optimizer subset: 4 roleplay experiments (one per model, smallest)
OPTIMIZER_EXPERIMENTS = [
    '20260217-162547-llama-2-7b-tp1-roleplay',
    '20260218-084319-llama-2-70b-tp4-roleplay',
    '20260218-141024-mixtral-8x7b-v0-1-tp2-roleplay',
    '20260218-155500-codellama-34b-tp2-roleplay',
]


def beta_features(pf: int, dc: int, model: str) -> np.ndarray:
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0
    return np.array([L, dc * L * kv_dim * 1e-6, (pf + dc) * is_moe, is_tp])


# ─── Phase A: Corrected β Fitting ────────────────────────

def fit_corrected_beta() -> tuple[np.ndarray, dict]:
    """Fit β from overhead-corrected consecutive-pair wall clock.

    1. Build consecutive pairs from step data (train only)
    2. Compute per-model median overhead = median(T_wall - β_iter3·features)
    3. Correct target: T_gpu = T_wall - overhead_median[model]
    4. Fit β via NNLS against corrected target
    """
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

    # Step 1: Compute per-model overhead using Iter 3 β as reference
    iter3_beta = np.array(ITER3_BETA)
    overheads = {}
    for model in MODEL_CONFIGS:
        mp = [p for p in pairs if p['model'] == model]
        if not mp:
            continue
        resids = []
        for p in mp:
            feat = beta_features(p['pf'], p['dc'], model)
            pred = float(np.dot(iter3_beta, feat))
            resids.append(p['t_wall'] - pred)
        overheads[model] = float(np.median(resids))

    # Step 2: Correct targets and fit β
    X = []
    y = []
    for p in pairs:
        feat = beta_features(p['pf'], p['dc'], p['model'])
        oh = overheads.get(p['model'], 0)
        t_gpu = p['t_wall'] - oh
        if t_gpu > 0:  # skip if correction overshoots
            X.append(feat)
            y.append(t_gpu)

    X = np.array(X)
    y = np.array(y)
    beta_corrected, _ = nnls(X, y)

    return beta_corrected, overheads


# ─── Replay Runner ───────────────────────────────────────

def run_replay(input_path: str, beta: list, alpha: list, delta: list | None = None) -> dict | None:
    """Run BLIS replay and return parsed output, or None on failure."""
    cmd = [
        REPLAY_BIN,
        '--input', input_path,
        '--beta', ','.join(f'{b:.6f}' for b in beta),
        '--alpha', ','.join(f'{a:.6f}' for a in alpha),
    ]
    if delta:
        cmd.extend(['--delta', ','.join(f'{d:.6f}' for d in delta)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def replay_ttft_mape(experiments: list[str], beta: list, alpha: list,
                     delta: list | None, ground_truth: dict) -> float:
    """Compute mean TTFT MAPE across experiments via BLIS replay."""
    errors = []
    for exp in experiments:
        input_path = os.path.join(DATA_DIR, 'replay_data', f'{exp}.json')
        if not os.path.exists(input_path):
            continue
        gt = ground_truth.get(exp)
        if not gt:
            continue

        output = run_replay(input_path, beta, alpha, delta)
        if output is None:
            return 1e6  # penalty for failed run

        sim_ttft = output['summary']['ttft_mean_ms']
        gt_ttft = gt['ttft']['mean_ms']
        if gt_ttft > 0:
            errors.append(abs(sim_ttft - gt_ttft) / gt_ttft)

    if not errors:
        return 1e6
    return float(np.mean(errors)) * 100  # percentage


# ─── Optimization ────────────────────────────────────────

def optimize_hmain(beta: list, ground_truth: dict, max_evals: int) -> dict:
    """Phase B: Optimize (α₀, δ₀) with fixed corrected β."""
    eval_count = [0]
    best_loss = [1e6]

    def objective(params):
        alpha_0, delta_0 = params
        if alpha_0 < 0 or delta_0 < 0:
            return 1e6
        alpha = [alpha_0, 0.0, 0.0]
        delta = [delta_0]
        loss = replay_ttft_mape(OPTIMIZER_EXPERIMENTS, beta, alpha, delta, ground_truth)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            sys.stderr.write(f'\r  H-main eval {eval_count[0]}: α₀={alpha_0:.0f} δ₀={delta_0:.0f} loss={loss:.2f}%')
            sys.stderr.flush()
        return loss

    # Warm start: α₀=10000, δ₀=5000
    x0 = [10000.0, 5000.0]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxfev': max_evals, 'xatol': 100, 'fatol': 0.1,
                               'initial_simplex': np.array([x0, [15000, 5000], [10000, 10000]])})
    sys.stderr.write('\n')

    return {
        'alpha_0': result.x[0],
        'delta_0': result.x[1],
        'loss': result.fun,
        'evals': eval_count[0],
        'success': result.success,
    }


def optimize_hablation(ground_truth: dict, max_evals: int) -> dict:
    """Optimize (α₀, δ₀, δ₁) with fixed Iter 3 β."""
    eval_count = [0]
    best_loss = [1e6]

    def objective(params):
        alpha_0, delta_0, delta_1 = params
        if alpha_0 < 0 or delta_0 < -5000:
            return 1e6
        alpha = [alpha_0, 0.0, 0.0]
        delta = [delta_0, delta_1]
        loss = replay_ttft_mape(OPTIMIZER_EXPERIMENTS, list(ITER3_BETA), alpha, delta, ground_truth)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            sys.stderr.write(f'\r  H-ablation eval {eval_count[0]}: α₀={alpha_0:.0f} δ₀={delta_0:.0f} δ₁={delta_1:.1f} loss={loss:.2f}%')
            sys.stderr.flush()
        return loss

    x0 = [5000.0, 3000.0, 0.0]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxfev': max_evals, 'xatol': 100, 'fatol': 0.1,
                               'initial_simplex': np.array([x0, [10000, 3000, 0], [5000, 8000, 0], [5000, 3000, -100]])})
    sys.stderr.write('\n')

    return {
        'alpha_0': result.x[0],
        'delta_0': result.x[1],
        'delta_1': result.x[2],
        'loss': result.fun,
        'evals': eval_count[0],
        'success': result.success,
    }


# ─── Full Evaluation ─────────────────────────────────────

@dataclass
class ArmConfig:
    name: str
    beta: list
    alpha: list
    delta: list | None


def evaluate_all(arms: list[ArmConfig], ground_truth: dict) -> dict:
    """Run all experiments for each arm, return structured results."""
    all_replay_files = sorted(glob.glob(os.path.join(DATA_DIR, 'replay_data', '*.json')))
    all_replay_files = [f for f in all_replay_files if '_ground_truth' not in f]

    results = {}
    for arm in arms:
        arm_results = {}
        sys.stderr.write(f'\nEvaluating {arm.name}...\n')
        for rf in all_replay_files:
            exp = os.path.basename(rf).replace('.json', '')
            gt = ground_truth.get(exp)
            if not gt:
                continue

            sys.stderr.write(f'  {exp}...')
            sys.stderr.flush()
            output = run_replay(rf, arm.beta, arm.alpha, arm.delta)
            if output is None:
                sys.stderr.write(' FAILED\n')
                continue

            sim_ttft = output['summary']['ttft_mean_ms']
            sim_e2e = output['summary']['e2e_mean_ms']
            gt_ttft = gt['ttft']['mean_ms']
            gt_e2e = gt['e2e']['mean_ms']

            arm_results[exp] = {
                'model': gt['model_short'],
                'profile': gt['profile'],
                'split': gt['split'],
                'gt_ttft': gt_ttft,
                'sim_ttft': sim_ttft,
                'ttft_err': (sim_ttft - gt_ttft) / gt_ttft * 100 if gt_ttft > 0 else 0,
                'gt_e2e': gt_e2e,
                'sim_e2e': sim_e2e,
                'e2e_err': (sim_e2e - gt_e2e) / gt_e2e * 100 if gt_e2e > 0 else 0,
                'completed': output['summary']['completed'],
                'throughput': output['summary']['responses_per_sec'],
            }
            sys.stderr.write(f' TTFT={sim_ttft:.1f}ms ({arm_results[exp]["ttft_err"]:+.1f}%)\n')

        results[arm.name] = arm_results
    return results


def print_results(results: dict, arms: list[ArmConfig]):
    """Print comprehensive results table."""
    print("\n" + "=" * 120)
    print("FULL EVALUATION RESULTS")
    print("=" * 120)

    for arm in arms:
        arm_data = results.get(arm.name, {})
        if not arm_data:
            continue

        # Separate by split and profile
        train_non_reason = {k: v for k, v in arm_data.items() if v['split'] == 'train' and v['profile'] != 'reasoning'}
        val = {k: v for k, v in arm_data.items() if v['split'] == 'validate' and v['profile'] != 'reasoning'}
        test_reason = {k: v for k, v in arm_data.items() if v['profile'] == 'reasoning'}
        test_non_reason = {k: v for k, v in arm_data.items() if v['split'] == 'test' and v['profile'] != 'reasoning'}

        print(f"\n--- {arm.name} ---")
        print(f"  β = [{', '.join(f'{b:.1f}' for b in arm.beta)}]")
        print(f"  α = [{', '.join(f'{a:.1f}' for a in arm.alpha)}]")
        if arm.delta:
            print(f"  δ = [{', '.join(f'{d:.1f}' for d in arm.delta)}]")
        else:
            print(f"  δ = none")

        for label, subset in [('TRAIN (non-reasoning)', train_non_reason),
                               ('VALIDATE', val),
                               ('TEST (reasoning)', test_reason)]:
            if not subset:
                continue
            ttft_errs = [abs(v['ttft_err']) for v in subset.values()]
            e2e_errs = [abs(v['e2e_err']) for v in subset.values()]
            ttft_bias = [v['ttft_err'] for v in subset.values()]
            e2e_bias = [v['e2e_err'] for v in subset.values()]

            print(f"\n  {label} ({len(subset)} experiments):")
            print(f"    TTFT MAE: {np.mean(ttft_errs):.1f}%  (bias: {np.mean(ttft_bias):+.1f}%)")
            print(f"    E2E  MAE: {np.mean(e2e_errs):.1f}%  (bias: {np.mean(e2e_bias):+.1f}%)")

            print(f"\n    {'Experiment':<55} {'GT TTFT':>8} {'Sim':>8} {'Err%':>7} | {'GT E2E':>8} {'Sim':>8} {'Err%':>7}")
            print(f"    {'-'*110}")
            for exp in sorted(subset.keys()):
                v = subset[exp]
                print(f"    {exp[:54]:<55} {v['gt_ttft']:>7.1f}ms {v['sim_ttft']:>7.1f}ms {v['ttft_err']:>+6.1f}% | "
                      f"{v['gt_e2e']:>7.0f}ms {v['sim_e2e']:>7.0f}ms {v['e2e_err']:>+6.1f}%")

    # Summary comparison
    print("\n" + "=" * 120)
    print("SUMMARY COMPARISON")
    print("=" * 120)
    print(f"\n{'Arm':<50} {'Train TTFT MAE':>15} {'Train E2E MAE':>15} {'Val TTFT MAE':>14} {'Test TTFT MAE':>15}")
    print("-" * 112)

    for arm in arms:
        arm_data = results.get(arm.name, {})
        train = [v for v in arm_data.values() if v['split'] == 'train' and v['profile'] != 'reasoning']
        val = [v for v in arm_data.values() if v['split'] == 'validate' and v['profile'] != 'reasoning']
        test = [v for v in arm_data.values() if v['profile'] == 'reasoning']

        t_ttft = f"{np.mean([abs(v['ttft_err']) for v in train]):.1f}%" if train else "—"
        t_e2e = f"{np.mean([abs(v['e2e_err']) for v in train]):.1f}%" if train else "—"
        v_ttft = f"{np.mean([abs(v['ttft_err']) for v in val]):.1f}%" if val else "—"
        te_ttft = f"{np.mean([abs(v['ttft_err']) for v in test]):.1f}%" if test else "—"

        print(f"{arm.name:<50} {t_ttft:>15} {t_e2e:>15} {v_ttft:>14} {te_ttft:>15}")

    # Per-model TTFT MAE
    print(f"\nPer-Model TTFT MAE (train, non-reasoning):")
    models = ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']
    print(f"{'Arm':<50}", end="")
    for m in models:
        print(f" {m:>14}", end="")
    print()
    print("-" * (50 + 15 * len(models)))

    for arm in arms:
        arm_data = results.get(arm.name, {})
        print(f"{arm.name:<50}", end="")
        for m in models:
            errs = [abs(v['ttft_err']) for v in arm_data.values()
                    if v['model'] == m and v['split'] == 'train' and v['profile'] != 'reasoning']
            if errs:
                print(f" {np.mean(errs):>13.1f}%", end="")
            else:
                print(f" {'—':>14}", end="")
        print()


# ─── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', type=int, default=200)
    parser.add_argument('--skip-optimize', action='store_true')
    args = parser.parse_args()

    print("=" * 120)
    print("Iter 5 — Replay-Calibrated Joint Coefficient Optimization")
    print("Strategy Evolution Phase 3: Execute")
    print("=" * 120)

    # Load ground truth
    ground_truth = {}
    for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        ground_truth[d['experiment']] = d
    print(f"\nLoaded {len(ground_truth)} ground truth files")

    # ─── Phase A: Corrected β ────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE A: Corrected β Fitting")
    print("=" * 80)

    beta_corrected, overheads = fit_corrected_beta()
    print(f"\n  Per-model overhead (Iter 3 residual):")
    for model, oh in sorted(overheads.items()):
        print(f"    {model}: {oh:.0f}µs")
    print(f"\n  Corrected β: [{', '.join(f'{b:.2f}' for b in beta_corrected)}]")
    print(f"  Iter 3 β:    [{', '.join(f'{b:.2f}' for b in ITER3_BETA)}]")
    print(f"\n  Changes:")
    labels = ['β₀(L)', 'β₁(KV)', 'β₂(MoE)', 'β₃(TP)']
    for i, label in enumerate(labels):
        ratio = beta_corrected[i] / ITER3_BETA[i] if ITER3_BETA[i] > 0 else float('inf')
        print(f"    {label}: {ITER3_BETA[i]:.1f} → {beta_corrected[i]:.1f} ({ratio:.2f}x)")

    # ─── Optimization ────────────────────────────────────
    if args.skip_optimize:
        # Use pre-computed values for debugging
        hmain_result = {'alpha_0': 10000, 'delta_0': 5000, 'evals': 0, 'loss': 0, 'success': True}
        hablation_result = {'alpha_0': 5000, 'delta_0': 3000, 'delta_1': 0, 'evals': 0, 'loss': 0, 'success': True}
    else:
        print("\n" + "=" * 80)
        print("PHASE B: H-main Optimization — (α₀, δ₀) with corrected β")
        print("=" * 80)
        t0 = time.time()
        hmain_result = optimize_hmain(beta_corrected.tolist(), ground_truth, args.max_evals)
        elapsed = time.time() - t0
        print(f"\n  Result: α₀={hmain_result['alpha_0']:.0f}µs, δ₀={hmain_result['delta_0']:.0f}µs")
        print(f"  Loss: {hmain_result['loss']:.2f}% TTFT MAPE")
        print(f"  Evaluations: {hmain_result['evals']}, Time: {elapsed:.0f}s, Success: {hmain_result['success']}")

        print("\n" + "=" * 80)
        print("H-ablation Optimization — (α₀, δ₀, δ₁) with Iter 3 β")
        print("=" * 80)
        t0 = time.time()
        hablation_result = optimize_hablation(ground_truth, args.max_evals)
        elapsed = time.time() - t0
        print(f"\n  Result: α₀={hablation_result['alpha_0']:.0f}µs, δ₀={hablation_result['delta_0']:.0f}µs, δ₁={hablation_result['delta_1']:.1f}µs/req")
        print(f"  Loss: {hablation_result['loss']:.2f}% TTFT MAPE")
        print(f"  Evaluations: {hablation_result['evals']}, Time: {elapsed:.0f}s, Success: {hablation_result['success']}")

    # ─── Build arm configs ───────────────────────────────
    arms = [
        ArmConfig(
            name='H-control (Iter 3, δ=0)',
            beta=list(ITER3_BETA),
            alpha=list(ITER3_ALPHA),
            delta=None,
        ),
        ArmConfig(
            name=f'H-main (corrected β, α₀={hmain_result["alpha_0"]:.0f}, δ₀={hmain_result["delta_0"]:.0f})',
            beta=beta_corrected.tolist(),
            alpha=[hmain_result['alpha_0'], 0.0, 0.0],
            delta=[hmain_result['delta_0']],
        ),
        ArmConfig(
            name=f'H-ablation (Iter3 β, α₀={hablation_result["alpha_0"]:.0f}, δ₀={hablation_result["delta_0"]:.0f}, δ₁={hablation_result["delta_1"]:.0f})',
            beta=list(ITER3_BETA),
            alpha=[hablation_result['alpha_0'], 0.0, 0.0],
            delta=[hablation_result['delta_0'], hablation_result['delta_1']],
        ),
    ]

    # ─── Full Evaluation ─────────────────────────────────
    print("\n" + "=" * 80)
    print("FULL EVALUATION (all experiments, all arms)")
    print("=" * 80)

    results = evaluate_all(arms, ground_truth)
    print_results(results, arms)

    # ─── Save coefficients ───────────────────────────────
    output = {
        'iter5_hmain': {
            'beta': beta_corrected.tolist(),
            'alpha': [hmain_result['alpha_0'], 0.0, 0.0],
            'delta': [hmain_result['delta_0']],
            'overheads_used': overheads,
        },
        'iter5_hablation': {
            'beta': list(ITER3_BETA),
            'alpha': [hablation_result['alpha_0'], 0.0, 0.0],
            'delta': [hablation_result['delta_0'], hablation_result['delta_1']],
        },
        'iter3_baseline': {
            'beta': list(ITER3_BETA),
            'alpha': list(ITER3_ALPHA),
            'delta': None,
        },
    }
    coeff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iter5_coefficients.json')
    with open(coeff_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nCoefficients saved to {coeff_path}")


if __name__ == '__main__':
    main()
