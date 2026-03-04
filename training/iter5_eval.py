"""
Iter 5 — Full Evaluation of Wall-Clock-Fitted Cross-Model β

Fits β from TRAIN-only consecutive-pair wall-clock data, then evaluates:
1. Step-level MAPE (predicted step time vs actual wall clock) on train/test/val
2. Analytical TTFT/E2E (journey-level, using 2.0x pipeline factor + refitted α)
3. Comparison with Iter 3 coefficients

The 2.0x pipeline factor means:
  predicted_single_step_prefill = 2.0 × β·features(step)

For TTFT:
  TTFT_pred = α₀ + 2.0 × β·features(prefill_step)

For E2E:
  E2E_pred = TTFT_pred + Σ(decode_steps) × T_per_decode_step
  where T_per_decode_step = β·features(decode_step) (NOT 2.0x — decode is continuous)

Usage:
    python3 training/iter5_eval.py
"""

from __future__ import annotations

import csv
import json
import glob
import os
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls


# ─── Model configs ───────────────────────────────────────

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}

ITER3_BETA = np.array([116.110, 1226.868, 19.943, 9445.157])
ITER3_ALPHA = np.array([13732.0, 0.0])  # α₀, α₁
ITER3_GAMMA = np.array([0.0, 860.6])    # γ₀, γ₁


def beta_features(pf: int, dc: int, model: str) -> np.ndarray:
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0
    return np.array([
        L,
        dc * L * kv_dim * 1e-6,
        (pf + dc) * is_moe,
        is_tp,
    ])


# ─── Data Loading ────────────────────────────────────────

def find_data_dir() -> str:
    """Find the training data directory."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training'),
        '/Users/sri/Documents/Projects/inference-sim/training',
        'training',
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'iter0_steps.csv')):
            return c
    raise FileNotFoundError("Cannot find training data directory")


def load_steps(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'experiment': row['experiment'],
                'model': row['model_short'],
                'split': row['split'],
                'step_id': int(row['step_id']),
                'duration_us': int(row['duration_us']),
                'ts_start_ns': int(row['ts_start_ns']),
                'ts_end_ns': int(row['ts_end_ns']),
                'prefill_tokens': int(row['prefill_tokens']),
                'decode_tokens': int(row['decode_tokens']),
                'running_depth': int(row['running_depth']),
                'scheduled_tokens': int(row['scheduled_tokens']),
            })
    return records


def load_journeys(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'experiment': row['experiment'],
                'model': row['model_short'],
                'split': row['split'],
                'scheduled_ns': int(row['scheduled_ns']),
                'first_token_ns': int(row['first_token_ns']),
                'finished_ns': int(row['finished_ns']),
                'scheduled_step': int(row['scheduled_step']),
                'first_token_step': int(row['first_token_step']),
                'finished_step': int(row['finished_step']),
                'prefill_total': int(row['prefill_total']),
                'decode_max': int(row['decode_max']),
                'num_preemptions': int(row['num_preemptions']),
            })
    return records


def load_ground_truth(data_dir: str) -> dict:
    """Load per-experiment ground truth from replay data."""
    gt = {}
    for f in glob.glob(os.path.join(data_dir, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        gt[d['experiment']] = d
    return gt


# ─── Step 1: Build consecutive-pair wall-clock data ─────

def build_consecutive_pairs(steps: list[dict]) -> list[dict]:
    """Build consecutive-pair data with wall-clock timing."""
    by_exp = defaultdict(list)
    for s in steps:
        by_exp[s['experiment']].append(s)

    pairs = []
    for exp, exp_steps in by_exp.items():
        exp_steps.sort(key=lambda s: s['step_id'])
        for i in range(len(exp_steps) - 1):
            curr = exp_steps[i]
            nxt = exp_steps[i + 1]
            if nxt['step_id'] - curr['step_id'] != 1:
                continue
            t_wall_us = (nxt['ts_start_ns'] - curr['ts_start_ns']) / 1000
            pairs.append({
                'model': curr['model'],
                'split': curr['split'],
                'experiment': curr['experiment'],
                'pf': curr['prefill_tokens'],
                'dc': curr['decode_tokens'],
                'running': curr['running_depth'],
                'tokens': curr['scheduled_tokens'],
                't_wall_us': t_wall_us,
                't_sched_us': curr['duration_us'],
            })
    return pairs


# ─── Step 2: Fit β (train only) ─────────────────────────

def fit_beta_train(pairs: list[dict]) -> tuple[np.ndarray, int]:
    """Fit global β from TRAIN-only consecutive-pair wall-clock data."""
    train = [p for p in pairs if p['split'] == 'train']
    X = np.array([beta_features(p['pf'], p['dc'], p['model']) for p in train])
    y = np.array([p['t_wall_us'] for p in train])
    beta, _ = nnls(X, y)
    return beta, len(train)


# ─── Step 3: Evaluate step-level accuracy ────────────────

def eval_step_accuracy(pairs: list[dict], beta: np.ndarray) -> dict:
    """Compute step-level MAPE by model and split."""
    results = defaultdict(lambda: {'pred': [], 'real': []})

    for p in pairs:
        features = beta_features(p['pf'], p['dc'], p['model'])
        pred = float(np.dot(beta, features))
        real = p['t_wall_us']
        key = (p['model'], p['split'])
        results[key]['pred'].append(pred)
        results[key]['real'].append(real)

    out = {}
    for (model, split), data in results.items():
        pred = np.array(data['pred'])
        real = np.array(data['real'])
        mape = np.mean(np.abs(pred - real) / np.maximum(real, 1)) * 100
        bias = np.mean((pred - real) / np.maximum(real, 1)) * 100
        rmse = np.sqrt(np.mean((pred - real) ** 2))
        out[(model, split)] = {
            'n': len(pred),
            'mape': mape,
            'bias': bias,
            'rmse': rmse,
            'pred_median': np.median(pred),
            'real_median': np.median(real),
        }
    return out


# ─── Step 4: Fit α from journey residuals ────────────────

def fit_alpha(journeys: list[dict], beta: np.ndarray, pipeline_factor: float) -> tuple[np.ndarray, dict]:
    """Fit α from TRAIN journey TTFT residuals.

    TTFT_pred = α₀ + α₁·input_tokens + pipeline_factor × β·features(prefill_step)

    For single-step prefills, features are estimated from prefill_total tokens
    and an average decode batch size.
    """
    train = [j for j in journeys
             if j['split'] == 'train'
             and j['num_preemptions'] == 0
             and j['first_token_step'] == j['scheduled_step']]  # single-step

    # We need to estimate the step features at the prefill step.
    # We don't have the exact batch composition, but we know:
    # - prefill_tokens for THIS request
    # - decode_tokens from co-batched requests (unknown)
    # We'll use a simplified approach: assume the step features are
    # dominated by THIS request's prefill tokens + some average decode load.
    # This is a known limitation — the exact step composition is in the
    # step trace data, but joining by step_id is sparse (10% sample).

    # For now, estimate step features using just this request's prefill:
    # β·features(pf=prefill_total, dc=0, model) as a lower bound
    # Then α absorbs the difference including co-batched decode tokens.

    X_alpha = []
    y_alpha = []

    for j in train:
        real_ttft_us = (j['first_token_ns'] - j['scheduled_ns']) / 1000
        if real_ttft_us <= 0:
            continue

        # Estimate GPU step time for this request's prefill
        pf = j['prefill_total']
        model = j['model']
        # Use pf tokens as cache-miss (conservative), 0 decode
        features = beta_features(pf, 0, model)
        gpu_time = pipeline_factor * float(np.dot(beta, features))

        residual = real_ttft_us - gpu_time

        X_alpha.append([1.0, pf])
        y_alpha.append(residual)

    X = np.array(X_alpha)
    y = np.array(y_alpha)
    alpha, _ = nnls(X, y)

    # Also compute per-model α for comparison
    per_model = {}
    for model in MODEL_CONFIGS:
        mask = [i for i, j in enumerate(train) if j['model'] == model]
        if len(mask) < 10:
            continue
        Xm = X[mask]
        ym = y[mask]
        am, _ = nnls(Xm, ym)
        per_model[model] = am

    return alpha, per_model


# ─── Step 5: Evaluate TTFT/E2E accuracy ─────────────────

def eval_journey_accuracy(
    journeys: list[dict],
    beta: np.ndarray,
    alpha: np.ndarray,
    pipeline_factor: float,
) -> dict:
    """Compute TTFT and E2E MAPE by model and split."""
    results = defaultdict(lambda: {
        'ttft_pred': [], 'ttft_real': [],
        'e2e_pred': [], 'e2e_real': [],
    })

    for j in journeys:
        if j['num_preemptions'] > 0:
            continue

        model = j['model']
        split = j['split']
        pf = j['prefill_total']
        dc = j['decode_max']

        # Real values
        real_ttft_us = (j['first_token_ns'] - j['scheduled_ns']) / 1000
        real_e2e_us = (j['finished_ns'] - j['scheduled_ns']) / 1000

        if real_ttft_us <= 0 or real_e2e_us <= 0:
            continue

        # Predict TTFT: α + pipeline_factor × β·features(prefill)
        pf_features = beta_features(pf, 0, model)
        ttft_pred = alpha[0] + alpha[1] * pf + pipeline_factor * float(np.dot(beta, pf_features))

        # Predict E2E: TTFT + decode_steps × step_time
        # Each decode step processes ~1 token per request in the batch
        # For simplicity, use dc decode steps with 1 decode token each
        dc_features = beta_features(0, 1, model)
        step_time_decode = float(np.dot(beta, dc_features))
        e2e_pred = ttft_pred + dc * step_time_decode

        key = (model, split)
        results[key]['ttft_pred'].append(ttft_pred)
        results[key]['ttft_real'].append(real_ttft_us)
        results[key]['e2e_pred'].append(e2e_pred)
        results[key]['e2e_real'].append(real_e2e_us)

    out = {}
    for (model, split), data in results.items():
        tp = np.array(data['ttft_pred'])
        tr = np.array(data['ttft_real'])
        ep = np.array(data['e2e_pred'])
        er = np.array(data['e2e_real'])

        ttft_mape = np.mean(np.abs(tp - tr) / np.maximum(tr, 1)) * 100
        ttft_bias = np.mean((tp - tr) / np.maximum(tr, 1)) * 100
        e2e_mape = np.mean(np.abs(ep - er) / np.maximum(er, 1)) * 100
        e2e_bias = np.mean((ep - er) / np.maximum(er, 1)) * 100

        out[(model, split)] = {
            'n': len(tp),
            'ttft_mape': ttft_mape,
            'ttft_bias': ttft_bias,
            'e2e_mape': e2e_mape,
            'e2e_bias': e2e_bias,
        }
    return out


# ─── Main ────────────────────────────────────────────────

def main():
    data_dir = find_data_dir()

    print("=" * 72)
    print("Iter 5 — Cross-Model Wall-Clock β Evaluation")
    print("=" * 72)

    # Load data
    print("\n[1] Loading data...")
    steps = load_steps(os.path.join(data_dir, 'iter0_steps.csv'))
    journeys = load_journeys(os.path.join(data_dir, 'iter3_journeys.csv'))
    ground_truth = load_ground_truth(data_dir)
    print(f"    {len(steps):,} steps, {len(journeys):,} journeys, {len(ground_truth)} ground truth files")

    # Build consecutive pairs
    print("\n[2] Building consecutive-pair wall-clock data...")
    pairs = build_consecutive_pairs(steps)
    train_pairs = [p for p in pairs if p['split'] == 'train']
    test_pairs = [p for p in pairs if p['split'] == 'test']
    val_pairs = [p for p in pairs if p['split'] == 'validate']
    print(f"    {len(pairs):,} total pairs (train={len(train_pairs)}, test={len(test_pairs)}, val={len(val_pairs)})")

    # Fit β from train only
    print("\n[3] Fitting cross-model β from TRAIN wall-clock data...")
    beta_wall, n_train = fit_beta_train(pairs)
    print(f"    N={n_train} training pairs")
    print(f"    β = [{', '.join(f'{b:.2f}' for b in beta_wall)}]")
    print(f"\n    Comparison:")
    print(f"    {'Feature':<20} {'Iter5(wall)':>12} {'Iter3(A+B)':>12} {'Ratio':>8}")
    print(f"    {'-'*54}")
    labels = ['β₀ (L)', 'β₁ (KV bw)', 'β₂ (MoE)', 'β₃ (TP)']
    for i, label in enumerate(labels):
        r = beta_wall[i] / ITER3_BETA[i] if ITER3_BETA[i] > 0 else float('inf')
        print(f"    {label:<20} {beta_wall[i]:>12.1f} {ITER3_BETA[i]:>12.1f} {r:>8.2f}")

    # Evaluate step-level accuracy
    print("\n" + "=" * 72)
    print("STEP-LEVEL ACCURACY (wall-clock prediction)")
    print("=" * 72)

    step_acc = eval_step_accuracy(pairs, beta_wall)
    step_acc_i3 = eval_step_accuracy(pairs, ITER3_BETA)

    print(f"\n  {'Model':<16} {'Split':<10} {'N':>6} {'MAPE%':>8} {'Bias%':>8} {'RMSE(µs)':>10}   | Iter3 MAPE%")
    print(f"  {'-'*88}")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        for split in ['train', 'validate', 'test']:
            key = (model, split)
            d = step_acc.get(key)
            d3 = step_acc_i3.get(key)
            if not d:
                continue
            i3_mape = f"{d3['mape']:.1f}" if d3 else "—"
            print(f"  {model:<16} {split:<10} {d['n']:>6} {d['mape']:>7.1f}% {d['bias']:>7.1f}% {d['rmse']:>10.0f}   | {i3_mape}%")

    # Fit α with 2.0x pipeline factor
    PIPELINE_FACTOR = 2.0
    print(f"\n" + "=" * 72)
    print(f"FITTING α (pipeline factor = {PIPELINE_FACTOR})")
    print("=" * 72)

    alpha, alpha_per_model = fit_alpha(journeys, beta_wall, PIPELINE_FACTOR)
    print(f"\n  Global α: α₀={alpha[0]:.0f}µs, α₁={alpha[1]:.2f}µs/tok")
    print(f"  Iter 3 α: α₀={ITER3_ALPHA[0]:.0f}µs, α₁={ITER3_ALPHA[1]:.2f}µs/tok")
    print(f"\n  Per-model α (for comparison):")
    for model, am in sorted(alpha_per_model.items()):
        print(f"    {model}: α₀={am[0]:.0f}µs, α₁={am[1]:.2f}µs/tok")

    # Evaluate journey-level TTFT/E2E
    print(f"\n" + "=" * 72)
    print("JOURNEY-LEVEL ACCURACY (TTFT / E2E)")
    print("=" * 72)

    j_acc = eval_journey_accuracy(journeys, beta_wall, alpha, PIPELINE_FACTOR)

    # Also eval with Iter 3 coefficients for comparison
    j_acc_i3 = eval_journey_accuracy(journeys, ITER3_BETA, ITER3_ALPHA, 1.0)

    print(f"\n  --- Iter 5 (wall-clock β, pipeline=2.0) ---")
    print(f"  {'Model':<16} {'Split':<10} {'N':>6} {'TTFT MAPE%':>12} {'TTFT bias%':>12} {'E2E MAPE%':>11} {'E2E bias%':>11}")
    print(f"  {'-'*82}")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        for split in ['train', 'validate', 'test']:
            key = (model, split)
            d = j_acc.get(key)
            if not d:
                continue
            print(f"  {model:<16} {split:<10} {d['n']:>6} {d['ttft_mape']:>11.1f}% {d['ttft_bias']:>11.1f}% {d['e2e_mape']:>10.1f}% {d['e2e_bias']:>10.1f}%")

    print(f"\n  --- Iter 3 (Block A+B β, no pipeline factor) ---")
    print(f"  {'Model':<16} {'Split':<10} {'N':>6} {'TTFT MAPE%':>12} {'TTFT bias%':>12} {'E2E MAPE%':>11} {'E2E bias%':>11}")
    print(f"  {'-'*82}")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        for split in ['train', 'validate', 'test']:
            key = (model, split)
            d = j_acc_i3.get(key)
            if not d:
                continue
            print(f"  {model:<16} {split:<10} {d['n']:>6} {d['ttft_mape']:>11.1f}% {d['ttft_bias']:>11.1f}% {d['e2e_mape']:>10.1f}% {d['e2e_bias']:>10.1f}%")

    # Ground truth comparison
    print(f"\n" + "=" * 72)
    print("GROUND TRUTH COMPARISON (aggregate TTFT/E2E by experiment)")
    print("=" * 72)

    # Aggregate journey predictions by experiment
    exp_predictions = defaultdict(lambda: {'ttft': [], 'e2e': []})
    for j in journeys:
        if j['num_preemptions'] > 0:
            continue
        model = j['model']
        pf = j['prefill_total']
        dc = j['decode_max']

        real_ttft_us = (j['first_token_ns'] - j['scheduled_ns']) / 1000
        real_e2e_us = (j['finished_ns'] - j['scheduled_ns']) / 1000
        if real_ttft_us <= 0 or real_e2e_us <= 0:
            continue

        pf_features = beta_features(pf, 0, model)
        ttft_pred = alpha[0] + alpha[1] * pf + PIPELINE_FACTOR * float(np.dot(beta_wall, pf_features))
        dc_features = beta_features(0, 1, model)
        step_decode = float(np.dot(beta_wall, dc_features))
        e2e_pred = ttft_pred + dc * step_decode

        exp_predictions[j['experiment']]['ttft'].append(ttft_pred / 1000)  # ms
        exp_predictions[j['experiment']]['e2e'].append(e2e_pred / 1000)    # ms

    print(f"\n  {'Experiment':<55} {'Split':<6} {'GT TTFT':>8} {'Pred':>8} {'Err%':>7} | {'GT E2E':>8} {'Pred':>8} {'Err%':>7}")
    print(f"  {'-'*120}")
    for exp, gt_data in sorted(ground_truth.items()):
        pred = exp_predictions.get(exp)
        if not pred or not pred['ttft']:
            continue
        gt_ttft = gt_data['ttft']['mean_ms']
        gt_e2e = gt_data['e2e']['mean_ms']
        pred_ttft = np.mean(pred['ttft'])
        pred_e2e = np.mean(pred['e2e'])
        ttft_err = (pred_ttft - gt_ttft) / gt_ttft * 100
        e2e_err = (pred_e2e - gt_e2e) / gt_e2e * 100
        split = gt_data['split']
        print(f"  {exp:<55} {split:<6} {gt_ttft:>7.1f}ms {pred_ttft:>7.1f}ms {ttft_err:>+6.1f}% | {gt_e2e:>7.0f}ms {pred_e2e:>7.0f}ms {e2e_err:>+6.1f}%")

    # Output coefficients as JSON
    print(f"\n" + "=" * 72)
    print("FITTED COEFFICIENTS (JSON)")
    print("=" * 72)
    coeffs = {
        'beta': beta_wall.tolist(),
        'beta_labels': ['L (per-layer)', 'dc*kv_dim*1e-6 (KV bandwidth)',
                        '(pf+dc)*I(MoE)', 'I(TP>1)'],
        'alpha': alpha.tolist(),
        'alpha_labels': ['α₀ (fixed overhead)', 'α₁ (per-input-token)'],
        'pipeline_factor': PIPELINE_FACTOR,
        'note': 'TTFT = α₀ + α₁*input_tokens + pipeline_factor * β·features(pf, 0)',
    }
    print(json.dumps(coeffs, indent=2))


if __name__ == '__main__':
    main()
