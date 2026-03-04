"""
Iter 5 — BLIS Replay Comparison

Runs the BLIS replay binary with multiple coefficient configurations and
compares against real vLLM ground truth.

Configurations tested:
  A: Iter 3 baseline (β_iter3, α_iter3, δ=0)
  B: Wall-clock β + no overhead (β_wall, α=0, δ=0)
  C: Wall-clock β + per-model median δ
  D: Wall-clock β + global median δ

The key question: does the wall-clock β + appropriate δ produce better
BLIS simulation accuracy than Iter 3, especially for TTFT?

Usage:
    python3 training/iter5_replay.py
"""

from __future__ import annotations

import csv
import json
import glob
import os
import subprocess
import sys
import numpy as np
from collections import defaultdict
from scipy.optimize import nnls


# ─── Model configs ───────────────────────────────────────

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}


def beta_features(pf, dc, model):
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0
    return np.array([L, dc * L * kv_dim * 1e-6, (pf + dc) * is_moe, is_tp])


def find_paths():
    """Locate data directory and replay binary."""
    data_dir = '/Users/sri/Documents/Projects/inference-sim/training'
    worktree = '/Users/sri/Documents/Projects/inference-sim/.worktrees/iter5-decomposed'
    replay_bin = os.path.join(worktree, 'replay')
    if not os.path.exists(replay_bin):
        raise FileNotFoundError(f"Replay binary not found at {replay_bin}")
    return data_dir, replay_bin


def fit_wall_clock_beta(data_dir):
    """Fit β from TRAIN consecutive-pair wall-clock data."""
    by_exp = defaultdict(list)
    with open(os.path.join(data_dir, 'iter0_steps.csv')) as f:
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
            pairs.append({
                'model': curr['model_short'],
                'pf': int(curr['prefill_tokens']),
                'dc': int(curr['decode_tokens']),
                't_wall': (int(nxt['ts_start_ns']) - int(curr['ts_start_ns'])) / 1000,
            })

    X = np.array([beta_features(p['pf'], p['dc'], p['model']) for p in pairs])
    y = np.array([p['t_wall'] for p in pairs])
    beta, _ = nnls(X, y)

    # Compute per-model median overhead (residual)
    overheads = {}
    for model in MODEL_CONFIGS:
        mp = [p for p in pairs if p['model'] == model]
        if not mp:
            continue
        resids = []
        for p in mp:
            feat = beta_features(p['pf'], p['dc'], model)
            pred = float(np.dot(beta, feat))
            resids.append(p['t_wall'] - pred)
        overheads[model] = {
            'median': float(np.median(resids)),
            'mean': float(np.mean(resids)),
            'p25': float(np.percentile(resids, 25)),
            'p75': float(np.percentile(resids, 75)),
        }

    # Compute per-model median wall clock
    wall_clocks = {}
    for model in MODEL_CONFIGS:
        mp = [p for p in pairs if p['model'] == model]
        if mp:
            walls = [p['t_wall'] for p in mp]
            wall_clocks[model] = float(np.median(walls))

    return beta, overheads, wall_clocks


def run_replay(replay_bin, input_path, beta, alpha, delta=None):
    """Run the BLIS replay binary and return parsed output."""
    cmd = [
        replay_bin,
        '--input', input_path,
        '--beta', ','.join(f'{b:.4f}' for b in beta),
        '--alpha', ','.join(f'{a:.4f}' for a in alpha),
    ]
    if delta is not None:
        cmd.extend(['--delta', ','.join(f'{d:.4f}' for d in delta)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return None, result.stderr

    try:
        output = json.loads(result.stdout)
        return output, result.stderr
    except json.JSONDecodeError:
        return None, f"JSON parse error: {result.stdout[:200]}"


def main():
    data_dir, replay_bin = find_paths()

    print("=" * 80)
    print("Iter 5 — BLIS Replay Comparison")
    print("=" * 80)

    # Fit wall-clock β
    print("\n[1] Fitting wall-clock β from train data...")
    beta_wall, overheads, wall_clocks = fit_wall_clock_beta(data_dir)
    print(f"    β_wall = [{', '.join(f'{b:.2f}' for b in beta_wall)}]")
    print(f"\n    Per-model overhead (wall - β_pred):")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        oh = overheads.get(model, {})
        wc = wall_clocks.get(model, 0)
        print(f"      {model}: median_residual={oh.get('median', 0):.0f}µs, median_wall={wc:.0f}µs")

    # Compute global median overhead across all models
    all_overheads = [overheads[m]['median'] for m in overheads]
    global_delta = float(np.median(all_overheads))
    print(f"\n    Global median δ: {global_delta:.0f}µs")

    # Define configurations to test
    ITER3_BETA = [116.110, 1226.868, 19.943, 9445.157]
    ITER3_ALPHA = [13732.0, 0.0, 860.6]

    configs = {
        'A_iter3': {
            'label': 'Iter 3 (baseline)',
            'beta': ITER3_BETA,
            'alpha': ITER3_ALPHA,
            'delta': None,
        },
        'B_wall_nodelta': {
            'label': 'Wall-clock β, no δ',
            'beta': beta_wall.tolist(),
            'alpha': [0.0, 0.0, 0.0],
            'delta': None,
        },
        'C_wall_walldelta': {
            'label': 'Wall-clock β, δ=median_wall (pipeline factor)',
            'beta': beta_wall.tolist(),
            'alpha': [0.0, 0.0, 0.0],
            'delta': None,  # set per-experiment below
        },
        'D_wall_globaldelta': {
            'label': f'Wall-clock β, δ={global_delta:.0f}µs (global)',
            'beta': beta_wall.tolist(),
            'alpha': [0.0, 0.0, 0.0],
            'delta': [global_delta],
        },
    }

    # Get replay data files (non-reasoning, non-saturated experiments)
    replay_files = sorted(glob.glob(os.path.join(data_dir, 'replay_data', '*.json')))
    replay_files = [f for f in replay_files if '_ground_truth' not in f]

    # Load ground truth
    ground_truth = {}
    for f in glob.glob(os.path.join(data_dir, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        ground_truth[d['experiment']] = d

    # Run replays
    print(f"\n[2] Running BLIS replays ({len(replay_files)} experiments × {len(configs)} configs)...")

    results = defaultdict(dict)  # results[config_name][experiment] = output

    for rf in replay_files:
        with open(rf) as fh:
            meta = json.load(fh)
        exp = meta['experiment']
        model = meta['model_short']
        profile = meta['profile']
        split = meta['split']

        # Skip reasoning experiments (saturated, >85% failure)
        if profile == 'reasoning':
            continue

        gt = ground_truth.get(exp)
        if not gt:
            continue

        for config_name, cfg in configs.items():
            beta = cfg['beta']
            alpha = cfg['alpha']
            delta = cfg['delta']

            # Config C: per-model δ = median wall clock
            if config_name == 'C_wall_walldelta':
                wc = wall_clocks.get(model, 8000)
                delta = [wc]

            sys.stderr.write(f"  {config_name}: {exp}...")
            sys.stderr.flush()
            output, stderr = run_replay(replay_bin, rf, beta, alpha, delta)

            if output is None:
                sys.stderr.write(f" FAILED: {stderr[:100]}\n")
                continue

            results[config_name][exp] = {
                'output': output,
                'model': model,
                'profile': profile,
                'split': split,
                'gt': gt,
            }
            sys.stderr.write(f" done (TTFT={output['summary']['ttft_mean_ms']:.1f}ms)\n")

    # ─── Results comparison ──────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS: BLIS Replay vs Real vLLM")
    print("=" * 80)

    # Table header
    print(f"\n{'Config':<35} {'Exp':<45} {'Split':<6} "
          f"{'GT TTFT':>8} {'Sim TTFT':>9} {'Err%':>7} | "
          f"{'GT E2E':>8} {'Sim E2E':>9} {'Err%':>7}")
    print("-" * 145)

    for config_name in ['A_iter3', 'B_wall_nodelta', 'C_wall_walldelta', 'D_wall_globaldelta']:
        cfg = configs[config_name]
        label = cfg['label']
        exps = results.get(config_name, {})

        ttft_errors = []
        e2e_errors = []

        for exp in sorted(exps.keys()):
            data = exps[exp]
            gt = data['gt']
            out = data['output']
            split = data['split']

            gt_ttft = gt['ttft']['mean_ms']
            gt_e2e = gt['e2e']['mean_ms']
            sim_ttft = out['summary']['ttft_mean_ms']
            sim_e2e = out['summary']['e2e_mean_ms']

            ttft_err = (sim_ttft - gt_ttft) / gt_ttft * 100
            e2e_err = (sim_e2e - gt_e2e) / gt_e2e * 100

            ttft_errors.append(ttft_err)
            e2e_errors.append(e2e_err)

            exp_short = exp[:44]
            print(f"{label:<35} {exp_short:<45} {split:<6} "
                  f"{gt_ttft:>7.1f}ms {sim_ttft:>8.1f}ms {ttft_err:>+6.1f}% | "
                  f"{gt_e2e:>7.0f}ms {sim_e2e:>8.0f}ms {e2e_err:>+6.1f}%")

        if ttft_errors:
            abs_ttft = [abs(e) for e in ttft_errors]
            abs_e2e = [abs(e) for e in e2e_errors]
            print(f"{'  → MEAN ABS ERROR':<35} {'':45} {'':6} "
                  f"{'':>8} {'':>9} {np.mean(abs_ttft):>6.1f}% | "
                  f"{'':>8} {'':>9} {np.mean(abs_e2e):>6.1f}%")
            print(f"{'  → MEAN SIGNED ERROR':<35} {'':45} {'':6} "
                  f"{'':>8} {'':>9} {np.mean(ttft_errors):>+6.1f}% | "
                  f"{'':>8} {'':>9} {np.mean(e2e_errors):>+6.1f}%")
        print()

    # ─── Summary table ───────────────────────────────────
    print("=" * 80)
    print("SUMMARY: Mean Absolute Error by Config")
    print("=" * 80)
    print(f"\n{'Config':<45} {'TTFT MAE%':>10} {'E2E MAE%':>10} {'TTFT bias%':>11} {'E2E bias%':>10}")
    print("-" * 88)

    for config_name in ['A_iter3', 'B_wall_nodelta', 'C_wall_walldelta', 'D_wall_globaldelta']:
        cfg = configs[config_name]
        exps = results.get(config_name, {})
        if not exps:
            continue

        ttft_errs = []
        e2e_errs = []
        for data in exps.values():
            gt = data['gt']
            out = data['output']
            gt_ttft = gt['ttft']['mean_ms']
            gt_e2e = gt['e2e']['mean_ms']
            sim_ttft = out['summary']['ttft_mean_ms']
            sim_e2e = out['summary']['e2e_mean_ms']
            ttft_errs.append((sim_ttft - gt_ttft) / gt_ttft * 100)
            e2e_errs.append((sim_e2e - gt_e2e) / gt_e2e * 100)

        abs_ttft = np.mean([abs(e) for e in ttft_errs])
        abs_e2e = np.mean([abs(e) for e in e2e_errs])
        bias_ttft = np.mean(ttft_errs)
        bias_e2e = np.mean(e2e_errs)

        print(f"{cfg['label']:<45} {abs_ttft:>9.1f}% {abs_e2e:>9.1f}% {bias_ttft:>+10.1f}% {bias_e2e:>+9.1f}%")

    # Per-model breakdown
    print(f"\n{'--- Per-Model TTFT MAE% ---'}")
    models_seen = set()
    for exps in results.values():
        for data in exps.values():
            models_seen.add(data['model'])

    print(f"{'Config':<45}", end="")
    for model in sorted(models_seen):
        print(f" {model:>14}", end="")
    print()
    print("-" * (45 + 15 * len(models_seen)))

    for config_name in ['A_iter3', 'B_wall_nodelta', 'C_wall_walldelta', 'D_wall_globaldelta']:
        cfg = configs[config_name]
        exps = results.get(config_name, {})
        model_errs = defaultdict(list)
        for data in exps.values():
            gt_ttft = data['gt']['ttft']['mean_ms']
            sim_ttft = data['output']['summary']['ttft_mean_ms']
            err = abs(sim_ttft - gt_ttft) / gt_ttft * 100
            model_errs[data['model']].append(err)

        print(f"{cfg['label']:<45}", end="")
        for model in sorted(models_seen):
            errs = model_errs.get(model, [])
            if errs:
                print(f" {np.mean(errs):>13.1f}%", end="")
            else:
                print(f" {'—':>14}", end="")
        print()


if __name__ == '__main__':
    main()
