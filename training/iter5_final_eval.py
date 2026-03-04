#!/usr/bin/env python3
"""Iter 5 — Final evaluation of all three arms with optimized coefficients."""

import json
import glob
import os
import subprocess
import sys
import numpy as np
from collections import defaultdict

REPLAY_BIN = '/Users/sri/Documents/Projects/inference-sim/.worktrees/iter5-decomposed/replay'
DATA_DIR = '/Users/sri/Documents/Projects/inference-sim/training'

# Load optimized coefficients
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iter5_coefficients.json')) as f:
    COEFFS = json.load(f)

ARMS = {
    'H-control (Iter 3, δ=0)': {
        'beta': COEFFS['iter3_baseline']['beta'],
        'alpha': COEFFS['iter3_baseline']['alpha'],
        'delta': None,
    },
    'H-main (corrected β + α₀/δ₀)': {
        'beta': COEFFS['iter5_hmain']['beta'],
        'alpha': COEFFS['iter5_hmain']['alpha'],
        'delta': COEFFS['iter5_hmain']['delta'],
    },
    'H-ablation (Iter3 β + α₀/δ₀/δ₁)': {
        'beta': COEFFS['iter5_hablation']['beta'],
        'alpha': COEFFS['iter5_hablation']['alpha'],
        'delta': COEFFS['iter5_hablation']['delta'],
    },
}


def run_replay(input_path, beta, alpha, delta=None):
    cmd = [REPLAY_BIN, '--input', input_path,
           '--beta', ','.join(f'{b:.6f}' for b in beta),
           '--alpha', ','.join(f'{a:.6f}' for a in alpha)]
    if delta:
        cmd.extend(['--delta', ','.join(f'{d:.6f}' for d in delta)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def main():
    # Load ground truth
    gt = {}
    for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*_ground_truth.json')):
        with open(f) as fh:
            d = json.load(fh)
        gt[d['experiment']] = d

    # Get all replay files (skip ground truth)
    replay_files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, 'replay_data', '*.json'))
                          if '_ground_truth' not in f)

    # Run all arms on all experiments
    results = {}
    for arm_name, cfg in ARMS.items():
        arm_results = {}
        for rf in replay_files:
            exp = os.path.basename(rf).replace('.json', '')
            g = gt.get(exp)
            if not g:
                continue

            sys.stderr.write(f'  {arm_name[:20]:20s} {exp[:50]:50s}')
            sys.stderr.flush()
            output = run_replay(rf, cfg['beta'], cfg['alpha'], cfg['delta'])
            if output is None:
                sys.stderr.write(' FAIL\n')
                continue

            sim_ttft = output['summary']['ttft_mean_ms']
            sim_e2e = output['summary']['e2e_mean_ms']
            gt_ttft = g['ttft']['mean_ms']
            gt_e2e = g['e2e']['mean_ms']

            arm_results[exp] = {
                'model': g['model_short'],
                'profile': g['profile'],
                'split': g['split'],
                'gt_ttft': gt_ttft, 'sim_ttft': sim_ttft,
                'ttft_err': (sim_ttft - gt_ttft) / gt_ttft * 100,
                'gt_e2e': gt_e2e, 'sim_e2e': sim_e2e,
                'e2e_err': (sim_e2e - gt_e2e) / gt_e2e * 100,
                'completed': output['summary']['completed'],
            }
            sys.stderr.write(f' TTFT={sim_ttft:.1f}ms ({arm_results[exp]["ttft_err"]:+.1f}%)\n')

        results[arm_name] = arm_results

    # ─── Print results ───────────────────────────────────
    print("=" * 130)
    print("ITER 5 — FULL EVALUATION (cross-model coefficients)")
    print("=" * 130)

    for arm_name in ARMS:
        cfg = ARMS[arm_name]
        ar = results[arm_name]
        print(f"\n{'─'*130}")
        print(f"  {arm_name}")
        print(f"  β=[{', '.join(f'{b:.1f}' for b in cfg['beta'])}]  α=[{', '.join(f'{a:.0f}' for a in cfg['alpha'])}]  δ={cfg['delta']}")
        print(f"{'─'*130}")

        for label, filt in [('TRAIN (non-reasoning)', lambda v: v['split']=='train' and v['profile']!='reasoning'),
                             ('VALIDATE (non-reasoning)', lambda v: v['split']=='validate' and v['profile']!='reasoning'),
                             ('TEST (reasoning)', lambda v: v['profile']=='reasoning')]:
            subset = {k: v for k, v in ar.items() if filt(v)}
            if not subset:
                continue
            ttft_mae = np.mean([abs(v['ttft_err']) for v in subset.values()])
            e2e_mae = np.mean([abs(v['e2e_err']) for v in subset.values()])
            ttft_bias = np.mean([v['ttft_err'] for v in subset.values()])
            print(f"\n  {label}: TTFT MAE={ttft_mae:.1f}% (bias={ttft_bias:+.1f}%), E2E MAE={e2e_mae:.1f}%")

            print(f"  {'Experiment':<58} {'GT TTFT':>8} {'Sim':>9} {'Err':>7} | {'GT E2E':>8} {'Sim':>9} {'Err':>7}")
            for exp in sorted(subset):
                v = subset[exp]
                print(f"  {exp:<58} {v['gt_ttft']:>7.1f}ms {v['sim_ttft']:>8.1f}ms {v['ttft_err']:>+6.1f}% | "
                      f"{v['gt_e2e']:>7.0f}ms {v['sim_e2e']:>8.0f}ms {v['e2e_err']:>+6.1f}%")

    # ─── Summary comparison ──────────────────────────────
    print(f"\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    print(f"\n{'Arm':<40} {'Train TTFT':>11} {'Train E2E':>10} {'Val TTFT':>10} {'Test TTFT':>11}")
    print("-" * 85)
    for arm_name in ARMS:
        ar = results[arm_name]
        train = [v for v in ar.values() if v['split']=='train' and v['profile']!='reasoning']
        val = [v for v in ar.values() if v['split']=='validate' and v['profile']!='reasoning']
        test = [v for v in ar.values() if v['profile']=='reasoning']
        t = f"{np.mean([abs(v['ttft_err']) for v in train]):.1f}%" if train else "—"
        te = f"{np.mean([abs(v['e2e_err']) for v in train]):.1f}%" if train else "—"
        vl = f"{np.mean([abs(v['ttft_err']) for v in val]):.1f}%" if val else "—"
        tt = f"{np.mean([abs(v['ttft_err']) for v in test]):.1f}%" if test else "—"
        print(f"{arm_name:<40} {t:>11} {te:>10} {vl:>10} {tt:>11}")

    # Per-model
    print(f"\nPer-Model Train TTFT MAE:")
    models = ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']
    print(f"{'Arm':<40}", end="")
    for m in models:
        print(f" {m:>14}", end="")
    print()
    print("-" * (40 + 15*4))
    for arm_name in ARMS:
        ar = results[arm_name]
        print(f"{arm_name:<40}", end="")
        for m in models:
            errs = [abs(v['ttft_err']) for v in ar.values()
                    if v['model']==m and v['split']=='train' and v['profile']!='reasoning']
            print(f" {np.mean(errs):>13.1f}%" if errs else f" {'—':>14}", end="")
        print()

    # Answer the cross-model question
    print(f"\n{'='*130}")
    print("CROSS-MODEL COEFFICIENTS")
    print(f"{'='*130}")
    hm = COEFFS['iter5_hmain']
    print(f"\nH-main (corrected β, 6 global parameters):")
    print(f"  β = [{', '.join(f'{b:.2f}' for b in hm['beta'])}]")
    print(f"  α₀ = {hm['alpha'][0]:.0f} µs")
    print(f"  δ₀ = {hm['delta'][0]:.0f} µs")
    print(f"\nThese are TRULY cross-model: same 6 coefficients for all 4 architectures.")
    print(f"No per-model fitting. Architecture features in β drive the differentiation.")


if __name__ == '__main__':
    main()
