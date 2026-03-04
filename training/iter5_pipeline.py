"""
Iter 5 — Pipeline Factor Investigation

The prototype found a universal 2.0x ratio between journey-observed prefill
time and step-to-step wall clock. But when used for TTFT prediction, it
catastrophically overshoots mixtral (+342%) while undershooting llama-2-7b.

This script investigates:
1. What the 2.0x ratio actually measures (timestamp alignment)
2. Whether it varies with batch composition, not just model
3. The correct way to convert step-level β to journey-level TTFT
4. Why the analytical TTFT formula breaks

Usage:
    python3 training/iter5_pipeline.py
"""

from __future__ import annotations

import csv
import os
import numpy as np
from collections import defaultdict


# ─── Model configs ───────────────────────────────────────

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}


def beta_features(pf: int, dc: int, model: str) -> np.ndarray:
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0
    return np.array([L, dc * L * kv_dim * 1e-6, (pf + dc) * is_moe, is_tp])


def find_data_dir() -> str:
    for c in ['/Users/sri/Documents/Projects/inference-sim/training', 'training']:
        if os.path.exists(os.path.join(c, 'iter0_steps.csv')):
            return c
    raise FileNotFoundError("Cannot find training data directory")


def main():
    data_dir = find_data_dir()

    print("=" * 72)
    print("Iter 5 — Pipeline Factor Investigation")
    print("=" * 72)

    # ─── Load step data ──────────────────────────────────
    steps_by_exp = defaultdict(dict)
    with open(os.path.join(data_dir, 'iter0_steps.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps_by_exp[row['experiment']][int(row['step_id'])] = {
                'ts_start': int(row['ts_start_ns']),
                'ts_end': int(row['ts_end_ns']),
                'dur': int(row['duration_us']),
                'pf': int(row['prefill_tokens']),
                'dc': int(row['decode_tokens']),
                'running': int(row['running_depth']),
                'tokens': int(row['scheduled_tokens']),
                'model': row['model_short'],
            }

    # ─── Build consecutive-pair wall clock ────────────────
    wall_by_step = {}
    for exp, exp_steps in steps_by_exp.items():
        sorted_ids = sorted(exp_steps.keys())
        for i in range(len(sorted_ids) - 1):
            sid = sorted_ids[i]
            nid = sorted_ids[i + 1]
            if nid - sid == 1:
                t = (exp_steps[nid]['ts_start'] - exp_steps[sid]['ts_start']) / 1000
                wall_by_step[(exp, sid)] = t

    # ─── Load journey data ────────────────────────────────
    journeys = []
    with open(os.path.join(data_dir, 'iter3_journeys.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['num_preemptions']) > 0:
                continue
            journeys.append({
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
            })

    # =================================================================
    # INVESTIGATION 1: Timestamp alignment
    # For matched journey-step pairs, where does scheduled_ns fall
    # relative to ts_start and ts_end of the step?
    # =================================================================
    print("\n" + "=" * 72)
    print("INVESTIGATION 1: Timestamp Alignment")
    print("=" * 72)
    print("For single-step prefills matched to sampled steps:")
    print("  offset_sched = scheduled_ns - ts_start[S] (how far into schedule())")
    print("  offset_ft    = first_token_ns - ts_end[S]  (how far after schedule())")
    print("  sched_dur    = ts_end[S] - ts_start[S]     (scheduler CPU time)")

    align_data = defaultdict(lambda: {
        'offset_sched': [], 'offset_ft': [], 'sched_dur': [],
        'ft_minus_start': [], 'wall': [],
    })

    for j in journeys:
        ss = j['scheduled_step']
        fs = j['first_token_step']
        if fs != ss:
            continue  # single-step only

        exp = j['experiment']
        step_data = steps_by_exp.get(exp, {}).get(ss)
        wall = wall_by_step.get((exp, ss))
        if step_data is None:
            continue

        model = j['model']
        ts_start = step_data['ts_start']
        ts_end = step_data['ts_end']

        offset_sched_us = (j['scheduled_ns'] - ts_start) / 1000
        offset_ft_us = (j['first_token_ns'] - ts_end) / 1000
        sched_dur_us = (ts_end - ts_start) / 1000
        ft_from_start_us = (j['first_token_ns'] - ts_start) / 1000

        align_data[model]['offset_sched'].append(offset_sched_us)
        align_data[model]['offset_ft'].append(offset_ft_us)
        align_data[model]['sched_dur'].append(sched_dur_us)
        align_data[model]['ft_minus_start'].append(ft_from_start_us)
        if wall is not None:
            align_data[model]['wall'].append(wall)

    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        d = align_data.get(model)
        if not d or not d['offset_sched']:
            continue
        os_arr = np.array(d['offset_sched'])
        oft_arr = np.array(d['offset_ft'])
        sd_arr = np.array(d['sched_dur'])
        ft_arr = np.array(d['ft_minus_start'])

        print(f"\n  {model} (N={len(os_arr)}):")
        print(f"    offset_sched (scheduled_ns - ts_start):  median={np.median(os_arr):>10.0f}µs")
        print(f"    sched_dur    (ts_end - ts_start):        median={np.median(sd_arr):>10.0f}µs")
        print(f"    offset_ft    (first_token_ns - ts_end):  median={np.median(oft_arr):>10.0f}µs")
        print(f"    ft_from_start (first_token_ns - ts_start): median={np.median(ft_arr):>10.0f}µs")
        if d['wall']:
            wall_arr = np.array(d['wall'])
            print(f"    step_wall    (ts_start[S+1] - ts_start[S]): median={np.median(wall_arr):>10.0f}µs")
            # The journey real_ttft is ft_from_start minus offset_sched
            real_ttft = ft_arr[:len(os_arr)] - os_arr
            print(f"    journey_ttft (ft_from_start - offset_sched): median={np.median(real_ttft):>10.0f}µs")
            print(f"    ratio (journey_ttft / step_wall):            median={np.median(real_ttft[:len(wall_arr)] / wall_arr):>10.3f}")

    # =================================================================
    # INVESTIGATION 2: Where does first_token_ns fall in the timeline?
    # Is it BEFORE or AFTER ts_start[S+1]?
    # =================================================================
    print("\n" + "=" * 72)
    print("INVESTIGATION 2: Does first_token_ns fall before or after ts_start[S+1]?")
    print("=" * 72)

    ft_position = defaultdict(lambda: {'before': 0, 'after': 0, 'overshoot': []})

    for j in journeys:
        ss = j['scheduled_step']
        fs = j['first_token_step']
        if fs != ss:
            continue

        exp = j['experiment']
        step_curr = steps_by_exp.get(exp, {}).get(ss)
        step_next = steps_by_exp.get(exp, {}).get(ss + 1)
        if step_curr is None or step_next is None:
            continue

        model = j['model']
        ft_ns = j['first_token_ns']
        next_start_ns = step_next['ts_start']

        if ft_ns < next_start_ns:
            ft_position[model]['before'] += 1
        else:
            ft_position[model]['after'] += 1
            overshoot_us = (ft_ns - next_start_ns) / 1000
            ft_position[model]['overshoot'].append(overshoot_us)

    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        d = ft_position.get(model)
        if not d:
            continue
        total = d['before'] + d['after']
        print(f"\n  {model}: {total} matched (step S AND S+1 both sampled)")
        print(f"    first_token BEFORE ts_start[S+1]: {d['before']} ({d['before']*100/total:.1f}%)")
        print(f"    first_token AFTER  ts_start[S+1]: {d['after']} ({d['after']*100/total:.1f}%)")
        if d['overshoot']:
            ov = np.array(d['overshoot'])
            print(f"    overshoot median: {np.median(ov):.0f}µs, mean: {np.mean(ov):.0f}µs")

    # =================================================================
    # INVESTIGATION 3: Is the 2.0x a batch-feature mismatch?
    # Compare journey real_ttft with β·features(BATCH), not β·features(request)
    # =================================================================
    print("\n" + "=" * 72)
    print("INVESTIGATION 3: Journey TTFT vs β·features(full batch)")
    print("=" * 72)
    print("The TTFT formula used β·features(pf=request_pf, dc=0)")
    print("But β was trained on β·features(pf=batch_pf, dc=batch_dc)")
    print("Let's compare journey TTFT directly with the matched step's features.")

    from scipy.optimize import nnls

    # Fit β from train consecutive pairs (same as iter5_eval.py)
    train_X, train_y = [], []
    for exp, exp_steps in steps_by_exp.items():
        sorted_ids = sorted(exp_steps.keys())
        for i in range(len(sorted_ids) - 1):
            sid = sorted_ids[i]
            nid = sorted_ids[i + 1]
            if nid - sid != 1:
                continue
            s = exp_steps[sid]
            # Only train split - check via model+profile in experiment name
            # (simplified: use all for now, the point is the structural analysis)
            features = beta_features(s['pf'], s['dc'], s['model'])
            wall_us = (exp_steps[nid]['ts_start'] - s['ts_start']) / 1000
            train_X.append(features)
            train_y.append(wall_us)

    X = np.array(train_X)
    y = np.array(train_y)
    beta_wall, _ = nnls(X, y)
    print(f"\n  Wall-clock β: [{', '.join(f'{b:.1f}' for b in beta_wall)}]")

    # For matched journey-step pairs, compute:
    # (a) β·features(batch) = what β predicts for the FULL step
    # (b) journey real_ttft
    # (c) the ratio real_ttft / β·features(batch)
    batch_ratios = defaultdict(list)
    batch_details = defaultdict(lambda: {
        'real_ttft': [], 'beta_batch': [], 'wall': [],
        'pf_batch': [], 'dc_batch': [], 'pf_req': [], 'running': [],
    })

    for j in journeys:
        ss = j['scheduled_step']
        fs = j['first_token_step']
        if fs != ss:
            continue

        exp = j['experiment']
        step_data = steps_by_exp.get(exp, {}).get(ss)
        wall = wall_by_step.get((exp, ss))
        if step_data is None or wall is None:
            continue

        model = j['model']
        real_ttft_us = (j['first_token_ns'] - j['scheduled_ns']) / 1000
        if real_ttft_us <= 0:
            continue

        # β prediction using FULL BATCH features
        features_batch = beta_features(step_data['pf'], step_data['dc'], model)
        beta_pred_batch = float(np.dot(beta_wall, features_batch))

        if beta_pred_batch > 0:
            ratio = real_ttft_us / beta_pred_batch
            batch_ratios[model].append(ratio)

        batch_details[model]['real_ttft'].append(real_ttft_us)
        batch_details[model]['beta_batch'].append(beta_pred_batch)
        batch_details[model]['wall'].append(wall)
        batch_details[model]['pf_batch'].append(step_data['pf'])
        batch_details[model]['dc_batch'].append(step_data['dc'])
        batch_details[model]['pf_req'].append(j['prefill_total'])
        batch_details[model]['running'].append(step_data['running'])

    print(f"\n  Ratio = real_journey_TTFT / β·features(full_batch_pf, full_batch_dc)")
    print(f"  {'Model':<16} {'N':>6} {'Ratio med':>10} {'Ratio mean':>11} {'Ratio p10':>10} {'Ratio p90':>10}")
    print(f"  {'-'*68}")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        r = batch_ratios.get(model)
        if not r:
            continue
        arr = np.array(r)
        print(f"  {model:<16} {len(arr):>6} {np.median(arr):>10.3f} {np.mean(arr):>11.3f} {np.percentile(arr, 10):>10.3f} {np.percentile(arr, 90):>10.3f}")

    # =================================================================
    # INVESTIGATION 4: Decompose the ratio by batch size
    # =================================================================
    print(f"\n  Breakdown by running_depth:")
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        d = batch_details.get(model)
        if not d or not d['real_ttft']:
            continue
        real = np.array(d['real_ttft'])
        beta_b = np.array(d['beta_batch'])
        wall = np.array(d['wall'])
        running = np.array(d['running'])
        pf_b = np.array(d['pf_batch'])
        dc_b = np.array(d['dc_batch'])
        pf_r = np.array(d['pf_req'])

        print(f"\n  {model}:")
        for lo, hi in [(1, 5), (5, 20), (20, 50), (50, 128)]:
            mask = (running >= lo) & (running < hi) & (beta_b > 0)
            if mask.sum() < 5:
                continue
            ratio_batch = real[mask] / beta_b[mask]
            ratio_wall = real[mask] / wall[mask]
            pf_frac = pf_b[mask] / (pf_b[mask] + dc_b[mask] + 1e-9)
            req_frac = pf_r[mask] / (pf_b[mask] + 1e-9)
            print(f"    running [{lo:>3},{hi:>3}): N={mask.sum():>4}"
                  f"  real/β_batch={np.median(ratio_batch):.3f}"
                  f"  real/wall={np.median(ratio_wall):.3f}"
                  f"  pf_frac={np.median(pf_frac):.3f}"
                  f"  req_pf/batch_pf={np.median(req_frac):.3f}")

    # =================================================================
    # INVESTIGATION 5: The correct TTFT formula
    # Instead of TTFT = α + factor × β·features(request),
    # what if TTFT = α + factor × wall_clock_of_step?
    # Or: TTFT = α + factor × β·features(batch)?
    # =================================================================
    print("\n" + "=" * 72)
    print("INVESTIGATION 5: Finding the correct TTFT formula")
    print("=" * 72)
    print("For each model, what multiplier of β·features(batch) best predicts TTFT?")
    print("And what multiplier of β·features(request_pf_only) best predicts TTFT?")

    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        d = batch_details.get(model)
        if not d or not d['real_ttft']:
            continue
        real = np.array(d['real_ttft'])
        beta_b = np.array(d['beta_batch'])
        wall = np.array(d['wall'])
        pf_r = np.array(d['pf_req'])

        # β·features(request_pf_only, dc=0)
        beta_req = np.array([
            float(np.dot(beta_wall, beta_features(int(pf), 0, model)))
            for pf in pf_r
        ])

        # Optimal multiplier via OLS (no intercept): TTFT ≈ k × predictor
        # k = Σ(real * pred) / Σ(pred²)
        valid_b = beta_b > 0
        valid_r = beta_req > 0

        if valid_b.sum() > 10:
            k_batch = np.sum(real[valid_b] * beta_b[valid_b]) / np.sum(beta_b[valid_b] ** 2)
            resid_b = real[valid_b] - k_batch * beta_b[valid_b]
            mape_b = np.mean(np.abs(resid_b) / real[valid_b]) * 100
        else:
            k_batch = float('nan')
            mape_b = float('nan')

        if valid_r.sum() > 10:
            k_req = np.sum(real[valid_r] * beta_req[valid_r]) / np.sum(beta_req[valid_r] ** 2)
            resid_r = real[valid_r] - k_req * beta_req[valid_r]
            mape_r = np.mean(np.abs(resid_r) / real[valid_r]) * 100
        else:
            k_req = float('nan')
            mape_r = float('nan')

        k_wall = np.sum(real * wall) / np.sum(wall ** 2) if len(wall) > 0 else float('nan')
        resid_w = real - k_wall * wall
        mape_w = np.mean(np.abs(resid_w) / real) * 100

        print(f"\n  {model}:")
        print(f"    TTFT ≈ k × wall_clock:            k={k_wall:.3f}, MAPE={mape_w:.1f}%")
        print(f"    TTFT ≈ k × β·features(batch):     k={k_batch:.3f}, MAPE={mape_b:.1f}%")
        print(f"    TTFT ≈ k × β·features(req_pf):    k={k_req:.3f}, MAPE={mape_r:.1f}%")

    # =================================================================
    # INVESTIGATION 6: Try TTFT = α + k × β·features(batch) per model
    # with α fitted from residuals
    # =================================================================
    print("\n" + "=" * 72)
    print("INVESTIGATION 6: TTFT = α₀ + k × β·features(batch) [per-model k, global β]")
    print("=" * 72)
    print("Fit k and α₀ per model, then check if k is model-independent.")

    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        d = batch_details.get(model)
        if not d or not d['real_ttft']:
            continue
        real = np.array(d['real_ttft'])
        beta_b = np.array(d['beta_batch'])

        valid = beta_b > 0
        if valid.sum() < 10:
            continue

        # OLS: real = α₀ + k * beta_batch
        X = np.column_stack([np.ones(valid.sum()), beta_b[valid]])
        y = real[valid]
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_0 = coeffs[0]
        k = coeffs[1]

        pred = alpha_0 + k * beta_b[valid]
        mape = np.mean(np.abs(pred - y) / y) * 100
        bias = np.mean((pred - y) / y) * 100

        print(f"\n  {model}: α₀={alpha_0:.0f}µs, k={k:.3f}, MAPE={mape:.1f}%, bias={bias:+.1f}%")


if __name__ == '__main__':
    main()
