"""Quick check: compare journey real prefill time with step wall clock."""
import csv
import numpy as np
from collections import defaultdict

steps_by_exp = defaultdict(dict)
with open('training/iter0_steps.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps_by_exp[row['experiment']][int(row['step_id'])] = {
            'ts_start': int(row['ts_start_ns']),
            'ts_end': int(row['ts_end_ns']),
            'dur': int(row['duration_us']),
            'pf': int(row['prefill_tokens']),
            'dc': int(row['decode_tokens']),
            'running': int(row['running_depth']),
        }

wall_by_step = {}
for exp, exp_steps in steps_by_exp.items():
    sorted_ids = sorted(exp_steps.keys())
    for i in range(len(sorted_ids) - 1):
        if sorted_ids[i+1] - sorted_ids[i] == 1:
            sid = sorted_ids[i]
            t = (exp_steps[sorted_ids[i+1]]['ts_start'] - exp_steps[sid]['ts_start']) / 1000
            wall_by_step[(exp, sid)] = t

model_results = defaultdict(lambda: {'real': [], 'wall': [], 'sched': [], 'pf': [], 'running': []})

with open('training/iter3_journeys.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['num_preemptions']) > 0:
            continue
        ss = int(row['scheduled_step'])
        fs = int(row['first_token_step'])
        if fs != ss:
            continue

        exp = row['experiment']
        model = row['model_short']
        real_us = (int(row['first_token_ns']) - int(row['scheduled_ns'])) / 1000

        wall = wall_by_step.get((exp, ss))
        step_data = steps_by_exp.get(exp, {}).get(ss)

        if wall is not None and step_data is not None:
            model_results[model]['real'].append(real_us)
            model_results[model]['wall'].append(wall)
            model_results[model]['sched'].append(step_data['dur'])
            model_results[model]['pf'].append(step_data['pf'])
            model_results[model]['running'].append(step_data['running'])

print('Single-step prefill: journey real time vs step wall clock')
print('(journey: first_token_ns - scheduled_ns  vs  step: ts_start[S+1] - ts_start[S])')
print()
print(f'{"Model":<16} {"N":>6} {"Real(us)":>10} {"Wall(us)":>10} {"Sched(us)":>10} {"Ratio":>8} {"Diff(us)":>10}')
print('-' * 76)
for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
    d = model_results.get(model)
    if not d or not d['real']:
        print(f'{model:<16} no matched data')
        continue
    real = np.array(d['real'])
    wall = np.array(d['wall'])
    sched = np.array(d['sched'])
    ratio = real / wall
    diff = real - wall
    n = len(real)
    print(f'{model:<16} {n:>6} {np.median(real):>10.0f} {np.median(wall):>10.0f} {np.median(sched):>10.0f} {np.median(ratio):>8.3f} {np.median(diff):>10.0f}')

print()
print('Ratio interpretation:')
print('  ~1.0 = journey sees same time as step-to-step gap')
print('  >1.0 = hidden overhead not captured by step cadence')
print('  <1.0 = prefill completes before next step starts (normal: first_token emits mid-step)')

# Breakdown by batch size for each model
for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
    d = model_results.get(model)
    if not d or not d['real']:
        continue
    real = np.array(d['real'])
    wall = np.array(d['wall'])
    running = np.array(d['running'])
    ratio = real / wall

    print(f'\n{model} breakdown by running_depth:')
    for lo, hi in [(1, 5), (5, 20), (20, 50), (50, 128)]:
        mask = (running >= lo) & (running < hi)
        if mask.sum() > 5:
            print(f'  [{lo:>3},{hi:>3}): N={mask.sum():>4}, ratio={np.median(ratio[mask]):.3f}, '
                  f'real={np.median(real[mask]):>8.0f}us, wall={np.median(wall[mask]):>8.0f}us, '
                  f'diff={np.median(real[mask] - wall[mask]):>8.0f}us')

# What fraction of wall clock is the prefill?
print('\n--- Prefill fraction of wall clock ---')
for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
    d = model_results.get(model)
    if not d or not d['real']:
        continue
    real = np.array(d['real'])
    wall = np.array(d['wall'])
    pf = np.array(d['pf'])
    running = np.array(d['running'])

    # For single-step prefills with mixed batch (pf > 0 and running > 1),
    # the wall clock includes decode tokens from other requests too.
    # The journey's real_prefill is just this request's prefill portion.
    mixed = (pf > 0) & (running > 1)
    pure_pf = (pf > 0) & (running == 1)

    if mixed.sum() > 5:
        print(f'  {model} mixed batch (running>1): N={mixed.sum()}, '
              f'ratio={np.median(real[mixed]/wall[mixed]):.3f}')
    if pure_pf.sum() > 5:
        print(f'  {model} pure prefill (running=1): N={pure_pf.sum()}, '
              f'ratio={np.median(real[pure_pf]/wall[pure_pf]):.3f}')
