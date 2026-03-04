"""
Iteration 5 — Decomposed Inter-Step Overhead Analysis (Approach D)

Strategy Evolution Phase 2 prototype: Journey-Step Correlation Fitting.

Instead of treating inter-step overhead as a constant δ₀ to add on top of
Iter 3 β (which caused entanglement because β already absorbed some overhead
via Block B journey constraints), this approach:

1. Measures the TOTAL wall-clock time per step from consecutive BATCH_SUMMARY
   timestamps: T_wall[i] = ts_start[i+1] - ts_start[i]

2. Decomposes T_wall into:
   - T_sched[i]  = step.duration_us (scheduler CPU time, directly measured)
   - T_gpu[i]    = β·features(step_i) (GPU forward pass, to be fitted)
   - T_overhead[i] = T_wall[i] - T_sched[i] - T_gpu[i] (residual: input prep,
     output processing, CUDA graph overhead, Python/GIL)

3. Validates via journey timestamps: for a request with scheduled_step=S and
   first_token_step=F, the real prefill time = first_token_ns - scheduled_ns.
   The predicted prefill time = Σ_{k=S}^{F} (T_gpu[k] + T_overhead[k]).
   The gap between these should be zero if the decomposition is correct.

Key insight from vLLM code inspection:
- step.duration_us measures ONLY scheduler.schedule() CPU time
- The gap between consecutive steps includes: execute_model() (GPU + CPU input
  prep), update_from_output() (CPU output processing), sample_tokens(), and
  Python overhead
- These overhead components scale with DIFFERENT features:
  * Scheduling scales with batch_size (KV block allocation is O(n_requests))
  * Input prep scales with total_tokens (tensor construction)
  * Output processing scales with num_finished (stop checks, event emission)

Usage:
    python3 training/iter5_decompose.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ─── Data Loading ────────────────────────────────────────

@dataclass
class StepRecord:
    experiment: str
    model: str
    profile: str
    split: str
    tp: int
    step_id: int
    duration_us: int       # scheduler CPU time
    ts_start_ns: int
    ts_end_ns: int
    prefill_tokens: int
    decode_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    scheduled_tokens: int
    num_finished: int
    num_preempted: int
    running_depth: int
    waiting_depth: int
    kv_usage_ratio: float
    kv_blocks_total: int
    kv_blocks_free: int


@dataclass
class JourneyRecord:
    experiment: str
    model: str
    profile: str
    split: str
    tp: int
    span_id: str
    request_id: str
    queued_ns: int
    scheduled_ns: int
    first_token_ns: int
    finished_ns: int
    queued_step: int
    scheduled_step: int
    first_token_step: int
    finished_step: int
    prefill_total: int
    decode_max: int
    num_preemptions: int


def load_steps(path: str) -> list[StepRecord]:
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(StepRecord(
                experiment=row['experiment'],
                model=row['model_short'],
                profile=row['profile'],
                split=row['split'],
                tp=int(row['tp']),
                step_id=int(row['step_id']),
                duration_us=int(row['duration_us']),
                ts_start_ns=int(row['ts_start_ns']),
                ts_end_ns=int(row['ts_end_ns']),
                prefill_tokens=int(row['prefill_tokens']),
                decode_tokens=int(row['decode_tokens']),
                num_prefill_reqs=int(row['num_prefill_reqs']),
                num_decode_reqs=int(row['num_decode_reqs']),
                scheduled_tokens=int(row['scheduled_tokens']),
                num_finished=int(row['num_finished']),
                num_preempted=int(row['num_preempted']),
                running_depth=int(row['running_depth']),
                waiting_depth=int(row['waiting_depth']),
                kv_usage_ratio=float(row['kv_usage_ratio']),
                kv_blocks_total=int(row['kv_blocks_total']),
                kv_blocks_free=int(row['kv_blocks_free']),
            ))
    return records


def load_journeys(path: str) -> list[JourneyRecord]:
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(JourneyRecord(
                experiment=row['experiment'],
                model=row['model_short'],
                profile=row['profile'],
                split=row['split'],
                tp=int(row['tp']),
                span_id=row['span_id'],
                request_id=row['request_id'],
                queued_ns=int(row['queued_ns']),
                scheduled_ns=int(row['scheduled_ns']),
                first_token_ns=int(row['first_token_ns']),
                finished_ns=int(row['finished_ns']),
                queued_step=int(row['queued_step']),
                scheduled_step=int(row['scheduled_step']),
                first_token_step=int(row['first_token_step']),
                finished_step=int(row['finished_step']),
                prefill_total=int(row['prefill_total']),
                decode_max=int(row['decode_max']),
                num_preemptions=int(row['num_preemptions']),
            ))
    return records


# ─── Model configs (from training data) ─────────────────

MODEL_CONFIGS = {
    'llama-2-7b':    {'L': 32, 'kv_dim': 4096, 'is_moe': False, 'tp': 1},
    'llama-2-70b':   {'L': 80, 'kv_dim': 8192, 'is_moe': False, 'tp': 4},
    'mixtral-8x7b':  {'L': 32, 'kv_dim': 4096, 'is_moe': True,  'tp': 2},
    'codellama-34b': {'L': 48, 'kv_dim': 8192, 'is_moe': False, 'tp': 2},
}

# Iter 3 global β coefficients (for reference/comparison)
ITER3_BETA = np.array([116.110, 1226.868, 19.943, 9445.157])


def beta_features(pf: int, dc: int, model: str) -> np.ndarray:
    """Compute Iter 3 β feature vector for a step."""
    cfg = MODEL_CONFIGS[model]
    L = cfg['L']
    kv_dim = cfg['kv_dim'] / cfg['tp']
    is_moe = 1.0 if cfg['is_moe'] else 0.0
    is_tp = 1.0 if cfg['tp'] > 1 else 0.0

    return np.array([
        L,                              # per-layer overhead
        dc * L * kv_dim * 1e-6,         # KV bandwidth (scaled)
        (pf + dc) * is_moe,             # MoE routing
        is_tp,                          # TP sync barrier
    ])


# ─── Analysis 1: Consecutive Step Gap Decomposition ─────

def analyze_consecutive_gaps(steps_by_exp: dict[str, list[StepRecord]]) -> dict:
    """For consecutive sampled steps, decompose the wall-clock gap.

    T_wall = ts_start[i+1] - ts_start[i]  (total time for step i)
    T_sched = duration_us[i]                (scheduler CPU, measured)
    T_gpu_pred = β·features(step_i)         (GPU forward pass, predicted)
    T_residual = T_wall - T_sched - T_gpu_pred  (everything else)
    """
    results = {}

    for exp, steps in steps_by_exp.items():
        steps_sorted = sorted(steps, key=lambda s: s.step_id)
        model = steps_sorted[0].model

        wall_times = []       # total wall clock per step
        sched_times = []      # scheduler CPU per step
        gpu_pred_times = []   # Iter 3 β predicted GPU time
        residuals = []        # T_wall - T_sched - T_gpu
        batch_sizes = []      # running_depth for regression
        total_tokens = []     # scheduled_tokens for regression
        finished_counts = []  # num_finished for regression

        for i in range(len(steps_sorted) - 1):
            s_curr = steps_sorted[i]
            s_next = steps_sorted[i + 1]

            # Only use consecutive step pairs (step_id differs by 1)
            if s_next.step_id - s_curr.step_id != 1:
                continue

            t_wall_us = (s_next.ts_start_ns - s_curr.ts_start_ns) / 1000
            t_sched_us = s_curr.duration_us
            features = beta_features(s_curr.prefill_tokens, s_curr.decode_tokens, model)
            t_gpu_us = float(np.dot(ITER3_BETA, features))
            t_resid_us = t_wall_us - t_sched_us - t_gpu_us

            wall_times.append(t_wall_us)
            sched_times.append(t_sched_us)
            gpu_pred_times.append(t_gpu_us)
            residuals.append(t_resid_us)
            batch_sizes.append(s_curr.running_depth)
            total_tokens.append(s_curr.scheduled_tokens)
            finished_counts.append(s_curr.num_finished)

        if not wall_times:
            continue

        results[exp] = {
            'model': model,
            'n_pairs': len(wall_times),
            'wall': _stats(wall_times),
            'sched': _stats(sched_times),
            'gpu_pred': _stats(gpu_pred_times),
            'residual': _stats(residuals),
            'batch_sizes': np.array(batch_sizes),
            'total_tokens': np.array(total_tokens),
            'finished_counts': np.array(finished_counts),
            'residuals_arr': np.array(residuals),
            'wall_arr': np.array(wall_times),
            'sched_arr': np.array(sched_times),
            'gpu_pred_arr': np.array(gpu_pred_times),
        }

    return results


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p10': np.percentile(values, 10),
        'p90': np.percentile(values, 90),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
    }


# ─── Analysis 2: Journey-Step Correlation ────────────────

def analyze_journey_step_correlation(
    steps_by_exp: dict[str, list[StepRecord]],
    journeys: list[JourneyRecord],
) -> dict:
    """Join journey timestamps with step data to validate decomposition.

    For each request journey:
    - Real prefill time = first_token_ns - scheduled_ns
    - Real decode time  = finished_ns - first_token_ns
    - Step-summed wall time = Σ wall_clock for steps in [scheduled_step, first_token_step]

    The ratio real_time / step_summed_time reveals how much time is
    NOT captured in step-level measurements (queueing, inter-step gaps).
    """
    # Build step lookup: (experiment, step_id) -> StepRecord
    step_lookup: dict[tuple[str, int], StepRecord] = {}
    for exp, steps in steps_by_exp.items():
        for s in steps:
            step_lookup[(exp, s.step_id)] = s

    # Build consecutive-step wall-clock lookup
    wall_clock_lookup: dict[tuple[str, int], float] = {}
    for exp, steps in steps_by_exp.items():
        steps_sorted = sorted(steps, key=lambda s: s.step_id)
        for i in range(len(steps_sorted) - 1):
            if steps_sorted[i + 1].step_id - steps_sorted[i].step_id == 1:
                t_wall = (steps_sorted[i + 1].ts_start_ns - steps_sorted[i].ts_start_ns) / 1000
                wall_clock_lookup[(exp, steps_sorted[i].step_id)] = t_wall

    results_by_model = defaultdict(lambda: {
        'prefill_real': [], 'prefill_steps': [],
        'decode_real': [], 'decode_steps': [],
        'prefill_overhead_ratios': [],
    })

    for j in journeys:
        if j.num_preemptions > 0:
            continue  # preemptions complicate the timeline

        model = j.model

        # Real timestamps
        prefill_real_us = (j.first_token_ns - j.scheduled_ns) / 1000
        decode_real_us = (j.finished_ns - j.first_token_ns) / 1000

        if prefill_real_us <= 0 or decode_real_us <= 0:
            continue

        # Count how many steps in prefill phase have wall-clock data
        prefill_wall_sum = 0.0
        prefill_steps_counted = 0
        prefill_steps_total = j.first_token_step - j.scheduled_step + 1

        for step_id in range(j.scheduled_step, j.first_token_step + 1):
            wc = wall_clock_lookup.get((j.experiment, step_id))
            if wc is not None:
                prefill_wall_sum += wc
                prefill_steps_counted += 1

        # Only use journeys where we have >50% step coverage
        if prefill_steps_total > 0 and prefill_steps_counted / prefill_steps_total > 0.5:
            # Scale up to estimate full prefill from sampled steps
            scale = prefill_steps_total / prefill_steps_counted
            estimated_prefill_wall = prefill_wall_sum * scale

            results_by_model[model]['prefill_real'].append(prefill_real_us)
            results_by_model[model]['prefill_steps'].append(estimated_prefill_wall)
            if estimated_prefill_wall > 0:
                ratio = prefill_real_us / estimated_prefill_wall
                results_by_model[model]['prefill_overhead_ratios'].append(ratio)

    return dict(results_by_model)


# ─── Analysis 3: Fit structured δ from residuals ────────

def fit_structured_delta(gap_results: dict) -> dict:
    """Fit δ(batch_size, total_tokens) from the residual after
    subtracting scheduler time and Iter 3 β-predicted GPU time.

    Model: T_overhead = δ₀ + δ₁·batch_size + δ₂·total_tokens
    """
    from numpy.linalg import lstsq

    fits = {}
    for exp, data in gap_results.items():
        resid = data['residuals_arr']
        bs = data['batch_sizes']
        tt = data['total_tokens']

        # OLS: residual = δ₀ + δ₁·batch_size + δ₂·total_tokens
        X = np.column_stack([np.ones(len(resid)), bs, tt])
        delta, residual_sum, _, _ = lstsq(X, resid, rcond=None)

        predicted = X @ delta
        fit_residuals = resid - predicted
        rmse = np.sqrt(np.mean(fit_residuals ** 2))
        r2 = 1 - np.sum(fit_residuals ** 2) / np.sum((resid - np.mean(resid)) ** 2)

        fits[exp] = {
            'model': data['model'],
            'delta_0': delta[0],
            'delta_1_batch': delta[1],
            'delta_2_tokens': delta[2],
            'rmse_us': rmse,
            'r2': r2,
            'n': len(resid),
        }

    return fits


# ─── Analysis 4: Re-fit β from step-only data ───────────

def fit_step_only_beta(steps_by_exp: dict[str, list[StepRecord]]) -> dict:
    """Fit β using ONLY step-level data (Block A only, no journey constraints).

    This isolates GPU compute time from inter-step overhead.
    The dependent variable is:
        T_gpu_estimated = T_wall - T_sched  (for consecutive pairs)

    where T_wall = ts_start[i+1] - ts_start[i] gives the TOTAL time per step,
    and T_sched = duration_us[i] is the scheduler CPU time.

    But wait — T_wall - T_sched still includes input_prep + output_proc + GPU.
    To isolate GPU, we use:
        T_gpu_approx ≈ step.duration_us (from the BATCH_SUMMARY perspective)

    Actually, step.duration_us is the scheduler time, NOT GPU time.
    The step-level observations in Block A use the raw duration_us which
    IS the scheduler-side measurement of the step.

    For this prototype, we fit β against the raw step.duration_us (which measures
    scheduler CPU time, a proxy for step complexity) AND against T_wall from
    consecutive pairs (which includes everything).
    """
    from numpy.linalg import lstsq

    results = {}

    # Approach 1: β from raw step.duration_us (scheduler CPU time only)
    # This tells us what scheduler.schedule() time correlates with
    all_X_sched, all_y_sched = [], []

    # Approach 2: β from consecutive-pair wall clock (total per-step time)
    # This gives the full cost including GPU + overhead
    all_X_wall, all_y_wall = [], []

    for exp, steps in steps_by_exp.items():
        steps_sorted = sorted(steps, key=lambda s: s.step_id)
        model = steps_sorted[0].model

        for s in steps_sorted:
            features = beta_features(s.prefill_tokens, s.decode_tokens, model)
            all_X_sched.append(features)
            all_y_sched.append(s.duration_us)

        for i in range(len(steps_sorted) - 1):
            s_curr = steps_sorted[i]
            s_next = steps_sorted[i + 1]
            if s_next.step_id - s_curr.step_id != 1:
                continue
            t_wall = (s_next.ts_start_ns - s_curr.ts_start_ns) / 1000
            features = beta_features(s_curr.prefill_tokens, s_curr.decode_tokens, model)
            all_X_wall.append(features)
            all_y_wall.append(t_wall)

    # Fit β against scheduler time
    X_s = np.array(all_X_sched)
    y_s = np.array(all_y_sched)
    from scipy.optimize import nnls
    beta_sched, _ = nnls(X_s, y_s)

    # Fit β against total wall-clock time
    X_w = np.array(all_X_wall)
    y_w = np.array(all_y_wall)
    beta_wall, _ = nnls(X_w, y_w)

    results['beta_sched_only'] = {
        'description': 'β fitted against scheduler.schedule() CPU time only',
        'beta': beta_sched.tolist(),
        'n': len(all_X_sched),
    }
    results['beta_wall_clock'] = {
        'description': 'β fitted against total wall-clock time per step',
        'beta': beta_wall.tolist(),
        'n': len(all_X_wall),
    }
    results['iter3_beta'] = {
        'description': 'Iter 3 β (Block A + Block B NNLS)',
        'beta': ITER3_BETA.tolist(),
    }

    return results


# ─── Main ────────────────────────────────────────────────

def main():
    # Data files are untracked — they live in the main repo, not worktrees.
    # Try the local directory first, fall back to the main repo.
    base = os.path.dirname(os.path.abspath(__file__))
    main_repo_training = os.path.join(os.path.dirname(base), 'training')
    if not os.path.exists(os.path.join(base, 'iter0_steps.csv')):
        # We're probably in a worktree; look in the main repo
        # Try common locations
        candidates = [
            main_repo_training,
            '/Users/sri/Documents/Projects/inference-sim/training',
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, 'iter0_steps.csv')):
                base = c
                break
    steps_path = os.path.join(base, 'iter0_steps.csv')
    journeys_path = os.path.join(base, 'iter3_journeys.csv')

    print("=" * 70)
    print("Iter 5 — Decomposed Inter-Step Overhead Analysis (Approach D)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading step data...")
    steps = load_steps(steps_path)
    steps_by_exp: dict[str, list[StepRecord]] = defaultdict(list)
    for s in steps:
        steps_by_exp[s.experiment].append(s)
    print(f"    {len(steps):,} steps across {len(steps_by_exp)} experiments")

    print("\n[2] Loading journey data...")
    journeys = load_journeys(journeys_path)
    print(f"    {len(journeys):,} journeys")

    # ─── Analysis 1: Consecutive Step Gap Decomposition ──
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Consecutive Step Gap Decomposition")
    print("=" * 70)
    print("For consecutive sampled steps, decompose wall clock into:")
    print("  T_wall = T_sched (measured) + T_gpu (β-predicted) + T_residual")

    gap_results = analyze_consecutive_gaps(steps_by_exp)

    # Aggregate by model
    model_agg = defaultdict(lambda: {
        'n': 0, 'wall': [], 'sched': [], 'gpu': [], 'resid': [],
    })
    for exp, data in gap_results.items():
        m = data['model']
        model_agg[m]['n'] += data['n_pairs']
        model_agg[m]['wall'].extend(data['wall_arr'].tolist())
        model_agg[m]['sched'].extend(data['sched_arr'].tolist())
        model_agg[m]['gpu'].extend(data['gpu_pred_arr'].tolist())
        model_agg[m]['resid'].extend(data['residuals_arr'].tolist())

    print(f"\n{'Model':<16} {'N':>6} {'T_wall':>10} {'T_sched':>10} {'T_gpu(β)':>10} {'T_resid':>10} {'resid%':>8}")
    print("-" * 72)
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        agg = model_agg.get(model)
        if not agg or not agg['wall']:
            continue
        w = statistics.median(agg['wall'])
        s = statistics.median(agg['sched'])
        g = statistics.median(agg['gpu'])
        r = statistics.median(agg['resid'])
        pct = r / w * 100 if w > 0 else 0
        print(f"{model:<16} {agg['n']:>6} {w:>9.0f}µs {s:>9.0f}µs {g:>9.0f}µs {r:>9.0f}µs {pct:>7.1f}%")

    print("\nKey: T_resid = T_wall - T_sched - T_gpu. This is the 'missing time' —")
    print("     input prep, output processing, CUDA overhead, Python/GIL.")

    # ─── Analysis 2: Fit structured δ from residuals ─────
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Structured δ Regression on Residuals")
    print("=" * 70)
    print("Model: T_residual = δ₀ + δ₁·batch_size + δ₂·total_tokens")

    delta_fits = fit_structured_delta(gap_results)

    print(f"\n{'Experiment':<55} {'Model':<16} {'δ₀(µs)':>10} {'δ₁(µs/req)':>12} {'δ₂(µs/tok)':>12} {'R²':>6}")
    print("-" * 115)
    for exp in sorted(delta_fits.keys()):
        d = delta_fits[exp]
        print(f"{exp:<55} {d['model']:<16} {d['delta_0']:>10.0f} {d['delta_1_batch']:>12.1f} {d['delta_2_tokens']:>12.2f} {d['r2']:>6.3f}")

    # Per-model aggregated δ fits
    print("\n--- Per-Model Aggregated δ Fits ---")
    model_residuals = defaultdict(lambda: {'resid': [], 'bs': [], 'tt': []})
    for exp, data in gap_results.items():
        m = data['model']
        model_residuals[m]['resid'].extend(data['residuals_arr'].tolist())
        model_residuals[m]['bs'].extend(data['batch_sizes'].tolist())
        model_residuals[m]['tt'].extend(data['total_tokens'].tolist())

    from numpy.linalg import lstsq as np_lstsq

    print(f"\n{'Model':<16} {'N':>6} {'δ₀(µs)':>10} {'δ₁(µs/req)':>12} {'δ₂(µs/tok)':>12} {'RMSE(µs)':>10} {'R²':>6}")
    print("-" * 76)
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        mr = model_residuals.get(model)
        if not mr or not mr['resid']:
            continue
        resid = np.array(mr['resid'])
        bs = np.array(mr['bs'])
        tt = np.array(mr['tt'])
        X = np.column_stack([np.ones(len(resid)), bs, tt])
        delta, _, _, _ = np_lstsq(X, resid, rcond=None)
        pred = X @ delta
        rmse = np.sqrt(np.mean((resid - pred) ** 2))
        r2 = 1 - np.sum((resid - pred) ** 2) / np.sum((resid - np.mean(resid)) ** 2)
        print(f"{model:<16} {len(resid):>6} {delta[0]:>10.0f} {delta[1]:>12.1f} {delta[2]:>12.2f} {rmse:>10.0f} {r2:>6.3f}")

    # ─── Analysis 3: Journey-Step Correlation ────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Journey-Step Correlation (Cross-Validation)")
    print("=" * 70)
    print("Compare real journey prefill time with step-summed wall clock.")
    print("Ratio > 1.0 means real vLLM is SLOWER than step sums predict")
    print("(due to inter-step gaps not captured in step-level data).")

    corr_results = analyze_journey_step_correlation(steps_by_exp, journeys)

    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        data = corr_results.get(model)
        if not data or not data['prefill_overhead_ratios']:
            print(f"\n  {model}: insufficient step coverage for correlation")
            continue

        ratios = data['prefill_overhead_ratios']
        reals = [r / 1000 for r in data['prefill_real']]  # convert to ms
        preds = [p / 1000 for p in data['prefill_steps']]

        print(f"\n  {model}:")
        print(f"    N journeys with >50% step coverage: {len(ratios)}")
        print(f"    Real prefill: median={statistics.median(reals):.1f}ms, mean={statistics.mean(reals):.1f}ms")
        print(f"    Step-summed:  median={statistics.median(preds):.1f}ms, mean={statistics.mean(preds):.1f}ms")
        print(f"    Ratio (real/step): median={statistics.median(ratios):.3f}, mean={statistics.mean(ratios):.3f}")
        print(f"    Ratio p10={np.percentile(ratios, 10):.3f}, p90={np.percentile(ratios, 90):.3f}")

    # ─── Analysis 4: Re-fit β without journey constraints ─
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Re-fit β (Step-Only vs Wall-Clock vs Iter 3)")
    print("=" * 70)
    print("Compare β fitted against different targets:")
    print("  (a) scheduler.schedule() CPU time only (step.duration_us)")
    print("  (b) total wall-clock per step (consecutive ts_start gaps)")
    print("  (c) Iter 3 β (Block A + Block B NNLS with journey constraints)")

    beta_results = fit_step_only_beta(steps_by_exp)

    print(f"\n{'Source':<50} {'β₀(L)':>10} {'β₁(KV)':>10} {'β₂(MoE)':>10} {'β₃(TP)':>10}")
    print("-" * 92)
    for key, data in beta_results.items():
        b = data['beta']
        label = data['description'][:48]
        print(f"{label:<50} {b[0]:>10.1f} {b[1]:>10.1f} {b[2]:>10.1f} {b[3]:>10.1f}")

    # ─── Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Implications for Fitting Strategy")
    print("=" * 70)

    # Check if residuals are large enough to matter
    for model in ['llama-2-7b', 'codellama-34b', 'llama-2-70b', 'mixtral-8x7b']:
        agg = model_agg.get(model)
        if not agg or not agg['wall']:
            continue
        resid_med = statistics.median(agg['resid'])
        wall_med = statistics.median(agg['wall'])
        gpu_med = statistics.median(agg['gpu'])
        overhead_pct = resid_med / wall_med * 100 if wall_med > 0 else 0
        print(f"  {model}: overhead is {resid_med:.0f}µs ({overhead_pct:.1f}% of wall time)")
        print(f"    → Over 150 steps: {resid_med * 150 / 1000:.0f}ms accumulated (vs TTFT ~45-60ms)")

    print("\n  If overhead is >10% of wall time AND batch-correlated (R²>0.3),")
    print("  then structured δ fitting (Approach D) is justified over constant δ₀.")


if __name__ == '__main__':
    main()
