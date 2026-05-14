#!/usr/bin/env python3
"""
Compute SI_robust (tail-averaged variant) from BLIS metrics JSON.
Based on suggestion in issue #1364 comment #4451941038.

SI_robust = TailMeanQ / MeanQ
where TailMeanQ = (1/W) ∫_{T-W}^{T} Q(t) dt
and W = T/10 (captures multiple burst cycles)
"""
import json
import sys
import math

def compute_si_robust(metrics_file):
    with open(metrics_file) as f:
        data = json.load(f)

    requests = data['requests']

    # Step 1: Find T (last arrival time)
    T = max(req['arrived_at'] * 1e6 for req in requests)

    # Step 2: Build event trace
    events = []
    completions_within_T = 0

    for req in requests:
        arrival_us = req['arrived_at'] * 1e6
        events.append({'time_us': arrival_us, 'delta': +1})

        # Only process completed requests (e2e_ms > 0)
        if req['e2e_ms'] is not None and req['e2e_ms'] > 0:
            completion_us = arrival_us + (req['e2e_ms'] * 1000)
            if completion_us <= T:
                events.append({'time_us': completion_us, 'delta': -1})
                completions_within_T += 1

    # Step 3: Sort events
    events.sort(key=lambda e: (e['time_us'], -e['delta']))

    # Step 4: Compute integral of Q(t) over [0, T] and tail window [T-W, T]
    W = T / 10  # Tail window = 10% of observation window
    tail_start = T - W

    area_full = 0      # ∫₀ᵀ Q(t) dt
    area_tail = 0      # ∫_{T-W}^{T} Q(t) dt
    count = 0
    prev_t = 0

    for ev in events:
        dt = ev['time_us'] - prev_t

        # Full window integral
        area_full += count * dt

        # Tail window integral (only count segments within [tail_start, T])
        if prev_t >= tail_start:
            # Entire segment is in tail window
            area_tail += count * dt
        elif ev['time_us'] > tail_start:
            # Segment straddles tail_start boundary
            if prev_t < tail_start:
                # Only count portion after tail_start
                tail_dt = ev['time_us'] - tail_start
                area_tail += count * tail_dt
            else:
                area_tail += count * dt

        count += ev['delta']
        prev_t = ev['time_us']

    # Final segment to T
    dt = T - prev_t
    area_full += count * dt
    if prev_t >= tail_start:
        area_tail += count * dt
    elif T > tail_start:
        tail_dt = T - max(prev_t, tail_start)
        area_tail += count * tail_dt

    # Step 5: Compute observables
    N = len(requests)
    MeanQ = area_full / T if T > 0 else 0
    TailMeanQ = area_tail / W if W > 0 else 0
    EndQ = N - completions_within_T

    SI_original = EndQ / MeanQ if MeanQ > 0 else 0
    SI_robust = TailMeanQ / MeanQ if MeanQ > 0 else 0
    CF = completions_within_T / N if N > 0 else 0

    # Classification for both
    def classify(si):
        if si <= 1.0:
            return "UNSATURATED"
        elif si < 2.0:
            return "TRANSIENT_BACKLOG"
        else:
            return "PERSISTENTLY_SATURATED"

    return {
        'file': metrics_file,
        'N': N,
        'completions_within_T': completions_within_T,
        'T_seconds': T / 1e6,
        'W_seconds': W / 1e6,
        'MeanQ': MeanQ,
        'TailMeanQ': TailMeanQ,
        'EndQ': EndQ,
        'SI_original': SI_original,
        'SI_robust': SI_robust,
        'CF': CF,
        'classification_original': classify(SI_original),
        'classification_robust': classify(SI_robust)
    }

if __name__ == '__main__':
    for metrics_file in sys.argv[1:]:
        result = compute_si_robust(metrics_file)
        print(f"\n{'='*70}")
        print(f"File: {result['file']}")
        print(f"{'='*70}")
        print(f"Total requests (N):        {result['N']}")
        print(f"Completed within [0,T]:    {result['completions_within_T']}")
        print(f"Still in-flight at T:      {result['EndQ']}")
        print(f"Observation window T:      {result['T_seconds']:.1f} seconds")
        print(f"Tail window W:             {result['W_seconds']:.1f} seconds (T/10)")
        print(f"")
        print(f"Mean in-flight (MeanQ):    {result['MeanQ']:.2f}")
        print(f"Tail mean (TailMeanQ):     {result['TailMeanQ']:.2f}")
        print(f"End in-flight (EndQ):      {result['EndQ']}")
        print(f"")
        print(f"SI_original (EndQ/MeanQ):  {result['SI_original']:.3f} → {result['classification_original']}")
        print(f"SI_robust (TailMeanQ/MeanQ): {result['SI_robust']:.3f} → {result['classification_robust']}")
        print(f"Completion Fraction (CF):  {result['CF']:.3f}")
