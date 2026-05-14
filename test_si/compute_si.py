#!/usr/bin/env python3
"""
Compute Saturation Index (SI) from BLIS metrics JSON output.
Based on algorithm from issue #1364.
"""
import json
import sys

def compute_si(metrics_file):
    with open(metrics_file) as f:
        data = json.load(f)

    requests = data['requests']

    # Step 1: Find T (last arrival time - arrived_at is in seconds, convert to microseconds)
    T = max(req['arrived_at'] * 1e6 for req in requests)

    # Step 2: Build event trace
    events = []
    completions_within_T = 0

    for req in requests:
        arrival_us = req['arrived_at'] * 1e6  # Convert seconds to microseconds
        events.append({'time_us': arrival_us, 'delta': +1})

        # Completion time = arrival + e2e_ms (only for completed requests, e2e_ms > 0)
        if req['e2e_ms'] is not None and req['e2e_ms'] > 0:
            completion_us = arrival_us + (req['e2e_ms'] * 1000)  # e2e_ms in milliseconds
            if completion_us <= T:
                events.append({'time_us': completion_us, 'delta': -1})
                completions_within_T += 1

    # Step 3: Sort events (arrivals before completions at same timestamp)
    events.sort(key=lambda e: (e['time_us'], -e['delta']))

    # Step 4: Compute integral of Q(t) over [0, T]
    area = 0
    count = 0
    prev_t = 0

    for ev in events:
        dt = ev['time_us'] - prev_t
        area += count * dt
        count += ev['delta']
        prev_t = ev['time_us']

    # Final segment to T
    area += count * (T - prev_t)

    # Step 5: Compute observables
    N = len(requests)
    MeanQ = area / T if T > 0 else 0
    EndQ = N - completions_within_T

    SI = EndQ / MeanQ if MeanQ > 0 else 0
    CF = completions_within_T / N if N > 0 else 0

    # Classification per issue #1364
    if SI <= 1.0:
        classification = "UNSATURATED"
    elif SI < 2.0:
        classification = "TRANSIENT_BACKLOG"
    else:
        classification = "PERSISTENTLY_SATURATED"

    return {
        'file': metrics_file,
        'N': N,
        'completions_within_T': completions_within_T,
        'T_seconds': T / 1e6,
        'MeanQ': MeanQ,
        'EndQ': EndQ,
        'SI': SI,
        'CF': CF,
        'classification': classification
    }

if __name__ == '__main__':
    for metrics_file in sys.argv[1:]:
        result = compute_si(metrics_file)
        print(f"\n{'='*60}")
        print(f"File: {result['file']}")
        print(f"{'='*60}")
        print(f"Total requests (N):        {result['N']}")
        print(f"Completed within [0,T]:    {result['completions_within_T']}")
        print(f"Still in-flight at T:      {result['EndQ']}")
        print(f"Observation window T:      {result['T_seconds']:.1f} seconds")
        print(f"")
        print(f"Mean in-flight (MeanQ):    {result['MeanQ']:.2f}")
        print(f"End in-flight (EndQ):      {result['EndQ']}")
        print(f"")
        print(f"Saturation Index (SI):     {result['SI']:.3f}")
        print(f"Completion Fraction (CF):  {result['CF']:.3f}")
        print(f"")
        print(f"Classification:            {result['classification']}")
