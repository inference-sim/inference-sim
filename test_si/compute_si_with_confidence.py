#!/usr/bin/env python3
"""
Compute Saturation Index (SI) with confidence scores from BLIS metrics JSON.
Based on algorithm from issue #1364 including confidence scores from comment.
"""
import json
import sys
import math

def compute_si_with_confidence(metrics_file):
    with open(metrics_file) as f:
        data = json.load(f)

    requests = data['requests']

    # Step 1: Find T (last arrival time)
    T = max(req['arrived_at'] * 1e6 for req in requests)
    boundary_threshold = T - (T / 100)  # Last 1% of window

    # Step 2: Build event trace
    events = []
    completions_within_T = 0
    boundary_arrivals = 0

    for req in requests:
        arrival_us = req['arrived_at'] * 1e6
        events.append({'time_us': arrival_us, 'delta': +1})

        if arrival_us > boundary_threshold:
            boundary_arrivals += 1

        if req['e2e_ms'] is not None:
            completion_us = arrival_us + (req['e2e_ms'] * 1000)
            if completion_us <= T:
                events.append({'time_us': completion_us, 'delta': -1})
                completions_within_T += 1

    # Step 3: Sort events
    events.sort(key=lambda e: (e['time_us'], -e['delta']))

    # Step 4: Compute integral and variance
    area = 0
    area_squared = 0
    count = 0
    prev_t = 0

    for ev in events:
        dt = ev['time_us'] - prev_t
        area += count * dt
        area_squared += (count * count) * dt
        count += ev['delta']
        prev_t = ev['time_us']

    # Final segment to T
    dt = T - prev_t
    area += count * dt
    area_squared += (count * count) * dt

    # Step 5: Compute primary statistics
    N = len(requests)
    MeanQ = area / T if T > 0 else 0
    EndQ = N - completions_within_T
    SI = EndQ / MeanQ if MeanQ > 0 else 0
    CF = completions_within_T / N if N > 0 else 0

    # Step 6: Compute confidence scores

    # 1. Boundary Integrity (BI)
    BI = 1.0 - (boundary_arrivals / N) if N > 0 else 1.0

    # 2. Signal Separation (SS)
    if SI <= 1.0:
        SS = 1.0 - SI if SI >= 0 else 0
    elif SI < 2.0:
        SS = min(SI - 1.0, 2.0 - SI)
    else:
        SS = SI - 2.0

    # 3. Signal-Noise Ratio (SNR)
    MeanQSq = area_squared / T if T > 0 else 0
    VarianceQ = max(0, MeanQSq - MeanQ * MeanQ)
    StdQ = math.sqrt(VarianceQ)
    SNR = abs(EndQ - MeanQ) / StdQ if StdQ > 0 else 0

    # 4. Agreement (AG)
    SI_signal = SI - 1.0
    CF_signal = 1.0 - CF
    AG = 1.0 if (SI_signal > 0 and CF_signal > 0) or (SI_signal <= 0 and CF_signal <= 0.01) else 0.0

    # Classification
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
        'classification': classification,
        'confidence': {
            'BI': BI,
            'SS': SS,
            'SNR': SNR,
            'AG': AG
        }
    }

if __name__ == '__main__':
    for metrics_file in sys.argv[1:]:
        result = compute_si_with_confidence(metrics_file)
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
        print(f"Classification:            {result['classification']}")
        print(f"")
        print(f"Confidence Scores:")
        print(f"  Boundary Integrity (BI): {result['confidence']['BI']:.3f}")
        print(f"  Signal Separation (SS):  {result['confidence']['SS']:.3f}")
        print(f"  Signal-Noise Ratio (SNR):{result['confidence']['SNR']:.3f}")
        print(f"  Agreement (AG):          {result['confidence']['AG']:.3f}")
