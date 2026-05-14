#!/usr/bin/env python3
"""
Compute Composite Saturation Detector score from BLIS metrics JSON.
Based on implementation guide in issue #1364 comment #4452288270.

Uses three signals:
- Rate Deficit (RD): 1 - (completions / arrivals)
- Growth Ratio (GR): (MeanQ_2nd_half / MeanQ_1st_half) - 1
- Standing Queue (SQ): 1 if queue never emptied, else 0
"""
import json
import sys
import math

def compute_composite(metrics_file):
    with open(metrics_file) as f:
        data = json.load(f)

    requests = data['requests']
    N = len(requests)

    # Step 1: Find observation window T
    T = max(req['arrived_at'] * 1e6 for req in requests)
    midpoint = T / 2

    # Step 2: Build event trace
    events = []
    completions_within_T = 0
    min_sojourn = float('inf')

    for req in requests:
        arrival_us = req['arrived_at'] * 1e6
        events.append({'time_us': arrival_us, 'delta': +1})

        # Only process completed requests (e2e_ms > 0)
        if req['e2e_ms'] is not None and req['e2e_ms'] > 0:
            completion_us = arrival_us + (req['e2e_ms'] * 1000)

            # Count completions within window for RD
            if completion_us <= T:
                completions_within_T += 1

            # Compute sojourn for SQ (scheduling_delay_ms is queue wait time)
            if 'scheduling_delay_ms' in req and req['scheduling_delay_ms'] is not None:
                sojourn_ms = req['scheduling_delay_ms']
                min_sojourn = min(min_sojourn, sojourn_ms)

            events.append({'time_us': completion_us, 'delta': -1})

    # Step 3: Sort events
    events.sort(key=lambda e: (e['time_us'], -e['delta']))

    # Step 4: Compute queue integrals for GR
    area_first_half = 0
    area_second_half = 0
    time_first_half = 0
    time_second_half = 0
    count = 0
    prev_t = 0

    for ev in events:
        dt = ev['time_us'] - prev_t

        if prev_t < midpoint:
            if ev['time_us'] <= midpoint:
                # Entire segment in first half
                area_first_half += count * dt
                time_first_half += dt
            else:
                # Spans midpoint
                dt_first = midpoint - prev_t
                dt_second = ev['time_us'] - midpoint
                area_first_half += count * dt_first
                time_first_half += dt_first
                area_second_half += count * dt_second
                time_second_half += dt_second
        else:
            # Entire segment in second half
            area_second_half += count * dt
            time_second_half += dt

        count += ev['delta']
        prev_t = ev['time_us']

    # Final segment to T
    dt = T - prev_t
    if prev_t < midpoint:
        if T <= midpoint:
            area_first_half += count * dt
            time_first_half += dt
        else:
            dt_first = midpoint - prev_t
            dt_second = T - midpoint
            area_first_half += count * dt_first
            time_first_half += dt_first
            area_second_half += count * dt_second
            time_second_half += dt_second
    else:
        area_second_half += count * dt
        time_second_half += dt

    # Step 5: Compute the three signals

    # Signal 1: Rate Deficit
    rate_deficit = max(0.0, 1.0 - (completions_within_T / N)) if N > 0 else 0.0

    # Signal 2: Growth Ratio
    mean_q_first = area_first_half / time_first_half if time_first_half > 0 else 0.0
    mean_q_second = area_second_half / time_second_half if time_second_half > 0 else 0.0

    if mean_q_first >= 1.0:
        growth_ratio = max(0.0, (mean_q_second / mean_q_first) - 1.0)
    elif mean_q_second > 0:
        growth_ratio = 1.0  # went from empty to non-empty
    else:
        growth_ratio = 0.0

    # Signal 3: Standing Queue
    # Queue never emptied if minimum sojourn > 0
    standing_queue = 1.0 if (completions_within_T >= 10 and min_sojourn > 0) else 0.0

    # Step 6: Composite score
    saturation_score = max(rate_deficit, min(growth_ratio, 1.0), standing_queue)

    # Step 7: Classification
    epsilon = (1.0 / math.sqrt(N)) if N > 0 else 1.0

    if saturation_score < epsilon:
        classification = "UNSATURATED"
    elif standing_queue == 1.0:
        classification = "DEEPLY_SATURATED"
    elif growth_ratio > epsilon:
        classification = "SATURATING"
    elif rate_deficit > epsilon:
        classification = "DEEPLY_SATURATED"
    else:
        classification = "UNSATURATED"

    # Step 8: Confidence
    confidence = min(1.0, N / 20.0)

    return {
        'file': metrics_file,
        'N': N,
        'completions_within_T': completions_within_T,
        'T_seconds': T / 1e6,
        'signals': {
            'rate_deficit': rate_deficit,
            'growth_ratio': min(growth_ratio, 1.0),
            'standing_queue': standing_queue
        },
        'saturation_score': saturation_score,
        'classification': classification,
        'confidence': confidence,
        'epsilon': epsilon,
        'mean_q_first': mean_q_first,
        'mean_q_second': mean_q_second,
        'min_sojourn_ms': min_sojourn if min_sojourn != float('inf') else 0
    }

if __name__ == '__main__':
    for metrics_file in sys.argv[1:]:
        result = compute_composite(metrics_file)
        print(f"\n{'='*70}")
        print(f"File: {result['file']}")
        print(f"{'='*70}")
        print(f"Requests: {result['N']}")
        print(f"Completed within [0,T]: {result['completions_within_T']}")
        print(f"Observation window: {result['T_seconds']:.1f}s")
        print(f"")
        print(f"Signals:")
        print(f"  Rate Deficit (RD):    {result['signals']['rate_deficit']:.4f}")
        print(f"  Growth Ratio (GR):    {result['signals']['growth_ratio']:.4f}")
        print(f"  Standing Queue (SQ):  {result['signals']['standing_queue']:.1f}")
        print(f"")
        print(f"Composite Score:        {result['saturation_score']:.4f}")
        print(f"Classification:         {result['classification']}")
        print(f"Confidence:             {result['confidence']:.4f}")
        print(f"Epsilon (noise floor):  {result['epsilon']:.4f}")
        print(f"")
        print(f"Debug:")
        print(f"  MeanQ first half:     {result['mean_q_first']:.2f}")
        print(f"  MeanQ second half:    {result['mean_q_second']:.2f}")
        print(f"  Min sojourn:          {result['min_sojourn_ms']:.2f}ms")
