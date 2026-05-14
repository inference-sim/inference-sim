#!/usr/bin/env python3
import sys
sys.path.insert(0, 'test_si')
from compute_si_with_confidence import compute_si_with_confidence

results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    file = f'test_si/metrics-{ratio}x.json'
    try:
        r = compute_si_with_confidence(file)
        results.append({
            'ratio': ratio,
            'SI': r['SI'],
            'CF': r['CF'],
            'BI': r['confidence']['BI'],
            'SS': r['confidence']['SS'],
            'SNR': r['confidence']['SNR'],
            'AG': r['confidence']['AG']
        })
    except FileNotFoundError:
        continue

print(f"{'Load':>6} | {'SI':>6} | {'CF':>5} | {'BI':>5} | {'SS':>5} | {'SNR':>6} | {'AG':>4}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}")
for r in results:
    print(f"{r['ratio']:>6.3f} | {r['SI']:>6.3f} | {r['CF']:>5.3f} | {r['BI']:>5.3f} | {r['SS']:>5.3f} | {r['SNR']:>6.2f} | {r['AG']:>4.1f}")

print("\nConfidence Score Definitions:")
print("  BI (Boundary Integrity): 1.0 = clean boundary, <0.5 = arrivals clustered at T")
print("  SS (Signal Separation): Distance from nearest classification threshold (1.0 or 2.0)")
print("  SNR (Signal-Noise Ratio): |EndQ - MeanQ| / StdQ (higher = stronger signal)")
print("  AG (Agreement): 1.0 = SI and CF agree on direction, 0.0 = disagreement")
