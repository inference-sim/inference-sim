#!/usr/bin/env python3
"""
Test composite saturation detector for monotonicity.
"""
import sys
sys.path.insert(0, 'test_si')
from compute_composite import compute_composite

results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    file = f'test_si/metrics-{ratio}x.json'
    try:
        r = compute_composite(file)
        results.append({
            'ratio': ratio,
            'score': r['saturation_score'],
            'classification': r['classification'],
            'RD': r['signals']['rate_deficit'],
            'GR': r['signals']['growth_ratio'],
            'SQ': r['signals']['standing_queue']
        })
    except FileNotFoundError:
        print(f"Warning: {file} not found, need to run experiments first")
        continue

if not results:
    print("No metrics files found. Run experiments to generate them.")
    sys.exit(1)

print(f"{'Load':>6} | {'Score':>6} | {'RD':>6} | {'GR':>6} | {'SQ':>4} | Classification")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*18}")
for r in results:
    print(f"{r['ratio']:>6.3f} | {r['score']:>6.4f} | {r['RD']:>6.4f} | {r['GR']:>6.4f} | {r['SQ']:>4.1f} | {r['classification']}")

print("\n=== Monotonicity Check ===")
violations = 0
for i in range(1, len(results)):
    prev = results[i-1]
    curr = results[i]
    if curr['score'] < prev['score']:
        print(f"  ✗ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: Score decreased from {prev['score']:.4f} to {curr['score']:.4f}")
        violations += 1
    else:
        delta = curr['score'] - prev['score']
        print(f"  ✓ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: Score increased from {prev['score']:.4f} to {curr['score']:.4f} (+{delta:.4f})")

print(f"\nTotal violations: {violations}/{len(results)-1}")

if violations == 0:
    print("\n✓✓✓ COMPOSITE DETECTOR IS PERFECTLY MONOTONIC! ✓✓✓")
else:
    print(f"\n✗ Composite still has {violations} monotonicity violations")
