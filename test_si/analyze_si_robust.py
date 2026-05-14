#!/usr/bin/env python3
"""
Compare SI_original vs SI_robust for monotonicity.
"""
import sys
sys.path.insert(0, 'test_si')
from compute_si_robust import compute_si_robust

results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    file = f'test_si/metrics-{ratio}x.json'
    try:
        r = compute_si_robust(file)
        results.append({
            'ratio': ratio,
            'N': r['N'],
            'CF': r['CF'],
            'SI_original': r['SI_original'],
            'SI_robust': r['SI_robust']
        })
    except FileNotFoundError:
        print(f"Warning: {file} not found, skipping...")
        continue

if not results:
    print("No metrics files found. Run experiments first.")
    sys.exit(1)

print(f"{'Load':>6} | {'CF':>5} | {'SI_original':>12} | {'SI_robust':>10}")
print(f"{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*10}")
for r in results:
    print(f"{r['ratio']:>6.3f} | {r['CF']:>5.3f} | {r['SI_original']:>12.3f} | {r['SI_robust']:>10.3f}")

print("\n=== Monotonicity Analysis: SI_original ===")
violations_orig = 0
for i in range(1, len(results)):
    prev = results[i-1]
    curr = results[i]
    if curr['SI_original'] < prev['SI_original']:
        print(f"  ✗ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_original']:.3f} → {curr['SI_original']:.3f}")
        violations_orig += 1
    else:
        delta = curr['SI_original'] - prev['SI_original']
        print(f"  ✓ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_original']:.3f} → {curr['SI_original']:.3f} (+{delta:.3f})")

print(f"\nTotal violations: {violations_orig}/{len(results)-1}")

print("\n=== Monotonicity Analysis: SI_robust ===")
violations_robust = 0
for i in range(1, len(results)):
    prev = results[i-1]
    curr = results[i]
    if curr['SI_robust'] < prev['SI_robust']:
        print(f"  ✗ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_robust']:.3f} → {curr['SI_robust']:.3f}")
        violations_robust += 1
    else:
        delta = curr['SI_robust'] - prev['SI_robust']
        print(f"  ✓ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_robust']:.3f} → {curr['SI_robust']:.3f} (+{delta:.3f})")

print(f"\nTotal violations: {violations_robust}/{len(results)-1}")

print("\n" + "="*60)
if violations_robust < violations_orig:
    print(f"✓ SI_robust reduces monotonicity violations from {violations_orig} to {violations_robust}")
    if violations_robust == 0:
        print("  SI_robust is PERFECTLY MONOTONIC!")
else:
    print(f"✗ SI_robust does not improve monotonicity ({violations_robust} violations vs {violations_orig} original)")
