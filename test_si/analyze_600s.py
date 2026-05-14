#!/usr/bin/env python3
import sys
sys.path.insert(0, 'test_si')
from compute_si import compute_si

results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    file = f'test_si/metrics-{ratio}x.json'
    try:
        r = compute_si(file)
        results.append({
            'ratio': ratio,
            'N': r['N'],
            'completed': r['completions_within_T'],
            'CF': r['CF'],
            'SI': r['SI']
        })
    except FileNotFoundError:
        print(f"Warning: {file} not found")
        continue

print(f"{'Load':>6} | {'Requests':>8} | {'Completed':>9} | {'CF':>5} | {'SI':>6}")
print(f"{'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*5}-+-{'-'*6}")
for r in results:
    print(f"{r['ratio']:>6.3f} | {r['N']:>8} | {r['completed']:>9} | {r['CF']:>5.3f} | {r['SI']:>6.3f}")

print("\n=== Monotonicity Analysis ===")
violations = 0
for i in range(1, len(results)):
    prev = results[i-1]
    curr = results[i]
    if curr['SI'] < prev['SI']:
        print(f"  ✗ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: SI decreased from {prev['SI']:.3f} to {curr['SI']:.3f}")
        violations += 1
    else:
        delta = curr['SI'] - prev['SI']
        print(f"  ✓ Load {prev['ratio']:.3f} → {curr['ratio']:.3f}: SI increased from {prev['SI']:.3f} to {curr['SI']:.3f} (+{delta:.3f})")

print(f"\nTotal monotonicity violations: {violations} out of {len(results)-1} transitions")
if violations == 0:
    print("✓ SI is perfectly monotonic!")
