#!/usr/bin/env python3
from compute_composite import compute_composite

results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    file = f'metrics-{ratio}x.json'
    try:
        r = compute_composite(file)
        results.append({
            'ratio': ratio,
            'N': r['N'],
            'completed': r['completions_within_T'],
            'timedout': r['N'] - r['completions_within_T'] - sum(1 for _ in range(r['N'])),  # Will calculate from JSON
            'score': r['saturation_score'],
            'RD': r['signals']['rate_deficit'],
            'GR': r['signals']['growth_ratio'],
            'SQ': r['signals']['standing_queue']
        })
    except Exception as e:
        print(f"# Error processing {ratio}x: {e}", file=sys.stderr)

print("## Composite Detector with 11-Minute Timeout")
print("")
print("**Configuration:**")
print("- Arrival window: 10 min (600s)")
print("- Request timeout: 11 min (660s)")
print("- Horizon: 21 min (1260s)")
print("")
print("```")
print(f"  Load | Requests | Completed |  Score |     RD |     GR |   SQ")
print(f"-------+----------+-----------+--------+--------+--------+------")
for r in results:
    print(f" {r['ratio']:>5.3f} | {r['N']:>8} | {r['completed']:>9} | {r['score']:>6.4f} | {r['RD']:>6.4f} | {r['GR']:>6.4f} | {r['SQ']:>4.1f}")
print("```")

if len(results) > 1:
    print("\n### Monotonicity Check:")
    print("```")
    violations = 0
    for i in range(1, len(results)):
        prev, curr = results[i-1], results[i]
        if curr['score'] < prev['score']:
            print(f"✗ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['score']:.4f} → {curr['score']:.4f}")
            violations += 1
        else:
            delta = curr['score'] - prev['score']
            print(f"✓ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['score']:.4f} → {curr['score']:.4f} (+{delta:.4f})")
    
    print(f"\nTotal violations: {violations}/{len(results)-1}")
    print("```")
    
    # Find the peak
    max_score = max(r['score'] for r in results)
    peak_loads = [r['ratio'] for r in results if r['score'] == max_score]
    print(f"\nPeak score: {max_score:.4f} at loads: {', '.join(f'{l:.3f}x' for l in peak_loads)}")

import sys
sys.path.insert(0, '.')
