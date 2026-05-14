#!/usr/bin/env python3
from compute_si import compute_si
from compute_si_robust import compute_si_robust

print("## SI (Saturation Index) with 11-Minute Timeout\n")
print("```")
print(f"  Load |   SI   |   CF   | Classification")
print(f"-------+--------+--------+-------------------")

si_results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    try:
        r = compute_si(f'metrics-{ratio}x.json')
        si_results.append({
            'ratio': ratio,
            'SI': r['SI'],
            'CF': r['CF'],
            'class': r['classification']
        })
        print(f" {ratio:>5.3f} | {r['SI']:>6.3f} | {r['CF']:>6.3f} | {r['classification']}")
    except Exception as e:
        print(f"# Error: {e}")

print("```")

if len(si_results) > 1:
    print("\n### Monotonicity:")
    print("```")
    violations = 0
    for i in range(1, len(si_results)):
        prev, curr = si_results[i-1], si_results[i]
        if curr['SI'] < prev['SI']:
            violations += 1
            print(f"✗ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI']:.3f} → {curr['SI']:.3f}")
        else:
            print(f"✓ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI']:.3f} → {curr['SI']:.3f}")
    
    print(f"\nViolations: {violations}/{len(si_results)-1}")
    print("```")

print("\n" + "="*70)
print("\n## SI_robust (Tail-Averaged) with 11-Minute Timeout\n")
print("```")
print(f"  Load | SI_robust |   CF   | Classification")
print(f"-------+-----------+--------+-------------------")

sir_results = []
for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    try:
        r = compute_si_robust(f'metrics-{ratio}x.json')
        sir_results.append({
            'ratio': ratio,
            'SI_robust': r['SI_robust'],
            'CF': r['CF'],
            'class': r['classification_robust']
        })
        print(f" {ratio:>5.3f} |   {r['SI_robust']:>6.3f} | {r['CF']:>6.3f} | {r['classification_robust']}")
    except Exception as e:
        print(f"# Error: {e}")

print("```")

if len(sir_results) > 1:
    print("\n### Monotonicity:")
    print("```")
    violations = 0
    for i in range(1, len(sir_results)):
        prev, curr = sir_results[i-1], sir_results[i]
        if curr['SI_robust'] < prev['SI_robust']:
            violations += 1
            print(f"✗ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_robust']:.3f} → {curr['SI_robust']:.3f}")
        else:
            print(f"✓ {prev['ratio']:.3f} → {curr['ratio']:.3f}: {prev['SI_robust']:.3f} → {curr['SI_robust']:.3f}")
    
    print(f"\nViolations: {violations}/{len(sir_results)-1}")
    print("```")
