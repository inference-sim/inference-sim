#!/usr/bin/env python3
import sys
sys.path.insert(0, 'test_si')
from compute_si_robust import compute_si_robust

print("Detailed comparison: SI_original vs SI_robust")
print("")
print(f"{'Load':>6} | {'EndQ':>6} | {'TailMeanQ':>10} | {'MeanQ':>7} | {'SI_orig':>8} | {'SI_robust':>10} | {'CF':>5}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*5}")

for ratio in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    file = f'test_si/metrics-{ratio}x.json'
    try:
        r = compute_si_robust(file)
        print(f"{ratio:>6.3f} | {r['EndQ']:>6} | {r['TailMeanQ']:>10.2f} | {r['MeanQ']:>7.2f} | {r['SI_original']:>8.3f} | {r['SI_robust']:>10.3f} | {r['CF']:>5.3f}")
    except FileNotFoundError:
        pass
