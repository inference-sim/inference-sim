#!/usr/bin/env python3
"""Fit host_launch_eager_per_kernel for DeepSeek-V4-Pro per chip from fit-role TTFT residuals (report-only).

THE FORM. The kernel charges the eager per-launch term once per planned launch of a non-graphed step, so a
unit's predicted TTFT is linear in the term's value v for a fixed unit (more precisely piecewise linear:
the step is the max of device and host time). Each fit unit is scored by the dashboard scorer at several values
of v, each arm a registry with one model-scoped host_launch_eager_per_kernel entry. This script interpolates
each unit's predicted TTFT between the arms and writes the v on a 2 us grid that minimises the mean
|pred / measured - 1| (the gate metric, TTFT MAPE) over the chip's units. A chip is reported WRITE only with at
least MIN_UNITS fit units and a bootstrap interval (BOOTSTRAPS resamples, fixed seed) narrower than MAX_WIDTH.

INPUT. --arm VALUE=points.csv (repeat; the scorer's points.csv, one row per unit, same units in every arm);
--fit-units is the published fit list (catalog/ledger/published/fit.yaml): a unit not on it is never used (D7).
--model selects the model column (default deepseek-v4-pro). The values are read from the csv; the units of the
rows are the scorer's microseconds.

    python scripts/fit_deepseek_v4_pro_eager.py --fit-units fit.yaml --arm 30=a30.csv --arm 60=a60.csv ...
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
from collections import defaultdict

MIN_UNITS = 30
MAX_WIDTH = 40.0   # us
BOOTSTRAPS = 400
SEED = 1


def load(arms, fit, model):
    units = defaultdict(dict)
    for v, path in arms:
        for r in csv.DictReader(open(path)):
            if r["point_id"] in fit and r["model"] == model:
                units[r["point_id"]][v] = r
    xs = sorted(v for v, _ in arms)
    out = []
    for pid, d in units.items():
        if set(d) != set(xs):
            continue
        r0 = d[xs[0]]
        out.append({"chip": r0["gpu"], "meas": float(r0["meas_ttft_us"]), "xs": xs,
                    "ys": [float(d[x]["pred_ttft_us"]) for x in xs]})
    return out


def pred(u, v):
    xs, ys = u["xs"], u["ys"]
    i = 1
    while i < len(xs) - 1 and v > xs[i]:
        i += 1
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (v - xs[i - 1]) / (xs[i] - xs[i - 1])


def best(units):
    grid = range(2 * -(-int(units[0]["xs"][0]) // 2), int(units[0]["xs"][-1]) + 1, 2)   # even values, 2 us step
    return min(grid, key=lambda v: statistics.fmean(abs(pred(u, v) / u["meas"] - 1) for u in units))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-units", required=True)
    ap.add_argument("--arm", action="append", required=True, help="VALUE=points.csv")
    ap.add_argument("--model", default="deepseek-v4-pro")
    a = ap.parse_args(argv[1:])
    fit = set(re.findall(r"^  - ([0-9a-f]{64})", open(a.fit_units).read(), re.M))
    arms = [(float(s.split("=")[0]), s.split("=", 1)[1]) for s in a.arm]
    units = load(arms, fit, a.model)
    by = defaultdict(list)
    for u in units:
        by[u["chip"]].append(u)
    for chip in sorted(by):
        us = by[chip]
        v = best(us)
        rng = random.Random(SEED)
        boot = sorted(best([rng.choice(us) for _ in us]) for _ in range(BOOTSTRAPS))
        lo, hi = boot[int(0.025 * BOOTSTRAPS)], boot[int(0.975 * BOOTSTRAPS) - 1]
        write = len(us) >= MIN_UNITS and (hi - lo) < MAX_WIDTH
        print(f"{chip}: {'WRITE' if write else 'NOT WRITTEN'} n={len(us)} v={v} us ci95=[{lo}, {hi}] "
              f"TTFT MAPE at v {100 * statistics.fmean(abs(pred(u, v) / u['meas'] - 1) for u in us):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
