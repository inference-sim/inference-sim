#!/usr/bin/env python3
"""Fit host_launch_per_kernel per chip from concurrency 1-4 fit-role units.

THE FORM. The kernel charges the per-launch term once per planned launch of the step, so a unit's predicted
TPOT is linear in it: pred(g) = pred(0) + n * g, with n the unit's planned launches. A unit scored twice by
the same build, once at g = 3.2 us and once at g = 0, gives n = (pred(3.2) - pred(0)) / 3.2 us and pred(0).
The fit is the g on a 0.05 us grid that minimises the mean |pred(g) / measured - 1| over a chip's units.

THE RULE FOR WRITING A VALUE (accuracy/preregistrations.md K1, VAL-11). A chip's value is emitted only when it has
at least MIN_UNITS fit units in at least MIN_FAMILIES model families and the bootstrap interval (2.5-97.5
percentiles of BOOTSTRAPS resamples of the units, fixed seed) is narrower than MAX_WIDTH us and stays above
the grid's zero edge (units whose error is mostly not the launch term pull a mean-absolute-error fit to zero
with a narrow interval). Otherwise the chip
is reported "not written" with its interval: a pooled value that one family drives is a scale error of that
family, not a property of the chip, and is not written. The leave-one-family-out values are printed so a reader
sees which family moves it.

INPUT. Two CSVs of the same units, `point_id,family,chip,concurrency,measured_us,pred_us` (the dashboard
scorer's points.csv joined to the unit list), one scored at 3.2 us and one at 0; `--fit-units` is the published
fit list (catalog/ledger/published/fit.yaml). A unit not on that list is never used (D7).

Usage:
    python scripts/fit_host_launch_per_chip.py --at-3p2 A.csv --at-0 B.csv --fit-units fit.yaml [--chip b200 ...]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

MIN_UNITS = 30
MIN_FAMILIES = 3
MAX_WIDTH = 1.6
BOOTSTRAPS = 400
SEED = 1
COMMITTED = 3.2
GRID = [round(i * 0.05, 2) for i in range(0, 101)]


def ape(u: dict, g: float) -> float:
    return abs((u["pred0"] + u["n"] * g) / u["measured"] - 1) * 100


def best(units: list[dict]) -> float:
    return min(GRID, key=lambda g: statistics.fmean(ape(u, g) for u in units))


def build_units(at_3p2: list[dict], at_0: list[dict], fit: set[str], max_conc: int = 4) -> list[dict]:
    zero = {r["point_id"]: r for r in at_0}
    out = []
    for r in at_3p2:
        z = zero.get(r["point_id"])
        if z is None or r["point_id"] not in fit or int(r["concurrency"]) > max_conc:
            continue
        out.append({"chip": r["chip"], "family": r["family"], "measured": float(r["measured_us"]),
                    "pred0": float(z["pred_us"]), "n": (float(r["pred_us"]) - float(z["pred_us"])) / COMMITTED})
    return out


def fit_chip(units: list[dict]) -> dict:
    """The chip's value and interval, and whether the rule lets it be written."""
    fams = Counter(u["family"] for u in units)
    g = best(units)
    rng = random.Random(SEED)
    boot = sorted(best([rng.choice(units) for _ in units]) for _ in range(BOOTSTRAPS))
    lo, hi = boot[int(0.025 * BOOTSTRAPS)], boot[int(0.975 * BOOTSTRAPS) - 1]
    lofo = {f: best([u for u in units if u["family"] != f]) for f in fams if len(units) - fams[f] >= 10}
    write = len(units) >= MIN_UNITS and len(fams) >= MIN_FAMILIES and (hi - lo) < MAX_WIDTH and lo > GRID[1]
    return {"n": len(units), "families": dict(fams), "g": g, "lo": lo, "hi": hi, "write": write, "lofo": lofo}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at-3p2", required=True)
    ap.add_argument("--at-0", required=True)
    ap.add_argument("--fit-units", required=True)
    ap.add_argument("--chip", action="append", default=[])
    a = ap.parse_args(argv[1:])
    fit = set(re.findall(r"^  - ([0-9a-f]{64})", open(a.fit_units).read(), re.M))
    units = build_units(list(csv.DictReader(open(a.at_3p2))), list(csv.DictReader(open(a.at_0))), fit)
    by = defaultdict(list)
    for u in units:
        by[u["chip"]].append(u)
    for chip in a.chip or sorted(by):
        r = fit_chip(by[chip])
        verdict = "WRITE" if r["write"] else "NOT WRITTEN"
        print(f"{chip}: {verdict} n={r['n']} families={r['families']} g={r['g']:.2f} us "
              f"ci95=[{r['lo']:.2f}, {r['hi']:.2f}]  leave-one-family-out={ {k: round(v, 2) for k, v in r['lofo'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
