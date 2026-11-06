#!/usr/bin/env python3
"""Hold out one sliding-window WIDTH and test the fit on it.

WHY THIS SCRIPT EXISTS. `fit_attention_by_kind.py` reports the error of a fit on the rows
it was fitted to. That is a residual, not a generalization test: a two-parameter form over
31,890 points will look good on its own data whether or not it describes the kernel. The
sliding-window sweep varies `window_size` over five widths, and the window is the whole
reason this kind is fitted apart from full attention — `kv_bytes` reads
`min(context, window)` positions, so a held-out width is a byte count the fit never saw.
Holding one out and scoring it is therefore a test along the dimension that defines the
kernel, which is what §4 of docs/methodology.md asks a split to do.

WHAT IT DOES NOT DO. It never writes a coefficient and never selects a lane. It reports,
so a relane can be defended with a held-out number rather than a residual.

THE WIDTHS ARE NOT BALANCED, and the report says so per fold: on h200 vllm/0.25.0 the
widths carry 26,160 / 1,972 / 1,334 / 1,256 / 1,168 rows at 128 / 1024 / 2048 / 8192 / 512.
Holding out 128 removes 82% of the data, so that fold tests a fit trained on an eighth of
the sweep; the other four folds each hold out 3-6%. Both facts are printed rather than
averaged away, because a mean over folds this uneven would describe none of them.

Reuses `attention_table_map.rows_for_kind` and `fit_attention_by_kind.{kv_bytes,fit}`
unchanged, so the fold and the shipped fit cannot disagree about which rows belong to a
kind or how bytes are counted.

Usage:
    python scripts/holdout_attention_swa.py --sku h200_sxm --chip h200 \
        --collection vllm/0.25.0
    python scripts/holdout_attention_swa.py --all --collection vllm/0.25.0
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fak = _load("fit_attention_by_kind")
atm = _load("attention_table_map")

DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))
PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100"), ("b200_sxm", "b200"),
         ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("l40s", "l40s")]


def geo_err(points, floor_us: float, fraction: float, peak: float) -> float:
    """The fitter's own objective: geometric mean of |log(predicted / measured)|."""
    floor = floor_us * 1e-6
    rate = peak * fraction
    return math.exp(
        sum(abs(math.log((floor + b / rate) / lat)) for b, lat in points) / len(points)
    )


def run(data: str, sku: str, chip: str, collection: str, catalog: Path) -> int:
    rows = atm.rows_for_kind(data, sku, "swa", collection=collection)
    if not rows:
        print(f"{chip:12} no sliding-window rows at {collection}")
        return 0
    # One lane, exactly as fit_kind picks it: the most populous.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["lane"]] = counts.get(r["lane"], 0) + 1
    lane = max(counts, key=lambda k: counts[k])
    rows = [r for r in rows if r["lane"] == lane
            and r["kv_heads"] and r["head_dim"] and r["window"]]
    if not rows:
        print(f"{chip:12} no usable windowed rows on lane {lane}")
        return 0

    facts = __import__("yaml").safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = facts["BwPeakTBs"] * 1e12
    widths = sorted({r["window"] for r in rows})

    pts_all = [(fak.kv_bytes(r), r["latency_ms"] * 1e-3) for r in rows]
    err_all, floor_all, frac_all = fak.fit(pts_all, peak)
    print(f"\n=== {chip} ({sku}) lane={lane} collection={collection}")
    print(f"  all widths      n={len(rows):6}  floor={floor_all:5.1f}us "
          f"frac={frac_all:.2f}  residual geo-err {err_all:.3f}x")
    print(f"  widths present  {widths}")
    print(f"  {'held-out w':>11} {'train n':>8} {'test n':>7} {'train err':>10} "
          f"{'TEST err':>9} {'floor':>7} {'frac':>6}  {'% held':>7}")

    test_errs = []
    for w in widths:
        train = [r for r in rows if r["window"] != w]
        test = [r for r in rows if r["window"] == w]
        if len(train) < fak.MIN_POINTS or not test:
            print(f"  {w:>11} too few training rows ({len(train)}) to fit")
            continue
        p_tr = [(fak.kv_bytes(r), r["latency_ms"] * 1e-3) for r in train]
        p_te = [(fak.kv_bytes(r), r["latency_ms"] * 1e-3) for r in test]
        err_tr, floor, frac = fak.fit(p_tr, peak)
        err_te = geo_err(p_te, floor, frac, peak)
        test_errs.append(err_te)
        print(f"  {w:>11} {len(train):8} {len(test):7} {err_tr:9.3f}x "
              f"{err_te:8.3f}x {floor:6.1f}us {frac:6.2f} "
              f"{100 * len(test) / len(rows):6.1f}%")
    if test_errs:
        print(f"  worst held-out width: {max(test_errs):.3f}x   "
              f"median across folds: {statistics.median(test_errs):.3f}x")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--collection", default="vllm/0.25.0")
    ap.add_argument("--sku")
    ap.add_argument("--chip")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv[1:])
    parts = PARTS if args.all else (
        [(args.sku, args.chip)] if args.sku and args.chip else None)
    if parts is None:
        ap.error("give --sku and --chip, or --all")
    rc = 0
    for sku, chip in parts:
        rc |= run(args.data, sku, chip, args.collection, Path(args.catalog))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
