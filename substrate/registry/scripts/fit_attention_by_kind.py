#!/usr/bin/env python3
"""Fit the decode-attention primitive SEPARATELY PER ATTENTION KIND.

    python scripts/fit_attention_by_kind.py --sku h200_sxm --chip h200
    python scripts/fit_attention_by_kind.py --all

# Why per kind

`blis-latency-kernel` prices every attention kind with one law,
`floor + kv_bytes / rate`, fitted per part. NVIDIA collects the kinds separately because they
are different kernels reading different bytes: full attention reads the whole context, a
sliding window reads a bounded slice of it, MLA reads one latent KV head, and sparse MLA reads
only the positions an index selects. One law across four kernels cannot be right for all four,
and the error it makes is SCATTER -- point-to-point inaccuracy in different directions -- which
is exactly the residual this project measured against AISimulate: identical mean log-error
(+0.0461 against +0.0462) and 1.5x the spread (0.1677 against 0.1119).

# The form, unchanged

    latency = floor + kv_bytes / rate

Same functional form the registry already carries, so a per-kind coefficient is a drop-in for
the unsuffixed one and a deployment with no per-kind fit prices exactly as before. What changes
is that `floor` and `rate` are fitted on the rows describing ONE kind.

# The lane rule

Every family carries `kernel_source`, and latencies for one shape differ by up to 5x between
lanes -- measured on the b200 MoE sweep, where pooling lanes gave a
prediction-over-measurement ratio of 0.309 against 0.625 for the lane a latency-sensitive
engine picks. So this fitter never pools: it fits the lane with the most rows and records which
one, or the lane the caller names. Where a collection has a single lane the choice is moot,
which is the case for the h200 trtllm collection the committed unsuffixed fit used.

# What is NOT fitted, and why

MLA and sparse MLA carry no `num_key_value_heads` or `head_dim` column: their byte count is a
property of the architecture, not of a width in the sweep. The `kv_bytes` this form needs
cannot be computed from those rows without assuming a model, so those kinds are REPORTED as
unfittable by this script rather than fitted from a guess. Their row counts are printed so the
gap is visible.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "attention_table_map", Path(__file__).with_name("attention_table_map.py")
)
atm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(atm)

# The same grids the unsuffixed fitter uses, so a per-kind fit is comparable to it.
FLOOR_GRID = [x * 0.5 for x in range(2, 81)]
# The rate is fitted as a FRACTION of the part's datasheet HBM bandwidth, capped at 1.0. A
# kernel cannot read faster than the memory delivers, so a fit above 1.0 is not a measurement
# of speed -- it means the byte count in `kv_bytes` is too large for those rows, which is
# exactly the case for a windowed kernel: it reads a bounded window, not the whole context.
# Leaving the grid open let the sliding-window fit land at 2.00 of peak, a physically
# impossible value that would have shipped as a coefficient.
FRACTION_GRID = [x * 0.02 for x in range(5, 51)]

# A fit on fewer points than this is refused: a handful of rows can be matched by many
# (floor, rate) pairs and the choice would be arbitrary.
MIN_POINTS = 200

# Kinds whose byte count is computable from THIS sweep's own columns.
#
# `mla` and `sparse_mla` are absent here and that remains correct for this script: the
# generation_attention tables carry no KV-head or head-dimension column for them, because
# an MLA kernel's byte count is a property of the checkpoint -- one latent vector per
# token of width kv_lora_rank + qk_rope_head_dim -- rather than of a head width the sweep
# varies.
#
# It is NOT correct as a statement about the data, and an earlier revision of
# docs/methodology.md read it that way. The vLLM MLA MODULE tables
# (mla/vllm/*/mla_generation_module_perf.parquet, 137,874 rows across parts) carry `model`
# and `architecture`, so the geometry is resolvable from blis-catalog and the fit is
# well-posed. scripts/fit_attention_mla.py does it and relane_attention_mla.py commits it
# for six parts. This constant means "not from these columns", not "not from this
# project's data".
FITTABLE = ("gqa", "swa")


def kv_bytes(row: dict) -> float:
    """Bytes the kernel reads for this shape: K and V, every position read, every KV head.

    For full attention the positions read are the whole context. For a SLIDING WINDOW they are
    bounded by the window: a kernel with a 128-token window over a 32768-token context reads
    128 positions, not 32768. Using the context for both is what made the windowed fit land at
    2.00x the datasheet bandwidth -- the form was being asked to explain a 256x byte
    overstatement with a rate, and the only way to do that is to exceed physics.

    `window_size` is the column the sweep records it in, and it is the whole reason gqa and swa
    are fitted separately rather than pooled.
    """
    width = 1.0 if row["kv_cache_dtype"] == "fp8" else 2.0
    positions = row["context"]
    if row.get("window") and row["window"] > 0:
        positions = min(positions, row["window"])
    return row["batch"] * positions * 2 * row["kv_heads"] * row["head_dim"] * width


def fit(points: list[tuple[float, float]], peak: float) -> tuple[float, float, float]:
    """Grid-search (floor us, bandwidth fraction) minimising mean absolute log ratio."""
    best = None
    for floor_us in FLOOR_GRID:
        floor = floor_us * 1e-6
        for fraction in FRACTION_GRID:
            rate = peak * fraction
            err = math.exp(
                sum(abs(math.log((floor + b / rate) / lat)) for b, lat in points)
                / len(points)
            )
            if best is None or err < best[0]:
                best = (err, floor_us, fraction)
    assert best is not None
    return best


def naive(points: list[tuple[float, float]], peak: float) -> float:
    """Error of the same form with no floor and no derate: what the fit is improving on."""
    return math.exp(
        sum(abs(math.log((b / peak) / lat)) for b, lat in points) / len(points)
    )


def fit_kind(data: str, sku: str, chip: str, kind: str, catalog: Path,
             lane: str | None, collection: str | None = None) -> dict | None:
    rows = atm.rows_for_kind(data, sku, kind, collection=collection)
    if not rows:
        return {"kind": kind, "status": "no rows"}

    # One lane. The most populous unless the caller names one.
    if lane is None:
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["lane"]] = counts.get(r["lane"], 0) + 1
        lane = max(counts, key=lambda k: counts[k])
    rows = [r for r in rows if r["lane"] == lane]

    if kind not in FITTABLE:
        return {"kind": kind, "status": "not fittable from these columns",
                "rows": len(rows), "lane": lane}
    rows = [r for r in rows if r["kv_heads"] and r["head_dim"]]
    if len(rows) < MIN_POINTS:
        return {"kind": kind, "status": f"only {len(rows)} usable rows, need {MIN_POINTS}",
                "rows": len(rows), "lane": lane}

    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = facts["BwPeakTBs"] * 1e12
    points = [(kv_bytes(r), r["latency_ms"] * 1e-3) for r in rows]
    err, floor_us, fraction = fit(points, peak)
    return {
        "kind": kind, "status": "ok", "lane": lane, "rows": len(rows),
        "floor_us": floor_us, "fraction": fraction, "rate_bytes_per_us": peak * fraction / 1e6,
        "geo_err": err, "naive_err": naive(points, peak), "peak_tbs": peak / 1e12,
    }


# The committed unsuffixed h200 coefficients, and the row count they were fitted on. A per-kind
# fitter that cannot reproduce these on the same rows is not a refactor of the old fit, it is a
# different fit -- and then nothing it produces for the other kinds can be trusted either.
#
# These are read from the registry rather than typed in, so a re-fit of the unsuffixed entry
# updates the gate with it.
REGRESSION_SKU = "h200_sxm"
REGRESSION_CHIP = "h200"
REGRESSION_KIND = "gqa"
REGRESSION_ROWS = 40367


def committed(catalog_unused: Path, chip: str) -> tuple[float, float]:
    """The committed unsuffixed decode-attention coefficients for one chip."""
    doc = yaml.safe_load(
        (Path(__file__).parent.parent / "coefficients" / "cost-model-attention.yaml").read_text()
    )
    floor = rate = None
    for entry in doc["coefficients"]:
        (name, body), = entry.items()
        if chip not in body["scope"].get("hardware", []):
            continue
        if name == "attention_decode_floor":
            floor = body["value"]
        elif name == "attention_decode_rate":
            rate = body["value"]
    if floor is None or rate is None:
        raise SystemExit(f"no committed unsuffixed decode coefficients for {chip}")
    return floor, rate


def check_regression(data: str, catalog: Path) -> int:
    """Refit the gqa kind on the pinned collection and compare with what is committed."""
    want_floor, want_rate = committed(catalog, REGRESSION_CHIP)
    got = fit_kind(data, REGRESSION_SKU, REGRESSION_CHIP, REGRESSION_KIND, catalog, None)
    if got is None or got["status"] != "ok":
        print(f"FAIL: gqa did not fit: {got and got['status']}", file=sys.stderr)
        return 1

    problems = []
    if got["rows"] != REGRESSION_ROWS:
        problems.append(
            f"row count {got['rows']:,} against the {REGRESSION_ROWS:,} the committed fit "
            f"used; the collection pin or the kind filter has changed the row set")
    # The grid resolution is 0.5 us on the floor and 0.02 of peak on the rate, so equality is
    # the right bar: the same rows on the same grid must land on the same cell.
    if abs(got["floor_us"] - want_floor) > 1e-9:
        problems.append(f"floor {got['floor_us']} against a committed {want_floor}")
    if abs(got["rate_bytes_per_us"] - want_rate) > 1.0:
        problems.append(f"rate {got['rate_bytes_per_us']:,.0f} against a committed "
                        f"{want_rate:,.0f}")

    print(f"regression check: gqa on {REGRESSION_CHIP} ({REGRESSION_SKU})")
    print(f"  committed floor {want_floor} us, rate {want_rate:,.0f} B/us")
    print(f"  refitted  floor {got['floor_us']} us, rate {got['rate_bytes_per_us']:,.0f} B/us"
          f"  n={got['rows']:,} lane={got['lane']} geo-err {got['geo_err']:.3f}x")
    if problems:
        print("FAIL:", file=sys.stderr)
        for pr in problems:
            print(f"  {pr}", file=sys.stderr)
        return 1
    print("PASS: the per-kind fitter reproduces the committed unsuffixed fit on the same rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=atm.DEFAULT_DATA)
    ap.add_argument("--catalog", default=os.environ.get(
        "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog")))
    ap.add_argument("--sku")
    ap.add_argument("--chip")
    ap.add_argument("--lane", default=None, help="kernel_source to fit; default the most populous")
    ap.add_argument("--collection", default=None,
                    help="framework/version collection, overriding the family pin in "
                         "attention_table_map.COLLECTIONS (e.g. vllm/0.25.0)")
    ap.add_argument("--all", action="store_true", help="every SKU with data")
    ap.add_argument("--check-regression", action="store_true",
                    help="assert the gqa fit reproduces the committed unsuffixed coefficients")
    args = ap.parse_args()

    if args.check_regression:
        return check_regression(args.data, Path(args.catalog))

    if args.all:
        targets = [(s, c) for s, c in sorted(atm.SKUS.items())
                   if Path(args.data, s).is_dir()]
    elif args.sku and args.chip:
        targets = [(args.sku, args.chip)]
    else:
        print("give --sku and --chip, or --all", file=sys.stderr)
        return 2

    rc = 0
    for sku, chip in targets:
        print(f"=== {chip} ({sku})")
        for kind in atm.KINDS:
            r = fit_kind(args.data, sku, chip, kind, Path(args.catalog), args.lane,
                         args.collection)
            if r is None or r["status"] != "ok":
                print(f"  {kind:12s} {r['status']}"
                      + (f"  ({r.get('rows', 0):,} rows, lane {r.get('lane')})"
                         if r.get("rows") else ""))
                continue
            print(f"  {kind:12s} floor {r['floor_us']:5.1f}us  "
                  f"rate {r['rate_bytes_per_us']:12,.0f} B/us "
                  f"({r['fraction']:.2f} of {r['peak_tbs']:.3f} TB/s)  "
                  f"geo-err {r['geo_err']:.3f}x (naive {r['naive_err']:.1f}x)  "
                  f"n={r['rows']:,} lane={r['lane']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
