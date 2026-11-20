#!/usr/bin/env python3
"""Score the kernel's band against FPM's MIXED prefill+decode rows.

WHY THIS EXISTS. FPM is 79.2% mixed rows -- 67,736 of 85,484, against 12,522 pure decode
and 5,226 pure prefill -- and until now only the pure regimes had been scored. The band
choice in docs/band-selection.md used 219 pure-DECODE points; the 1.273 prefill ratio in
the calibration companion used pure-PREFILL rows. A batch carrying both kinds is the regime
the kernel is least tested on, and it is where a regime-interaction error shows: the
chunked-prefill pair count was wrong by 1.995x and `kernel.go:285` records that FPM's mixed
rows are what exposed it, because at `Computed == 0` the right and wrong forms agree.

WHAT IT DOES NOT DO. It never fits or writes a coefficient, and it refuses InferenceX paths
through the same guard `select_overlap_band.py` uses. FPM's role in this registry is
validation and model selection, not fitting.

THE TWO UNIT TRAPS, both load-bearing here:
  * `total_kv_read_tokens` and `total_prefill_tokens` are summed OVER THE BATCH. The
    kernel takes PER-REQUEST shapes, so both divide by `batch_size`. Reading them as
    per-request overstates a batch-4 row's work fourfold.
  * `latency_ms` is milliseconds, and so is the band CSV this reads. Neither is seconds.

The band CSV comes from `blis-latency-kernel/cmd/bandprobe`, which owns step composition.
This script only joins and reports; it does not re-implement the step model.

Usage:
    # 1. emit the grid this script needs, from the FPM artifact
    python scripts/score_fpm_mixed.py --emit-grid FPM.parquet > grid.csv
    # 2. price it with the kernel (from the blis-latency-kernel checkout)
    go run ./cmd/bandprobe -scenario SCEN.yaml < grid.csv > band.csv
    # 3. score
    python scripts/score_fpm_mixed.py --band band.csv --fpm FPM.parquet
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _guard():
    spec = importlib.util.spec_from_file_location(
        "select_overlap_band", HERE / "select_overlap_band.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.refuse_evaluation_data


refuse_evaluation_data = _guard()


def mixed_rows(fpm: Path) -> list[dict]:
    """FPM's mixed rows, folded to PER-REQUEST shapes and medianed per grid point.

    A median rather than a mean, for the reason select_overlap_band.decode_rows gives:
    the sidecar records one measurement repeat and no warmup, so single rows carry
    run-to-run scatter a mean would chase.
    """
    import pyarrow.parquet as pq

    d = pq.read_table(fpm).to_pydict()
    buckets: dict[tuple[int, int, int], list[float]] = {}
    for i in range(len(d["latency_ms"])):
        pre = d["total_prefill_tokens"][i] or 0
        kv = d["total_kv_read_tokens"][i] or 0
        if pre <= 0 or kv <= 0:
            continue                      # not a mixed row
        batch = d["batch_size"][i]
        if batch <= 0 or d["latency_ms"][i] <= 0:
            continue
        # Both totals are summed over the batch; the kernel wants per-request.
        per_pre, per_ctx = pre // batch, kv // batch
        if per_pre <= 0 or per_ctx <= 0:
            continue
        buckets.setdefault((batch, per_pre, per_ctx), []).append(d["latency_ms"][i])
    return [{"batch": b, "prefill": p, "context": c,
             "latency_ms": statistics.median(v), "rows": len(v)}
            for (b, p, c), v in sorted(buckets.items())]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpm", type=Path, required=True)
    ap.add_argument("--emit-grid", action="store_true",
                    help="write the bandprobe input CSV for this artifact and exit")
    ap.add_argument("--band", type=Path,
                    help="bandprobe output for the same grid")
    ap.add_argument("--max-points", type=int, default=0,
                    help="cap the grid (0 = no cap), for a quick pass")
    args = ap.parse_args(argv[1:])

    refuse_evaluation_data(args.fpm)
    grid = mixed_rows(args.fpm)
    if args.max_points:
        grid = grid[:args.max_points]
    if not grid:
        print(f"{args.fpm}: no mixed rows", file=sys.stderr)
        return 1

    if args.emit_grid:
        for g in grid:
            print(f"{g['batch']},{g['prefill']},{g['context']}")
        return 0

    if args.band is None:
        ap.error("give --band, or --emit-grid")
    refuse_evaluation_data(args.band)

    import csv
    priced: dict[tuple[int, int, int], tuple[float, float, str]] = {}
    with args.band.open() as fh:
        for r in csv.DictReader(fh):
            priced[(int(r["batch"]), int(r["prefill_tokens"]), int(r["context"]))] = (
                float(r["overlap_us"]) / 1000.0,
                float(r["nooverlap_us"]) / 1000.0,
                r.get("bottleneck", ""))

    rows = []
    for g in grid:
        key = (g["batch"], g["prefill"], g["context"])
        if key not in priced:
            continue
        ov, no, bn = priced[key]
        rows.append({**g, "overlap_ms": ov, "nooverlap_ms": no, "bottleneck": bn,
                     "ov_rel": ov / g["latency_ms"] - 1,
                     "no_rel": no / g["latency_ms"] - 1})
    if not rows:
        print("no grid point appears in both the band and the measurements",
              file=sys.stderr)
        return 1

    def summarise(label: str, sel: list[dict]) -> None:
        if not sel:
            return
        for edge, key in (("Overlap", "ov_rel"), ("NoOverlap", "no_rel")):
            vals = [r[key] for r in sel]
            absm = statistics.mean(abs(v) for v in vals) * 100
            sgn = statistics.mean(vals) * 100
            over = 100 * sum(1 for v in vals if v > 0) / len(vals)
            print(f"  {label:28} {edge:10} n={len(sel):5} "
                  f"mean|e|={absm:7.2f}%  signed={sgn:+8.2f}%  over={over:5.1f}%")

    print(f"# {args.fpm.name}: {len(rows)} mixed grid points scored")
    summarise("all mixed", rows)
    # The question the pure regimes cannot answer: does the error depend on how much of
    # the batch is prefill?
    print()
    for lo, hi, name in ((0, 0.25, "prefill < 25% of tokens"),
                         (0.25, 0.75, "prefill 25-75%"),
                         (0.75, 1.01, "prefill > 75%")):
        sel = [r for r in rows
               if lo <= r["prefill"] / (r["prefill"] + r["context"]) < hi]
        summarise(name, sel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
