#!/usr/bin/env python3
"""Fit the per-request per-step host cost from FPM's pure-decode rows.

THE GAP. `blis-latency-kernel`'s `hostPerStep()` charges a cost that depends only on the
graph mode and the layer count, so it is CONSTANT in batch size: 2.828 ms at every batch
on MiniMax-M2.7 h200 pure-tp4. vLLM's per-step CPU path is not constant -- it loops over
the request count several times per step (`v1/worker/gpu_model_runner.py:1097`,
`:1777`, `:1823`), preparing input ids, block tables and sampling metadata per request.

THE EVIDENCE. FPM's pure-decode residual is FLAT in context and RISES with batch, which
is the signature of a per-request term and not of a KV-read or weight-read error:

    batch=64   ctx 2..256 -> 0.92, 0.92, 0.94, 0.94, 0.95, 0.97, 0.98, 0.96
    batch=256  ctx 2..256 -> 1.21, 1.21, 1.21, 1.22, 1.21, 1.21, 1.22, 1.22
    batch=512  ctx 2..256 -> 1.30, 1.31, 1.31, 1.31, 1.30, 1.30, 1.31, 1.29

Expert coverage was the competing explanation and is ruled out by arithmetic: with 256
experts at top_k 8, `ExpertsTouched` reaches 98.3% of all experts by batch 128 and
saturates, while the residual keeps growing to batch 512.

THE FORM. `host_step_per_request`, in us per request, added to the step's host term. One
parameter, fitted by least squares on the residual against batch, with the intercept
pinned to zero because the constant part is already charged by `hostPerStep`.

SPLIT DESIGN. Grouped-randomized over CELLS, not over rows. A random row split is invalid
here: FPM's grid has consecutive batch sizes 32/33, 40/41, 48/49 -- 0.2% apart -- so a
random split trains on batch 32 and tests on batch 33, which measures interpolation. The
holdout unit is the cell (model x topology), and the seed is reported so the draw is
reproducible. FPM's model-system collinearity means a cell holdout is also a chip holdout;
that is stated rather than hidden, and the fit is deliberately NOT per-cell.

HELD-OUT EVIDENCE, never fitted: FPM's five GLM-5.3-Flash vLLM 0.31.0 GB300 leaves pair each
held-out step with a "clean" timer excluding schedule() and update_from_output; decode
(FPM - clean) rises 20-33 us per request over batch 1-32 (SGLang control: -1), though the
slope is not identified at batch <= 8. Grace hosts, separate runs. Methodology section 10.7.

Fitted on PURE-DECODE rows, where there is no prefill attention and no causal-FLOPs term,
so the batch-dependent residual is isolated. Validated on the MIXED rows, which no fit
sees.

Usage:
    python scripts/fit_host_step_per_request.py --cell NAME=FPM.parquet=BAND.csv ... \
        [--holdout-seed 7] [--holdout 2]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
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


def decode_points(fpm: Path, band: Path) -> list[tuple[int, float, float]]:
    """(batch, predicted_ms, measured_ms) for every pure-decode grid point."""
    import pyarrow.parquet as pq

    priced: dict[tuple[int, int], float] = {}
    with band.open() as fh:
        for r in csv.DictReader(fh):
            if int(r["prefill_tokens"]) != 0:
                continue
            priced[(int(r["batch"]), int(r["context"]))] = float(r["nooverlap_us"]) / 1000.0
    d = pq.read_table(fpm).to_pydict()
    buckets: dict[tuple[int, int], list[float]] = {}
    for i in range(len(d["latency_ms"])):
        pre = d["total_prefill_tokens"][i] or 0
        kv = d["total_kv_read_tokens"][i] or 0
        b = d["batch_size"][i]
        if pre != 0 or kv <= 0 or b <= 0 or d["latency_ms"][i] <= 0:
            continue
        buckets.setdefault((b, kv // b), []).append(d["latency_ms"][i])
    out = []
    for (b, c), v in buckets.items():
        if (b, c) in priced:
            out.append((b, priced[(b, c)], statistics.median(v)))
    return out


def fit(points: list[tuple[int, float, float]]) -> float:
    """Least squares through the origin: deficit_ms ~= c * batch, c in ms/request."""
    num = sum(b * (m - p) for b, p, m in points)
    den = sum(b * b for b, _, _ in points)
    return num / den if den else 0.0


def score(points: list[tuple[int, float, float]], c: float) -> dict:
    rel = [((p + c * b) / m - 1) for b, p, m in points]
    return {"n": len(rel),
            "abs": statistics.mean(abs(x) for x in rel) * 100,
            "signed": statistics.mean(rel) * 100}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True,
                    metavar="NAME=FPM=BAND",
                    help="cell name, its FPM artifact, and its bandprobe output")
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument("--holdout", type=int, default=2,
                    help="how many cells to hold out of the fit")
    ap.add_argument("--max-batch", type=int, default=0,
                    help="drop points above this batch (0 = keep all)")
    args = ap.parse_args(argv[1:])

    cells: dict[str, list] = {}
    for spec in args.cell:
        name, fpm, band = spec.split("=", 2)
        refuse_evaluation_data(Path(fpm))
        refuse_evaluation_data(Path(band))
        pts = decode_points(Path(fpm), Path(band))
        if args.max_batch:
            pts = [p for p in pts if p[0] <= args.max_batch]
        if not pts:
            print(f"{name}: no pure-decode points", file=sys.stderr)
            continue
        cells[name] = pts

    if len(cells) < 2:
        print("need at least two cells to hold one out", file=sys.stderr)
        return 1

    names = sorted(cells)
    rng = random.Random(args.holdout_seed)
    held = set(rng.sample(names, min(args.holdout, len(names) - 1)))
    train = [p for n in names if n not in held for p in cells[n]]
    print(f"# grouped-randomized cell holdout, seed={args.holdout_seed}")
    print(f"# train cells: {[n for n in names if n not in held]}")
    print(f"# HELD OUT:    {sorted(held)}")
    if args.max_batch:
        print(f"# batches above {args.max_batch} dropped")

    c = fit(train)
    print(f"\nhost_step_per_request = {c * 1000:.2f} us/request   "
          f"(fitted on {len(train)} training points)")

    print(f"\n{'cell':16} {'role':9} {'n':>6} {'before abs':>11} {'after abs':>10} "
          f"{'before signed':>14} {'after signed':>13}")
    for n in names:
        pts = cells[n]
        b0, b1 = score(pts, 0.0), score(pts, c)
        role = "HELD OUT" if n in held else "train"
        print(f"{n:16} {role:9} {b0['n']:6} {b0['abs']:10.2f}% {b1['abs']:9.2f}% "
              f"{b0['signed']:+13.2f}% {b1['signed']:+12.2f}%")
    allp = [p for n in names for p in cells[n]]
    a0, a1 = score(allp, 0.0), score(allp, c)
    print(f"{'POOLED':16} {'':9} {a0['n']:6} {a0['abs']:10.2f}% {a1['abs']:9.2f}% "
          f"{a0['signed']:+13.2f}% {a1['signed']:+12.2f}%")
    hp = [p for n in held for p in cells[n]]
    h0, h1 = score(hp, 0.0), score(hp, c)
    print(f"{'HELD-OUT ONLY':16} {'':9} {h0['n']:6} {h0['abs']:10.2f}% {h1['abs']:9.2f}% "
          f"{h0['signed']:+13.2f}% {h1['signed']:+12.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
