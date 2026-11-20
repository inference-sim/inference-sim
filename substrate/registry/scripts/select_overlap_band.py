#!/usr/bin/env python3
"""Choose between the kernel's two step-time band edges on measured evidence.

    select_overlap_band.py --band BAND.csv --fpm DIR [--cell NAME=FILE]...

blis-latency-kernel reports a band rather than a point estimate: `Overlap` sums the max
over resources within each layer and then sums layers, `NoOverlap` sums every resource.
Which edge a consumer should use is a model-selection question, and it is answerable by
measurement rather than assertion.

NVIDIA's FPM dataset measures one synchronized whole-forward iteration at a known batch
and KV-token count, so it prices the same quantity the band brackets, with no scheduler
in between. That makes it the right evidence for this choice and the wrong evidence for
fitting a coefficient -- it constrains a composition, not a constant.

Two traps this script exists to avoid, both of which produced wrong answers by hand:

  * FPM's `total_kv_read_tokens` is summed OVER THE BATCH. The kernel's DecodeBatch takes
    a PER-SEQUENCE context. Joining the two directly is a batch-fold error: at batch 256
    it compares a 256x KV difference. The join here is on (batch, batch*ctx == kv_total).
  * FPM's `latency_ms` is milliseconds. So is the band CSV this reads. Neither is seconds.

The band CSV is produced by the kernel repository, which owns step-time composition. This
script only selects between its columns; it never recomputes them.

Separation of evidence, enforced by what this script will read: FPM for selection,
AISimulate operator tables for coefficient fitting, and the InferenceX corpus for
evaluation only. It refuses a path under an `inferencex` directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def refuse_evaluation_data(path: Path | str) -> None:
    """Fail loudly if asked to select against the evaluation corpus.

    Accepts a str as well as a Path: a caller that passes a string would
    otherwise get an AttributeError from .parts rather than the refusal, which
    reads as a bug in the caller instead of the deliberate block it is.
    """
    parts = {p.lower() for p in Path(path).parts}
    if any("inferencex" in p or "semianalysis" in p for p in parts):
        raise SystemExit(
            f"{path}: InferenceX/SemiAnalysis data is evaluation-only and must not "
            f"inform model selection. Use NVIDIA FPM for the band choice."
        )


def decode_rows(fpm: Path) -> pd.DataFrame:
    """Return an FPM artifact's decode rows, medianed per (batch, kv_total).

    A median rather than a mean: the sidecar records
    `measurement_policy: dynamo_native_single_sample_v1` with one repeat and no warmup,
    so individual rows carry run-to-run scatter a mean would chase.
    """
    d = pd.read_parquet(fpm)
    dec = d[d["workload_kind"] == "decode"]
    if dec.empty:
        raise SystemExit(f"{fpm}: no decode rows")
    return (
        dec.groupby(["batch_size", "total_kv_read_tokens"])["latency_ms"]
        .median()
        .reset_index()
    )


def compare(band: pd.DataFrame, measured: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Join a scenario's band against measured latency on the batch-folded key."""
    b = band[band["scenario"] == scenario]
    if b.empty:
        raise SystemExit(f"no band rows for scenario {scenario!r}")
    m = b.merge(
        measured,
        left_on=["batch", "kv_total"],
        right_on=["batch_size", "total_kv_read_tokens"],
    )
    if m.empty:
        raise SystemExit(
            f"{scenario}: no (batch, kv_total) pair appears in both the band and the "
            f"measurements; the band sweep does not reach this cell's grid"
        )
    m = m.assign(
        overlap_rel=m["overlap_ms"] / m["latency_ms"] - 1,
        nooverlap_rel=m["nooverlap_ms"] / m["latency_ms"] - 1,
    )
    return m


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--band", required=True, type=Path)
    ap.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="SCENARIO=FPM_PARQUET",
        help="a band scenario name and the FPM artifact that measured it",
    )
    args = ap.parse_args(argv[1:])

    refuse_evaluation_data(args.band)
    band = pd.read_csv(args.band, comment="#")

    frames = []
    print(f"{'cell':32s} {'n':>4} {'Overlap':>9} {'NoOverlap':>10}  winner")
    print("-" * 72)
    for spec in args.cell:
        if "=" not in spec:
            raise SystemExit(f"--cell wants SCENARIO=FILE, got {spec!r}")
        scenario, path = spec.split("=", 1)
        fpm = Path(path)
        refuse_evaluation_data(fpm)
        m = compare(band, decode_rows(fpm), scenario)
        eo = m["overlap_rel"].abs().mean() * 100
        en = m["nooverlap_rel"].abs().mean() * 100
        frames.append(m)
        print(
            f"{scenario:32s} {len(m):4d} {eo:8.2f}% {en:9.2f}%  "
            f"{'NoOverlap' if en < eo else 'Overlap'}"
        )

    allm = pd.concat(frames)
    eo = allm["overlap_rel"]
    en = allm["nooverlap_rel"]
    print("-" * 72)
    print(
        f"{'POOLED':32s} {len(allm):4d} {eo.abs().mean()*100:8.2f}% "
        f"{en.abs().mean()*100:9.2f}%  "
        f"{'NoOverlap' if en.abs().mean() < eo.abs().mean() else 'Overlap'}"
    )
    print()
    print(f"  Overlap    signed mean = {eo.mean()*100:+6.2f}%")
    print(f"  NoOverlap  signed mean = {en.mean()*100:+6.2f}%")
    print(f"  NoOverlap closer on {(en.abs() < eo.abs()).sum()}/{len(allm)} points")
    print()
    print("A signed mean near zero is the figure that matters: it says the edge is")
    print("unbiased rather than merely close on average.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
