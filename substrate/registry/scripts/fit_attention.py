#!/usr/bin/env python3
"""Fit the decode-attention primitive from NVIDIA AISimulate's generation sweeps.

The cost model prices decode attention as a KV read at derated HBM bandwidth. Measured
against these sweeps that is wrong by 9.7x to 35.8x depending on the part, and the reason
is structural rather than a bad constant: an attention kernel has a launch and setup floor
that a pure-bandwidth model charges nothing for, and it reaches only 58 to 90 percent of
datasheet bandwidth rather than the 80 percent the generic HBM derate assumes.

The form that fits is the same shape the collectives use:

    latency = floor + kv_bytes / rate

with both fitted per part by minimising the mean absolute log ratio, which weighs an
under-prediction and an over equally. Across five parts this takes the geometric error from
9.7-35.8x down to 1.51-1.94x.

Only FULL attention is fitted. The sweeps also cover sliding windows (window_size 128, 2048,
8192), where the bytes read are bounded by the window rather than the context, and mixing
those in would fit a curve to two different byte counts. A windowed model needs its own
entry and does not have one yet.

Usage:
    python scripts/fit_attention.py <data>/h200_sxm/attention/trtllm/1.3.0rc20 \
        --chip h200 [--catalog <blis-catalog>]
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import yaml

# The floor grid, in microseconds. The measured floors sit between 11 and 18.5, so this
# spans well past both ends at a resolution finer than the value is reported to.
FLOOR_GRID = [x * 0.5 for x in range(2, 81)]
# The rate grid, as a fraction of the part's datasheet HBM bandwidth. Fitted as a fraction
# rather than an absolute so the value is comparable across parts and so a reader can see
# at once how far below peak an attention kernel runs. The grid stops at 1.0: a kernel cannot
# read faster than the datasheet peak (SUB-5), and the sibling fitters stop there too.
FRACTION_GRID = [x * 0.02 for x in range(5, 51)]


def check_fraction(fraction: float) -> float:
    """Return the fraction, or refuse one outside (0, 1.0]. relane_attention_decode.py writes
    fraction * peak unchanged, so a value above 1.0 would be committed as a rate above the
    part's datasheet bandwidth."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"bandwidth fraction {fraction} is outside (0, 1.0]: "
                         "a kernel cannot read faster than the datasheet peak (SUB-5)")
    return fraction


def read(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    columns = list(table)
    return [{c: table[c][i] for c in columns} for i in range(len(table[columns[0]]))]


def kv_bytes(row: dict) -> float:
    """Bytes the attention kernel reads for this shape.

    Keys and values, for every position in the context, for every KV head. `step` is the
    context length in these sweeps and `isl` is 1, because a generation step appends one
    token; `batch_size` is the request count.
    """
    width = 1.0 if row["kv_cache_dtype"] == "fp8" else 2.0
    return (
        row["batch_size"]
        * row["step"]
        * 2
        * row["num_key_value_heads"]
        * row["head_dim"]
        * width
    )


def log_error(points: list[tuple[float, float]], predict) -> float:
    return math.exp(
        sum(abs(math.log(predict(b) / lat)) for b, lat in points) / len(points)
    )


def main(argv: list[str]) -> int:
    args = argv[1:]
    chip = None
    catalog = Path(os.environ.get("BLIS_CATALOG",
                                 str(Path(__file__).resolve().parents[2] / "catalog")))
    if "--chip" in args:
        i = args.index("--chip")
        chip = args[i + 1]
        args = args[:i] + args[i + 2:]
    if "--catalog" in args:
        i = args.index("--catalog")
        catalog = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 1 or chip is None:
        print(__doc__, file=sys.stderr)
        return 2

    base = Path(args[0])
    path = base / "generation_attention_perf.parquet"
    if not path.is_file():
        print(f"no generation_attention_perf.parquet under {base}", file=sys.stderr)
        return 1
    rows = read(path)

    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = facts["BwPeakTBs"] * 1e12

    # window_size is absent from the older collections -- vLLM 0.14.0 on A100 predates
    # the sliding-window sweep -- and a missing column means every row is full
    # attention. Defaulting to 0 keeps those rows rather than failing on the key, and
    # cannot silently admit a windowed row from a collection that does sweep them.
    points = [
        (kv_bytes(r), r["latency"] * 1e-3)  # the file records milliseconds
        for r in rows
        if r.get("window_size", 0) == 0 and r["latency"] > 0
    ]
    if len(points) < 100:
        print(f"only {len(points)} full-attention points; too few to fit",
              file=sys.stderr)
        return 1

    best: tuple[float, float, float] | None = None
    for floor_us in FLOOR_GRID:
        floor = floor_us * 1e-6
        for fraction in FRACTION_GRID:
            rate = peak * fraction
            err = log_error(
                points, lambda b, f=floor, r=rate: f + b / r
            )
            if best is None or err < best[0]:
                best = (err, floor_us, fraction)
    assert best is not None
    err, floor_us, fraction = best
    check_fraction(fraction)

    # What the current model gets, for comparison: a pure KV read at the generic derate.
    naive = log_error(points, lambda b: max(b / (peak * 0.8), 1e-12))

    print(f"# {rows[0]['device']}, {rows[0]['framework']} {rows[0]['version']}")
    print(f"# {len(points)} full-attention decode points of {len(rows)} rows")
    print(f"attention_decode_floor      {floor_us:6.1f} us")
    print(f"attention_decode_bw_fraction {fraction:6.2f} of {peak / 1e12:.3f} TB/s "
          f"= {peak * fraction / 1e12:.2f} TB/s")
    print(f"geometric error              {err:6.3f}x  "
          f"(pure KV read at 0.8 derate: {naive:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
