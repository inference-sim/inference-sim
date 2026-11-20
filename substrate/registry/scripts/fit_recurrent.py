#!/usr/bin/env python3
"""Fit the recurrent (linear-attention) primitive from AISimulate's decode sweeps.

The cost model prices a recurrent layer's state update at zero. That is wrong by whatever
the kernel costs, and on the two hybrid models in the catalog the kernel fires on 44 percent
(Nemotron-3-Ultra) and 100 percent (Kimi-K3) of layers.

The form is the same shape the collectives and attention use — a launch-and-setup floor that
a large batch amortizes:

    latency = floor + tokens / rate

with both fitted per (model family, kernel) by minimising the mean absolute log ratio.

WHAT THE DATA COVERS, AND WHAT IT DOES NOT. This matters more here than anywhere else in
the registry:

  KDA (Kimi-K3) is complete. The sweep carries the convolution and the recurrent scan as
  separate kernels, plus a fused decode variant, so a layer's cost is the sum of the two
  or the fused figure.

  MAMBA2 (Nemotron-3-Ultra) is NOT complete. Its sweep carries only `causal_conv1d_fn` and
  `causal_conv1d_update` — the convolution. The selective-scan kernel that is the rest of a
  Mamba2 layer is absent from every collection in the tree. So the mamba figures here are a
  LOWER BOUND on the layer, not the layer, and they are labelled to say so.

Usage:
    python scripts/fit_recurrent.py <sweep>.parquet [--model <model_name>]

    python scripts/fit_recurrent.py <data>/h100_sxm/kda/sglang/0.5.16/kda_perf.parquet
    python scripts/fit_recurrent.py <data>/h100_sxm/linear_attention/trtllm/1.3.0rc20/mamba2_perf.parquet
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

# Floor grid in microseconds, and rate grid in tokens per microsecond. Both span well past
# the fitted values at a resolution finer than the values are reported to.
FLOOR_GRID = [x * 0.1 for x in range(10, 301)]
RATE_GRID = [x * 0.25 for x in range(1, 401)]


def read(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    columns = list(table)
    return [{c: table[c][i] for c in columns} for i in range(len(table[columns[0]]))]


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Return (floor_us, rate_tokens_per_us, geometric_error)."""
    best: tuple[float, float, float] | None = None
    for floor in FLOOR_GRID:
        for rate in RATE_GRID:
            err = math.exp(
                sum(abs(math.log((floor + t / rate) / lat)) for t, lat in points)
                / len(points)
            )
            if best is None or err < best[0]:
                best = (err, floor, rate)
    assert best is not None
    err, floor, rate = best
    return floor, rate, err


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=Path, help="a *_perf.parquet under <sku>/<family>/...")
    # Pooling models is wrong when their geometries differ: the committed mamba2 entries
    # are fitted on Nemotron-3-Ultra alone, because d_model/nheads set how much work one
    # token is and a fit over six geometries describes none of them. The sweep's own
    # model_name column is the filter, so the citation and the command can agree.
    ap.add_argument("--model", dest="only_model",
                    help="fit only rows whose model_name matches exactly")
    # The same hazard along a second axis. vLLM's KDA sweep varies num_k_heads over
    # 12/24/48/96 where SGLang's holds it at 12, so a fit pooled over the vLLM file
    # describes no geometry: it lands at geo-err 1.733x where each width alone fits to
    # 1.169-1.196x. A per-width fit is the comparable one.
    ap.add_argument("--num-k-heads", dest="num_k_heads", type=int,
                    help="fit only rows with this num_k_heads")
    ap.add_argument("--group-by", choices=["kernel", "kernel+heads"], default="kernel",
                    help="'kernel+heads' reports one fit per (kernel, num_k_heads)")
    args = ap.parse_args(argv[1:])
    path, only_model = args.sweep, args.only_model
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    rows = read(path)
    generation = [r for r in rows if r.get("phase") == "generation" and r["latency"] > 0]
    if not generation:
        print(f"no generation-phase rows in {path.name}", file=sys.stderr)
        return 1

    if only_model is not None:
        generation = [r for r in generation
                      if str(r.get("model_name", "?")) == only_model]
        if not generation:
            print(f"no generation rows for model {only_model}", file=sys.stderr)
            return 1
    if args.num_k_heads is not None:
        generation = [r for r in generation
                      if r.get("num_k_heads") == args.num_k_heads]
        if not generation:
            print(f"no generation rows with num_k_heads={args.num_k_heads}",
                  file=sys.stderr)
            return 1
    models = sorted({str(r.get("model_name", "?")) for r in generation})
    print(f"# {rows[0]['device']}, {rows[0]['framework']} {rows[0]['version']}, "
          f"op {rows[0]['op_name']}")
    print(f"# {len(generation)} generation rows; models: {models}")

    by_kernel: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for r in generation:
        key = r["kernel_source"]
        if args.group_by == "kernel+heads" and r.get("num_k_heads") is not None:
            key = f"{key} hk={r['num_k_heads']}"
        by_kernel[key].append((r["num_tokens"], r["latency"] * 1000))
    for kernel in sorted(by_kernel):
        points = by_kernel[kernel]
        if len(points) < 8:
            print(f"{kernel:38s} n={len(points):3d}  too few to fit")
            continue
        floor, rate, err = fit(points)
        print(f"{kernel:38s} n={len(points):3d} floor={floor:5.1f}us "
              f"rate={rate:6.2f} tok/us geo-err {err:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
