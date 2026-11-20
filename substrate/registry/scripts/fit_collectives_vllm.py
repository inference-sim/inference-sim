#!/usr/bin/env python3
"""Fit all-reduce collectives from vLLM's own custom kernel, not from NCCL.

    python scripts/fit_collectives_vllm.py <data>/h200_sxm/comm/vllm/0.24.0
    python scripts/fit_collectives_vllm.py --all <data>

Every `collective_*` coefficient this registry ships was fitted from NCCL sweeps. BLIS
prices vLLM serving, and vLLM does not use NCCL for a tensor-parallel all-reduce that fits
in its custom kernel's size limit: it uses its own one-shot/two-shot implementation, which
AISimulate measures separately in `custom_allreduce_perf.parquet`.

The two are not close. Matching 63 (num_gpus, message_size) pairs on h200 between NCCL
2.29.2 and vLLM 0.24.0, the custom kernel under CUDA graph is FASTER by a median factor of
0.635 -- so an NCCL-derived coefficient over-charges a graph-mode all-reduce by about 1.6x.
Eager mode runs the other way, a median 1.271x slower than NCCL.

That difference is why this script exists, and why it fits per BACKEND. vLLM captures
decode in a CUDA graph, so `vllm_graph` is the lane a decode step runs in; `vllm_eager` is
reported alongside rather than averaged in, because averaging two backends whose ratio
spans 2x would describe neither.

Estimators are deliberately identical to `fit_collectives.py` -- floor as the minimum over
the flat region, peak as the maximum size/latency, transition rate by grid search over the
three-parameter form -- so a difference in output is a difference in the data rather than
in the method. The flat-region bound and the rate grid are imported from that module for
the same reason.

DTYPE. The vLLM tables measure bfloat16 only, where the committed NCCL fits carry fp16 and
int8. These are the same two-byte wire payload: NCCL's own table records
`nccl_dtype: half` against `wire_dtype: bfloat16` for every all-reduce row, so a bf16
measurement prices the fp16 entry the kernel asks for. An int8 all-reduce is NOT covered
here and keeps its NCCL fit; this script refuses to emit one rather than halve a 2-byte
figure and call it measured.

SCOPE. all-reduce only. vLLM's custom kernel implements no all-gather, reduce-scatter or
all-to-all, so those three keep their NCCL fits, which is correct rather than a gap: they
really do run on NCCL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fit_collectives import FLAT_REGION_BYTES, RATE_GRID, log_error

# vLLM's custom all-reduce has a size ceiling; above it the engine falls back to NCCL. The
# sweeps stop well below any such limit, so every row here is the custom kernel.
WIRE_BYTES_PER_ELEMENT = 2


class Group:
    """One (backend, num_gpus) sweep, fitted with fit_collectives.py's estimators."""

    def __init__(self, backend: str, gpus: int, points: list[tuple[float, float]]):
        self.backend = backend
        self.gpus = gpus
        self.points = sorted(points)
        flat = [lat for size, lat in self.points if size <= FLAT_REGION_BYTES]
        self.flat_count = len(flat)
        self.floor_us = min(flat) if flat else None
        self.peak_bytes_per_us = max(size / lat for size, lat in self.points)
        self.transition_rate: float | None = None
        self.error_two_param: float | None = None
        self.error_three_param: float | None = None
        if self.floor_us is not None:
            self._fit_transition()

    def _fit_transition(self) -> None:
        assert self.floor_us is not None
        floor, peak = self.floor_us, self.peak_bytes_per_us
        self.error_two_param = log_error(
            self.points, lambda size: max(size / peak, floor)
        )
        best: tuple[float, float] | None = None
        for rate in RATE_GRID:
            err = log_error(
                self.points,
                lambda size, rate=rate: max(floor + size / rate, size / peak),
            )
            if best is None or err < best[0]:
                best = (err, rate)
        assert best is not None
        self.error_three_param, self.transition_rate = best

    @property
    def usable(self) -> bool:
        return self.floor_us is not None


def read(base: Path) -> list[Group]:
    path = base / "custom_allreduce_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"no custom_allreduce_perf.parquet under {base}")
    d = pd.read_parquet(path)
    dtypes = sorted(d["allreduce_dtype"].unique())
    if dtypes != ["bfloat16"]:
        raise SystemExit(
            f"{path}: expected bfloat16 only, found {dtypes}. The fp16-equivalence "
            f"argument in this script's docstring covers a two-byte payload and nothing "
            f"else; widen it deliberately rather than by accident."
        )
    groups = []
    for (backend, gpus), g in d.groupby(["backend", "num_gpus"]):
        # The file records MILLISECONDS, as fit_collectives.py also notes for the NCCL
        # sweeps. Omitting this conversion yields a peak of 143,314 GB/s on h200 and a
        # three-parameter error worse than the two-parameter one -- both of which are
        # how the mistake announces itself.
        pts = [
            (float(r.message_size), float(r.latency) * 1e3)
            for r in g.itertuples()
            if r.latency > 0
        ]
        if pts:
            groups.append(Group(str(backend), int(gpus), pts))
    return sorted(groups, key=lambda x: (x.backend, x.gpus))


def report(base: Path, groups: list[Group]) -> None:
    print(f"# {base.parent.parent.parent.name}, {base.parent.name} {base.name}, "
          f"all_reduce, bfloat16")
    print(f"# {'backend':12s} {'ranks':>5s} {'floor us':>9s} {'transition':>11s} "
          f"{'peak':>10s} {'2-param':>8s} {'3-param':>8s} {'flat n':>7s}")
    for g in groups:
        if not g.usable:
            print(f"# {g.backend:12s} {g.gpus:5d}   no flat-region point; states no floor")
            continue
        assert g.transition_rate is not None
        print(f"# {g.backend:12s} {g.gpus:5d} {g.floor_us:9.3f} "
              f"{g.transition_rate / 1000:10.2f} {g.peak_bytes_per_us / 1000:9.2f} "
              f"{g.error_two_param:7.3f}x {g.error_three_param:7.3f}x {g.flat_count:7d}")
    print("#")
    print("# floor in microseconds; transition and peak in GB/s. Errors are geometric,")
    print("# two-param against max(size/peak, floor) and three-param against the form the")
    print("# kernel evaluates. vllm_graph is the decode lane: vLLM captures decode in a")
    print("# CUDA graph, so an eager figure does not price a decode step.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", type=Path)
    ap.add_argument("--all", action="store_true",
                    help="treat base as the data root and fit every vllm comm collection")
    args = ap.parse_args(argv[1:])

    if args.all:
        bases = sorted(args.base.glob("*/comm/vllm/*"))
        if not bases:
            raise SystemExit(f"no */comm/vllm/* under {args.base}")
    else:
        bases = [args.base]

    for b in bases:
        if not (b / "custom_allreduce_perf.parquet").is_file():
            continue
        report(b, read(b))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
