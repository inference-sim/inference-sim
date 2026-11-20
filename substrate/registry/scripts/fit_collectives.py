#!/usr/bin/env python3
"""Derive collective floors and peak rates from NVIDIA AISimulate's NCCL sweeps.

Every collective coefficient in the registry comes from here, so each is
reproducible: point this at the collection a registry entry cites and it prints
that entry's value.

Source. NVIDIA AISimulate ships `comm/nccl/<version>/nccl_perf.parquet` for every
SKU the catalog holds: 256 B to 512 MiB across four operations, two dtypes and 2,
4 and 8 ranks. It is one methodology applied uniformly, which is what makes
cross-SKU and cross-width comparisons meaningful.

Three axes, all of which the measurements show matter:

  * OPERATION. An all-reduce floor is 1.6x an all-to-all floor on the same part,
    because a reduction synchronizes where a shuffle does not. One floor for all
    four operations is wrong in both directions at once.
  * DTYPE. int8 all-reduce reaches 201 GB/s on H200 where half reaches 132,
    because the reduction arithmetic sits in the critical path. A rate with no
    dtype axis describes one of the two and misprices the other by 1.5x.
  * RANK COUNT. all-reduce floors run 6.0 / 9.4 / 16.0 us at 2 / 4 / 8 ranks
    while all-to-all runs 6.4 / 6.6 / 7.4. The ring-shaped reduction's floor grows
    with width; the shuffle's barely does. That difference is the span the cost
    model prices, measured rather than assumed.

Estimators, and why these.

  FLOOR is the minimum latency over messages at or below 4 KiB, the region where
  latency does not depend on size. The minimum rather than the median because the
  sweeps carry occasional single-point noise — h100 all-to-all at 4 ranks reads
  17.67 us at 512 B among neighbours at 6.5 to 7.6 us — and a floor is a lower
  bound on achievable latency, so an outlier high reading should not raise it. The
  spread over the flat region is reported so a reader can see the noise rather
  than having it hidden.

  PEAK RATE is the maximum of message_size / latency over the whole sweep. Where
  the rate is still climbing at the largest message measured, the figure is a
  lower bound on the asymptote rather than the asymptote, and it is flagged: a
  sweep that has not saturated cannot state one. A lower bound is the safe
  direction for a rate (it overstates transfer time), which is why the value is
  still usable.

Usage:
    python scripts/fit_collectives.py <data>/h200_sxm/comm/nccl/2.29.2
    python scripts/fit_collectives.py --all <data>      # every SKU, one table
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Messages at or below this size sit on the flat part of the curve, where latency
# is independent of size. Verified against every sweep in the set: the knee is
# above 4 KiB on all of them.
FLAT_REGION_BYTES = 4096

# A rate whose last three points span more than this is still climbing, so the
# maximum is a lower bound on the asymptote rather than the asymptote.
SATURATION_TOLERANCE = 1.15

# The grid the transition rate is searched over, in bytes per microsecond. One GB/s is
# 1000 B/us, so this spans 0.025 to 1200 GB/s — finer than the precision the value is
# reported to, so the fit is reproducible to the digits committed.
#
# THE LOWER BOUND IS LOAD-BEARING, and it used to be wrong. The grid began at 4 * 250 =
# 1000 B/us, which is 1 GB/s, and five L40S 8-rank entries landed exactly there because
# the search was CLAMPED rather than converged: the optimum was below the grid and the
# fitter reported its floor. On all_reduce fp16 the true optimum is near 400 B/us at
# 1.1617x geometric error against 1.4183x at the clamped 1000 — a 22% accuracy loss,
# invisible to a re-derivation gate because the fitter reproduced its own clamp exactly.
#
# An L40S has no NVLink, so an 8-rank all-reduce crosses PCIe and sustains well under
# 1 GB/s. A grid that cannot express that is a grid that cannot fit the slowest real
# topology in the tree. The 25 B/us step below 1000 keeps resolution where the slow
# parts live; the 250 B/us step above it preserves every figure already committed on the
# NVLink parts, so this widening changes no other value.
#
# test_coefficient_properties.py asserts transition <= peak, which is the invariant a
# clamped fit violates and the reason the defect was found.
RATE_GRID = (
    [x * 25.0 for x in range(1, 40)]          # 25 to 975 B/us, step 25
    + [x * 250.0 for x in range(4, 4801)]     # 1,000 to 1,200,000 B/us, step 250
)

# AISimulate SKU directory to catalog chip name. Explicit rather than derived: an
# SKU mapped by guess would attach a measurement to the wrong part.
SKU_TO_CHIP = {
    "h100_sxm": "h100",
    "h200_sxm": "h200",
    "a100_sxm": "a100-sxm",
    "a100_pcie": "a100-80",
    "l40s": "l40s",
    "gb200": "gb200-nvl72",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
}


class Group:
    """One (operation, dtype, rank count) sweep, reduced."""

    def __init__(self, key: tuple[str, str, int], points: list[tuple[int, float]]):
        self.op, self.dtype, self.ranks = key
        self.points = sorted(points)
        flat = [lat for size, lat in self.points if size <= FLAT_REGION_BYTES]
        self.flat_count = len(flat)
        self.floor_us = min(flat) if flat else None
        self.flat_spread = (max(flat) / min(flat)) if flat else None
        self.peak_bytes_per_us = max(size / lat for size, lat in self.points)
        tail = [size / lat for size, lat in self.points[-3:]]
        self.saturated = max(tail) / min(tail) <= SATURATION_TOLERANCE
        self.transition_rate: float | None = None
        self.error_two_param: float | None = None
        self.error_three_param: float | None = None
        if self.floor_us is not None:
            self._fit_transition()

    def _fit_transition(self) -> None:
        """Fit the transition rate, and record what it buys over the two-parameter form.

        Both errors are reported so a reader can see the improvement per configuration
        rather than taking an aggregate on trust.
        """
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
        """A group with no flat-region measurement states no floor."""
        return self.floor_us is not None and self.flat_count >= 3


def log_error(points: list[tuple[int, float]], predict) -> float:
    """Geometric mean of the absolute ratio between prediction and measurement.

    The log makes a 2x under-prediction and a 2x over weigh equally, and the mean over
    points rather than over bytes keeps a long large-message tail from dominating a fit
    that has to be right at decode sizes too.
    """
    total = 0.0
    for size, latency in points:
        total += abs(math.log(predict(size) / latency))
    return math.exp(total / len(points))


def read_sweep(base: Path) -> list[dict]:
    import pyarrow.parquet as pq

    path = base / "nccl_perf.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no nccl_perf.parquet under {base}")
    table = pq.read_table(path).to_pydict()
    columns = list(table)
    return [{c: table[c][i] for c in columns} for i in range(len(table[columns[0]]))]


def reduce_sweep(rows: list[dict]) -> list[Group]:
    groups: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    for r in rows:
        latency_us = r["latency"] * 1e3  # the file records milliseconds
        if latency_us <= 0:
            continue
        key = (r["op_name"], r["nccl_dtype"], r["num_gpus"])
        groups.setdefault(key, []).append((r["message_size"], latency_us))
    return [Group(k, v) for k, v in sorted(groups.items())]


def report(base: Path, rows: list[dict], show_header: bool = True) -> None:
    sku = next((p for p in reversed(base.parts) if p in SKU_TO_CHIP), None)
    chip = SKU_TO_CHIP.get(sku or "", "<unmapped>")
    print(
        f"# {rows[0]['device']} -> catalog {chip}; NCCL {rows[0]['version']} via "
        f"{rows[0]['framework']}, {len(rows)} rows"
    )
    if show_header:
        print(
            f"{'operation':15s} {'dtype':5s} {'ranks':>5s} {'floor_us':>8s} "
            f"{'spread':>6s} {'peak_GB/s':>9s} {'trans_GB/s':>10s} "
            f"{'err2':>6s} {'err3':>6s} {'note':s}"
        )
    for g in reduce_sweep(rows):
        if not g.usable:
            print(
                f"{g.op:15s} {g.dtype:5s} {g.ranks:5d} {'-':>8s} {'-':>6s} "
                f"{g.peak_bytes_per_us / 1000:9.2f} {'-':>10s} {'-':>6s} {'-':>6s} "
                f"no flat-region points, so no floor"
            )
            continue
        note = "" if g.saturated else "peak is a LOWER BOUND (not saturated)"
        print(
            f"{g.op:15s} {g.dtype:5s} {g.ranks:5d} {g.floor_us:8.2f} "
            f"{g.flat_spread:5.2f}x {g.peak_bytes_per_us / 1000:9.2f} "
            f"{g.transition_rate / 1000:10.2f} {g.error_two_param:5.3f}x "
            f"{g.error_three_param:5.3f}x {note}"
        )


def main(argv: list[str]) -> int:
    args = argv[1:]
    if len(args) == 2 and args[0] == "--all":
        root = Path(args[1])
        for sku in sorted(SKU_TO_CHIP):
            comm = root / sku / "comm" / "nccl"
            if not comm.is_dir():
                print(f"# {sku}: no comm/nccl collection")
                continue
            # Newest NCCL version present, which is the one a current deployment
            # runs. Sorted as strings, which orders these correctly.
            version = sorted(comm.iterdir())[-1]
            try:
                report(version, read_sweep(version))
            except FileNotFoundError as e:
                print(f"# {sku}: {e}")
            print()
        return 0
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    base = Path(args[0])
    try:
        report(base, read_sweep(base))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
