#!/usr/bin/env python3
"""Write the NCCL lane of the fp16 tensor-parallel all-reduce (inference-sim/blis-substrate#116).

    python scripts/emit_allreduce_nccl_lane.py [--data <aisimulate systems/data>] [--check]

WHY. The kernel prices a tensor-parallel all-reduce from the lane of the kernel the engine
resolved (kernel/new.go, liftCollectiveFloors). The plain `collective_*_all_reduce_fp16_*` names
are vLLM's custom all-reduce on every part that has a custom sweep; a deployment that runs NCCL
instead (custom all-reduce disabled, a width its kernel refuses, a group that leaves the node)
was priced from those same entries. This writer adds the lane that prices it:

    collective_{floor,peak_rate,transition_rate}_all_reduce_nccl_fp16_<N>rank_<part>

THE ESTIMATORS are fit_collectives.py's, unchanged (floor = min over messages <= 4 KiB, peak =
max size/latency, transition rate grid-searched on max(floor + size/rate, size/peak)), applied
to the all_reduce / half rows of AISimulate's NCCL sweep for the part. NCCL's `half` payload is
the two-byte activation the custom sweeps call bfloat16: the same bytes.

THE SOURCE of each part, cited by the sweep's own `version` column and not by its directory
(insert_collectives_wide.py explains why the directory is not the provenance):

  h100, h200, b200, b300, gb300       NCCL 2.29.2 (the registry's pinned collection), 2 and 4 ranks;
                                      8 ranks too where the sweep carries it (not gb300)
  a100-sxm, a100-80 (an alias), l40s  NCCL 2.27.3
  gb200-nvl72                         2 and 4 ranks: 2.29.2; 8 and 16 ranks: 2.27.7, in the file
                                      named 2.23 (the version-mix caveat of insert_collectives_wide)

A width the sweep does not carry has no entry. The kernel then prices that width from the custom
lane and records an assumption (kernel all_reduce_lane); it never borrows another width.

THE 2.30.7 TABLE. On b200 and gb200, and on h100 and h200, AISimulate's comm/nccl/2.30.7
file measures the same all-reduce at 2 and 4 ranks through torch.distributed (a different
collector, not a different NCCL alone). It is not a lane here: no engine states the NCCL version,
and the collector's flat ~26-115 us floor is the eager call, not NCCL. What this writer does is
record, on the 2- and 4-rank NCCL entries of those parts, how 2.30.7 compares with the cited
sweep: the ratio of the floors, and the median latency ratio over the messages at or above 16 MiB
that both sweeps carry. A reader choosing a lane sees the disagreement on the entry.

Usage: see above. `--check` exits 1 if the committed file would change.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402
import fit_collectives as fc  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-collectives.yaml"
AISIMULATE_SHA = "6fdf6fb6b2e1c15b8f5cd1ba35fd2d50cdb2eb18"
DEFAULT_DATA = (os.environ.get("AISIMULATE_DATA")
                or "/tmp/aisim/python/aisim/src/aisim_core/systems/data")

# chip -> (sku, [(directory under comm/nccl, version column value, widths)], 2.30.7 comparable)
PARTS = {
    "h100": ("h100_sxm", [("2.29.2", "2.29.2", (2, 4, 8))], True),
    "h200": ("h200_sxm", [("2.29.2", "2.29.2", (2, 4, 8))], True),
    "b200": ("b200_sxm", [("2.29.2", "2.29.2", (2, 4, 8))], True),
    "b300": ("b300_sxm", [("2.29.2", "2.29.2", (2, 4, 8))], False),
    "gb300": ("gb300", [("2.29.2", "2.29.2", (2, 4))], False),
    "a100-sxm": ("a100_sxm", [("2.27.3", "2.27.3", (2, 4, 8))], False),
    "a100-80": ("a100_sxm", [("2.27.3", "2.27.3", (2, 4, 8))], False),
    "l40s": ("l40s", [("2.27.3", "2.27.3", (2, 4, 8))], False),
    "gb200-nvl72": ("gb200", [("2.29.2", "2.29.2", (2, 4)), ("2.23", "2.27.7", (8, 16))], True),
}
LARGE = 16 * 1024 * 1024  # bytes; the issue's ">= 16 MiB"


def sweep(data: Path, sku: str, directory: str, version: str, widths) -> dict[int, fc.Group]:
    rows = [r for r in fc.read_sweep(data / sku / "comm" / "nccl" / directory)
            if r["op_name"] == "all_reduce" and r["nccl_dtype"] == "half"
            and str(r.get("version", version)) == version and r["num_gpus"] in widths]
    out = {}
    for g in fc.reduce_sweep(rows):
        if g.usable:
            out[g.ranks] = g
    return out


def _interp(pts, size):
    """The cited sweep's latency at `size` bytes, linear in log-log between its neighbours (the
    two sweeps do not share message sizes)."""
    import bisect
    import math
    xs = [p[0] for p in pts]
    i = bisect.bisect_left(xs, size)
    if xs[i] == size:
        return pts[i][1]
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    t = (math.log(size) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


def collector_ratio(data: Path, sku: str, directory: str, version: str, w: int, base: fc.Group):
    """2.30.7 against the cited sweep at w ranks: (floor ratio, median >= 16 MiB latency ratio, n)."""
    path = data / sku / "comm" / "nccl" / "2.30.7"
    if not (path / "nccl_perf.parquet").is_file():
        return None
    rows = [r for r in fc.read_sweep(path) if r["op_name"] == "all_reduce"
            and r["nccl_dtype"] == "half" and r["num_gpus"] == w and r["latency"] > 0]
    if not rows:
        return None
    # The 2.30.7 file has no message at or below 4 KiB (its smallest is 5 KiB), so its "floor" is
    # the least latency over its three smallest messages.
    b = {r["message_size"]: r["latency"] * 1e3 for r in rows}
    smallest = sorted(b)[:3]
    pts = base.points  # (bytes, us) ascending
    ratios = [b[s] / _interp(pts, s) for s in sorted(b) if s >= LARGE and pts[0][0] <= s <= pts[-1][0]]
    if not ratios:
        return None
    return min(b[s] for s in smallest) / base.floor_us, statistics.median(ratios), len(ratios)


def entry(name, value, units, fitted, chip, cite, rationale):
    return [f"  - {name}:", f"      value: {value}", f"      units: {units}",
            "      method: measured", f"      fitted: {'true' if fitted else 'false'}",
            f"      scope: {{hardware: [{chip}]}}", "      sources:",
            f'        - {{kind: model, cite: "{cite}", role: primary}}',
            "      rationale: >", f"        {rationale}"]


def lane_entries(chip, sku, directory, version, w, g, versionnote):
    stem = f"all_reduce_nccl_fp16_{w}rank_{chip.replace('-', '_')}"
    cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/nccl/{directory}/nccl_perf.parquet "
            f"(aisimulate {AISIMULATE_SHA[:8]}; version column: {version}), all_reduce, half, "
            f"{w} ranks, {len(g.points)} messages")
    lane = ("The NCCL lane of the tensor-parallel all-reduce, read when the engine resolves "
            "NCCL (custom all-reduce disabled, a width its kernel refuses, a group that leaves "
            "the node); the plain all_reduce_fp16 entries are the custom kernel's lane. "
            "scripts/emit_allreduce_nccl_lane.py. ")
    return [
        entry(f"collective_floor_{stem}", round(g.floor_us, 2), "us_per_transfer", False, chip,
              cite, f"Minimum latency over the {g.flat_count} messages at or below 4 KiB, where "
              f"latency is independent of size. {lane}{versionnote}"),
        entry(f"collective_peak_rate_{stem}", round(g.peak_bytes_per_us, 1), "bytes_per_us",
              False, chip, cite, f"Maximum of message_size/latency over the sweep, "
              f"{g.peak_bytes_per_us / 1e3:.2f} GB/s. {lane}"),
        entry(f"collective_transition_rate_{stem}", round(g.transition_rate, 1), "bytes_per_us",
              True, chip, cite, f"The rate that prices a forward pass's all-reduce, "
              f"grid-searched on the three-parameter form the kernel evaluates; geometric error "
              f"{g.error_three_param:.3f}x against {g.error_two_param:.3f}x under "
              f"max(size/peak, floor). {g.transition_rate / 1e3:.2f} GB/s. {lane}"),
    ]


def build(data: Path, before: str) -> str:
    blocks = cb.split(before)
    for chip, (sku, sources, compare) in PARTS.items():
        scope = f"{{hardware: [{chip}]}}"
        key = cb.scope_key(scope)

        def anchor(b, key=key):
            n = cb.name_of(b)
            return (n is not None and cb.scope_of(b) is not None and cb.scope_key(cb.scope_of(b)) == key
                    and (n.startswith("collective_transition_rate_all_reduce_fp16_")
                         or n.startswith("collective_transition_rate_all_reduce_nccl_fp16_")))

        for directory, version, widths in sources:
            groups = sweep(data, sku, directory, version, widths)
            for w in widths:
                g = groups.get(w)
                if g is None:
                    continue
                note = ""
                cmp = collector_ratio(data, sku, directory, version, w, g) if compare else None
                if cmp:
                    note = (f"NCCL 2.30.7 (comm/nccl/2.30.7, torch.distributed through "
                            f"ProcessGroupNCCL, a different collector) measures this width's floor "
                            f"(least latency over its three smallest messages, 5 KiB and up) at {cmp[0]:.2f}x this entry and, over its {cmp[2]} messages at or "
                            f"above 16 MiB (the cited sweep interpolated log-log), a median latency {cmp[1]:.3f}x; it is "
                            f"not a lane (no engine states the NCCL version). ")
                for e in lane_entries(chip, sku, directory, version, w, g, note):
                    cb.upsert(blocks, e, after=anchor)
    return cb.join(blocks)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    before = SET_PATH.read_text(encoding="utf-8")
    after = build(Path(args.data), before)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
