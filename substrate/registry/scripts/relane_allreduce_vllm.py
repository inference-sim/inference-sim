#!/usr/bin/env python3
"""Move the h100 and h200 fp16 all-reduce triples onto vLLM v0.31.0's own custom all-reduce.

    python scripts/relane_allreduce_vllm.py [--data <aisimulate systems/data>] [--check]

WHY. Every committed h100 and h200 `collective_*_all_reduce_fp16_{2,4,8}rank_*` entry was
fitted (fit_collectives_vllm.py) on AISimulate's vLLM 0.24.0 `custom_allreduce_perf.parquet`,
CUDA-graph lane. AISimulate ai-dynamo/aisimulate 7980e9500c00d9f85d7f296451d12768a743c70e
(2026-10-29) adds the same sweep at vLLM 0.31.0, the engine BLIS prices
(`systems/data/{h100_sxm,h200_sxm}/comm/vllm/0.31.0/custom_allreduce_perf.parquet`, image
vllm/vllm-openai:v0.31.0, collector 572c7ba3). Same 69 graph rows per part (23 messages x
2/4/8 ranks), bfloat16. On matched messages 0.31.0 is a median 0.69-0.87x of 0.24.0's
latency, so the 0.24.0 fit over-prices a v0.31.0 tensor-parallel all-reduce.

THE ESTIMATORS are fit_collectives_vllm.py's Group (floor = min over messages <= 4 KiB,
peak = max size/latency, transition rate grid-searched on max(floor + size/rate,
size/peak)), unchanged, so a different value is a different measurement.

THE SPLIT AND GATE, declared here before the comparison was run. Per (part, width) the
messages are sorted by size; every second one (odd index) is held out. The 0.31.0 triple is
refitted on the rest, and both it and the committed 0.24.0 triple are scored on the held-out
messages by fit_collectives.log_error (geometric mean |ratio|) of the kernel's form. The
relane lands only if the 0.31.0 fit's held-out error is lower at every width on both parts;
the committed values are then the 0.31.0 triple fitted on all messages, as the other
all-reduce entries are. If any width fails, nothing is written.

INT8 is not touched: the vLLM tables measure bfloat16 only (fit_collectives_vllm.py's guard),
and the int8 triples stay on NCCL. The 16-rank h100/h200 fp16 all-reduce entries are `assumed`
extrapolations from the committed 2/4/8 floors (extrapolate_collective_width.py), so they are
re-derived from the relaned floors by that script, in place, after this one runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402
import fit_collectives_vllm as fcv  # noqa: E402
from fit_collectives import log_error  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-collectives.yaml"
AISIMULATE_SHA = "7980e9500c00d9f85d7f296451d12768a743c70e"
# Pinned separately from the other families (rederive.yml AISIMULATE_ALLREDUCE_SHA): the
# 0.31.0 sweep arrives at aisimulate 7980e95, and the main pin predates it.
DEFAULT_DATA = (os.environ.get("AISIMULATE_ALLREDUCE_DATA") or os.environ.get("AISIMULATE_DATA")
                or "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
NEW, OLD = "0.31.0", "0.24.0"
PARTS = {"h100": ("h100_sxm", "NVIDIA H100 80GB HBM3"), "h200": ("h200_sxm", "NVIDIA H200")}
WIDTHS = (2, 4, 8)


def graph(data: Path, sku: str, version: str) -> dict[int, fcv.Group]:
    return {g.gpus: g for g in fcv.read(data / sku / "comm" / "vllm" / version)
            if g.backend == "vllm_graph"}


def form(floor, peak, rate):
    return lambda size: max(floor + size / rate, size / peak)


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(name, value, units, fitted, chip, cite, rat):
    return [f"  - {name}:", f"      value: {value}", f"      units: {units}",
            "      method: measured", f"      fitted: {'true' if fitted else 'false'}",
            f"      scope: {{hardware: [{chip}]}}", "      sources:",
            f'        - {{kind: model, cite: "{cite}", role: primary}}',
            "      rationale: >", f"        {rat}"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    data = Path(args.data)
    before = SET_PATH.read_text(encoding="utf-8")
    # The gate compares against the 0.24.0-fitted values, which this writer replaces; so it
    # reads them from the 0.24.0 sweep itself (fit_collectives_vllm.py reproduces them
    # exactly, validator/test_generated_from_aisimulate.py), not from the file.
    results, ok = {}, True
    for chip, (sku, device) in PARTS.items():
        path = data / sku / "comm" / "vllm" / NEW / "custom_allreduce_perf.parquet"
        if not path.is_file():
            print(f"{chip}: {path} absent; the AISimulate {AISIMULATE_SHA[:8]} tree is "
                  f"required", file=sys.stderr)
            return 2
        new, old = graph(data, sku, NEW), graph(data, sku, OLD)
        for w in WIDTHS:
            pts = new[w].points
            held = pts[1::2]
            half = fcv.Group("vllm_graph", w, pts[0::2])
            o = old[w]
            e_old = log_error(held, form(round(o.floor_us, 2), o.peak_bytes_per_us,
                                         o.transition_rate))
            e_new = log_error(held, form(half.floor_us, half.peak_bytes_per_us,
                                         half.transition_rate))
            win = e_new < e_old
            ok &= win
            results[(chip, w)] = (new[w], e_old, e_new, len(held), sha256(path), device, sku)
            print(f"{chip} {w}rank: held-out ({len(held)} messages) 0.24.0 {e_old:.3f}x, "
                  f"0.31.0 {e_new:.3f}x ({'better' if win else 'NOT better'}); 0.31.0 "
                  f"floor {new[w].floor_us:.2f} us peak {new[w].peak_bytes_per_us / 1e3:.1f} "
                  f"transition {new[w].transition_rate / 1e3:.2f} GB/s")
    if not ok:
        print("not every width improves on held-out messages: nothing written")
        after = before
    else:
        blocks = cb.split(before)
        for (chip, w), (g, e_old, e_new, nh, sha, device, sku) in results.items():
            stem = f"all_reduce_fp16_{w}rank_{chip}"
            cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/vllm/{NEW}/"
                    f"custom_allreduce_perf.parquet (aisimulate {AISIMULATE_SHA[:8]}, sha256 "
                    f"{sha[:12]}; vLLM {NEW} custom all-reduce, CUDA-graph lane, {device}), "
                    f"{len(g.points)} messages at {w} ranks")
            ho = (f"Relaned from vLLM {OLD}'s sweep to {NEW}'s: on every second message, held "
                  f"out, the {NEW} fit scores {e_new:.3f}x against the {OLD} values' "
                  f"{e_old:.3f}x ({nh} messages; scripts/relane_allreduce_vllm.py).")
            scope = f"{{hardware: [{chip}]}}"
            for name, value, units, fitted, rat in (
                    (f"collective_floor_{stem}", round(g.floor_us, 2), "us_per_transfer", True,
                     f"Minimum latency over the {g.flat_count} messages at or below 4 KiB, where "
                     f"latency is independent of size, on vLLM's OWN custom all-reduce in its "
                     f"CUDA-graph lane: vLLM does not call NCCL for a tensor-parallel all-reduce "
                     f"its custom kernel can serve. {ho}"),
                    (f"collective_peak_rate_{stem}", round(g.peak_bytes_per_us, 1),
                     "bytes_per_us", True,
                     f"Maximum of message_size/latency over the sweep, on vLLM's custom "
                     f"all-reduce in its CUDA-graph lane. {g.peak_bytes_per_us / 1e3:.2f} GB/s. "
                     f"{ho}"),
                    (f"collective_transition_rate_{stem}", round(g.transition_rate, 1),
                     "bytes_per_us", True,
                     f"The rate that prices a forward pass's all-reduce, grid-searched on the "
                     f"three-parameter form the kernel evaluates. Geometric error "
                     f"{g.error_three_param:.3f}x against {g.error_two_param:.3f}x under "
                     f"max(size/peak, floor). {g.transition_rate / 1e3:.2f} GB/s. {ho}")):
                cb.upsert(blocks, entry(name, value, units, fitted, chip, cite, rat),
                          after=lambda b: False)
        after = cb.join(blocks)
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
