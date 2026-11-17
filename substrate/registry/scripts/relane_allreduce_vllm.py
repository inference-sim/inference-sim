#!/usr/bin/env python3
"""Move the fp16 all-reduce triples onto vLLM v0.31.0's own custom all-reduce (h100, h200, gb200, gb300).

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

GB200 AND GB300 (added with the aisim 6fdf6fb6 re-pin). AISimulate carries the same vLLM
0.31.0 custom all-reduce sweep for both Grace-Blackwell parts, at 2 and 4 ranks (one tray).
gb200's committed 2- and 4-rank triples were NCCL 2.29.2 (fit_collectives.py), a kernel vLLM
does not call for an all-reduce its custom kernel serves; gb300's were vLLM 0.27.0's sweep.
Each is relaned under the same split and gate, against the triple it replaces, re-derived
from that triple's own sweep.

At 8 and 16 ranks a GB200 tensor-parallel group spans trays, and vLLM's CustomAllreduce
serves it over multi-node NVLink (vLLM v0.33.0 custom_all_reduce.py, `mnnvl_only`), so the
committed NCCL 2.23 triples price a kernel vLLM does not run (VAL-8; inference-sim/
blis-substrate#116). No GB200 sweep measures the custom kernel at those widths. gb300's vLLM
0.27.0 sweep does, on the same NVL72 NVLink-5 fabric, so gb200 takes gb300's 8- and 16-rank
triples as `transferred`, superseding the NCCL entries. The transfer is checked first, where
both parts are measured: gb300's 0.31.0 triple scores gb200's 0.31.0 messages at 2 and 4 ranks,
and it must come within TRANSFER_GATE of gb200's own fit there, or nothing is written.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402
import fit_collectives as fc  # noqa: E402
import fit_collectives_vllm as fcv  # noqa: E402
from fit_collectives import log_error  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-collectives.yaml"
# Every sweep this writer reads is in the main AISimulate pin (rederive.yml AISIMULATE_SHA):
# the h100/h200 0.31.0 files there are byte-identical to the ones the earlier all-reduce pin
# (7980e950) carried.
AISIMULATE_SHA = "6fdf6fb6b2e1c15b8f5cd1ba35fd2d50cdb2eb18"
DEFAULT_DATA = (os.environ.get("AISIMULATE_DATA")
                or "/tmp/aisim/python/aisim/src/aisim_core/systems/data")
NEW = "0.31.0"
# chip: (sku, device, widths relaned onto NEW, the lane the replaced triple was fitted on)
PARTS = {
    "h100": ("h100_sxm", "NVIDIA H100 80GB HBM3", (2, 4, 8), ("vllm", "0.24.0")),
    "h200": ("h200_sxm", "NVIDIA H200", (2, 4, 8), ("vllm", "0.24.0")),
    "gb200-nvl72": ("gb200", "NVIDIA GB200", (2, 4), ("nccl", "2.29.2")),
    "gb300": ("gb300", "NVIDIA GB300", (2, 4), ("vllm", "0.27.0")),
}
# gb200's cross-tray widths, transferred from gb300's sweep that measures them.
TRANSFER = {"to": "gb200-nvl72", "from": "gb300", "version": "0.27.0", "widths": (8, 16),
            "check_widths": (2, 4)}
# The transferred triple may score the target's own rows at most this much worse than the
# target's own fit, at every width both parts measure (geometric-mean |ratio|).
TRANSFER_GATE = 1.10


def graph(data: Path, sku: str, version: str) -> dict[int, fcv.Group]:
    return {g.gpus: g for g in fcv.read(data / sku / "comm" / "vllm" / version)
            if g.backend == "vllm_graph"}


def nccl_all_reduce_fp16(data: Path, sku: str, version: str) -> dict[int, object]:
    """The NCCL fp16 all-reduce triples fit_collectives.py fits, by width."""
    groups = fc.reduce_sweep(fc.read_sweep(data / sku / "comm" / "nccl" / version))
    return {g.ranks: g for g in groups if g.op == "all_reduce" and g.dtype == "half"}


def form(floor, peak, rate):
    return lambda size: max(floor + size / rate, size / peak)


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(name, value, units, fitted, chip, cite, rat, transfer=None):
    method = "transferred" if transfer else "measured"
    out = [f"  - {name}:", f"      value: {value}", f"      units: {units}",
           f"      method: {method}",
           f"      fitted: {'true' if fitted and not transfer else 'false'}",
           f"      scope: {{hardware: [{chip}]}}"]
    if transfer:
        out += [f'      transferred_from: "{transfer[0]}"',
                f'      transfer_rule: "{transfer[1]}"',
                f'      supersedes: "{transfer[2]}"']
    return out + ["      sources:", f'        - {{kind: model, cite: "{cite}", role: primary}}',
                  "      rationale: >", f"        {rat}"]


def triple(stem, g, chip, cite, ho, transfer=None):
    return (
        entry(f"collective_floor_{stem}", round(g.floor_us, 2), "us_per_transfer", True, chip,
              cite, f"Minimum latency over the {g.flat_count} messages at or below 4 KiB, where "
              f"latency is independent of size, on vLLM's OWN custom all-reduce in its "
              f"CUDA-graph lane: vLLM does not call NCCL for a tensor-parallel all-reduce "
              f"its custom kernel can serve. {ho}", transfer),
        entry(f"collective_peak_rate_{stem}", round(g.peak_bytes_per_us, 1), "bytes_per_us",
              True, chip, cite, f"Maximum of message_size/latency over the sweep, on vLLM's "
              f"custom all-reduce in its CUDA-graph lane. {g.peak_bytes_per_us / 1e3:.2f} GB/s. "
              f"{ho}", transfer),
        entry(f"collective_transition_rate_{stem}", round(g.transition_rate, 1), "bytes_per_us",
              True, chip, cite, f"The rate that prices a forward pass's all-reduce, "
              f"grid-searched on the three-parameter form the kernel evaluates. Geometric error "
              f"{g.error_three_param:.3f}x against {g.error_two_param:.3f}x under "
              f"max(size/peak, floor). {g.transition_rate / 1e3:.2f} GB/s. {ho}", transfer),
    )


def stem_of(w: int, chip: str) -> str:
    return f"all_reduce_fp16_{w}rank_{chip.replace('-', '_')}"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    data = Path(args.data)
    before = SET_PATH.read_text(encoding="utf-8")
    # The gate compares against the triple this writer replaces; so it re-derives that triple
    # from its own sweep, not from the file, which after the first run holds the new values.
    results, ok = {}, True
    for chip, (sku, device, widths, (lane, old_version)) in PARTS.items():
        path = data / sku / "comm" / "vllm" / NEW / "custom_allreduce_perf.parquet"
        if not path.is_file():
            print(f"{chip}: {path} absent; the AISimulate {AISIMULATE_SHA[:8]} tree is "
                  f"required", file=sys.stderr)
            return 2
        new = graph(data, sku, NEW)
        old = (graph(data, sku, old_version) if lane == "vllm"
               else nccl_all_reduce_fp16(data, sku, old_version))
        for w in widths:
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
            results[(chip, w)] = (new[w], e_old, e_new, len(held), sha256(path), device, sku,
                                  f"{'vLLM' if lane == 'vllm' else 'NCCL'} {old_version}")
            print(f"{chip} {w}rank: held-out ({len(held)} messages) {lane} {old_version} "
                  f"{e_old:.3f}x, vLLM {NEW} {e_new:.3f}x ({'better' if win else 'NOT better'}); "
                  f"{NEW} floor {new[w].floor_us:.2f} us peak {new[w].peak_bytes_per_us / 1e3:.1f} "
                  f"transition {new[w].transition_rate / 1e3:.2f} GB/s")
    # The cross-tray transfer, checked where both parts are measured.
    to_sku, src_sku = PARTS[TRANSFER["to"]][0], PARTS[TRANSFER["from"]][0]
    to_new, src_new = graph(data, to_sku, NEW), graph(data, src_sku, NEW)
    checks = []
    for w in TRANSFER["check_widths"]:
        own, other = to_new[w], src_new[w]
        e_own = log_error(own.points, form(own.floor_us, own.peak_bytes_per_us,
                                           own.transition_rate))
        e_tr = log_error(own.points, form(other.floor_us, other.peak_bytes_per_us,
                                          other.transition_rate))
        good = e_tr <= e_own * TRANSFER_GATE
        ok &= good
        checks.append((w, e_own, e_tr))
        print(f"transfer check {w}rank: {TRANSFER['from']} {NEW} triple on {TRANSFER['to']}'s "
              f"rows {e_tr:.3f}x against its own fit {e_own:.3f}x "
              f"({'within' if good else 'OUTSIDE'} {TRANSFER_GATE}x)")
    src_path = data / src_sku / "comm" / "vllm" / TRANSFER["version"] / "custom_allreduce_perf.parquet"
    src_wide = graph(data, src_sku, TRANSFER["version"])
    if not ok:
        print("a gate failed: nothing written")
        after = before
    else:
        blocks = cb.split(before)
        for (chip, w), (g, e_old, e_new, nh, sha, device, sku, was) in results.items():
            cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/vllm/{NEW}/"
                    f"custom_allreduce_perf.parquet (aisimulate {AISIMULATE_SHA[:8]}, sha256 "
                    f"{sha[:12]}; vLLM {NEW} custom all-reduce, CUDA-graph lane, {device}), "
                    f"{len(g.points)} messages at {w} ranks")
            ho = (f"Relaned from {was}'s sweep to vLLM {NEW}'s: on every second message, held "
                  f"out, the {NEW} fit scores {e_new:.3f}x against the {was} values' "
                  f"{e_old:.3f}x ({nh} messages; scripts/relane_allreduce_vllm.py).")
            for e in triple(stem_of(w, chip), g, chip, cite, ho):
                cb.upsert(blocks, e, after=lambda b: False)
        tchip, schip = TRANSFER["to"], TRANSFER["from"]
        shown = "; ".join(f"{w} ranks {tr:.3f}x against {own:.3f}x" for w, own, tr in checks)
        for w in TRANSFER["widths"]:
            g = src_wide[w]
            cite = (f"NVIDIA AISimulate systems/data/{src_sku}/comm/vllm/{TRANSFER['version']}/"
                    f"custom_allreduce_perf.parquet (aisimulate {AISIMULATE_SHA[:8]}, sha256 "
                    f"{sha256(src_path)[:12]}; vLLM {TRANSFER['version']} custom all-reduce, "
                    f"CUDA-graph lane, NVIDIA GB300), {len(g.points)} messages at {w} ranks")
            ho = (f"TRANSFERRED from {schip}: no {tchip} sweep measures vLLM's custom "
                  f"all-reduce at {w} ranks, where the group spans trays and vLLM serves it over "
                  f"multi-node NVLink; the NCCL 2.23 triple it supersedes prices a kernel vLLM "
                  f"does not call (inference-sim/blis-substrate#116). Checked where both parts "
                  f"are measured, vLLM {NEW} at 2 and 4 ranks, {schip}'s triple scoring "
                  f"{tchip}'s messages: {shown} (gate {TRANSFER_GATE}x). "
                  f"scripts/relane_allreduce_vllm.py.")
            rule = (f"the same NVL72 NVLink-5 fabric and vLLM custom all-reduce kernel; "
                    f"{schip}'s triple at this width is used unchanged, checked on {tchip}'s "
                    f"own rows at 2 and 4 ranks within {TRANSFER_GATE}x")
            stem = stem_of(w, tchip)
            for e in triple(stem, g, tchip, cite, ho,
                            (f"hardware={schip}", rule, "")):
                name = cb.name_of(e)
                e = [x for x in e if x != f'      supersedes: ""']
                e.insert(e.index("      sources:"),
                         f'      supersedes: "{name} {{hardware: [{tchip}]}} method measured, '
                         f'NCCL 2.23 (inference-sim/blis-substrate#116)"')
                j = cb.find(blocks, name, f"{{hardware: [{tchip}]}}", "measured")
                if j is not None:
                    blocks[j] = e
                else:
                    cb.upsert(blocks, e, after=lambda b: False)
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
