#!/usr/bin/env python3
"""Fit host_link_bandwidth per part from InferenceX CollectiveX swap_blocks, and re-derive the
DeepSeek-V4.1 host-resident gather rate from a measured H2D payload ceiling.

WHAT THE KERNEL MEANS BY IT. blis-latency-kernel caps a CPU tier's transfer rate at
host_link_bandwidth (TierTime: rate = min(tier rate, host link)), and uses it as the UVA
gather rate when no host_gather_rate_uva is stated. So it is the payload bytes per second a
GPU<->host copy can sustain over the link, per direction, at large transfers.

THE DATA. CollectiveX times vLLM's `swap_blocks` (vllm/vllm-openai:v0.27.1 on h100,
v0.25.1 elsewhere; pinned host memory) per direction, layout, block size and count. vLLM
v0.31.0's offload calls swap_blocks_batch (one cuMemcpyBatchAsync) over the same copy
engines; csrc swap_blocks is unchanged v0.27.1 -> v0.31.0 and swap_blocks_batch only adds a
descriptor-chunking loop that is a no-op on CUDA with VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS
unset, so the LARGE-payload rate transfers (the small-copy floor does not).

ONE FIT, NOT TWO. blis-catalog's cpu_dram (5f6ecab) is fitted to the same h100 rows:
contiguous, payload >= 64 MiB, median payload rate per direction (H2D 27,058, D2H 46,089
MB/s). The cap over it is that fit's larger direction, so the cap never binds below either
fitted tier rate and states the link's measured ceiling. Other parts: the only block size
every part measures is 1 MiB, so the estimator is the same median at 1 MiB blocks; on
h100 that gives 46,008 against 46,089 (0.2%), so the two estimators agree where both apply.

THE GATHER RATE. host_gather_rate_uva is an SM-issued UVA read of random 264 B rows. Its
ceiling is the SM-driven H2D read rate, measured by vLLM PR #42212 on H100 (Triton swap
kernel, onload, about 51 GB/s at every SM count >= 20) and by CollectiveX H2D at 1 GiB
blocks (51.6 GB/s): 51 GB/s x 264/288 useful bytes. A measured payload rate already pays
the PCIe line code, so no 128/130 factor. h200 shares the PCIe Gen5 x16 host link and has no
SM-driven measurement: transferred, stated.

THE BODIES MUST BE THE CITED ONES. CollectiveX's API serves only its latest 15 runs (bodies
from https://inferencex.semianalysis.com/api/v1/collectivex/runs/<id>?version=1; the bare
`version=1` query is required). The entries cite the sha256 of the bodies they were fitted on;
if the bodies present hash to anything else, this refuses (exit 3) rather than re-fit and
re-cite the entries on different runs.

Usage:
    python scripts/fit_host_link.py --data <dir of CollectiveX run bodies> [--check]
    (or set COLLECTIVEX_DATA)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
# registry scope -> CollectiveX sku
PARTS = {"h100": "h100-dgxc", "h200": "h200-dgxc", "b200": "b200-nscale", "b300": "b300",
         "gb200-nvl72": "gb200", "gb300": "gb300"}
MIN_PAYLOAD = 64 << 20
H100_ESTIMATOR = "all block sizes"  # the catalog cpu_dram fit's estimator, on its own part
BLOCK = 1 << 20
SM_DRIVEN_H2D = 51000.0  # bytes/us, vLLM PR #42212 (H100, Triton onload ceiling)
ENGRAM_ROW_BYTES, ENGRAM_FETCHED_BYTES = 256 + 8, 256 + 32

_spec = importlib.util.spec_from_file_location("coefficient_blocks", HERE / "coefficient_blocks.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def rows(data: Path) -> tuple[list[tuple], list[str]]:
    out, runs = set(), set()
    for f in sorted(glob.glob(str(data / "cx_run_*.body"))):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        d = d.get("data", d)
        for s in d.get("swap_blocks") or []:
            if s.get("sku") not in PARTS.values():
                continue
            runs.add(Path(f).name)
            for p in s["points"]:
                if p.get("host_memory", "pinned") != "pinned":
                    continue
                out.add((s["sku"], p["direction"], p["layout"], p["block_bytes"],
                         p["payload_bytes"], p["latency_us"]["p50"]))
    return sorted(out), sorted(runs)


def rate(rs, sku, direction, block=None) -> tuple[float, int]:
    v = [b / l for s, d, lay, bb, b, l in rs if s == sku and d == direction
         and lay == "contiguous" and b >= MIN_PAYLOAD and (block is None or bb == block)]
    return statistics.median(v), len(v)


def link_entry(part, value, sku, est, h2d, d2h, sha):
    return [
        "  - host_link_bandwidth:",
        f"      value: {value}",
        "      units: bytes_per_us",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{part}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "InferenceX CollectiveX swap_blocks (api/v1/collectivex/runs, '
        f'contract v1; bodies sha256 {sha}), sku {sku}, pinned host memory, contiguous, payload '
        f'>= 64 MiB, {est}", role: primary}}',
        "      rationale: >",
        f"        The payload rate a GPU<->host copy sustains over this part's host link, as the "
        f"kernel caps a CPU tier's transfer with it. Median payload rate of vLLM swap_blocks "
        f"over contiguous points of at least 64 MiB ({est}): host-to-device {h2d[0]:,.0f} "
        f"bytes/us (n {h2d[1]}), device-to-host {d2h[0]:,.0f} (n {d2h[1]}); the larger is "
        f"committed, so the cap never binds below either direction's measured rate. The same "
        f"rows fit blis-catalog's cpu_dram tier (h100), so tier and cap are one fit, not two. "
        f"Only the large-payload rate transfers to v0.31.0's swap_blocks_batch (one "
        f"cuMemcpyBatchAsync over the same copy engines); the per-call floor does not. "
        f"Replaces the descriptor's nominal PCIe rate. Reproduce with scripts/fit_host_link.py.",
    ]


def cited_sha(text: str) -> set[str]:
    """The body sha256 each committed CollectiveX host_link_bandwidth entry cites."""
    import re
    import yaml
    out = set()
    for e in yaml.safe_load(text)["coefficients"]:
        (name, body), = e.items()
        if name != "host_link_bandwidth":
            continue
        for s in body.get("sources") or []:
            m = re.search(r"CollectiveX swap_blocks .*?bodies sha256 ([0-9a-f]{12})", s["cite"])
            if m:
                out.add(m.group(1))
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("COLLECTIVEX_DATA"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    if not args.data or not Path(args.data).is_dir():
        print("no CollectiveX data: pass --data or set COLLECTIVEX_DATA", file=sys.stderr)
        return 2
    rs, runs = rows(Path(args.data))
    h = hashlib.sha256()
    for r in runs:
        h.update((Path(args.data) / r).read_bytes())
    sha = h.hexdigest()[:12]
    before = SET_PATH.read_text(encoding="utf-8")
    cited = cited_sha(before)
    if cited != {sha}:
        print(f"refusing: the cited CollectiveX bodies are not present (the entries cite bodies "
              f"sha256 {', '.join(sorted(cited))}; the swap_blocks bodies present hash to {sha}). "
              "The API serves only its latest 15 runs; restore the cited bodies to re-derive.",
              file=sys.stderr)
        return 3
    blocks = cb.split(before)
    for part, sku in PARTS.items():
        block = None if part == "h100" else BLOCK
        est = H100_ESTIMATOR if block is None else "1 MiB blocks"
        h2d, d2h = rate(rs, sku, "h2d", block), rate(rs, sku, "d2h", block)
        value = float(round(max(h2d[0], d2h[0])))
        if part == "h100":
            alt = max(rate(rs, sku, "h2d", BLOCK)[0], rate(rs, sku, "d2h", BLOCK)[0])
            print(f"h100 at 1 MiB blocks {alt:,.0f} (vs {value:,.0f}: {alt / value - 1:+.1%})")
        print(f"{part:12s} h2d {h2d[0]:9,.0f} (n {h2d[1]:2d}) d2h {d2h[0]:9,.0f} (n {d2h[1]:2d}) -> {value:,.0f}")
        cb.upsert(blocks, link_entry(part, value, sku, est, h2d, d2h, sha), after=lambda b: False)
    after = cb.join(blocks)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches the fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
