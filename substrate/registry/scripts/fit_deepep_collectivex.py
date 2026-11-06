#!/usr/bin/env python3
"""Fit DeepEP dispatch + combine on GB200-NVL72 (EP8, EP16) and H200 (EP16, two nodes) from
InferenceX CollectiveX.

WHAT THIS CLOSES (inference-sim/blis-registry#60, gap 2). GB200-NVL72 carried one DeepEP
entry, low-latency at 4 ranks (AISimulate, eager). An expert-parallel deployment on an NVL72
rack runs wider than that, and fell back to the generic routed all-to-all. H200 carried DeepEP
at 8 ranks on one node only.

THE DATA. InferenceX CollectiveX run bodies, fetched from
    https://inferencex.semianalysis.com/api/v1/collectivex/runs/<run_id>?version=1
The query must be the bare integer `version=1`: without it, or with `v1`, the API answers
{"error":"Unknown version"}. The API serves only its latest 15 runs. Backend `deepep-v2`,
precision bf16 (BF16 dispatch and combine), shape hidden 7168, top-6 of 384 experts, uniform
routing; the roundtrip (dispatch + combine) p50 per tokens-per-rank point. Within one series
every body that carries it publishes byte-identical points (asserted below), so each entry
cites one body by run id and that body's full sha256: the run it already cites while that
body is present, else the newest carrying the series. A newer run that publishes the same
points therefore leaves the entry unchanged; one that publishes different points fails.

WHAT EACH MODE RUNS, AND HOW IT IS TIMED (InferenceX ea6f44d6eaa0908674e6208d61acc8cb5dc3945b,
the source_sha of the newest run; collectivex/bench and collectivex/docs/methodology.md):
  * `low-latency` drives DeepEP's legacy `deep_ep.Buffer` low-latency kernels, built as vLLM's
    `deepep_low_latency` builds them (num_qps_per_rank = local experts, NVLink allowed; on the
    GB pools `allow_mnnvl=True`, ep_deepep_v2.py:336-350, as vLLM does when
    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL is set), each collective split into a send kernel and a
    receive hook. TIMED BY CUDA-GRAPH REPLAY: CUDA_GRAPH_MODES includes it
    (ep_deepep_v2.py:134-135) and graph replay is the default (ep_backend.py:183-188,
    COLLX_CUDA_GRAPH defaults to 1). There is no eager low-latency series, so no eager pair
    (moe_a2a_*_deepep_low_latency_<N>rank_*) is written. The bodies do not carry
    kernel_generation, so the regime is the harness default's, not read off the row.
  * `normal` is DeepEP v2's ElasticBuffer as vLLM's `deepep_v2` runs it with CUDA graphs on
    (use_cudagraph = not enforce_eager): no CPU sync, no expand, a receive bounded at the next
    power of two of the batch, overlap off, single reduction (ep_deepep_v2.py:166-171,
    197-201; vLLM v0.31.0 prepare_finalize/deepep_v2.py:180-207, unchanged at v0.32.0).
    Decode is replayed as a graph; prefill runs eager on the same no-sync path
    (ep_deepep_v2.py:180-185). That is what the kernel calls `deepep_v2_graph`, the variant
    vLLM runs at every step size when enforce_eager is off, so it is written under that name.
    It is NOT DeepEP's legacy high-throughput (`deepep_high_throughput`) kernels, which
    CollectiveX does not run on these parts, so no high-throughput entry is written.

WHICH DeepEP. CollectiveX runs DeepEP 01dc3aaac820 (library_version in every series); vLLM
v0.31.0 pins d4f41e4e93. The GitHub compare d4f41e4e93...01dc3aa is 6 commits ahead, 0 behind.
None touches csrc/kernels/legacy, where the low-latency kernels live. Four touch the elastic
(v2) path the `normal` rows run (#674, #688, #715, #178), so the v2 entries measure a DeepEP
six commits newer than v0.31.0's; that residual is unmeasured and is stated on them.

THE FORM: latency_us = floor + tokens_per_rank x hidden x 2 x top_k / rate, the basis the
kernel prices a DeepEP layer on (deepep.go). Fitted by mean |log ratio| over a log grid
refined three times (the fit_deepep_a2a.py search).

THE SPLIT, declared here before any held-out value was computed:
  * low-latency (decode ladder 1-256): fit tokens {1, 4, 16, 64, 256}; held out {2, 8, 32, 128},
    the split fit_deepep_ll_graph.py uses.
  * normal (decode ladder 1-512, prefill ladder 1024-8192): fit {1, 4, 16, 64, 256, 1024, 4096};
    held out {2, 8, 32, 128, 512, 2048, 8192}, every other rung of both ladders.
  * band: held-out median |rel err| <= 15%. A series outside the band is printed and not written.

SCOPE. GB200-NVL72 entries are {hardware: [gb200-nvl72]}: EP8 is 2 trays and EP16 is 4 trays of
one NVL72 rack, all over MNNVL with no scale-out, so the link is the chip's. The H200 EP16
entries are two nodes over ConnectX-7 400G InfiniBand (CollectiveX h200-dgxc), so the value
depends on the fabric and is scoped {hardware: [h200], fabric: [ib-400g]}. The rank count rides
in the name, as the kernel builds it.

WHAT THE H200 EP16 ROWS MEASURE, AND WHAT THEY DO NOT. CollectiveX's h200-dgxc nodes are KVM
guests. Its methodology records that there strict-ordered GPU-NIC writes cap near 10 GB/s per
GPU; the `normal` rows run its relaxed-ordering registration patch (`rdma_relaxed_ordering` in
the pool's network block), which stock DeepEP does not apply. The low-latency rows ride NVSHMEM
IBGDA, which wants gdrdrv on the node; only b200-nscale's bare metal is stated to have it. The
h200 low-latency EP16 rate (about 5 GB/s of logical payload) is far below b200-nscale's on the
same 400G InfiniBand. So these are measurements of that pool's path, not of the fabric. Neither
is the InfiniBand fabric's efficiency: scripts/emit_fabric.py does not use them.

READERS. blis-latency-kernel (deepep.go liftDeepEP) reads the graph pair only beside an eager
low-latency pair of the same width, and refuses DeepEP for an expert group that spans nodes.
So, as of its 726f64b: the gb200 deepep_v2_graph entries are read; the gb200 low-latency graph
entries are read once an eager 8- or 16-rank pair exists or the kernel accepts a graph-only
pair; the h200 16-rank entries are read once it prices a cross-node DeepEP group.

Usage:
    python scripts/fit_deepep_collectivex.py --data <dir of CollectiveX run bodies> [--check]
    (or set COLLECTIVEX_DATA)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
SET_PATH = REPO / "coefficients" / "cost-model-collectives-inferencex.yaml"
URL = "https://inferencex.semianalysis.com/api/v1/collectivex/runs/{run}?version=1"
HIDDEN, TOP_K, EXPERTS = 7168, 6, 384
BACKEND, PRECISION = "deepep-v2", "bf16"

# The declared split and band (see the module docstring); written before the fit was read.
SPLIT = {
    "low-latency": ({1, 4, 16, 64, 256}, {2, 8, 32, 128}),
    "normal": ({1, 4, 16, 64, 256, 1024, 4096}, {2, 8, 32, 128, 512, 2048, 8192}),
}
BAND = 0.15

# CollectiveX mode -> the kernel's variant name.
VARIANT = {"low-latency": "deepep_low_latency_graph", "normal": "deepep_v2_graph"}

# (registry part, CollectiveX sku, ranks, nodes, fabric or None). nodes is CollectiveX's host
# count: a GB200 tray is a host, so EP8 is 2 and EP16 is 4, all inside one MNNVL domain.
SERIES = [
    ("gb200-nvl72", "gb200", 8, 2, None),
    ("gb200-nvl72", "gb200", 16, 4, None),
    ("h200", "h200-dgxc", 16, 2, "ib-400g"),
]

LIBRARY = "01dc3aaac82068020353dce2c302e38153c0bfaa"
DIFF = {
    "low-latency": ("DeepEP d4f41e4e93 (vLLM v0.31.0's pin)...01dc3aa: 6 commits ahead, 0 "
                    "behind, none under csrc/kernels/legacy, where these kernels live."),
    "normal": ("DeepEP d4f41e4e93 (vLLM v0.31.0's pin)...01dc3aa: 6 commits ahead, 0 behind, "
               "four of them (#674, #688, #715, #178) on the elastic path these rows run, so "
               "this measures a DeepEP six commits newer than v0.31.0's; that residual is "
               "unmeasured."),
}
WHAT = {
    "low-latency": (
        "DeepEP low-latency (legacy Buffer kernels, send kernel plus receive hook, as vLLM's "
        "deepep_low_latency builds them) dispatch plus combine as one MoE layer's price under "
        "CUDA-graph replay, which is how CollectiveX times this mode (ep_deepep_v2.py:134-135, "
        "ep_backend.py:183-188, InferenceX ea6f44d); there is no eager series, so no eager pair "
        "is written."),
    "normal": (
        "DeepEP v2 ElasticBuffer dispatch plus combine with vLLM deepep_v2's CUDA-graphs-on "
        "flags (no CPU sync, no expand, receive bounded at the next power of two of the batch, "
        "overlap off, single reduction; vLLM v0.31.0 prepare_finalize/deepep_v2.py:180-207), "
        "which is the kernel's deepep_v2_graph. CollectiveX replays the decode ladder as a "
        "graph and runs the prefill ladder eager on the same path (ep_deepep_v2.py:166-185, "
        "InferenceX ea6f44d), as vLLM runs a large prefill batch uncaptured; the fit spans "
        "both. Not DeepEP's legacy high-throughput kernels, which CollectiveX does not run."),
}
WHERE = {
    ("gb200-nvl72", 8): ("EP8 over 2 GB200 trays (4 GPUs each) inside one NVL72 MNNVL domain, "
                         "no scale-out."),
    ("gb200-nvl72", 16): ("EP16 over 4 GB200 trays inside one NVL72 MNNVL domain, no "
                          "scale-out."),
    ("h200", 16): (
        "EP16 over two 8-GPU H200 nodes, ConnectX-7 400G InfiniBand (CollectiveX h200-dgxc), "
        "so the value depends on the fabric and is scoped to ib-400g. CAVEAT: h200-dgxc's nodes "
        "are KVM guests; CollectiveX records that there strict-ordered GPU-NIC writes cap near "
        "10 GB/s per GPU, and its normal-mode rows run a relaxed-ordering registration patch "
        "stock DeepEP does not; its low-latency rows ride NVSHMEM IBGDA, which wants gdrdrv, "
        "stated present only on b200-nscale's bare metal. So this measures that pool's path, "
        "not the InfiniBand fabric, and is not the fabric's efficiency. blis-latency-kernel "
        "(726f64b) refuses DeepEP for a group that spans nodes, so this has no reader yet."),
}

_spec = importlib.util.spec_from_file_location(
    "coefficient_blocks", Path(__file__).parent / "coefficient_blocks.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def stem_chip(part: str) -> str:
    return part.replace("-", "_")


def bodies(data: Path) -> list[tuple[str, Path, dict]]:
    out = []
    for f in sorted(glob.glob(str(data / "cx_run_*.body"))):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        out.append((d["run"]["run_id"], Path(f), d))
    return sorted(out, key=lambda x: int(x[0]))


def series(data: Path, sku: str, mode: str, ranks: int, nodes: int, prefer: str | None = None):
    """The series' points {tokens: roundtrip p50 us}, the run cited for it, the other runs
    that publish it, and the cited body's sha256. The cited run is `prefer` (the run the
    committed entry already cites) while that body is present, else the newest run carrying
    the series, so a newer run publishing the same points does not move a citation. Refuses
    a series whose points differ between bodies, since then one body is not the measurement."""
    phases = ("decode",) if mode == "low-latency" else ("decode", "prefill")
    found: dict[str, dict[int, float]] = {}
    paths = {}
    for run, path, d in bodies(data):
        pts: dict[int, float] = {}
        for s in d.get("series") or []:
            sy = s.get("system") or {}
            if (s.get("backend"), s.get("mode"), s.get("precision"), sy.get("sku"),
                    sy.get("ep_size"), sy.get("nodes")) != (BACKEND, mode, PRECISION, sku,
                                                            ranks, nodes):
                continue
            if s.get("phase") not in phases:
                continue
            if not str(s.get("library_version", "")).endswith(LIBRARY):
                raise SystemExit(f"{s['series_id']} in run {run}: DeepEP "
                                 f"{s.get('library_version')}, not {LIBRARY}")
            for p in s["points"]:
                pts[p["tokens_per_rank"]] = p["components"]["roundtrip"]["latency_us"]["p50"]
        if pts:
            found[run], paths[run] = pts, path
    if not found:
        return None
    newest = prefer if prefer in found else max(found, key=int)
    for run, pts in found.items():
        if pts != found[newest]:
            raise SystemExit(f"{sku} {mode} EP{ranks}: run {run} publishes different points "
                             f"from run {newest}; one body is not the measurement")
    sha = hashlib.sha256(paths[newest].read_bytes()).hexdigest()
    # The corroborating runs named are those up to the cited one, so a later run that
    # republishes the same points does not rewrite the entry.
    others = sorted((r for r in found if r != newest and int(r) < int(newest)), key=int)
    return found[newest], newest, others, sha


def fit(b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """floor (us) and rate (B/us) minimising mean |log(pred / y)|, by a deterministic log
    grid refined three times around its best point (fit_deepep_a2a.py fit_phase)."""
    lo_f, hi_f = np.log(max(y.min() * 0.2, 1e-3)), np.log(y.min() * 1.2)
    lo_r, hi_r = np.log(1e2), np.log(1e8)
    best = None
    for _ in range(4):
        fl = np.exp(np.linspace(lo_f, hi_f, 241))
        rt = np.exp(np.linspace(lo_r, hi_r, 241))
        pred = fl[:, None, None] + b[None, None, :] / rt[None, :, None]
        loss = np.mean(np.abs(np.log(pred / y[None, None, :])), axis=2)
        i, j = np.unravel_index(np.argmin(loss), loss.shape)
        best = (float(fl[i]), float(rt[j]), float(loss[i, j]))
        sf, sr = (hi_f - lo_f) / 24, (hi_r - lo_r) / 24
        lo_f, hi_f = np.log(fl[i]) - sf, np.log(fl[i]) + sf
        lo_r, hi_r = np.log(rt[j]) - sr, np.log(rt[j]) + sr
    return best


def basis(t: int) -> float:
    return t * HIDDEN * 2.0 * TOP_K


def fit_series(pts: dict[int, float], mode: str):
    fit_t, held_t = SPLIT[mode]
    missing = (fit_t | held_t) - set(pts)
    if missing:
        raise SystemExit(f"{mode}: the declared split names tokens {sorted(missing)} the "
                         f"series does not carry")
    ts = sorted(fit_t)
    fl, rt, loss = fit(np.array([basis(t) for t in ts]), np.array([pts[t] for t in ts]))
    floor, rate = round(fl, 1), float(f"{rt:.4g}")
    held = {t: (floor + basis(t) / rate) / pts[t] - 1 for t in sorted(held_t)}
    med = statistics.median(abs(x) for x in held.values())
    return floor, rate, loss, held, med


def entry(name, value, units, part, fabric, mode, ranks, run, others, sha, n_fit, loss,
          held, med):
    scope = (f"{{hardware: [{part}], fabric: [{fabric}]}}" if fabric
             else f"{{hardware: [{part}]}}")
    fit_t, held_t = SPLIT[mode]
    line = ", ".join(f"T{t} {x:+.1%}" for t, x in held.items())
    also = (f"; the same points are published byte-identically in runs "
            f"{', '.join(others)}" if others else "")
    cite = (f"InferenceX CollectiveX run {run} ({URL.format(run=run)}; body sha256 {sha}"
            f"{also}), backend deepep-v2 mode {mode}, bf16, EP{ranks}, hidden {HIDDEN} top-"
            f"{TOP_K} of {EXPERTS} uniform, roundtrip p50")
    rat = (f"{WHAT[mode]} {WHERE[(part, ranks)]} floor + tokens x hidden x 2 x top_k / rate, "
           f"per-rank tokens, fitted by mean |log ratio| ({loss:.3f}) on tokens "
           f"{'/'.join(map(str, sorted(fit_t)))} ({n_fit} points). Held out (tokens "
           f"{'/'.join(map(str, sorted(held_t)))}, declared before the fit): median |rel err| "
           f"{med:.1%}; {line}. {DIFF[mode]} Reproduce with scripts/fit_deepep_collectivex.py.")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {scope}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


OWNED = re.compile(r"^moe_a2a_(floor|rate)_(deepep_low_latency_graph|deepep_v2_graph)_"
                   r"(\d+)rank_([a-z0-9_]+)$")
CITE = re.compile(r"InferenceX CollectiveX run (\d+) \(\S+; body sha256 ([0-9a-f]{64})")


def owned(name: str) -> bool:
    m = OWNED.match(name)
    if not m:
        return False
    return any(int(m.group(3)) == r and m.group(4) == stem_chip(p) for p, _, r, _, _ in SERIES)


def cited_runs(text: str) -> dict[str, str]:
    """The run each committed entry this writer owns cites, by entry name."""
    out = {}
    for e in yaml.safe_load(text)["coefficients"]:
        (name, body), = e.items()
        if owned(name):
            for s in body.get("sources") or []:
                m = CITE.search(s["cite"])
                if m:
                    out[name] = m.group(1)
    return out


def cited_bodies_absent(text: str, data: Path) -> str:
    problems = []
    for e in yaml.safe_load(text)["coefficients"]:
        (name, body), = e.items()
        if not owned(name):
            continue
        for s in body.get("sources") or []:
            m = CITE.search(s["cite"])
            if not m:
                continue
            path = data / f"cx_run_{m.group(1)}.body"
            if not path.is_file():
                problems.append(f"run {m.group(1)} absent")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != m.group(2):
                problems.append(f"run {m.group(1)} body differs from sha256 {m.group(2)[:12]}")
    return "; ".join(sorted(set(problems)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("COLLECTIVEX_DATA"),
                    help="directory of CollectiveX run bodies (default $COLLECTIVEX_DATA)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    if not args.data or not Path(args.data).is_dir():
        print("no CollectiveX data: pass --data or set COLLECTIVEX_DATA", file=sys.stderr)
        return 2
    data = Path(args.data)
    before = SET_PATH.read_text(encoding="utf-8")
    missing = cited_bodies_absent(before, data)
    if missing:
        print(f"refusing: the cited CollectiveX bodies are not present ({missing}). The API "
              "serves only its latest 15 runs; restore the cited bodies to re-derive, or fit "
              "new runs into new entries rather than re-citing these.", file=sys.stderr)
        return 3
    cited = cited_runs(before)
    blocks = cb.split(before)
    for part, sku, ranks, nodes, fabric in SERIES:
        for mode in ("low-latency", "normal"):
            stem = f"{VARIANT[mode]}_{ranks}rank_{stem_chip(part)}"
            got = series(data, sku, mode, ranks, nodes, cited.get(f"moe_a2a_floor_{stem}"))
            if got is None:
                raise SystemExit(f"{sku} {mode} EP{ranks}: no series in {data}")
            pts, run, others, sha = got
            floor, rate, loss, held, med = fit_series(pts, mode)
            ok = med <= BAND
            line = ", ".join(f"T{t} {x:+.1%}" for t, x in held.items())
            print(f"{part:12s} EP{ranks:<2d} {mode:11s} run {run}: floor {floor:7.1f} us "
                  f"rate {rate / 1e3:7.1f} GB/s fit {loss:.3f}; held out {line}; median "
                  f"{med:.1%} max {max(abs(x) for x in held.values()):.1%} "
                  f"({'pass' if ok else 'FAIL, not written'})")
            if not ok:
                continue
            for name, value, units in ((f"moe_a2a_floor_{stem}", floor, "us_per_transfer"),
                                       (f"moe_a2a_rate_{stem}", rate, "bytes_per_us")):
                cb.upsert(blocks, entry(name, value, units, part, fabric, mode, ranks, run,
                                        others, sha, len(SPLIT[mode][0]), loss, held, med),
                          after=lambda b: b[0] == "coefficients:" or cb.name_of(b) is not None)
    after = cb.join(blocks)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches the fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    yaml.safe_load(after)
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
