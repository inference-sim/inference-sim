#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash host costs: the eager per-kernel launch cost of an uncompiled model,
and the front end's admission cost, each derived from a vendored measurement.

    python scripts/fit_deepseek_v41_host.py                 # print, rewrite the host set
    python scripts/fit_deepseek_v41_host.py --check         # exit 1 if the set would change
    python scripts/fit_deepseek_v41_host.py --extract-anchor <report.html>
        # re-derive the anchor's inputs from a local copy of the source report

    V41_EAGER_ANCHOR_REPORT=<report.html> python scripts/fit_deepseek_v41_host.py --check
        # also verify the vendored inputs against the full report (sha256 and values)

Owns, in cost-model-host-overheads.yaml, scoped to model deepseek-v4.1-flash:

  host_launch_eager_per_kernel  {hardware: [h200]}       assumed  backed out of an H200 enforce-eager anchor
  host_launch_eager_per_kernel  {hardware: [8 others]}   copied   the h200 value: an assumed transfer
  host_admission_per_token      {hardware: [all]}   assumed  vLLM v0.31.0's V4.1 encoder +
                                                             tokenizer + block hash, timed on an
                                                             M1 Max: a transfer to every part
  host_admission_per_token_rust_frontend
                                {hardware: [all]}   assumed  the per_token value / fastokens'
                                                             stated speedup: what the vllm-rs
                                                             frontend (engine.frontend: rust)
                                                             charges, a vendor README claim
  host_admission_per_request    {hardware: [all]}   not_charged  0.0: an inference from what
                                                             was timed (the schema requires this
                                                             method for 0), not a structural zero

THE TWO INPUTS ARE VENDORED (validator/testdata/anchors/), not retyped:
  * v41_h200_enforce_eager_anchor.json -- what the fit needs from an internal
    enforce-eager DeepSeek-V4.1-Flash H200 serving run, and nothing else: the decode
    levels over which ITL is flat, the median, minimum and maximum mean ITL over them, the
    setup, and the sha256 of the source report. The report itself is not redistributed.
    --extract-anchor derives the file from a local copy; with V41_EAGER_ANCHOR_REPORT set,
    --check re-derives it and refuses a mismatch, and without it --check verifies the
    derived inputs' internal consistency.
  * v41_frontend_bench.json -- the rows diagnosis/frontend/bench_frontend.py printed.
  * v41_kernel_eager_basis.json -- the KERNEL's own device time and launch count for the
    anchor's step, from diagnosis/kdevice, via --record-device. --check refuses a basis
    recorded for another anchor, scenario or host composition. After any kernel or
    registry change that moves that step, regenerate it:
        (cd diagnosis/kdevice && go run . -kernel <kernel> -registry <registry>) > dev.json
        python scripts/fit_deepseek_v41_host.py --record-device dev.json
        python scripts/fit_deepseek_v41_host.py

WHY THE EAGER TERM IS `assumed`. Nothing here observes a kernel launch: the value is backed
out of end-to-end ITL, as host_launch_per_kernel is, so whatever else the model gets wrong at
one decode request lands in it. That is the registry's rule for such a value (methodology:
`assumed`, fitted: false), and `derived` is not in the closed method vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

# fastokens, the tokenizer the vllm-rs frontend loads ("loading tokenizer with fastokens"):
# its README states "on average achieves a 10x+ faster tokenization compared to the
# `tokenizers` library" (crusoecloud/fastokens 7923de6c, README blob e0bb3b78, Apache-2.0).
FASTOKENS = "crusoecloud/fastokens 7923de6c76753347d4090c444e22f6c95a15f01b README.md (blob e0bb3b7811029d8c7d7f97823ad179c87cc9a715)"
FASTOKENS_SPEEDUP = 10
# The public cold-encode range of the Python path: vllm PR #47583's DeepSeek-V3 tokenizer
# benchmark, 35.66 ms at 32K and 160.40 ms at 128K tokens (1.11-1.25 us/token).
PY_PATH_LOW_US = 1.11

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

REPO = HERE.parent
HOST_PATH = REPO / "coefficients" / "cost-model-host-overheads.yaml"
ANCHORS = REPO / "validator" / "testdata" / "anchors"
ANCHOR_JSON = ANCHORS / "v41_h200_enforce_eager_anchor.json"
ANCHOR_SOURCE = ("an internal DeepSeek-V4.1-Flash H200 enforce-eager serving run's report "
                 "(not redistributed)")
ANCHOR_ENV = "V41_EAGER_ANCHOR_REPORT"
BENCH_JSON = ANCHORS / "v41_frontend_bench.json"
KERNEL_BASIS_JSON = ANCHORS / "v41_kernel_eager_basis.json"
MODEL = "deepseek-v4.1-flash"

# The decode levels over which the report's ITL is flat (host-bound, no queueing): c1-c64.
ANCHOR_LEVELS = (1, 2, 4, 8, 16, 32, 64)

# THE COMPOSITION IT MUST MATCH: ADDITIVE. The kernel charges an eager step's launches and its
# device time in sequence, step = launches x (eager + dispatch) + device, so the device time is
# subtracted. A launch-overlap form, step ~ max(launches, device), was tried in the kernel
# and REJECTED: it does not reproduce the H200 DSpark arm's capture-ceiling collapse at c32,
# whose measured throughput the additive form matches (blis-latency-kernel 07ed424,
# TestTheEagerDecodeAnchorIsAdditive). The basis records the composition, and --check refuses
# any other.
EAGER_COMPOSITION = "additive: step = launches x (eager + dispatch) + device"
# THE KERNEL BASIS: the eager constant is (ITL - device) / launches - host_launch_per_kernel,
# and both the device time and the launch count are the KERNEL's own for the anchor's step
# (TP8-no-DSpark, one decode request, one token, ctx 90,000). They change with the kernel
# build and with this registry's device entries, so they are a recorded input, vendored with
# their provenance in KERNEL_BASIS_JSON and regenerated by --record-device from the output
# of diagnosis/kdevice (public kernel API only). The launch count is read off the same step:
# its graph-mode host time is launches x host_launch_per_kernel + host_replay_graph_per_step.
KERNEL_BASIS_SCENARIO = ("the TP8-no-DSpark H200 transfer scenario, decode n=1, 1 token, "
                         "ctx 90000, no speculation")
KDEVICE_SCENARIO_SUFFIX = ", decode n=1, 1 token, ctx 90000, no speculation"
# THE ANCHOR'S STEP TYPE FIXES THE COUNT. The anchor is a decode step with no speculation,
# and vLLM launches the graph's `phase: prefill` nodes (prefill_dequant_gather,
# index_prefill_k_gather: 46 launches) only in a step with a prefill row
# (vllm/models/deepseek_v41/nvidia/flashmla.py:143-165, sparse_attn_indexer.py:416-417 at
# v0.31.0). The kernel's decode-only target count is 972 and a step with a prefill row
# 1018; kernel 993ddeb used 1018 on every step, which put 46 prefill-only launches into the
# divisor. So the basis must say it priced a decode step and carry this count; a kernel whose
# decode count legitimately changes needs this pin moved with the reason.
DECODE_STEP_LAUNCHES = 972
# AISimulate's measured device sum for one request x 6 tokens at ctx 90,000 (diagnosis/aisim),
# the upper end of the device a one-token step can have; used only for the sensitivity.
DEVICE_AISIM_MS = 7.34

# The part whose nodes ran the anchor.
ANCHOR_PART = "h200"
ALL_PARTS = ["a100-80", "a100-sxm", "b200", "b300", "gb200-nvl72", "gb300", "h100", "h200",
             "l40s"]
# The workload the transfer check runs (weka agentic replay) has a mean prompt of ~89.6k
# tokens; the bench row nearest it supplies the committed per-token value.
WORKLOAD_PROMPT_TOKENS = 89_600


def js_array(html: str, name: str) -> list[dict]:
    m = re.search(rf"const {name}\s*=\s*\[", html)
    if not m:
        raise SystemExit(f"the report has no const {name}")
    i, depth = m.end() - 1, 0
    for j in range(i, len(html)):
        depth += {"[": 1, "]": -1}.get(html[j], 0)
        if depth == 0:
            break
    lit = html[i:j + 1]
    lit = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', lit)
    lit = re.sub(r",(\s*[}\]])", r"\1", lit)
    return json.loads(lit)


def derive_anchor(report: Path) -> dict:
    """The anchor's derived inputs from the source report: only what the fit needs."""
    raw = report.read_bytes()
    rows = js_array(raw.decode(), "WK")
    by_c = {r["c"]: r["itl"] for r in rows}
    missing = [c for c in ANCHOR_LEVELS if c not in by_c]
    if missing:
        raise SystemExit(f"the report lacks levels {missing}")
    itl = [by_c[c] for c in ANCHOR_LEVELS]
    return {
        "source": ANCHOR_SOURCE,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "setup": ("2 replicas behind the llm-d EPP router, TP8 per replica, 8x H200 each, "
                  "vLLM nightly ddd6fbc, --enforce-eager, no speculative decoding, CPU KV "
                  "offload + Engram CPU offload, an agentic multi-turn trace replay"),
        "plateau": {"levels": list(ANCHOR_LEVELS), "unit": "ms, mean ITL per level",
                    "median_itl_ms": statistics.median(itl), "min_itl_ms": min(itl),
                    "max_itl_ms": max(itl)},
        "derived_by": "scripts/fit_deepseek_v41_host.py --extract-anchor <report.html>",
    }


def extract_anchor(report: Path) -> int:
    doc = derive_anchor(report)
    ANCHORS.mkdir(parents=True, exist_ok=True)
    ANCHOR_JSON.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"anchor inputs -> {ANCHOR_JSON}")
    return 0


def verify_anchor(anchor: dict, env: dict) -> str:
    """Against the full report when a local copy is named by ANCHOR_ENV; otherwise the
    derived inputs' own consistency."""
    pl = anchor["plateau"]
    if tuple(pl["levels"]) != ANCHOR_LEVELS:
        raise SystemExit(f"anchor plateau levels {pl['levels']}, expected {ANCHOR_LEVELS}")
    if not pl["min_itl_ms"] <= pl["median_itl_ms"] <= pl["max_itl_ms"]:
        raise SystemExit("anchor plateau median is outside its own range")
    if not re.fullmatch(r"[0-9a-f]{64}", anchor.get("sha256", "")):
        raise SystemExit("anchor sha256 is not a sha256")
    path = env.get(ANCHOR_ENV)
    if not path:
        return "derived inputs only (set " + ANCHOR_ENV + " to verify against the report)"
    full = derive_anchor(Path(path))
    if full != anchor:
        raise SystemExit(f"the vendored anchor does not re-derive from {ANCHOR_ENV}")
    return "re-derived from the report"


def committed(name: str, chip: str) -> float:
    doc = yaml.safe_load(HOST_PATH.read_text())
    for e in doc["coefficients"]:
        ((n, body),) = e.items()
        sc = body["scope"]
        if n == name and chip in sc.get("hardware", []) and "model" not in sc:
            return float(body["value"])
    raise SystemExit(f"no committed part-wide {name} on {chip}")


def lines(name, value, units, method, fitted, scope, sources, rationale, copied_from=None):
    out = [f"  - {name}:", f"      value: {value}", f"      units: {units}",
           f"      method: {method}"]
    if copied_from:
        out.append(f'      copied_from: "{copied_from}"')
    out += [f"      fitted: {'true' if fitted else 'false'}",
            f"      scope: {scope}", "      sources:"]
    out += [f'        - {{kind: {k}, cite: "{c}", role: {r}}}' for k, c, r in sources]
    return out + ["      rationale: >", f"        {rationale}"]


def eager(anchor: dict) -> dict:
    pl = anchor["plateau"]
    lo, med, hi = pl["min_itl_ms"], pl["median_itl_ms"], pl["max_itl_ms"]
    if hi / lo > 1.15:
        raise SystemExit(f"anchor ITL is not flat over c1-c64 ({lo}-{hi} ms); the step is "
                         f"not host-bound there and the derivation does not hold")
    launch = committed("host_launch_per_kernel", "h200")
    kb = kernel_basis(anchor, launch)
    n, dev = kb["kernel_count"], kb["device_ms"]
    if kb.get("composition") != EAGER_COMPOSITION:
        raise SystemExit(f"the kernel basis records composition {kb.get('composition')!r}; "
                         f"this constant is backed out under {EAGER_COMPOSITION!r}")
    e = (med - dev) / n * 1e3 - launch
    e_lo = (lo - dev) / n * 1e3 - launch
    e_hi = (hi - dev) / n * 1e3 - launch
    e_aisim = (med - DEVICE_AISIM_MS) / n * 1e3 - launch
    return {"median": med, "e": round(e, 2), "e_lo": e_lo, "e_hi": e_hi, "e_aisim": e_aisim,
            "launch": launch, "lo": lo, "hi": hi, "source": anchor["source"],
            "sha": anchor["sha256"], "setup": anchor["setup"], "per_ms": 1e3 / n,
            "n": n, "dev": dev, "kb": kb}


def launches_of(host_ms: float, launch: float, replay: float) -> int:
    """The graph-mode step's launch count: host = launches x launch + replay."""
    x = (host_ms * 1e3 - replay) / launch
    if abs(x - round(x)) > 0.05:
        raise SystemExit(f"graph-mode host {host_ms} ms is not launches x {launch} us + "
                         f"{replay} us replay ({x:.3f} launches): the kernel's host "
                         f"composition changed; read the count another way")
    return int(round(x))


def kernel_basis(anchor: dict, launch: float) -> dict:
    """The vendored kernel basis, refused unless it is the one this anchor is divided by."""
    if not KERNEL_BASIS_JSON.is_file():
        raise SystemExit(f"{KERNEL_BASIS_JSON} is absent; run --record-device")
    kb = json.loads(KERNEL_BASIS_JSON.read_text())
    problems = []
    if kb.get("anchor_sha256") != anchor["sha256"]:
        problems.append(f"recorded for anchor {kb.get('anchor_sha256')}, the vendored anchor "
                        f"is {anchor['sha256']}")
    if kb.get("scenario") != KERNEL_BASIS_SCENARIO:
        problems.append(f"scenario {kb.get('scenario')!r}, expected {KERNEL_BASIS_SCENARIO!r}")
    if kb.get("step") != "decode":
        problems.append(f"step {kb.get('step')!r}: the anchor is a decode step, so the count "
                        f"must be a decode step's")
    if kb.get("kernel_count") != DECODE_STEP_LAUNCHES:
        problems.append(f"kernel_count {kb.get('kernel_count')} is not the decode-only count "
                        f"{DECODE_STEP_LAUNCHES} (1018 counts the 46 phase: prefill launches a "
                        f"decode step never runs)")
    if not kb.get("prefill_row_host_ms", 0) > res_host(kb):
        problems.append("the basis lacks a prefill-row step pricing above the decode step's "
                        "host, so the count cannot be tied to the step type")
    for what in ("kernel_commit", "registry_commit", "catalog_commit"):
        if not clean_commit(kb.get(what, "")):
            problems.append(f"{what} {kb.get(what)!r} is not a clean git commit, so the basis "
                            f"cannot be re-derived")
    if kb.get("graph_mode") != "FULL":
        problems.append(f"graph mode {kb.get('graph_mode')!r}: the count is read off a FULL "
                        f"graph step")
    res = kb.get("per_resource_ms", {})
    device = sum(v for r, v in res.items() if r != "host")
    if abs(device - kb.get("device_ms", -1)) > 1e-6:
        problems.append(f"device_ms {kb.get('device_ms')} is not the non-host sum {device}")
    replay = committed("host_replay_graph_per_step", "h200")
    if "host" not in res or launches_of(res["host"], launch, replay) != kb.get("kernel_count"):
        problems.append(f"kernel_count {kb.get('kernel_count')} does not match the recorded "
                        f"graph-mode host time")
    if problems:
        raise SystemExit("the vendored kernel basis does not match the anchor record: "
                         + "; ".join(problems))
    return kb


def res_host(kb: dict) -> float:
    return float(kb.get("per_resource_ms", {}).get("host", math.inf))


def clean_commit(c: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", c or "") is not None


def record_device(probe: Path, kernel_commit: str | None) -> int:
    """Vendor diagnosis/kdevice's output as the kernel basis, with its provenance."""
    doc = json.loads(probe.read_text())
    anchor = json.loads(ANCHOR_JSON.read_text())
    sc = doc.get("scenario", "")
    if not (sc.endswith(KDEVICE_SCENARIO_SUFFIX) and "TP8-no-DSpark" in sc):
        raise SystemExit(f"probe priced {sc!r}, not the anchor's step")
    doc["scenario"] = KERNEL_BASIS_SCENARIO
    launch = committed("host_launch_per_kernel", "h200")
    replay = committed("host_replay_graph_per_step", "h200")
    doc["kernel_count"] = launches_of(doc["per_resource_ms"]["host"], launch, replay)
    if kernel_commit:
        doc["kernel_commit"] = kernel_commit
    for what in ("kernel_commit", "registry_commit", "catalog_commit"):
        if not clean_commit(doc.get(what, "")):
            raise SystemExit(f"{what} {doc.get(what)!r} is not a clean git commit; commit "
                             f"the checkout and re-run kdevice")
    doc["anchor_sha256"] = anchor["sha256"]
    if doc.get("step") != "decode" or doc["kernel_count"] != DECODE_STEP_LAUNCHES:
        raise SystemExit(f"probe step {doc.get('step')!r} with {doc['kernel_count']} launches; "
                         f"the anchor's decode step has {DECODE_STEP_LAUNCHES}")
    doc["composition"] = EAGER_COMPOSITION
    doc["recorded_by"] = ("diagnosis/kdevice (go run . -kernel <checkout> -registry <checkout>)"
                          " | scripts/fit_deepseek_v41_host.py --record-device")
    KERNEL_BASIS_JSON.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"kernel basis: device {doc['device_ms']:.3f} ms, {doc['kernel_count']} launches, "
          f"kernel {doc['kernel_commit']} -> {KERNEL_BASIS_JSON}")
    return 0


def admission(bench: dict) -> dict:
    rows = bench["rows"]
    per = [(r["tokens"], r["total_ms"] * 1e3 / r["tokens"]) for r in rows]
    nearest = min(rows, key=lambda r: abs(r["tokens"] - WORKLOAD_PROMPT_TOKENS))
    value = round(nearest["total_ms"] * 1e3 / nearest["tokens"], 2)
    # Least-squares line over the rows: its intercept is the fixed per-request cost.
    n = len(rows)
    xs = [r["tokens"] for r in rows]
    ys = [r["total_ms"] * 1e3 for r in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    icpt = my - slope * mx
    share = {k: sum(r[k] for r in rows) / sum(r["total_ms"] for r in rows)
             for k in ("tokenize_ms", "encode_ms", "hash_ms", "json_ms")}
    return {"value": value, "nearest": nearest, "per": per, "slope": slope, "icpt": icpt,
            "share": share, "lo": min(p for _, p in per), "hi": max(p for _, p in per),
            "procedure": bench["procedure"], "host": bench["host"]}


def entries(ea: dict, ad: dict) -> list:
    out = []
    src_anchor = [("discussion", f"{ea['source']}, sha256 {ea['sha'][:12]}; derived inputs "
                   f"in validator/testdata/anchors/{ANCHOR_JSON.name}: mean ITL c1-c64", "primary"),
                  ("discussion", f"validator/testdata/anchors/{KERNEL_BASIS_JSON.name}: the "
                   f"kernel's own device time and launch count for the anchor's step, "
                   f"blis-latency-kernel {ea['kb']['kernel_commit']}", "supporting")]
    form = (
        f"THE FORM the kernel evaluates on a step that runs with no CUDA graph (cudagraph mode "
        f"NONE): host = launches x (host_launch_eager_per_kernel + host_launch_per_kernel), "
        f"charged per launch, launches being the kernel's own eager count for the step "
        f"(target, plus the drafter when it also runs eager). For V4.1 NONE is PURE EAGER "
        f"PyTorch, not compiled-without-graph: vLLM v0.31.0 'DeepseekV41ForCausalLM cannot "
        f"torch.compile' (vllm/config/vllm.py:120) and the runner calls the raw module, so "
        f"every op pays Python dispatch; that is what this term is, and why it is model-"
        f"scoped rather than a part-wide launch constant.")
    deriv = (
        f"DERIVATION, from the vendored anchor (validator/testdata/anchors/"
        f"{ANCHOR_JSON.name}, derived from {ANCHOR_SOURCE}): {ea['setup']}. Its mean ITL "
        f"is flat from c1 to c64 -- {ea['lo']:g}-{ea['hi']:g} ms -- so the step is "
        f"host-bound and "
        f"one decode step costs the median, {ea['median']:.2f} ms. THE COMPOSITION is the "
        f"kernel's ADDITIVE one, step = launches x (eager + dispatch) + device; a "
        f"launch-overlap form (step ~ max(launches, device)) was rejected because it does not "
        f"reproduce the H200 DSpark arm's capture-ceiling collapse at c32. e = (ITL - device) / "
        f"launches - host_launch_per_kernel = ({ea['median']:.2f} - {ea['dev']:.2f} ms) / "
        f"{ea['n']} - {ea['launch']:g} us = {ea['e']} us, with {ea['n']} launches and "
        f"{ea['dev']:.2f} ms of device time the KERNEL's own for that step (TP8-no-DSpark, "
        f"one decode request, one token, ctx 90,000), recorded from blis-latency-kernel "
        f"{ea['kb']['kernel_commit']} with this registry at {ea['kb']['registry_commit'][:12]} "
        f"(validator/testdata/anchors/{KERNEL_BASIS_JSON.name}). THE CONSTANT IS VALID ONLY "
        f"WITH THAT KERNEL BUILD: a kernel or registry change that moves the step's device "
        f"time or launch count requires re-recording the basis and re-running this writer, "
        f"which refuses a basis recorded for another anchor or step type. SENSITIVITY: the "
        f"plateau's fastest and slowest levels give {ea['e_lo']:.1f}-{ea['e_hi']:.1f} us; each 1 ms of device time moves e by {ea['per_ms']:.2f} us, so "
        f"AISimulate's measured {DEVICE_AISIM_MS} ms device sum (one request x 6 tokens, the "
        f"upper end for a one-token step) would give {ea['e_aisim']:.1f} us; the anchor pins "
        f"launches x e ({ea['n'] * ea['e'] / 1e3:.1f} ms), so the constant must be "
        f"re-derived if the kernel's count changes. CAVEATS: the anchor is a vLLM nightly "
        f"(ddd6fbc), not v0.31.0, behind the llm-d EPP with two replicas; it is consistent "
        f"with v0.31.0's own graph-mode arms where they run eager (no-DSpark c128 ITL ~72 "
        f"ms at n > 64 tokens; DSpark c64 ~37 ms x AL 3.51 ~ 130 ms a step). The count is "
        f"the kernel's primitive count, a documented lower bound on V4.1's real launches "
        f"(a source reading gives 1.2-1.9k), so e absorbs the difference and is valid only "
        f"with the count it was divided by. HELD-OUT CHECKS (not used to derive it) were "
        f"run in diagnosis at the previous count (88.9 us x 1018, the same product): the "
        f"DSpark arm's c32 interactivity collapse, BLIS 27.1 against measured 27.2 tok/s/"
        f"user, and the no-DSpark arm's c128, 7.4 against 7.1; they are re-run against the "
        f"integrated kernel and are not re-derived by this script. Previously 7.0 us (the "
        f"conventional 5-10 us launch, host_launch_eager_per_layer), which priced V4.1's "
        f"eager step at 12.7 ms against the ~95 ms measured.")
    scope_all = f"{{hardware: [{', '.join(ALL_PARTS)}], model: [{MODEL}]}}"
    others = [c for c in ALL_PARTS if c != ANCHOR_PART]
    head = (f"ASSUMED (backed out of an end-to-end anchor, as host_launch_per_kernel is): "
            f"{ea['e']} us of per-launch Python dispatch in DeepSeek-V4.1-Flash's eager step "
            f"on the H200 nodes' host CPU, on top of host_launch_per_kernel. DERIVED ON H200 "
            f"NODES ONLY: this entry is the only one the anchor supports; every other part "
            f"carries it as a copy (host_launch_eager_per_kernel, method copied), which is an "
            f"assumption about host CPU and driver launch cost, not a derivation. ")
    out.append(lines("host_launch_eager_per_kernel", ea["e"], "us_per_transfer", "assumed",
                     False, f"{{hardware: [{ANCHOR_PART}], model: [{MODEL}]}}", src_anchor,
                     head + form + " " + deriv))
    copy_head = (
        f"COPIED, AN ASSUMED TRANSFER, NOT A DERIVATION: {ea['e']} us, the value derived on "
        f"H200 nodes (scope hardware={ANCHOR_PART}, model={MODEL}; see that entry for the "
        f"anchor, the form and the derivation). The term is host-CPU and CUDA-driver work, "
        f"not GPU work, so carrying it to another part assumes that part's nodes dispatch a "
        f"Python op and launch a kernel as fast as the H200 nodes behind the anchor did. "
        f"Nothing measures that: no enforce-eager run of this model exists on any other part, "
        f"and no host CPU is stated for any part this registry can read. DIRECTION unknown: a "
        f"slower host CPU or older driver makes it optimistic, a faster one pessimistic. It is "
        f"carried rather than left absent because absent would fall back to the part-wide "
        f"host_launch_eager_per_layer (7 us), which the H200 anchor shows is about 13x too "
        f"small for this model. ON H100, the campaign's part, TO REPLACE IT: an H100 "
        f"enforce-eager ITL anchor -- the same model, TP8, --enforce-eager, no speculative "
        f"decoding, c1-c64 on H100 nodes -- with e = (ITL - device) / launches - "
        f"host_launch_per_kernel, re-derived by this writer. ")
    out.append(lines("host_launch_eager_per_kernel", ea["e"], "us_per_transfer", "copied",
                     False, f"{{hardware: [{', '.join(others)}], model: [{MODEL}]}}",
                     src_anchor, copy_head + form,
                     copied_from=f"hardware={ANCHOR_PART},model={MODEL}"))
    nr = ad["nearest"]
    share = ad["share"]
    bench_src = [("discussion", f"validator/testdata/anchors/{BENCH_JSON.name}: "
                  f"{ad['procedure']}", "primary")]
    out.append(lines(
        "host_admission_per_token", ad["value"], "us_per_token", "assumed", False, scope_all,
        bench_src,
        f"ASSUMED: A MEASUREMENT ON ANOTHER HOST, TRANSFERRED. {ad['value']} us per prompt "
        f"token was measured on {ad['host']}, with synthetic token text -- NOT on any serving "
        f"node: no part this entry is scoped to matches the host it was measured on, so on "
        f"every part it is an assumed transfer of that measurement (the schema has no way to "
        f"say 'measured on another host'). DIRECTION: a server core slower per thread than "
        f"the M1 Max makes it optimistic; real conversation text tokenizes at a different "
        f"rate than random vocabulary ids, direction unknown. TO REPLACE IT: the same timing "
        f"(diagnosis/frontend/bench_frontend.py: vLLM v0.31.0's V4.1 encoder, tokenizer and "
        f"block hash) on the serving node's CPU, ideally over real corpus prompts. WHAT WAS "
        f"TIMED: front-end work before a "
        f"DeepSeek-V4.1-Flash request can be scheduled, on vLLM v0.31.0's own code path "
        f"for this model: json.loads of the request body, deepseek_v41_encoding."
        f"encode_messages (the v0.31.0 file, verbatim), the HF tokenizers encode of the "
        f"rendered prompt (DeepSeek-V4.1 tokenizer.json), and sha256 block hashing over "
        f"pickled 64-token blocks (vllm/config/cache.py:144-150). The value is the row "
        f"nearest the transfer workload's ~{WORKLOAD_PROMPT_TOKENS // 1000}k-token mean "
        f"prompt: {nr['tokens']:,} tokens in {nr['total_ms']} ms. Across the bench it is "
        f"{ad['lo']:.2f}-{ad['hi']:.2f} us/token ({', '.join(f'{t:,}: {p:.2f}' for t, p in ad['per'])}): "
        f"it rises with length, so a short-prompt workload is over-priced by up to "
        f"{ad['value'] / ad['lo'] - 1:.0%}. Tokenization is {share['tokenize_ms']:.0%} of "
        f"it, rendering {share['encode_ms']:.1%}, hashing {share['hash_ms']:.1%}, JSON "
        f"{share['json_ms']:.1%}. CROSS-CHECKS, not used to derive it: an internal v0.30.0 "
        f"TP8 run of this model at an 8,030-token prompt, 99.6% hit, c1 measures TTFT 37.15 ms "
        f"in total, bounding any fixed + per-token front-end cost at 8k below 37 ms (this "
        f"value charges ~{8030 * ad['value'] / 1e3:.0f} ms there); and the in-sample "
        f"calibration on the transfer check's TP8 c8 TTFT gave 2.24 us/token, which is not "
        f"used. MODEL-SCOPED because the cost is V4.1's Python encoder and tokenizer; nothing "
        f"here shows it general. Not modelled: vLLM's renderer is one FIFO thread shared by "
        f"every request, so under load requests also wait for each other's admission, which "
        f"a per-request cost cannot carry. Reproduce with scripts/fit_deepseek_v41_host.py."))
    rust = round(ad["value"] / FASTOKENS_SPEEDUP, 3)
    out.append(lines(
        "host_admission_per_token_rust_frontend", rust, "us_per_token", "assumed", False,
        scope_all, [("discussion", f"{FASTOKENS}: 'on average achieves a 10x+ faster "
                     f"tokenization compared to the tokenizers library'", "primary"),
                    ("discussion", "vllm-project/vllm PR #47583 benchmark: DeepSeek-V3 "
                     "tokenizer cold encode 35.66 ms at 32K, 160.40 ms at 128K tokens",
                     "supporting")],
        f"ASSUMED: A VENDOR CLAIM APPLIED TO THE PYTHON-PATH VALUE. The per-prompt-token "
        f"admission cost when the pool serves through vLLM's Rust frontend (blis-schemas "
        f"engine.frontend: rust; vLLM launches vllm-rs only with VLLM_USE_RUST_FRONTEND=1, "
        f"vllm/envs.py:169 and vllm/entrypoints/cli/serve.py:64 at v0.31.0), which renders "
        f"and tokenizes in Rust and loads its tokenizer through fastokens. Value = "
        f"host_admission_per_token ({ad['value']} us, the Python path) / {FASTOKENS_SPEEDUP}, "
        f"fastokens' stated average speedup over HF tokenizers = {rust} us/token. RANGE "
        f"{PY_PATH_LOW_US / FASTOKENS_SPEEDUP:.2f}-{rust} us/token: the low end divides the "
        f"public Python-path rate (vllm PR #47583, DeepSeek-V3 tokenizer, {PY_PATH_LOW_US} us/"
        f"token) instead, and the README says '10x+' and that the gap widens with prompt "
        f"length, so this value is the slow end. NOT MEASURED on any serving node, not fitted, "
        f"and NOT derived from any InferenceX TTFT (those request pairs ran this frontend and "
        f"are held out to test it). It is the same rate for the Rust frontend's rendering, "
        f"which the Python path spends in encode_messages (0.2% there). Use: pricing "
        f"deployments that state engine.frontend: rust; every other deployment takes "
        f"host_admission_per_token. Reproduce with scripts/fit_deepseek_v41_host.py."))
    out.append(lines(
        "host_admission_per_request", "0.0", "us_per_request", "not_charged", False,
        scope_all, bench_src,
        f"NOT CHARGED (0.0) for DeepSeek-V4.1-Flash -- but the zero is an empirical inference, "
        f"not a structural zero; it carries not_charged only because blis-schemas requires that "
        f"method for a value of 0, and is an assumption in substance. What was timed has no fixed per-request component above noise -- a "
        f"least-squares line over the {len(ad['per'])} bench rows has slope "
        f"{ad['slope']:.2f} us/token and intercept {ad['icpt'] / 1e3:.1f} ms (not positive), "
        f"and every timed step (JSON, render, tokenize, hash) scales with the prompt -- but "
        f"the bench does not time HTTP handling, request validation, the IPC to EngineCore or "
        f"scheduler admission, so it cannot show those carry no fixed cost. THE BOUND that "
        f"is the actual argument is end-to-end: an internal v0.30.0 TP8 run of this model at "
        f"an 8,030-token prompt, 99.6% prefix hit, c1, measures TTFT 37.15 ms in total, so a "
        f"fixed per-request cost is well under 37 ms there, which rules out the part-wide "
        f"40 ms (backed out of InferenceX TTFT ratios; its own rationale calls it misnamed "
        f"if read as front-end overhead) but not a smaller one. DIRECTION: optimistic, by an "
        f"unmeasured fixed cost; it matters most on short-prompt, high-QPS workloads. TO "
        f"REPLACE IT: the client-observed TTFT of short prompts at c1 against the timed "
        f"front end on the serving node. Paired with host_admission_per_token; see it for the measurement, the host it was taken on (not a serving node) and its caveats."))
    return out


def render(text: str, new: list) -> str:
    """Each entry beside its part-wide namesake; the eager one after the per-layer term
    it supersedes for this model."""
    blocks = cb.split(text)
    for e in new:
        name = cb.name_of(e)
        anchors = {name, "host_launch_eager_per_layer"} if name.startswith("host_launch") \
            else {name, "host_admission_per_token"} if name.endswith("_rust_frontend") \
            else {name}
        cb.upsert(blocks, e, after=lambda b, a=anchors: cb.name_of(b) in a)
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--extract-anchor", metavar="REPORT_HTML",
                    help="derive the anchor's inputs from a local copy of the source report")
    ap.add_argument("--record-device", metavar="KDEVICE_JSON",
                    help="vendor diagnosis/kdevice's output as the kernel basis, then exit")
    ap.add_argument("--kernel-commit", help="with --record-device: override the recorded "
                    "kernel identity (e.g. for a non-git kernel tree)")
    args = ap.parse_args(argv[1:])
    if args.extract_anchor:
        return extract_anchor(Path(args.extract_anchor))
    if args.record_device:
        return record_device(Path(args.record_device), args.kernel_commit)
    anchor = json.loads(ANCHOR_JSON.read_text())
    print(f"anchor            {verify_anchor(anchor, dict(os.environ))}")
    bench = json.loads(BENCH_JSON.read_text())
    ea, ad = eager(anchor), admission(bench)
    print(f"eager per kernel  e={ea['e']} us (median ITL {ea['median']:.2f} ms, device "
          f"{ea['dev']:.3f} ms, {ea['n']} launches, launch {ea['launch']} us); plateau "
          f"{ea['e_lo']:.1f}-{ea['e_hi']:.1f}; device {DEVICE_AISIM_MS} ms -> {ea['e_aisim']:.1f}")
    print(f"admission         per_token={ad['value']} us (row {ad['nearest']['tokens']}); "
          f"range {ad['lo']:.2f}-{ad['hi']:.2f}; LS slope {ad['slope']:.2f} intercept "
          f"{ad['icpt'] / 1e3:.1f} ms")
    t0 = HOST_PATH.read_text(encoding="utf-8")
    t1 = render(t0, entries(ea, ad))
    if args.check:
        if t0 != t1:
            print(f"would change: {HOST_PATH.name}", file=sys.stderr)
            return 1
        print("the host set already matches")
        return 0
    HOST_PATH.write_text(t1, encoding="utf-8")
    print("the host set written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
