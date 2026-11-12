#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash on SM90 (H200, H100): the attention, indexer, weight-only GEMM and
host-gather coefficients vLLM v0.31.0's kernels need, each measured where a kernel-level
measurement of the right kernel exists and declared `assumed` where none does.

    python scripts/fit_deepseek_v41_h200.py              # print the fits, rewrite both sets
    python scripts/fit_deepseek_v41_h200.py --check      # exit 1 if either set would change
    python scripts/fit_deepseek_v41_h200.py --part h100  # one part only

Named for h200, the part it was written for; every part in PARTS is fitted from its own
AISimulate rows, catalog descriptor and committed siblings, never from another part's
values. Run fit_marlin_grouped_ramp.py first: the W8A16 rationale cites the part's grouped
Marlin ramp.

Owns six entries per part, two fitted and four assumed (inference-sim/blis-registry#37,
#38, #39):

  cost-model-attention.yaml, scoped {hardware: [<part>], model: [deepseek-v4.1-flash]}
    attention_decode_rate_sparse_mla           measured  FlashMLA sparse decode
    attention_decode_floor_block_index_scores  measured  indexer MQA-logits scan floor
    attention_decode_rate_block_index_scores   assumed   indexer scan rate (empirical HBM)
  cost-model-primitives.yaml, scoped {hardware: [<part>]}
    gemm_eps_max_w8a16_mxfp8, gemm_m_half_w8a16_mxfp8   assumed  Marlin MXFP8 W8A16 dense
    host_gather_rate_uva                                assumed  engram rows over UVA

WHAT vLLM v0.31.0 RUNS ON SM90 FOR THIS MODEL (read from the tag, db9527a4):
  * Sparse attention: with no explicit backend, SM90 falls through to
    `DeepseekV4FlashMLAAttention` (`_select_dsv4_attn_cls`, nvidia/model.py: FlashInfer is
    SM12x or an explicit FLASHINFER_* backend, mega attention SM100 only), whose
    `_forward_decode` (nvidia/flashmla.py) makes ONE `flash_mla_with_kvcache` call per
    layer: the sliding window (`indices=swa_indices`) plus the indexer's top-k as an extra
    cache (`extra_indices_in_kvcache`), FP8 KV, query heads padded to 64
    (`get_padded_num_q_heads`: 64 for any count up to 64, so V4.1's 64 heads hit the h64
    kernel at EVERY tensor-parallel width). The KV record is the V4 one on every arch but
    SM100 -- 448 FP8 NoPE + 64 BF16 RoPE + 8 scale/pad bytes, 584 B
    (`_use_v41_mxfp8_kv_record`, attention.py).
  * Indexer scan: `SparseAttnIndexer` -> DeepGEMM `fp8_fp4_paged_mqa_logits` over the
    compressed index keys, FP8 with an FP32 scale, 132 B per key
    (`_indexer_k_cache_head_dim`). MXFP4 keys and the sparse-logits path both require
    SM100 (`dsa_indexer_uses_fp4`, `AttentionConfig.indexer_sparse_logits`), so on SM90
    the scan is dense over ceil(context / compress_ratio) keys per request.
  * Dense linears: `MarlinMxfp8LinearKernel`, "MXFP8 W8A16 GEMM via Marlin"
    (model_executor/kernels/linear/mxfp8/marlin.py); the FlashInfer MXFP8 kernels ahead of
    it require SM100.
  * Engram: tables in pinned host memory by default (`EngramConfig.cpu_offload = True`),
    gathered by `_engram_lookup_kernel` (models/deepseek_v41/common/engram.py) through UVA.

WHY THE vLLM 0.25.0 SWEEPS MEASURE THE v0.31.0 KERNELS. AISimulate has no vLLM collection
newer than 0.25.0 for h200_sxm or h100_sxm. For the two attention kernels the device code
was compared across the pins the two tags build:
  * FlashMLA (vllm-project/FlashMLA a6ec2ba -> 0eee43b): `csrc/sm90/decode/sparse_fp8/*`
    became `csrc/kernels/sm90/decode/sparse/*` with only namespace, include-path and
    MODEL1 -> V4 renames; the 584 B `BYTES_PER_TOKEN` assert is unchanged.
  * DeepGEMM (deepseek-ai/DeepGEMM a6b593d -> vllm-project/DeepGEMM e1f418c; v0.31.0
    builds from vLLM's fork, deepgemm.cmake): `sm90_fp8_paged_mqa_logits.cuh` and its
    scheduler differ by renames and by bounds handling for zero-length contexts and
    block-table rows past a request's pages; the 64-token compute loop is unchanged.

THE TABLES, AND WHY EACH IS (OR IS NOT) A KERNEL MEASUREMENT.
  * `dsv4_hca_attn_module_perf` (vllm/0.25.0, FLASHMLA_SPARSE_DSV4): times `forward_mqa`,
    i.e. the FlashMLA call with no projections, floor ~10 us -- below the 13.5 us (h200) and
    14.5 us (h100) attention floors, so it is the kernel and not the block (methodology
    section 10.1). Fitted
    on the DeepSeek-V4-Flash-FP8 rows: 64 query heads, the same h64 instantiation and 584 B
    record V4.1 runs. Its rows are compress_ratio 128, whose compressed tier is read in
    full; the kernel walks SWA blocks then extra blocks in one 64-token-block loop
    (`num_orig_kv_blocks`, splitkv_mla.cuh), so the cost of a request is its BLOCK count
    whatever tier the blocks come from. V4.1 reads at most 2 + 8 = 10 blocks (window 128,
    top-512) per query, inside the 1-130 blocks the table covers; both counts are derived
    from the configs, and the writer refuses if V4.1's would leave the measured range.
  * `dsv4_paged_mqa_logits_module_perf` (vllm/0.25.0): times `fp8_fp4_paged_mqa_logits`
    itself under CUDA graphs. Its FLOOR is the kernel's. Its SLOPE is not an HBM rate: the
    collector gives every query row the same block table, so the unique key footprint is at
    most 34.6 MB, and the fitted throughput is 1.12x the HBM datasheet peak on h200 and
    1.92x on h100 -- impossible from HBM, and consistent with the shared keys being served from cache. In serving each
    request's keys are its own and HBM-resident, so the scan is HBM-bound, and NVIDIA's
    empirical HBM rate is what is committed.
  * `dsv41/sglang/dev-1aa0e962.../dsv41_module_perf` is NOT used: it is SGLang, eager (no
    CUDA graph), a module timing (`MQALayer.forward`, projections included), batch <= 3 and
    context <= 2,049, flat at 0.8-2.0 ms -- launch-bound, not a kernel rate.

UNITS. `latency` is MILLISECONDS in every AISimulate parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

ATTN_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
PRIM_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
# The repository's own catalog, so the writer runs in CI; BLIS_CATALOG overrides it
# with another catalog directory.
DEFAULT_CATALOG = (os.environ.get("BLIS_CATALOG")
                   or str(HERE.parents[1] / "catalog"))

MODEL = "deepseek-v4.1-flash"
COLLECTION = "vllm/0.25.0"

# Per part: the AISimulate system, the name the datasheet gives it, and what an end-to-end
# arbiter (methodology section 3) can say about it -- for V4.1 on vLLM, nothing yet.
# `sibling` names a part on the same die whose rows are compared to this part's, so the
# rationale states what a rate is not.
PARTS = {
    "h200": {"sku": "h200_sxm", "name": "H200", "sibling": None,
             "e2e": ("NO END-TO-END CHECK EXISTS FOR vLLM: AISimulate FPM (revision "
                     "03e006de) carries DeepSeek-V4.1-Flash on h200 under SGLang only -- "
                     "another engine, MoE kernel and checkpoint revision -- so it cannot "
                     "arbitrate vLLM v0.31.0 pricing; blis-registry#42 has no arbiter until "
                     "a vLLM V4.1 forward-pass measurement exists, see methodology section "
                     "10.5.")},
    "h100": {"sku": "h100_sxm", "name": "H100", "sibling": "h200",
             "e2e": ("NO END-TO-END CHECK ON THIS PART: no forward-pass dataset carries "
                     "DeepSeek-V4.1-Flash on h100 under any engine, and FPM's h200 V4.1 rows "
                     "are SGLang (revision 03e006de), so nothing arbitrates this entry "
                     "(methodology section 10.5).")},
}


def model_scope(chip: str) -> str:
    return f"{{hardware: [{chip}], model: [{MODEL}]}}"


def part_scope(chip: str) -> str:
    return f"{{hardware: [{chip}]}}"

FLASHMLA_TABLE = "dsv4_hca_attn_module_perf.parquet"
FLASHMLA_LANE = "FLASHMLA_SPARSE_DSV4"
FLASHMLA_FIT_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
INDEXER_TABLE = "dsv4_paged_mqa_logits_module_perf.parquet"
INDEXER_LANE = "vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits"

# FlashMLA's TOPK_BLOCK_SIZE, its unit of work. Also the collector's indexer page size
# (`block_kv = 64`, collect_dsv4_attn.py), which is why one constant serves both.
BLOCK = 64
INDEXER_RATIO = 4     # the measured indexer rows are DeepSeek-V4's compress_ratio 4

# Engram rows: 256 FP8 values in one table, 8 UE8M0 scales in a separate one
# (ParallelEngramEmbedding._allocate_weights). The value row is 256 B at a 256 B stride,
# two whole 128 B lines; the 8 scale bytes cost at least one 32 B sector -- the minimum
# a global load moves (CUDA C++ Best Practices Guide, "Coalesced Access to Global Memory").
ENGRAM_ROW_BYTES = 256 + 8
ENGRAM_FETCHED_BYTES = 256 + 32
# AISimulate's pcie_bw (64 GB/s, annotated "pcie 5.0") is the raw signalling rate per
# direction; PCIe 5.0's 128b/130b line code leaves 128/130 of it for transactions.
PCIE_LINE_CODE = 128 / 130

RATE_FRACTIONS = [x / 1000 for x in range(5, 1001)]
# Bytes per call above which the marginal bandwidth is reported.
VOLUME_THRESHOLDS = (1e6, 1e7)
INDEXER_MODELS = {"sgl-project/DeepSeek-V4-Flash-FP8", "sgl-project/DeepSeek-V4-Pro-FP8"}


def interior(value, grid, what: str) -> None:
    """A fit pinned to its grid's edge is the grid's answer, not the data's."""
    if not math.isfinite(value) or value <= grid[0] or value >= grid[-1]:
        raise SystemExit(f"{what} {value} is at the edge of its grid [{grid[0]}, {grid[-1]}]")


def model_config(data: Path, name: str) -> dict:
    path = data.parent.parent / "model_configs" / f"{name}_config.json"
    cfg = json.loads(path.read_text())
    return cfg.get("text_config", cfg)


def committed(path: Path, name: str, chip: str) -> float:
    doc = yaml.safe_load(path.read_text())
    for e in doc["coefficients"]:
        ((n, body),) = e.items()
        if n == name and body["scope"].get("hardware") == [chip] and \
                "model" not in body["scope"]:
            return float(body["value"])
    raise SystemExit(f"{chip}: no committed part-wide {name}")


def catalog_facts(catalog: Path, chip: str) -> tuple[float, float]:
    """(HBM datasheet bandwidth in bytes per microsecond, dense FP8 peak in TFLOPS)."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts["BwPeakTBs"]) * 1e6, float(facts["TFlopsFP8"])


def read(data: Path, sku: str, table: str) -> dict:
    path = data / sku / "sparse_attention" / COLLECTION / table
    if not path.is_file():
        raise SystemExit(f"{path} is absent; the pinned collection is required")
    return pq.read_table(path).to_pydict()


def geo(points, floor, rate) -> tuple[float, float]:
    """(geometric mean |error|, geometric mean signed ratio) of floor + work/rate."""
    logs = [math.log((floor + w / rate) / y) for w, y in points]
    return (math.exp(sum(abs(x) for x in logs) / len(logs)),
            math.exp(sum(logs) / len(logs)))


def check_geometry(data: Path) -> dict:
    """Refuse unless V4.1 and the fitted checkpoints share the kernel geometry."""
    v41 = model_config(data, "deepseek-ai--DeepSeek-V4.1-Flash")
    flash = model_config(data, "sgl-project--DeepSeek-V4-Flash-FP8")
    pro = model_config(data, "sgl-project--DeepSeek-V4-Pro-FP8")

    def padded(h):  # DeepseekV4FlashMLAAttention.get_padded_num_q_heads
        return 64 if h <= 64 else 128

    if not padded(v41["num_attention_heads"]) == padded(flash["num_attention_heads"]) == 64:
        raise SystemExit("V4.1 and the fitted V4-Flash rows do not share the h64 kernel")
    for k in ("head_dim", "qk_rope_head_dim", "sliding_window"):
        if v41[k] != flash[k]:
            raise SystemExit(f"{k} differs between V4.1 and V4-Flash; the record differs")
    if set(v41["compress_ratios"]) - {0, 1, 2}:
        raise SystemExit(f"V4.1 compress_ratios {set(v41['compress_ratios'])}")
    for c in (flash, pro):
        if c["index_head_dim"] != v41["index_head_dim"]:
            raise SystemExit("indexer head_dim differs; the 132 B key does not hold")
    if flash["index_n_heads"] != pro["index_n_heads"]:
        raise SystemExit("the two measured checkpoints differ in index heads; pooling them "
                         "would pool two kernel configurations")
    nope = v41["head_dim"] - v41["qk_rope_head_dim"]
    window, topk = v41["sliding_window"], v41["index_topk"]
    return {
        # FlashMLA SM90 V4 BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8
        "record": nope + 2 * v41["qk_rope_head_dim"] + 8,
        "window": window, "topk": topk,
        # The most a V4.1 sparse layer reads per query, in FlashMLA blocks and tokens.
        "v41_blocks": math.ceil(window / BLOCK) + math.ceil(topk / BLOCK),
        "v41_tokens": window + topk,
        "key": v41["index_head_dim"] + v41["index_head_dim"] // 128 * 4,
        "v41_index_heads": v41["index_n_heads"],
        "fit_index_heads": flash["index_n_heads"],
    }


def fit_flashmla(data: Path, sku: str, g: dict, peak: float, floor: float) -> dict:
    t = read(data, sku, FLASHMLA_TABLE)
    pts, blocks_of, batch_of, rows = [], [], [], {}
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1 or t["model"][i] != FLASHMLA_FIT_MODEL:
            continue
        # Every fitted row must be the configuration V4.1 runs; anything else refuses
        # rather than pooling silently.
        if (t["kernel_source"][i] != FLASHMLA_LANE or t["compress_ratio"][i] != 128
                or t["kv_cache_dtype"][i] != "fp8" or t["local_num_heads"][i] != 64
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected FlashMLA row {i}: {t['kernel_source'][i]}, ratio "
                             f"{t['compress_ratio'][i]}, kv {t['kv_cache_dtype'][i]}, heads "
                             f"{t['local_num_heads'][i]}, latency {t['latency'][i]}")
        ctx = t["step"][i] + 1
        blocks = (math.ceil(min(ctx, g["window"]) / BLOCK)
                  + math.ceil((ctx // 128) / BLOCK))
        pts.append((t["batch_size"][i] * blocks * BLOCK * g["record"],
                    t["latency"][i] * 1000.0))
        blocks_of.append(blocks)
        batch_of.append(t["batch_size"][i])
        rows[(t["batch_size"][i], t["step"][i])] = t["latency"][i]
    if len(pts) < 150:
        raise SystemExit(f"only {len(pts)} FlashMLA decode rows")
    best = min((geo(pts, floor, peak * f)[0], f) for f in RATE_FRACTIONS)
    frac = best[1]
    interior(frac, RATE_FRACTIONS, "FlashMLA rate fraction")
    rate = int(round(peak * frac))
    err, signed = geo(pts, floor, rate)
    # Free-floor fit, reported to show what pinning the floor costs.
    free = min((geo(pts, fl / 10, peak * f)[0], fl / 10, f)
               for fl in range(50, 200) for f in RATE_FRACTIONS[::5])
    # Hold out the long-extra regime: train on <= 3 blocks per request, test on >= 5.
    tr = [p for p, b in zip(pts, blocks_of) if b <= 3]
    te = [p for p, b in zip(pts, blocks_of) if b >= 5]
    trf = min((geo(tr, floor, peak * f)[0], f) for f in RATE_FRACTIONS)[1]
    v41 = [p for p, b in zip(pts, blocks_of) if b == g["v41_blocks"]]
    if g["v41_blocks"] > max(blocks_of):
        raise SystemExit(f"V4.1 reads {g['v41_blocks']} blocks per query, beyond the "
                         f"{max(blocks_of)} measured; the rate would be an extrapolation")
    # Two richer forms, fitted on the same rows, to test whether the batch bias below is a
    # missing term: a per-request cost, t = f + c * batch + bytes / r; and an occupancy
    # form, t = f + a * ceil(batch * blocks / P), P fitted -- whole waves of P blocks.
    us = np.array([y for _, y in pts]); wb = np.array([w for w, _ in pts])
    nb = np.array(batch_of, dtype=float); tiles = nb * np.array(blocks_of, dtype=float)
    rs = np.geomspace(5e5, 5e6, 200)
    per_req = min(
        (float(np.min(np.mean(np.abs(np.log((f + c * nb[:, None] + wb[:, None] / rs[None, :])
                                              / us[:, None])), axis=0))), f, c)
        for f in np.arange(5, 15, 0.25) for c in np.arange(0, 0.3, 0.002))
    amps = np.geomspace(0.05, 20, 300)
    wave = min(
        (float(np.min(np.mean(np.abs(np.log((f + amps[None, :] * np.ceil(tiles / P)[:, None])
                                              / us[:, None])), axis=0))), P, f)
        for P in range(32, 400, 4) for f in np.arange(5, 15, 0.25))
    # The overall error hides a batch-dependent bias: report it per batch, signed.
    by_batch = {b: geo([p for p, x in zip(pts, batch_of) if x == b], floor, rate)[1]
                for b in sorted(set(batch_of))}
    # Marginal bandwidth at volume: the least-squares slope of latency on bytes over rows
    # reading at least 1 MB and at least 10 MB, and the largest call net of the floor.
    marginal = {}
    for thr in VOLUME_THRESHOLDS:
        sel = wb >= thr
        slope = np.polyfit(wb[sel], us[sel], 1)[0]
        marginal[thr] = (int(sel.sum()), 1.0 / slope)
    at_max = wb == wb.max()
    max_fracs = wb[at_max] / (us[at_max] - floor) / peak
    # Short-context rows the pinned floor over-prices: batch 1, at most one block.
    short = [y for (w, y), b, n in zip(pts, blocks_of, batch_of) if n == 1 and b <= 1]
    return {"n": len(pts), "rate": rate, "frac": frac, "err": err, "signed": signed,
            "per_req_err": math.exp(per_req[0]), "wave_err": math.exp(wave[0]),
            "wave_p": wave[1], "by_batch": by_batch, "short_lo": min(short), "short_hi": max(short),
            "hold_batch_max": max(n for n, b in zip(batch_of, blocks_of) if b >= 5),
            "v41_batch_max": max((n for n, b in zip(batch_of, blocks_of)
                                  if b == g["v41_blocks"]), default=0),
            "free_err": free[0], "free_floor": free[1], "rows": rows,
            "hold_n": len(te), "hold": geo(te, floor, peak * trf),
            "v41_n": len(v41), "v41": geo(v41, floor, rate), "floor": floor,
            "max_blocks": max(blocks_of), "marginal": marginal,
            "under_1mb": int((wb < VOLUME_THRESHOLDS[0]).sum()),
            "max_call": float(wb.max()), "max_call_n": int(at_max.sum()),
            "max_call_batch": sorted({int(b) for b in nb[at_max]}),
            "max_call_frac": (float(max_fracs.min()), float(max_fracs.max())),
            "max_batch": max(batch_of)}


def fit_indexer(data: Path, sku: str, g: dict) -> dict:
    t = read(data, sku, INDEXER_TABLE)
    pts, uniq, models = [], [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1:
            continue
        if (t["kernel_source"][i] != INDEXER_LANE or t["compress_ratio"][i] != INDEXER_RATIO
                or t["kv_cache_dtype"][i] != "fp8" or t["mla_dtype"][i] != "fp8_e4m3"
                or t["model"][i] not in INDEXER_MODELS
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected indexer row {i}: {t['kernel_source'][i]}, "
                             f"{t['model'][i]}, keys {t['mla_dtype'][i]}, latency "
                             f"{t['latency'][i]}")
        b, s = t["batch_size"][i], t["step"][i]
        # The collector's context_lens: (past_kv + row + 1) // 4, at least 1, per row.
        row_keys = sum(max((s + r) // INDEXER_RATIO, 1) for r in range(1, b + 1))
        pts.append((row_keys, t["latency"][i] * 1000.0))
        models.append(t["model"][i])
        # Every row shares one block table, so the unique keys are one request's.
        uniq.append(math.ceil(max((s + b) // INDEXER_RATIO, 1) / BLOCK) * BLOCK * g["key"])
    if len(pts) < 150 or set(models) != INDEXER_MODELS:
        raise SystemExit(f"only {len(pts)} indexer decode rows over {sorted(set(models))}")
    floors = [x / 100 for x in range(200, 800)]
    rates = [10 ** (3 + k / 500) for k in range(1501)]  # 1e3 .. 1e6 row-keys per us
    best = min((geo(pts, fl, r)[0], fl, r) for fl in floors[::10] for r in rates[::10])
    _, fl0, r0 = best
    best = min((geo(pts, fl, r)[0], fl, r)
               for fl in floors if abs(fl - fl0) <= 0.2
               for r in rates if 0.8 * r0 <= r <= 1.25 * r0)
    err, floor, rate = best
    interior(floor, floors, "indexer floor")
    interior(rate, rates, "indexer work rate")
    floor = round(floor, 1)
    implied = rate * g["key"]  # bytes per microsecond, if each row-key were a key read
    small = [p for p in pts if p[0] / rate < 0.05 * floor]
    # Each checkpoint on its own, to show the floor is the kernel's rather than one model's.
    per_model = {}
    for m in sorted(set(models)):
        mp = [p for p, x in zip(pts, models) if x == m]
        per_model[m] = min((geo(mp, fl, r)[0], fl, r)
                           for fl in floors if abs(fl - floor) <= 0.3
                           for r in rates[::5] if 0.8 * rate <= r <= 1.25 * rate)[1]
    return {"n": len(pts), "floor": floor, "rate_rowkeys": rate, "err": err,
            "per_model": per_model,
            "signed": geo(pts, floor, rate)[1], "implied": implied,
            "max_unique": max(uniq), "small_n": len(small),
            "small": geo(small, floor, rate)}


def lines(name, value, units, method, fitted, scope, sources, rationale, rate_basis=None):
    out = [f"  - {name}:", f"      value: {value}", f"      units: {units}",
           f"      method: {method}", f"      fitted: {'true' if fitted else 'false'}"]
    if rate_basis:
        out.append(f"      rate_basis: {rate_basis}")
    out += [f"      scope: {scope}", "      sources:"]
    out += [f'        - {{kind: {k}, cite: "{c}", role: {r}}}' for k, c, r in sources]
    return out + ["      rationale: >", f"        {rationale}"]


def volume_note(fm: dict, peak: float) -> str:
    """How far from HBM-bound the largest measured reads are."""
    m = fm["marginal"]
    lo, hi = fm["max_call_frac"]
    per = "; ".join(f"{m[t][1] / 1e6:.2f} TB/s, {m[t][1] / peak:.0%} of peak, over the "
                    f"{m[t][0]} rows of at least {t / 1e6:g} MB" for t in VOLUME_THRESHOLDS)
    return (f"AT VOLUME THE READ IS STILL FAR FROM HBM-BOUND: the marginal bandwidth (the "
            f"least-squares slope of latency on bytes) is {per}; the {fm['max_call_n']} "
            f"largest calls, {fm['max_call'] / 1e6:.0f} MB at batch "
            f"{'/'.join(map(str, fm['max_call_batch']))}, read at "
            f"{lo:.1%}-{hi:.1%} of peak net of the floor. ")


def sibling_note(chip: str, fm: dict, peak: float, sib: dict | None) -> str:
    """For a part fitted beside a same-die sibling: the two parts' latency on the rows both
    measure, against the ratio an HBM-bound read would show."""
    if sib is None:
        return ""
    common = sorted(set(fm["rows"]) & set(sib["fm"]["rows"]))
    if len(common) < 0.9 * len(fm["rows"]):
        raise SystemExit(f"{chip} and {sib['chip']} share only {len(common)} FlashMLA rows")
    r = sorted(fm["rows"][k] / sib["fm"]["rows"][k] for k in common)
    top = sorted(common, key=lambda k: sib["fm"]["rows"][k])[-(len(common) // 10):]
    r_top = sorted(fm["rows"][k] / sib["fm"]["rows"][k] for k in top)
    mine, theirs = fm["marginal"], sib["fm"]["marginal"]
    lo = VOLUME_THRESHOLDS[0]
    sizes = " (".join(f"{t / 1e6:g} MB" for t in VOLUME_THRESHOLDS) + ")"
    ratios = " (".join(f"{mine[t][1] / theirs[t][1]:.2f}x" for t in VOLUME_THRESHOLDS) + ")"
    theirs_tb = " and ".join(f"{theirs[t][1] / 1e6:.2f}" for t in VOLUME_THRESHOLDS)
    weighting = (
        f"This part's fitted rate is {fm['rate'] / sib['fm']['rate']:.3f}x {sib['chip']}'s, a "
        f"ratio that depends on row weighting: the fit is geometric over all {fm['n']} rows, {fm['under_1mb']} of "
        f"them under 1 MB; on rows of at least {sizes} this part's marginal bandwidth is "
        f"{ratios} {sib['chip']}'s ({theirs_tb} TB/s there, "
        f"{theirs[lo][1] / sib['peak']:.0%} of its peak at 1 MB). ")
    return (f"THE SAME KERNEL AS ON {sib['chip'].upper()}, NOT A SCALED HBM RATE: on the "
            f"{len(common)} rows both parts measure, this part's latency is a median "
            f"{r[len(r) // 2]:.2f}x {sib['chip']}'s ({r_top[len(r_top) // 2]:.2f}x over the "
            f"slowest tenth), where a read bound on HBM would be "
            f"{sib['peak'] / peak:.2f}x ({sib['peak'] / 1e6:g} against {peak / 1e6:g} TB/s); "
            f"so this rate is close to {sib['chip']}'s ({sib['fm']['rate'] / 1e6:.2f} TB/s) "
            f"and a larger fraction of this part's peak -- the cost per byte of reads this "
            f"short, not a bandwidth -- and must not be rescaled between parts by "
            f"BwPeak. {weighting}")


def entries(chip: str, g: dict, fm: dict, ix: dict, peak: float, fp8_tflops: float,
            derate: float, bf16_eps: float, bf16_mh: int, grouped_eps: float,
            grouped_mh: int, host_bw: float, sib: dict | None) -> tuple[list, list]:
    part = PARTS[chip]
    sku, name = part["sku"], part["name"]
    MODEL_SCOPE, PART_SCOPE = model_scope(chip), part_scope(chip)
    tb = peak / 1e6
    ridge = fp8_tflops * 1e6 / peak                       # FLOP per byte at the FP8 peak
    fit_heads = g["fit_index_heads"]
    measured_tflops = ix["rate_rowkeys"] * fit_heads * 256 / 1e6
    # Speculative decoding scores k+1 rows per key read; at the measured compute rate,
    # the key rate that sustains, as a fraction of the committed HBM rate, for k = 5.
    spec_keys = measured_tflops * 1e6 / (6 * g["v41_index_heads"] * 256)
    spec_frac = spec_keys * g["key"] / (peak * derate)
    floors = ", ".join(f"{v:.2f} us" for v in ix["per_model"].values())
    aisim = f"NVIDIA AISimulate systems/data/{sku}/sparse_attention/vllm/0.25.0/"
    hbm_rate = int(round(peak * derate))
    gather = int(round(host_bw * PCIE_LINE_CODE * ENGRAM_ROW_BYTES / ENGRAM_FETCHED_BYTES))
    v41_call = fm["max_batch"] * g["v41_blocks"] * BLOCK * g["record"]
    attn = [
        lines("attention_decode_rate_sparse_mla", f"{fm['rate']}.0", "bytes_per_us",
              "measured", True, MODEL_SCOPE,
              [("model", f"{aisim}{FLASHMLA_TABLE} (NVIDIA {name}), {FLASHMLA_LANE}, "
                f"{FLASHMLA_FIT_MODEL} (64 query heads, the h64 kernel), "
                f"{fm['n']} decode points", "primary"),
               ("model", "vLLM v0.31.0 models/deepseek_v41/nvidia/flashmla.py "
                "DeepseekV4FlashMLAAttention._forward_decode", "supporting")],
              f"{fm['rate'] / 1e6:.2f} TB/s, {fm['frac']:.3f} of this part's {tb:g} TB/s "
              f"datasheet bandwidth, for vLLM v0.31.0's FlashMLA sparse decode on SM90 -- "
              f"one flash_mla_with_kvcache call per layer over the sliding window plus the "
              f"indexer's top-k, query heads padded to 64 at every TP width, the "
              f"{g['record']} B V4 KV record. BYTES ARE COUNTED IN FLASHMLA'S 64-TOKEN "
              f"BLOCKS: per request, {g['record']} x 64 x (ceil(window "
              f"tokens / 64) + ceil(selected tokens / 64)), which once the context fills the "
              f"window and the top-{g['topk']} is {g['v41_blocks']} blocks, "
              f"{g['v41_blocks'] * BLOCK * g['record']:,} B, at ONE query token per request "
              f"(the sweep's decode rows are isl 1; a multi-token decode, as under DSpark "
              f"speculation, is outside what was measured). THE FLOOR is this part's "
              f"part-wide attention_decode_floor ({fm['floor']:g} us), which "
              f"blis-latency-kernel inherits for a kind with a rate and no floor. It "
              f"over-prices the table's shortest reads: batch-1 rows of at most one block "
              f"measure {fm['short_lo']:.1f}-{fm['short_hi']:.1f} us, so pinning costs "
              f"{fm['floor'] / fm['short_hi'] - 1:.0%}-{fm['floor'] / fm['short_lo'] - 1:.0%} "
              f"there; a free floor lands at {fm['free_floor']:.1f} us ({fm['free_err']:.3f}x "
              f"against {fm['err']:.3f}x) only because a linear form cannot follow the step "
              f"between one and two blocks. THE ERROR DEPENDS ON BATCH, which the overall "
              f"{fm['err']:.3f}x (signed {fm['signed']:.3f}, {fm['n']} points, "
              f"1-{fm['max_blocks']} blocks per request) hides. Signed by batch: "
              f"{', '.join(f'{b}: {v:.2f}' for b, v in fm['by_batch'].items())}. Neither a "
              f"per-request term (floor + c x batch + bytes / rate: {fm['per_req_err']:.3f}x) "
              f"nor an occupancy form (floor + a x ceil(batch x blocks / P), P fitted at "
              f"{fm['wave_p']}: {fm['wave_err']:.3f}x) removes it, so it is batch-dependent "
              f"scheduling inside the call, not a missing term this form can carry. "
              f"Trained on requests of <= 3 blocks and held out on >= 5 the rate predicts "
              f"{fm['hold'][0]:.3f}x (signed {fm['hold'][1]:.3f}, n={fm['hold_n']}, all at "
              f"batch <= {fm['hold_batch_max']}); at exactly V4.1's {g['v41_blocks']} blocks "
              f"{fm['v41'][0]:.3f}x (signed {fm['v41'][1]:.3f}, n={fm['v41_n']}, batch <= "
              f"{fm['v41_batch_max']}). {volume_note(fm, peak)}{sibling_note(chip, fm, peak, sib)}SO V4.1'S OPERATING POINT IS INSIDE THE MEASURED RANGE "
              f"ON THE BLOCK AXIS ONLY: it reads {g['v41_blocks']} blocks per query at any "
              f"context past about 1k tokens, at batches the sweep's batch x context <= 1M "
              f"grid never pairs with {g['v41_blocks']} blocks; across the measured batches "
              f"the signed error spans the range above. Nor on call volume: at the "
              f"sweep's largest batch, {fm['max_batch']}, V4.1's {g['v41_blocks']} blocks "
              f"per request read {v41_call / 1e6:.0f} MB per call, "
              f"{v41_call / fm['max_call']:.1f}x the largest measured call. WHY THIS SHIPS WHEN "
              f"DEEPSEEK-V4-PRO'S WAS WITHDRAWN (methodology section 10.3): that rate was "
              f"fitted on Pro's 128-head rows (the h128 instance, which Pro runs only below "
              f"TP 2) and was to be charged against the kernel's selection rule for Pro's "
              f"csa4 layers -- up to 262,912 tokens at a 1M context, far outside any measured "
              f"row on the block axis. Every V4.1 sparse layer reads at most "
              f"{g['v41_tokens']} tokens per query ({g['window']} window + top-{g['topk']}) "
              f"on the h64 instance at every TP width. SCOPED TO THIS MODEL so no other "
              f"sparse_mla deployment changes. WHAT THE TIMED OP DOES NOT CONTAIN: the "
              f"0.25.0 ratio-128 rows take their indices from metadata, while v0.31.0's V4.1 "
              f"decode also launches compute_global_topk_indices_and_lens (a Triton kernel "
              f"translating the indexer's top-k to cache slots) on every compressed "
              f"(compress_ratio 1 or 2) layer -- a "
              f"known gap, one small launch per layer not priced here. The FlashMLA SM90 "
              f"sparse decode device code is unchanged between the pins vLLM 0.25.0 and "
              f"0.31.0 build (a6ec2ba, 0eee43b: renames only, the 584 B assert identical); "
              f"its split-KV combine kernel's grid moves from (b, s_q, h) to (b*s_q, 1, h), "
              f"inert at one query token, and gains 192/224/256-split cases that only "
              f"requests needing more than 160 splits reach; the host API was rewritten, "
              f"including skipping the split workspace for unsplit requests. {part['e2e']} "
              f"Reproduce with scripts/fit_deepseek_v41_h200.py."),
        lines("attention_decode_floor_block_index_scores", ix["floor"], "us_per_transfer",
              "measured", True, MODEL_SCOPE,
              [("model", f"{aisim}{INDEXER_TABLE} (NVIDIA {name}), {INDEXER_LANE}, "
                f"DeepSeek-V4-Flash-FP8 and V4-Pro-FP8 (64 index heads, head_dim 128, "
                f"FP8 keys), {ix['n']} decode points", "primary"),
               ("model", "vLLM v0.31.0 model_executor/layers/sparse_attn_indexer.py, "
                "fp8_fp4_paged_mqa_logits decode call", "supporting")],
              f"The launch-and-schedule floor of DeepGEMM's fp8_fp4_paged_mqa_logits, the "
              f"lightning indexer's scoring scan that vLLM v0.31.0 runs on SM90 for every "
              f"index-source layer of DeepSeek-V4.1 (FP8 keys and dense logits: MXFP4 keys "
              f"and the sparse-logits path need SM100). Timed under CUDA graphs on the "
              f"kernel itself, not a module. Fitted jointly with a per-row-key work rate "
              f"over {ix['n']} points (geometric error {ix['err']:.3f}x); the floor is "
              f"identified by the {ix['small_n']} points whose work term is under 5% of it "
              f"({ix['small'][0]:.3f}x, signed {ix['small'][1]:.3f}). The work rate is NOT "
              f"committed -- see attention_decode_rate_block_index_scores. The measured "
              f"checkpoints carry {fit_heads} index heads where V4.1 carries "
              f"{g['v41_index_heads']}; a floor is a launch cost, and fitted on each "
              f"checkpoint alone it is {floors}. The kernel and scheduler are unchanged across the "
              f"DeepGEMM pins of vLLM 0.25.0 and 0.31.0 except for zero-length bounds "
              f"handling. Covers the logits kernel only: the top-k and V4.1's candidate "
              f"select/mask kernels are separate launches this does not include. Consumed "
              f"by no reader yet: the kernel prices a role: block_index_scores node with "
              f"its layer kind's pair and needs to look this name up "
              f"(attention_decode_<floor|rate>_<role>). Reproduce with "
              f"scripts/fit_deepseek_v41_h200.py."),
        lines("attention_decode_rate_block_index_scores", f"{hbm_rate}.0", "bytes_per_us",
              "assumed", False, MODEL_SCOPE,
              [("model", f"{aisim}{INDEXER_TABLE} (NVIDIA {name}): fitted throughput "
                f"{ix['implied'] / 1e6:.2f} TB/s-equivalent at {g['key']} B/key, above the "
                f"{tb:g} TB/s HBM peak", "upper_bound"),
               ("model", f"NVIDIA AISimulate systems/{sku}.yaml "
                "mem_bw_empirical_scaling_factor (hbm_derate)", "primary")],
              f"ASSUMED: NVIDIA's empirical HBM rate, BwPeak x hbm_derate = {tb:g} TB/s x "
              f"{derate:g} -- the descriptor's observed correction, not a bound; the bound is "
              f"BwPeak itself. "
              f"Bytes are {g['key']} per compressed index key per request ({g['key'] - 4} "
              f"FP8 + one FP32 scale, _indexer_k_cache_head_dim), over ceil(context / "
              f"compress_ratio) keys -- V4.1's compress ratios are 1 and 2, so the scan "
              f"grows with context while the attention it feeds does not. WHY NOT THE "
              f"MEASURED SLOPE: the only kernel-level sweep gives every query row the same "
              f"block table, so its unique key footprint is at most "
              f"{ix['max_unique'] / 1e6:.1f} MB, and its fitted throughput, "
              f"{ix['implied'] / 1e6:.2f} TB/s-equivalent, exceeds the HBM datasheet peak, "
              f"which an HBM read cannot -- consistent with the shared keys being served "
              f"from cache. In serving each request's keys are "
              f"its own and HBM-resident. Compute does not bind at one query token per "
              f"request: {g['v41_index_heads']} heads x 128 x 2 = "
              f"{g['v41_index_heads'] * 256:,} FLOP per {g['key']} B key is "
              f"{g['v41_index_heads'] * 256 / g['key']:.0f} FLOP/B against an FP8 ridge of "
              f"{ridge:.0f}. DIRECTION: unknown at this fraction -- optimistic if this "
              f"kernel streams index pages below NVIDIA's empirical HBM fraction, "
              f"pessimistic if above. The real rate cannot exceed BwPeak, so this rate can "
              f"be pessimistic by at most {1 / derate:.2f}x (scan time over-priced); how "
              f"optimistic it can be is not bounded. Under speculative decoding each request scores k+1 query rows against "
              f"each key, so k = 5 brings it to {6 * g['v41_index_heads'] * 256 / g['key']:.0f} "
              f"FLOP/B -- still under the ridge at datasheet peak, but at the "
              f"{measured_tflops:.0f} TFLOPS this kernel was measured sustaining (a lower "
              f"bound on what it can) the scan could become compute-bound at as little as "
              f"{spec_frac:.2f} of this rate. TO REPLACE IT: the same "
              f"sweep with a distinct block table per query row, so the keys are read from "
              f"HBM. Scoped to this model; consumed by no reader yet (see the floor)."),
    ]
    prim = [
        lines("gemm_eps_max_w8a16_mxfp8", bf16_eps, "dimensionless", "assumed", False,
              PART_SCOPE,
              [("model", f"NVIDIA AISimulate systems/data/{sku}/gemm/vllm/0.25.0/"
                "gemm_perf.parquet via gemm_eps_max_bf16 (cuBLAS BF16 envelope)",
                "upper_bound"),
               ("model", "vLLM v0.31.0 model_executor/kernels/linear/mxfp8/marlin.py "
                "MarlinMxfp8LinearKernel", "supporting")],
              f"ASSUMED: this part's measured BF16 GEMM asymptote, as an UPPER BOUND. On "
              f"SM90 vLLM v0.31.0 runs every MXFP8 dense linear through "
              f"MarlinMxfp8LinearKernel (W8A16; the FlashInfer MXFP8 kernels ahead of it "
              f"require SM100): FP8 weights dequantized in registers, BF16 activations, "
              f"mma.sync. Its compute rate is therefore a fraction of the BF16 peak, and "
              f"the measured cuBLAS BF16 envelope ({bf16_eps} of peak) bounds it above. "
              f"No sweep measures dense Marlin on any part, so the fraction it reaches is "
              f"unknown; the same Marlin MMA core in its grouped (MoE) form reaches "
              f"{grouped_eps:g} (gemm_eps_max_w4a16_mxfp4_grouped); if dense Marlin reaches "
              f"what the grouped form does, the bound is loose by about "
              f"{bf16_eps / grouped_eps:.1f}x at prefill-sized M. At decode-sized M "
              f"the kernel's HBM term -- the FP8 "
              f"weight read -- dominates, and this term barely enters. DIRECTION: optimistic "
              f"for compute-bound GEMMs. TO REPLACE IT: an {chip} GEMM sweep with "
              f"MarlinMxfp8LinearKernel forced (AISimulate's vLLM GEMM collector has no "
              f"Marlin path). Read by no consumer until blis-latency-kernel selects a ramp "
              f"by weight format and chip (blis-latency-kernel#34)."),
        lines("gemm_m_half_w8a16_mxfp8", bf16_mh, "tokens", "assumed", False, PART_SCOPE,
              [("model", f"NVIDIA AISimulate systems/data/{sku}/gemm/vllm/0.25.0/"
                "gemm_perf.parquet via gemm_m_half_bf16", "primary")],
              f"ASSUMED: this part's measured cuBLAS BF16 knee (gemm_m_half_bf16), paired "
              f"with gemm_eps_max_w8a16_mxfp8 so the two describe one ramp rather than an "
              f"asymptote from one kernel and a knee from another. The knee is where a GEMM "
              f"stops being operand-starved; Marlin's grouped form measures {grouped_mh} "
              f"tokens on this part (gemm_m_half_w4a16_mxfp4_grouped), against {bf16_mh} "
              f"here, so the dense Marlin knee is unmeasured within that range. DIRECTION: "
              f"NOT a bound. A knee above Marlin's own lowers the ramp at mid M, so this "
              f"half is likely pessimistic there while the asymptote is optimistic at large "
              f"M; the two partly offset and neither direction is established. At decode "
              f"M the kernel's HBM term -- the FP8 weight read -- dominates either way. TO "
              f"REPLACE IT: the same MarlinMxfp8LinearKernel sweep as the asymptote; both "
              f"come from one fit."),
        lines("host_gather_rate_uva", f"{gather}.0", "bytes_per_us", "assumed", False,
              PART_SCOPE,
              [("model", f"NVIDIA AISimulate systems/{sku}.yaml pcie_bw "
                "(the nominal link rate, not host_link_bandwidth)", "primary"),
               ("model", "vLLM v0.31.0 models/deepseek_v41/common/engram.py "
                "_engram_lookup_kernel, ParallelEngramEmbedding", "supporting")],
              f"ASSUMED: an UPPER BOUND on the useful-byte rate of DeepSeek-V4.1's engram "
              f"lookup when the tables are in pinned host memory (EngramConfig.cpu_offload, "
              f"the v0.31.0 default) and read through UVA. Derived as the descriptor's nominal "
              f"pcie_bw ({host_bw / 1e3:g} GB/s, AISimulate's 'pcie 5.0' figure: the raw rate per "
              f"direction) x 128/130 (PCIe 5.0's 128b/130b line code, leaving "
              f"{host_bw * PCIE_LINE_CODE / 1e3:.1f} GB/s) x {ENGRAM_ROW_BYTES}/"
              f"{ENGRAM_FETCHED_BYTES}: a row is {ENGRAM_ROW_BYTES} useful bytes -- 256 FP8 "
              f"values in one table at a 256 B stride (two whole 128 B lines) and 8 UE8M0 "
              f"scales in a separate table, which cost at least one 32 B sector -- so at "
              f"least {ENGRAM_FETCHED_BYTES} B cross the link per row. Per token and engram "
              f"layer a rank reads cdiv(24, shards) rows. WHAT THE BOUND IGNORES, each "
              f"making the real rate LOWER: PCIe transaction-layer overhead; GPU address "
              f"translation for random rows over a table of 384 million rows, ~101 GB per layer "
              f"before TP sharding (vLLM's own comment: 'The "
              f"table dwarfs TLB reach'); the latency-bound regime at decode batch sizes, "
              f"where 24 rows are a few kilobytes and a per-launch round trip dominates -- "
              f"no floor is stated because none is measured; and host-side contention when "
              f"every GPU of a node gathers from one host's memory. The lookup runs on a "
              f"side stream at half the SMs, so how much of it a step exposes is a "
              f"composition question, not this rate's. TO REPLACE IT: time "
              f"ParallelEngramEmbedding.lookup(background=True) with cpu_offload on an {name} "
              f"host over 1-8,192 tokens at TP 1 and 8, and fit floor + bytes / rate. Read "
              f"by no consumer until the schema can state a host-resident gather "
              f"(blis-schemas#48, blis-latency-kernel#38). NOT host_link_bandwidth, which is now "
              f"the measured copy-engine payload rate (fit_host_link.py); re-deriving this bound "
              f"from a measured SM-driven ceiling (vLLM PR #42212, ~51 GB/s on H100) was tried and "
              f"failed its held-out gate (h100 capped-replay ITL worse), so it is parked."),
    ]
    return attn, prim


def render(attn_text: str, prim_text: str, attn: list, prim: list) -> tuple[str, str]:
    a = cb.split(attn_text)
    for e in attn:
        # After the last attention entry, so the family sits at the end of the set.
        cb.upsert(a, e, after=lambda b: cb.name_of(b) is not None)
    p = cb.split(prim_text)
    for e in prim:
        stem = "gemm_" if cb.name_of(e).startswith("gemm_") else "host_link_bandwidth"
        scope = cb.scope_of(e)
        cb.upsert(p, e, after=lambda b, s=stem, sc=scope: (cb.name_of(b) or "").startswith(s)
                  and cb.scope_of(b) == sc)
    return cb.join(a), cb.join(p)


def fit_part(data: Path, catalog: Path, chip: str, g: dict, fitted: dict) -> tuple[list, list]:
    """Fit one part from its own rows, descriptor and committed siblings, print the fits,
    and return its (attention, primitives) entries. `fitted` holds the parts already fitted
    this run, so a part with a sibling can compare rows against it."""
    sku = PARTS[chip]["sku"]
    peak, fp8_tflops = catalog_facts(catalog, chip)
    floor = committed(ATTN_PATH, "attention_decode_floor", chip)
    fm = fit_flashmla(data, sku, g, peak, floor)
    ix = fit_indexer(data, sku, g)
    if ix["implied"] <= peak:
        raise SystemExit(
            f"{chip}: the indexer sweep's throughput no longer exceeds the HBM peak "
            f"(footprint {ix['max_unique'] / 1e6:.1f} MB, throughput "
            f"{ix['implied'] / 1e6:.2f} TB/s against {peak / 1e6:g} TB/s): fit its rate "
            f"instead of assuming NVIDIA's HBM rate")
    derate = committed(PRIM_PATH, "hbm_derate", chip)
    bf16_eps = committed(PRIM_PATH, "gemm_eps_max_bf16", chip)
    bf16_mh = int(committed(PRIM_PATH, "gemm_m_half_bf16", chip))
    # The descriptor's nominal pcie_bw, read directly: host_link_bandwidth is now a measured
    # copy-engine rate (fit_host_link.py), not the raw link rate this bound starts from.
    host_bw = yaml.safe_load((data.parent / f"{sku}.yaml").read_text())["node"]["pcie_bw"] / 1e6
    grouped_eps = committed(PRIM_PATH, "gemm_eps_max_w4a16_mxfp4_grouped", chip)
    grouped_mh = int(committed(PRIM_PATH, "gemm_m_half_w4a16_mxfp4_grouped", chip))
    fitted[chip] = {"chip": chip, "fm": fm, "peak": peak}
    sib_chip = PARTS[chip]["sibling"]
    sib = None
    if sib_chip is not None:
        if sib_chip not in fitted:
            raise SystemExit(f"{chip} is compared against {sib_chip}; fit both together")
        sib = fitted[sib_chip]

    print(f"[{chip}]")
    print(f"flashmla sparse decode  rate={fm['rate']:,} B/us ({fm['frac']:.3f} of peak) "
          f"floor pinned {floor:g} us  {fm['err']:.3f}x signed {fm['signed']:.3f} "
          f"(n={fm['n']}); free floor {fm['free_floor']:.1f} us {fm['free_err']:.3f}x; "
          f"holdout >=5 blocks {fm['hold'][0]:.3f}x (n={fm['hold_n']}); "
          f"10 blocks {fm['v41'][0]:.3f}x (n={fm['v41_n']})")
    print(f"{'':24}signed by batch: "
          + ", ".join(f"{b}: {v:.2f}" for b, v in fm["by_batch"].items())
          + f"; batch-1 <= 1 block measures {fm['short_lo']:.1f}-{fm['short_hi']:.1f} us"
          + f"; per-request form {fm['per_req_err']:.3f}x, occupancy form (P={fm['wave_p']}) "
          + f"{fm['wave_err']:.3f}x")
    print(f"indexer scan            floor={ix['floor']} us  {ix['err']:.3f}x (n={ix['n']}) "
          f"per checkpoint {ix['per_model']}; "
          f"throughput {ix['rate_rowkeys']:,.0f} row-keys/us = "
          f"{ix['implied'] / 1e6:.2f} TB/s-equivalent, footprint <= "
          f"{ix['max_unique'] / 1e6:.1f} MB, above the HBM peak: rate assumed at NVIDIA's HBM rate "
          f"{peak * derate:,.0f} B/us")
    print(f"w8a16 mxfp8 ramp        assumed = bf16 ({bf16_eps:g}, {bf16_mh})")
    print(f"host gather (UVA)       assumed = {host_bw:,.0f} x "
          f"128/130 x {ENGRAM_ROW_BYTES}/{ENGRAM_FETCHED_BYTES}")
    return entries(chip, g, fm, ix, peak, fp8_tflops, derate, bf16_eps, bf16_mh,
                   grouped_eps, grouped_mh, host_bw, sib)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if either set would change; write nothing")
    ap.add_argument("--part", choices=sorted(PARTS),
                    help="fit one part only (default: every part)")
    args = ap.parse_args(argv[1:])
    data, catalog = Path(args.data), Path(args.catalog)

    g = check_geometry(data)
    # A part with a sibling is compared against it, so the sibling is fitted too.
    chips = list(PARTS) if args.part is None else [args.part]
    need = set(chips) | {PARTS[c]["sibling"] for c in chips if PARTS[c]["sibling"]}
    fitted: dict = {}
    attn, prim = [], []
    for chip in PARTS:  # dict order puts every sibling before the part compared to it
        if chip not in need:
            continue
        a, p = fit_part(data, catalog, chip, g, fitted)
        if chip in chips:
            attn += a
            prim += p
    a0, p0 = ATTN_PATH.read_text(encoding="utf-8"), PRIM_PATH.read_text(encoding="utf-8")
    a1, p1 = render(a0, p0, attn, prim)
    if args.check:
        stale = [p.name for p, x, y in ((ATTN_PATH, a0, a1), (PRIM_PATH, p0, p1)) if x != y]
        if stale:
            print(f"would change: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("both sets already match")
        return 0
    ATTN_PATH.write_text(a1, encoding="utf-8")
    PRIM_PATH.write_text(p1, encoding="utf-8")
    print("both sets written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
