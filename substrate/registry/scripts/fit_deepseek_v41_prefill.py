#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash on SM90 (H200, H100): the PREFILL attention, PREFILL indexer and
mHC coefficients, each fitted on the part's own AISimulate rows of the kernel vLLM runs, or
declared `assumed` where only a neighbouring kernel was measured.

    python scripts/fit_deepseek_v41_prefill.py              # print the fits, rewrite both sets
    python scripts/fit_deepseek_v41_prefill.py --check      # exit 1 if either set would change
    python scripts/fit_deepseek_v41_prefill.py --part h100  # one part only

The companion of fit_deepseek_v41_h200.py (the decode entries); it reuses that writer's
geometry check, table reader and entry renderer, so the two cannot disagree about V4.1's
kernel geometry. Owns, per part, scoped {hardware: [<part>], model: [deepseek-v4.1-flash]}:

  cost-model-attention.yaml
    attention_prefill_floor_sparse_mla          measured  FlashMLA sparse prefill
    attention_prefill_per_request_sparse_mla    measured
    attention_prefill_per_token_sparse_mla      measured
    attention_prefill_rate_sparse_mla           measured
    attention_prefill_floor_block_index_scores      assumed  indexer logits, prefill
    attention_prefill_per_token_block_index_scores  assumed  (paged-kernel proxy)
    attention_prefill_rate_block_index_scores       assumed
  cost-model-primitives.yaml
    mhc_floor_pre, mhc_rate_pre, mhc_floor_post, mhc_rate_post   measured  TileLang mHC

The per-call forms the kernel evaluates are stated in each rationale (and in the contract
the kernel reader is written against).

WHY THESE ENTRIES. Before them the kernel priced V4.1's sparse-MLA prefill attention and the
indexer's prefill logits with the GENERIC full-attention prefill pair (attention_prefill_floor
/ attention_prefill_work_scale, fitted on context_attention_perf -- a different kernel), and
priced mHC as Elementwise bytes at the HBM rate with no floor: at one decode token that is
0.07 ms a step against 1.05 ms measured.

UNITS. `latency` is MILLISECONDS in every AISimulate parquet. Fits minimise the mean absolute
log ratio, the registry's criterion; the optimiser is a deterministic coordinate search in log
parameter space, so --check reproduces byte for byte with numpy alone.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402
import fit_deepseek_v41_h200 as base  # noqa: E402

ATTN_PATH = base.ATTN_PATH
PRIM_PATH = base.PRIM_PATH
PARTS = {chip: {"sku": p["sku"], "name": p["name"], "e2e": p["e2e"]}
         for chip, p in base.PARTS.items()}
MODEL = base.MODEL
COLLECTION = base.COLLECTION
BLOCK = base.BLOCK

MHC_TABLE = "mhc_module_perf.parquet"
MHC_LANES = {"pre": "vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang",
             "post": "vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang"}
MHC_SITES = 2            # collect_mhc_module.py MHC_NUM_SITES: one row times two sites
# Token bands the mHC error is reported over: decode, mid, prefill.
MHC_BANDS = ((1, 64), (65, 2048), (2049, 1 << 30))


def geo(pred, y) -> tuple[float, float]:
    """(geometric mean |error|, geometric mean signed ratio) of predictions against y."""
    lg = np.log(np.asarray(pred, float) / np.asarray(y, float))
    return float(math.exp(np.mean(np.abs(lg)))), float(math.exp(np.mean(lg)))


def fit_positive(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Positive theta minimising mean |log(X theta / y)|: relative least squares to start,
    then a deterministic coordinate search over log(theta) with halving steps."""
    th0, *_ = np.linalg.lstsq(X / y[:, None], np.ones_like(y), rcond=None)
    scale = np.median(y) / np.maximum(np.median(X, axis=0), 1e-30)
    lt = np.log(np.where(th0 > 0, th0, 1e-3 * scale))

    def obj(v):
        return float(np.mean(np.abs(np.log((X @ np.exp(v)) / y))))

    best, step = obj(lt), 2.0
    while step > 1e-7:
        improved = False
        for j in range(len(lt)):
            for d in (step, -step):
                trial = lt.copy()
                trial[j] += d
                v = obj(trial)
                if v < best - 1e-15:
                    lt, best, improved = trial, v, True
        if not improved:
            step /= 2
    return np.exp(lt)


def signed_by(keys, pred, y) -> dict:
    out: dict = {}
    for k in sorted(set(keys)):
        sel = np.asarray(keys) == k
        out[int(k)] = geo(pred[sel], y[sel])[1]
    return out


def fmt_signed(d: dict) -> str:
    return ", ".join(f"{k}: {v:.2f}" for k, v in d.items())


# --------------------------------------------------------------------------------------
# FlashMLA sparse prefill
# --------------------------------------------------------------------------------------

def sparse_rows(data: Path, sku: str, g: dict) -> dict:
    """The hca_attn PREFILL rows (isl > 1) of the h64 checkpoint, with the regressors of
    floor + per_request x requests + per_token x query tokens + W / rate."""
    t = base.read(data, sku, base.FLASHMLA_TABLE)
    cols = {k: [] for k in ("b", "s", "P", "Q", "blk", "W", "y")}
    for i in range(len(t["latency"])):
        if t["isl"][i] <= 1 or t["model"][i] != base.FLASHMLA_FIT_MODEL:
            continue
        if (t["kernel_source"][i] != base.FLASHMLA_LANE or t["compress_ratio"][i] != 128
                or t["kv_cache_dtype"][i] != "fp8" or t["local_num_heads"][i] != 64
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected FlashMLA prefill row {i}: {t['kernel_source'][i]}, "
                             f"ratio {t['compress_ratio'][i]}, heads {t['local_num_heads'][i]}")
        b, s, P = t["batch_size"][i], t["isl"][i], t["step"][i]
        p = np.arange(P + 1, P + s + 1)
        # Ratio-128 rows read the window and the whole compressed stream (C128A); both
        # tiers are walked in FlashMLA's 64-token blocks, as in the decode fit.
        blk = (np.ceil(np.minimum(p, g["window"]) / BLOCK)
               + np.ceil((p // 128) / BLOCK))
        cols["b"].append(b)
        cols["s"].append(s)
        cols["P"].append(P)
        cols["Q"].append(b * s)
        cols["blk"].append(int(blk.max()))
        cols["W"].append(b * float(blk.sum()) * BLOCK * g["record"])
        cols["y"].append(t["latency"][i] * 1000.0)
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    if len(out["y"]) < 1000:
        raise SystemExit(f"{sku}: only {len(out['y'])} FlashMLA prefill rows")
    return out


def sparse_X(r: dict) -> np.ndarray:
    return np.stack([np.ones_like(r["y"]), r["b"], r["Q"], r["W"]], 1)


def sub(r: dict, sel) -> dict:
    return {k: v[sel] for k, v in r.items()}


def fit_sparse(data: Path, sku: str, g: dict) -> dict:
    allr = sparse_rows(data, sku, g)
    # V4.1 reads at most v41_blocks per query: fit on the rows inside that read.
    r = sub(allr, allr["blk"] <= g["v41_blocks"])
    if g["v41_blocks"] > r["blk"].max():
        raise SystemExit("no prefill row reaches V4.1's block count")
    th = fit_positive(sparse_X(r), r["y"])
    floor = round(float(th[0]), 1)
    per_req = round(float(th[1]), 3)
    per_tok = round(float(th[2]), 4)
    rate = int(round(1.0 / float(th[3])))
    thr = np.array([floor, per_req, per_tok, 1.0 / rate])

    def pred(rr):
        return sparse_X(rr) @ thr

    y = r["y"]
    err, signed = geo(pred(r), y)
    # Held out: (i) the block axis, (ii) the chunk-length axis.
    holds = {}
    for name, tr, te in (("blocks", r["blk"] <= 6, r["blk"] >= 7),
                         ("isl", r["s"] <= 2048, r["s"] >= 3072)):
        t_tr, t_te = sub(r, tr), sub(r, te)
        th_tr = fit_positive(sparse_X(t_tr), t_tr["y"])
        holds[name] = (int(te.sum()), *geo(sparse_X(t_te) @ th_tr, t_te["y"]))
    # Two-parameter form (floor + W / rate), to show what the extra terms buy.
    th2 = fit_positive(sparse_X(r)[:, [0, 3]], y)
    two = geo(sparse_X(r)[:, [0, 3]] @ th2, y)
    # Beyond V4.1's read: long-chunk rows extrapolate; short-chunk, high-block rows do not.
    out = sub(allr, allr["blk"] > g["v41_blocks"])
    long_ = out["s"] >= 1024
    beyond_long = (int(long_.sum()), *geo(pred(sub(out, long_)), out["y"][long_]))
    beyond_short = (int((~long_).sum()), *geo(pred(sub(out, ~long_)), out["y"][~long_]))
    at_v41 = r["blk"] == g["v41_blocks"]
    big = (r["b"] == 1) & (r["s"] == 8192)
    big_rows = sorted((int(P), float(m), float(p)) for P, m, p in
                      zip(r["P"][big], y[big], pred(r)[big]))
    # One 8,192-token chunk at V4.1's full read (10 blocks, 640 keys a query).
    v41_chunk = floor + per_req + per_tok * 8192 + 8192 * g["v41_tokens"] * g["record"] / rate
    return {"n": len(y), "floor": floor, "per_req": per_req, "per_tok": per_tok, "rate": rate,
            "err": err, "signed": signed, "holds": holds, "two": two,
            "two_rate": 1.0 / float(th2[1]), "two_floor": float(th2[0]),
            "by_batch": signed_by(r["b"], pred(r), y),
            "by_blocks": signed_by(r["blk"], pred(r), y),
            "by_isl": signed_by(r["s"], pred(r), y),
            "at_v41": (int(at_v41.sum()), *geo(pred(r)[at_v41], y[at_v41]),
                       int(r["s"][at_v41].max())),
            "beyond_long": beyond_long, "beyond_short": beyond_short,
            "big_rows": big_rows, "v41_chunk": v41_chunk,
            "max_batch": int(r["b"].max()), "max_isl": int(r["s"].max())}


# --------------------------------------------------------------------------------------
# Indexer logits, prefill shapes (proxy: the paged kernel)
# --------------------------------------------------------------------------------------

def fit_indexer_prefill(data: Path, sku: str, g: dict) -> dict:
    t = base.read(data, sku, base.INDEXER_TABLE)
    m_, rk, P_, y_, mod = [], [], [], [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] <= 1:
            continue
        if (t["kernel_source"][i] != base.INDEXER_LANE
                or t["compress_ratio"][i] != base.INDEXER_RATIO
                or t["kv_cache_dtype"][i] != "fp8" or t["mla_dtype"][i] != "fp8_e4m3"
                or t["model"][i] not in base.INDEXER_MODELS
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected indexer prefill row {i}")
        b, s, P = t["batch_size"][i], t["isl"][i], t["step"][i]
        m = b * s
        # The collector packs b x s query rows into one causal sequence:
        # context_lens = (past_kv + row + 1) // 4, at least 1 (_bench_paged_mqa_logits_kernel).
        rows = np.arange(1, m + 1)
        m_.append(m)
        rk.append(float(np.maximum((P + rows) // base.INDEXER_RATIO, 1).sum()))
        P_.append(P)
        y_.append(t["latency"][i] * 1000.0)
        mod.append(t["model"][i])
    m_, rk, P_, y = (np.asarray(v, float) for v in (m_, rk, P_, y_))
    if len(y) < 1000 or set(mod) != base.INDEXER_MODELS:
        raise SystemExit(f"{sku}: only {len(y)} indexer prefill rows")
    W = rk * g["key"]
    X = np.stack([np.ones_like(y), m_, W], 1)
    th = fit_positive(X, y)
    floor = round(float(th[0]), 1)
    per_tok = round(float(th[1]), 5)
    rate = int(round(1.0 / float(th[2])))
    thr = np.array([floor, per_tok, 1.0 / rate])
    pred = X @ thr
    err, signed = geo(pred, y)
    holds = {}
    for name, tr, te in (("context", P_ <= 8192, P_ >= 16384),
                         ("rows", m_ <= 2048, m_ >= 4096)):
        th_tr = fit_positive(X[tr], y[tr])
        holds[name] = (int(te.sum()), *geo(X[te] @ th_tr, y[te]))
    th2 = fit_positive(X[:, [0, 2]], y)
    two = geo(X[:, [0, 2]] @ th2, y)
    per_model = {}
    mod = np.asarray(mod)
    for mm in sorted(set(mod)):
        sel = mod == mm
        per_model[mm] = geo(pred[sel], y[sel])
    # Throughput as FLOPs: 64 heads x 128 x 2 per row-key.
    tflops = (1.0 / th[2]) / g["key"] * g["fit_index_heads"] * 256 / 1e6
    return {"n": len(y), "floor": floor, "per_tok": per_tok, "rate": rate, "err": err,
            "signed": signed, "holds": holds, "two": two, "per_model": per_model,
            "by_rows": signed_by(m_, pred, y), "by_ctx": signed_by(P_, pred, y),
            "tflops": tflops}


RAGGED_TABLE = "glm5_mqa_logits_module_perf.parquet"
RAGGED_COLLECTION = "sglang/0.5.14"
RAGGED_LANE = "deep_gemm.fp8_mqa_logits"
RAGGED_MODEL = "zai-org/GLM-5"


def ragged_gap(data: Path, sku: str, g: dict, ix: dict) -> dict:
    """The committed proxy, applied to AISimulate's timing of the RAGGED DeepGEMM logits
    kernel (SGLang 0.5.14 lane, GLM-5: 32 index heads x 128, FP8 keys, ratio 1) -- the
    kernel family vLLM's prefill indexer calls. Not fitted (the lane decision is open);
    reported so the proxy's error against the real kernel is stated, not guessed."""
    path = data / sku / "sparse_attention" / RAGGED_COLLECTION / RAGGED_TABLE
    if not path.is_file():
        raise SystemExit(f"{path} is absent")
    t = pq.read_table(path).to_pydict()
    cfg = base.model_config(data, "zai-org--GLM-5")
    heads, hd = cfg["index_n_heads"], cfg["index_head_dim"]
    if hd != 128 or heads != g["v41_index_heads"]:
        raise SystemExit(f"GLM-5 index geometry {heads}x{hd} is not V4.1's")
    pred, y, rows, ctx = [], [], [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] <= 1:
            continue
        if (t["kernel_source"][i] != RAGGED_LANE or t["model"][i] != RAGGED_MODEL
                or t["compress_ratio"][i] != 1
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected ragged row {i}")
        b, s, P = t["batch_size"][i], t["isl"][i], t["step"][i]
        # Each of b requests scores keys [0, P + p] for local position p = 1..s.
        keys = b * (s * P + s * (s + 1) / 2)
        w = keys * g["key"] * heads / g["fit_index_heads"]
        pred.append(ix["floor"] + ix["per_tok"] * b * s + w / ix["rate"])
        y.append(t["latency"][i] * 1000.0)
        rows.append(b * s)
        ctx.append(P)
    pred, y, rows, ctx = (np.asarray(v, float) for v in (pred, y, rows, ctx))
    big = (rows >= 4096) & (ctx >= 8192)
    chunk = rows == 8192
    return {"n": len(y), "all": geo(pred, y), "big_n": int(big.sum()),
            "big": geo(pred[big], y[big]), "chunk_n": int(chunk.sum()),
            "chunk": geo(pred[chunk], y[chunk]) if chunk.any() else (math.nan, math.nan),
            "heads": heads}


# --------------------------------------------------------------------------------------
# mHC
# --------------------------------------------------------------------------------------

def check_mhc_geometry(data: Path) -> dict:
    v41 = base.model_config(data, "deepseek-ai--DeepSeek-V4.1-Flash")
    for k in ("hc_mult", "hc_sinkhorn_iters", "hidden_size"):
        if k not in v41:
            raise SystemExit(f"V4.1 config has no {k}")
    return {"hc_mult": v41["hc_mult"], "sinkhorn": v41["hc_sinkhorn_iters"],
            "hidden": v41["hidden_size"]}


def fit_mhc(data: Path, sku: str, mg: dict) -> dict:
    path = data / sku / "mhc" / COLLECTION / MHC_TABLE
    if not path.is_file():
        raise SystemExit(f"{path} is absent; the pinned collection is required")
    t = pq.read_table(path).to_pydict()
    per_hidden = 2 * mg["hc_mult"]          # BF16 x hc_mult residual streams
    out = {}
    for op, lane in MHC_LANES.items():
        T, H, y = [], [], []
        for i in range(len(t["latency"])):
            if t["op_name"][i] != op:
                continue
            if (t["kernel_source"][i] != lane or t["num_sites"][i] != MHC_SITES
                    or t["hc_mult"][i] != mg["hc_mult"]
                    or t["sinkhorn_iters"][i] != mg["sinkhorn"]
                    or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
                raise SystemExit(f"unexpected mHC row {i}: {t['kernel_source'][i]}, sites "
                                 f"{t['num_sites'][i]}, hc_mult {t['hc_mult'][i]}")
            T.append(t["num_tokens"][i])
            H.append(t["hidden_size"][i])
            # One row times MHC_SITES sites; the coefficient is per site.
            y.append(t["latency"][i] * 1000.0 / MHC_SITES)
        T, H, y = (np.asarray(v, float) for v in (T, H, y))
        hiddens = sorted(set(H))
        if len(hiddens) < 2 or not hiddens[0] < mg["hidden"] < hiddens[-1]:
            raise SystemExit(f"{op}: V4.1's hidden {mg['hidden']} is not inside the measured "
                             f"{hiddens}; the rate would be an extrapolation")
        W = T * H * per_hidden
        X = np.stack([np.ones_like(y), W], 1)
        th = fit_positive(X, y)
        floor = round(float(th[0]), 1)
        rate = int(round(1.0 / float(th[1])))
        thr = np.array([floor, 1.0 / rate])
        pred = X @ thr
        err, signed = geo(pred, y)
        bands = {}
        for lo, hi in MHC_BANDS:
            sel = (T >= lo) & (T <= hi)
            bands[(lo, hi)] = (int(sel.sum()), *geo(pred[sel], y[sel]))
        cross = {}
        for h in hiddens:
            tr, te = H != h, H == h
            th_tr = fit_positive(X[tr], y[tr])
            cross[int(h)] = (int(te.sum()), *geo(X[te] @ th_tr, y[te]))
        per_h = {}
        for h in hiddens:
            sel = H == h
            per_h[int(h)] = fit_positive(X[sel], y[sel])
        out[op] = {"n": len(y), "floor": floor, "rate": rate, "err": err, "signed": signed,
                   "bands": bands, "cross": cross,
                   "per_h": {h: (float(v[0]), 1.0 / float(v[1])) for h, v in per_h.items()},
                   "hiddens": [int(h) for h in hiddens], "max_tokens": int(T.max())}
    w1 = mg["hidden"] * per_hidden
    out["site_us"] = {T: sum(out[op]["floor"] + T * w1 / out[op]["rate"] for op in MHC_LANES)
                      for T in (6, 8192)}
    out["bytes_per_token"] = w1
    return out


# --------------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------------

def entries(chip: str, g: dict, mg: dict, sp: dict, ix: dict, rg: dict, mh: dict, peak: float,
            fp8_tflops: float, gen_floor: float, gen_scale: float,
            hbm_rate: float) -> tuple[list, list]:
    part = PARTS[chip]
    sku, name = part["sku"], part["name"]
    scope = base.model_scope(chip)
    aisim = f"NVIDIA AISimulate systems/data/{sku}/sparse_attention/vllm/0.25.0/"
    lines = base.lines
    tb = peak / 1e6
    form_sp = (
        f"THE FORM the kernel evaluates per layer, per rank, per step holding prefill: "
        f"t = attention_prefill_floor_sparse_mla + attention_prefill_per_request_sparse_mla x "
        f"prefill requests + attention_prefill_per_token_sparse_mla x prefill query tokens + "
        f"W / attention_prefill_rate_sparse_mla, W = sum over prefill query tokens of "
        f"64 x (ceil(min(p, window) / 64) + ceil(min(p / ratio, index_topk) / 64)) x "
        f"{g['record']} B x padded_heads / 64, p the query's position + 1 -- the decode "
        f"rate's block-and-record count, per query. For V4.1 (window {g['window']}, "
        f"top-{g['topk']}) every query past about 1k tokens reads {g['v41_blocks']} blocks, "
        f"{g['v41_tokens'] * g['record']:,} B. The four replace the generic "
        f"attention_prefill_floor / attention_prefill_work_scale pair for this kind; all four "
        f"or none are used.")
    common_sp = (
        f"Fitted jointly over the {sp['n']} prefill rows (isl > 1) of {base.FLASHMLA_FIT_MODEL} "
        f"in {base.FLASHMLA_TABLE} ({name}) that read at most V4.1's {g['v41_blocks']} blocks "
        f"a query (batch 1-{sp['max_batch']}, chunks 4-{sp['max_isl']} tokens): geometric error "
        f"{sp['err']:.3f}x, signed {sp['signed']:.3f}. The two-parameter form, floor + W / "
        f"rate, gives {sp['two'][0]:.3f}x and biases with batch; the per-request and "
        f"per-token terms are what remove it. Signed by batch: {fmt_signed(sp['by_batch'])}. "
        f"Signed by blocks a query: {fmt_signed(sp['by_blocks'])}. HELD OUT: trained on <= 6 "
        f"blocks, predicting 7-{g['v41_blocks']} blocks {sp['holds']['blocks'][1]:.3f}x (signed "
        f"{sp['holds']['blocks'][2]:.3f}, n={sp['holds']['blocks'][0]}); trained on chunks <= "
        f"2,048 tokens, predicting >= 3,072 {sp['holds']['isl'][1]:.3f}x (signed "
        f"{sp['holds']['isl'][2]:.3f}, n={sp['holds']['isl'][0]}). AT V4.1'S READ: the "
        f"{sp['at_v41'][0]} rows at exactly {g['v41_blocks']} blocks are all short chunks (<= "
        f"{sp['at_v41'][3]} tokens, at step 65,536) and the fit under-prices them "
        f"{sp['at_v41'][1]:.3f}x (signed {sp['at_v41'][2]:.3f}); long chunks at V4.1's read are "
        f"not in the table (an 8,192-token chunk at step 65,536 crosses 11 blocks), and the "
        f"{sp['beyond_long'][0]} long-chunk (>= 1,024 tokens) rows beyond it, 11-67 blocks, "
        f"extrapolate at {sp['beyond_long'][1]:.3f}x (signed {sp['beyond_long'][2]:.3f}) while "
        f"the {sp['beyond_short'][0]} short-chunk rows there do not ({sp['beyond_short'][1]:.2f}x, "
        f"signed {sp['beyond_short'][2]:.2f}: few queries over many blocks under-fill the "
        f"part). One {8192:,}-token chunk at V4.1's full read prices at "
        f"{sp['v41_chunk'] / 1e3:.2f} ms a layer. WHAT THE TIMED OP CONTAINS: vLLM 0.25.0 "
        f"DeepseekV4FlashMLAAttention.forward_mqa on the prefill branch under a CUDA graph -- "
        f"the dequantize-and-gather of the SWA window and of the (ratio-128, at most 512-"
        f"entry) compressed stream into a BF16 workspace, combine_topk_swa_indices, and "
        f"flash_mla_sparse_fwd (SM90 sparse prefill, D_QK 512). NOT CONTAINED: projections, "
        f"q/kv norms and rope (catalog GEMM and Elementwise nodes). V4.1's ratio-1/2 layers "
        f"gather their FULL-length compressed stream before the kernel (v0.31.0 flashmla.py "
        f"_forward_prefill, 'every cr>0 layer gathers a full-length compressed region'), which "
        f"the measured ratio-128 rows hold only at 1/128 of the context; that gather stays "
        f"priced by the catalog's prefill_dequant_gather node. VERSIONS: the SM90 sparse "
        f"prefill device code is unchanged between the FlashMLA pins vLLM 0.25.0 and 0.31.0 "
        f"build (a6ec2ba -> 0eee43b: csrc/sm90/prefill/sparse moved under csrc/kernels with "
        f"include-path and namespace renames only); v0.31.0's _forward_prefill adds "
        f"preallocated index workspaces and the full-length gather above, and keeps "
        f"combined_topk = round_up(top_k + window, 128) = {math.ceil(g['v41_tokens'] / 128) * 128}. "
        f"SCOPED TO THIS MODEL: fitted on the h64 instance V4.1 runs at every TP width. "
        f"{part['e2e']} Reproduce with scripts/fit_deepseek_v41_prefill.py.")
    sp_src = [("model", f"{aisim}{base.FLASHMLA_TABLE} (NVIDIA {name}), {base.FLASHMLA_LANE}, "
               f"{base.FLASHMLA_FIT_MODEL}, {sp['n']} prefill points", "primary"),
              ("model", "vLLM v0.31.0 models/deepseek_v41/nvidia/flashmla.py "
               "DeepseekV4FlashMLAAttention._forward_prefill", "supporting")]
    attn = [
        lines("attention_prefill_floor_sparse_mla", sp["floor"], "us_per_transfer",
              "measured", True, scope, sp_src,
              f"The per-call floor of V4.1's FlashMLA sparse PREFILL on SM90: {sp['floor']} us, "
              f"charged once per layer per rank in a step with prefill (a mixed step also pays "
              f"the decode floor: FlashMLA launches the two regimes separately). Against the "
              f"generic attention_prefill_floor of {gen_floor:g} us that priced this kernel "
              f"before. {form_sp} {common_sp}"),
        lines("attention_prefill_per_request_sparse_mla", sp["per_req"], "us_per_request",
              "measured", True, scope, sp_src,
              f"{sp['per_req']} us per prefill request per layer: the per-request work of the "
              f"prefill branch (one gather and index-combine segment per request). Fitted with "
              f"the floor, per-token term and rate; see attention_prefill_floor_sparse_mla for "
              f"the form, the errors and the held-out results."),
        lines("attention_prefill_per_token_sparse_mla", sp["per_tok"], "us_per_token",
              "measured", True, scope, sp_src,
              f"{sp['per_tok']} us per prefill query token per layer, independent of how many "
              f"keys the query reads: the query's own traffic -- 64 padded heads x 512 BF16 in "
              f"and out, the gather and dequantize of its own window entry, its combined index "
              f"row. At {sp['per_tok']} us a token this is the cost of moving about "
              f"{sp['per_tok'] * hbm_rate / 1e3:.0f} KB at this part's empirical HBM rate, the "
              f"size of that per-query traffic. Fitted with the floor, per-request term and "
              f"rate; see attention_prefill_floor_sparse_mla."),
        lines("attention_prefill_rate_sparse_mla", f"{sp['rate']}.0", "bytes_per_us",
              "measured", True, scope, sp_src,
              f"{sp['rate'] / 1e6:.2f} TB/s-equivalent: the rate over W, the {g['record']} B "
              f"record counted once per (query, key-slot) pair in 64-token blocks -- a WORK "
              f"rate, not an HBM read, because in prefill every query of a request re-reads "
              f"the same gathered workspace from cache; it is {sp['rate'] / peak:.2f}x this "
              f"part's {tb:g} TB/s datasheet bandwidth and must not be rescaled by BwPeak. "
              f"Free of the floor and per-query terms the rate is {sp['two_rate'] / 1e6:.2f} "
              f"TB/s ({sp['two'][0]:.3f}x), which is why the four are used together. See "
              f"attention_prefill_floor_sparse_mla for the form and every error."),
    ]
    per_model = ", ".join(f"{k.split('/')[-1]} {v[0]:.3f}x"
                          for k, v in ix["per_model"].items())
    ix_note = (
        f"ASSUMED -- A PROXY KERNEL, fitted on measured rows: vLLM v0.31.0's prefill indexer "
        f"runs DeepGEMM fp8_fp4_mqa_logits (the ragged kernel, sparse_attn_indexer.py prefill "
        f"chunks, after cp_gather_indexer_k_quant_cache), and this is fitted on the PAGED kernel "
        f"fp8_fp4_paged_mqa_logits that decode runs, at prefill shapes "
        f"({base.INDEXER_TABLE}, isl > 1: b x isl query rows packed into one causal sequence, "
        f"every row on one block table, 64 index heads). THE RAGGED KERNEL IS MEASURED, on "
        f"another lane: AISimulate's sglang/0.5.14 {RAGGED_TABLE} times "
        f"{RAGGED_LANE} at GLM-5's indexer geometry, {rg['heads']} index heads x 128, FP8 keys, "
        f"ratio 1 -- V4.1's head count. Against those {rg['n']} rows this proxy is "
        f"{rg['all'][0]:.3f}x, signed {rg['all'][1]:.3f}; on the {rg['big_n']} rows of at least "
        f"4,096 query rows at a prefix of at least 8,192, signed {rg['big'][1]:.3f} "
        f"({rg['big'][0]:.3f}x); at V4.1's 8,192-token chunk ({rg['chunk_n']} rows) signed "
        f"{rg['chunk'][1]:.3f}, evaluated as the kernel will charge it, W scaled to "
        f"{rg['heads']} of {g['fit_index_heads']} heads (an earlier review that read the "
        f"table's num_heads, GLM-5's 64 ATTENTION heads, as index heads found the proxy "
        f"over-pricing by 1.4-1.7x; at the config's {rg['heads']} index heads it is not). So "
        f"on the real kernel the proxy is within about 20% either way, "
        f"{'under' if rg['big'][1] < 1 else 'over'}-pricing long chunks over long prefixes "
        f"and {'under' if rg['chunk'][1] < 1 else 'over'}-pricing V4.1's chunk size; on its "
        f"own paged rows it is signed {ix['by_rows'].get(8192, float('nan')):.2f} at 8,192 "
        f"rows. NOT FITTED on the ragged rows because the lane decision (methodology section "
        f"2) is open: SGLang 0.5.14 calls upstream DeepGEMM's fp8_mqa_logits, vLLM v0.31.0 its "
        f"fork's fp8_fp4_mqa_logits on the FP8 path, and the device code of the two pins has "
        f"not been compared; the SGLang collector also splits large batches along the query "
        f"axis. THE FORM "
        f"per index-source layer, per rank, per step with prefill: t = "
        f"attention_prefill_floor_block_index_scores + "
        f"attention_prefill_per_token_block_index_scores x prefill query tokens + W / "
        f"attention_prefill_rate_block_index_scores, W = sum over prefill query tokens of "
        f"ceil(p / ratio) x {g['key']} B x n_q / {g['fit_index_heads']} -- every compressed key "
        f"the query can see, at the {g['key']} B FP8 key; n_q is the node's index heads "
        f"({g['v41_index_heads']} for V4.1 against the measured {g['fit_index_heads']}), a "
        f"LINEAR head scaling that is itself assumed (the rows are compute-bound: the fitted "
        f"rate is {ix['tflops']:.0f} TFLOPS at 64 heads x 128 x 2 FLOP a row-key, against "
        f"{fp8_tflops:g} FP8 dense peak). The three replace the generic prefill pair for this "
        f"role. Fitted over {ix['n']} rows (V4-Flash-FP8 and V4-Pro-FP8, both 64 index heads, "
        f"head_dim 128, FP8 keys): {ix['err']:.3f}x, signed {ix['signed']:.3f}; floor + rate "
        f"alone {ix['two'][0]:.3f}x. Signed by query rows: {fmt_signed(ix['by_rows'])}. Per "
        f"checkpoint: {per_model}. "
        f"HELD OUT: trained on step <= 8,192, predicting >= 16,384 {ix['holds']['context'][1]:.3f}x "
        f"(signed {ix['holds']['context'][2]:.3f}, n={ix['holds']['context'][0]}); trained on "
        f"<= 2,048 rows, predicting >= 4,096 {ix['holds']['rows'][1]:.3f}x (signed "
        f"{ix['holds']['rows'][2]:.3f}, n={ix['holds']['rows'][0]}). NOT CONTAINED: the K gather "
        f"(catalog index_prefill_k_gather), top_k_per_row_prefill and V4.1's candidate "
        f"select/mask (catalog index_logits_topk), the indexer projections (catalog GEMMs). "
        f"VERSIONS: the paged kernel and scheduler differ between the DeepGEMM pins of vLLM "
        f"0.25.0 and 0.31.0 only by renames and zero-length bounds handling (see the decode "
        f"floor's rationale). TO REPLACE IT: a sweep of fp8_fp4_mqa_logits over (query rows, "
        f"keys) at 32 and 64 index heads on {name}, or a recorded lane decision for the ragged "
        f"rows above. Scoped to this model. "
        f"Reproduce with scripts/fit_deepseek_v41_prefill.py.")
    ix_src = [("model", f"{aisim}{base.INDEXER_TABLE} (NVIDIA {name}), {base.INDEXER_LANE}, "
               f"{ix['n']} prefill-shaped points (a proxy for fp8_fp4_mqa_logits)", "primary"),
              ("model", "vLLM v0.31.0 model_executor/layers/sparse_attn_indexer.py prefill "
               "chunk loop, fp8_fp4_mqa_logits", "supporting")]
    attn += [
        lines("attention_prefill_floor_block_index_scores", ix["floor"], "us_per_transfer",
              "assumed", False, scope, ix_src,
              f"{ix['floor']} us per index-source layer per rank in a step with prefill. "
              f"{ix_note}"),
        lines("attention_prefill_per_token_block_index_scores", ix["per_tok"], "us_per_token",
              "assumed", False, scope, ix_src,
              f"{ix['per_tok']} us per prefill query token (one row: its query, weights and "
              f"logits row setup), not scaled by heads. Fitted with the floor and rate; see "
              f"attention_prefill_floor_block_index_scores for the form, the proxy and every "
              f"error."),
        lines("attention_prefill_rate_block_index_scores", f"{ix['rate']}.0", "bytes_per_us",
              "assumed", False, scope, ix_src,
              f"{ix['rate'] / 1e6:.2f} TB/s-equivalent over W (row-keys x {g['key']} B x n_q / "
              f"{g['fit_index_heads']}): a compute rate expressed per key byte -- every query "
              f"row of a request reads the same keys, which is why it exceeds the "
              f"{tb:g} TB/s HBM peak and why, unlike the decode rate, the shared-block-table "
              f"sweep is representative here. See attention_prefill_floor_block_index_scores."),
    ]
    pre, post = mh["pre"], mh["post"]
    bpt = mh["bytes_per_token"]

    def band_text(d):
        return "; ".join(f"{lo}-{hi if hi < 1 << 29 else 'max'} tokens {n} rows {e:.3f}x "
                         f"(signed {s:.3f})" for (lo, hi), (n, e, s) in d["bands"].items())

    def cross_text(d):
        return "; ".join(f"hidden {h} predicted from the other {e:.3f}x (signed {s:.3f}, n={n})"
                         for h, (n, e, s) in d["cross"].items())

    mhc_common = (
        f"THE FORM the kernel evaluates for every Elementwise node whose role starts with "
        f"mhc_, per rank, per step: t = sum over the node's components (pre if the role ends "
        f"in _pre, post if it contains post) of mhc_floor_<c> + W / mhc_rate_<c>, W = rows x "
        f"hidden_size x {2 * mg['hc_mult']} B (hc_mult {mg['hc_mult']} BF16 residual streams; "
        f"{bpt:,} B a token at V4.1's {mg['hidden']}), rows the tokens the node runs on this "
        f"rank (its token shard under sequence parallelism). This REPLACES the node's "
        f"bytes_per_token charge, and the hc_*_fn projection GEMM after it is NOT charged "
        f"separately: the timed pre includes the TF32 split-K prenorm GEMM "
        f"(tf32_hc_prenorm_gemm) over the same fn. A post+pre site at V4.1's geometry prices "
        f"at {mh['site_us'][6]:.1f} us for 6 tokens and {mh['site_us'][8192] / 1e3:.3f} ms for "
        f"8,192. MEASURED: {MHC_TABLE} (vllm/0.25.0, {name}), TileLang mhc_pre / mhc_post "
        f"under CUDA graphs, hc_mult {mg['hc_mult']}, Sinkhorn {mg['sinkhorn']} iterations, "
        f"hidden {pre['hiddens'][0]} and {pre['hiddens'][1]} (V4.1's {mg['hidden']} lies between: "
        f"W carries the hidden size, so one floor and one rate are fitted over both and "
        f"interpolate by bytes); each row times {MHC_SITES} sites and is divided by "
        f"{MHC_SITES}. VERSIONS: on SM90 vLLM v0.31.0 runs V4.1's mHC as "
        f"mhc_fused_post_pre_delayed_tilelang (the DeepGEMM mega-mHC path and the overlap "
        f"stream need SM100); above {32} tokens it runs 'the same post kernel and split-k "
        f"GEMM as the unfused pair' measured here, and at <= 32 tokens it folds the post into "
        f"the prenorm GEMM, one launch fewer, so at decode sizes this pair is an UPPER bound "
        f"by about one launch. The measured pre is the norm_weight=None variant; the fused "
        f"serving path also applies the attention RMSNorm, which AISimulate's collector "
        f"records as 2-3% of pre on H20. {part['e2e']} Reproduce with "
        f"scripts/fit_deepseek_v41_prefill.py.")
    mhc_src = [("model", f"NVIDIA AISimulate systems/data/{sku}/mhc/vllm/0.25.0/{MHC_TABLE} "
                f"(NVIDIA {name}), {{}} rows", "primary")]
    # cost-model-primitives is single-sourced on AISimulate; the vLLM v0.31.0 code read
    # (mhc_fused_post_pre_delayed_tilelang, deepseek_v41/nvidia/ops/mhc.py) is in the text.

    def src(op):
        return [(k, c.replace("{}", f"{mh[op]['n']} {op}"), r) for k, c, r in mhc_src]

    prim = []
    for op, d in (("pre", pre), ("post", post)):
        what = ("the RMS prenorm GEMM over the hc_mult x hidden residual, the Sinkhorn "
                "mixing and the layer-input write" if op == "pre" else
                "the residual update from the sublayer output with the post and comb mixes")
        prim += [
            lines(f"mhc_floor_{op}", d["floor"], "us_per_transfer", "measured", True, scope,
                  src(op),
                  f"The per-site floor of V4.1's mHC {op} ({what}): {d['floor']} us. Fitted "
                  f"jointly with mhc_rate_{op} over {d['n']} sites, 1-{d['max_tokens']:,} "
                  f"tokens: {d['err']:.3f}x, signed {d['signed']:.3f}. By band: "
                  f"{band_text(d)}. HELD OUT across hidden: {cross_text(d)}. {mhc_common}"),
            lines(f"mhc_rate_{op}", f"{d['rate']}.0", "bytes_per_us", "measured", True, scope,
                  src(op),
                  f"{d['rate'] / 1e6:.2f} TB/s over the residual-stream bytes W for the mHC "
                  f"{op}, {d['rate'] / peak:.2f} of this part's {tb:g} TB/s datasheet "
                  f"bandwidth. Fitted on each hidden size alone it is "
                  f"{', '.join(f'{h}: {v[1] / 1e6:.2f} TB/s (floor {v[0]:.1f} us)' for h, v in d['per_h'].items())}"
                  f" -- the bytes form carries the hidden size. See mhc_floor_{op} for the "
                  f"errors and the form."),
        ]
    return attn, prim


def render(attn_text: str, prim_text: str, attn: list, prim: list) -> tuple[str, str]:
    a = cb.split(attn_text)
    for e in attn:
        cb.upsert(a, e, after=lambda b: cb.name_of(b) is not None)
    p = cb.split(prim_text)
    for e in prim:
        scope = cb.scope_of(e)
        # After the model-scoped primitives of this part (the W8A16 pair / host gather are
        # part-scoped, so anchor on the last entry of any kind).
        cb.upsert(p, e, after=lambda b, sc=scope: cb.name_of(b) is not None)
    return cb.join(a), cb.join(p)


def fit_part(data: Path, catalog: Path, chip: str, g: dict, mg: dict) -> tuple[list, list]:
    sku = PARTS[chip]["sku"]
    peak, fp8_tflops = base.catalog_facts(catalog, chip)
    sp = fit_sparse(data, sku, g)
    ix = fit_indexer_prefill(data, sku, g)
    rg = ragged_gap(data, sku, g, ix)
    print(f"ragged check     proxy on {RAGGED_TABLE}: {rg['all'][0]:.3f}x signed "
          f"{rg['all'][1]:.3f} (n={rg['n']}); >=4096 rows at >=8k prefix signed "
          f"{rg['big'][1]:.3f} (n={rg['big_n']}); 8192-row chunks signed {rg['chunk'][1]:.3f}")
    mh = fit_mhc(data, sku, mg)
    gen_floor = base.committed(ATTN_PATH, "attention_prefill_floor", chip)
    gen_scale = base.committed(ATTN_PATH, "attention_prefill_work_scale", chip)
    derate = base.committed(PRIM_PATH, "hbm_derate", chip)
    print(f"[{chip}]")
    print(f"sparse prefill   floor={sp['floor']} us per_req={sp['per_req']} us "
          f"per_tok={sp['per_tok']} us rate={sp['rate']:,} B/us  {sp['err']:.3f}x signed "
          f"{sp['signed']:.3f} (n={sp['n']}); 2-param {sp['two'][0]:.3f}x; holdout blocks "
          f"{sp['holds']['blocks'][1]:.3f}x isl {sp['holds']['isl'][1]:.3f}x; at 10 blocks "
          f"{sp['at_v41'][1]:.3f}x signed {sp['at_v41'][2]:.3f}; 8192 chunk "
          f"{sp['v41_chunk'] / 1e3:.3f} ms")
    print(f"                 by batch: {fmt_signed(sp['by_batch'])}")
    print(f"                 b1 s8192: " + ", ".join(f"P{P}: {m:.0f}/{p:.0f}"
                                                     for P, m, p in sp["big_rows"]))
    print(f"indexer prefill  floor={ix['floor']} us per_tok={ix['per_tok']} us "
          f"rate={ix['rate']:,} B/us  {ix['err']:.3f}x signed {ix['signed']:.3f} "
          f"(n={ix['n']}); holdout ctx {ix['holds']['context'][1]:.3f}x rows "
          f"{ix['holds']['rows'][1]:.3f}x; {ix['tflops']:.0f} TFLOPS")
    for op in MHC_LANES:
        d = mh[op]
        print(f"mhc {op:4s}         floor={d['floor']} us rate={d['rate']:,} B/us "
              f"{d['err']:.3f}x signed {d['signed']:.3f} (n={d['n']}); cross-hidden "
              + ", ".join(f"{h}: {e:.3f}x" for h, (_, e, _) in d["cross"].items()))
    print(f"mhc site         6 tok {mh['site_us'][6]:.1f} us, 8192 tok "
          f"{mh['site_us'][8192]:.0f} us")
    hbm_rate = peak * derate
    return entries(chip, g, mg, sp, ix, rg, mh, peak, fp8_tflops, gen_floor, gen_scale,
                   hbm_rate)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=base.DEFAULT_DATA)
    ap.add_argument("--catalog", default=base.DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if either set would change; write nothing")
    ap.add_argument("--part", choices=sorted(PARTS),
                    help="fit one part only (default: every part)")
    args = ap.parse_args(argv[1:])
    data, catalog = Path(args.data), Path(args.catalog)
    g = base.check_geometry(data)
    mg = check_mhc_geometry(data)
    chips = list(PARTS) if args.part is None else [args.part]
    attn, prim = [], []
    for chip in chips:
        a, p = fit_part(data, catalog, chip, g, mg)
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
