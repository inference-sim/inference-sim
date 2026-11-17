#!/usr/bin/env python3
"""Write attention_prefill_work_scale_sparse_mla_kv_fp8 for the GLM-5.3 family on h200: how much
dearer a sparse-MLA prefill is per key slot over an fp8_ds_mla cache than over a BF16 one.

WHY. The family's prefill form (scripts/fit_glm53_attention.py) was fitted on a BF16 cache. Its W
counts the layer's record, and a DSA layer takes the engine's cell for it, 656 B at fp8_ds_mla
against GLM-5.3's 1152 B BF16 latent, so without this entry an fp8 cache pays 656/1152 = 0.569 of
the BF16 cost, as if time scaled with the bytes. vLLM does not run it that way: with fewer than 32
heads a rank (GLM-5.3 at TP8 runs 8), FlashMLA sparse runs an fp8_ds_mla prefill through its FP8
DECODE kernel in one mixed batch, its heads padded (MIN_HEADS_FOR_BF16_PREFILL, fp8_use_mixed_batch,
vllm/v1/attention/backends/mla/flashmla_sparse.py:66-75 and :443-447 at v0.32.0). The kernel prices
W at the cell as the BF16 record's W times this scale (kernel/prefill_forms.go packedFP8Prefill).

DATA. aisim's h200 dsa_context_module_perf.parquet for vLLM 0.30.0 measures zai-org/GLM-5.2
(GlmMoeDsaForCausalLM, GLM-5.3's attention: a 512 latent + 64 rope, top-k 2048) both ways on one
collector: a BF16 cache served by FLASH_ATTN_MLA_SPARSE and an fp8 cache by FLASHMLA_SPARSE. On
op dsa_context_module_skip_indexer (the indexer is priced apart, as the scorer), 8 heads a rank,
fp8_block projections (GLM-5.3's checkpoint), rows whose cached prefix + new tokens exceed the top-k,
each cache's latency is fitted by ordinary least squares as a + b x K, K = batch x sum over query
positions p of 64 x ceil(min(p, 2048) / 64) (key slots). The scale is b_fp8 / b_bf16.

NAMED SUBSTITUTIONS: GLM-5.2's rows for GLM-5.3 and GLM-5.3-Flash (the Flash record is 1024 B, so
it is moved further); vLLM 0.30.0's kernels for the engine's; FLASH_ATTN_MLA_SPARSE's BF16 path as
the reference for the family's FlashInfer SM90 fit (both are BF16 sparse prefill; their per-slot
costs are reported in the rationale). Hence transferred, fitted false.

Pre-registered as fits prereg 03 (S1). --check exits 1 if the committed set would change.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REG / "scripts"))
import coefficient_blocks as cb  # noqa: E402

ATTN_PATH = REG / "coefficients" / "cost-model-attention.yaml"
TABLE = Path("h200_sxm/sparse_attention/vllm/0.30.0/dsa_context_module_perf.parquet")
TABLE_SHA256 = "316941f994bebe7bcff6de1407dd3d21f345eefabb4280165c2aba3a25a2a001"
AISIM_SHA = "6fdf6fb6b2e1c15b8f5cd1ba35fd2d50cdb2eb18"
NAME = "attention_prefill_work_scale_sparse_mla_kv_fp8"
MODELS = ("glm-5.3", "glm-5.3-flash")
OP = "dsa_context_module_skip_indexer"
SOURCE_MODEL = "zai-org/GLM-5.2"
HEADS = 8
GEMM = "fp8_block"
TOPK = 2048
MAX_CONTEXT = 262144
PATHS = {"bfloat16": "FLASH_ATTN_MLA_SPARSE", "fp8": "FLASHMLA_SPARSE"}
MARKER = "  # BEGIN trtllm-gen prefill lane"


def slots(prefix: int, new: int) -> int:
    """Key slots the queries at positions prefix+1 .. prefix+new read: 64 x ceil(min(p, topk)/64)."""
    lo, hi = prefix + 1, prefix + new
    q = np.arange(lo, min(hi, TOPK - 1) + 1)
    s = int((64 * np.ceil(q / 64)).sum()) if len(q) else 0
    if hi >= max(lo, TOPK):
        s += (hi - max(lo, TOPK) + 1) * TOPK
    return s


def rows(data: Path) -> dict:
    path = data / TABLE
    if not path.is_file():
        raise SystemExit(f"{path} is absent: set AISIM_GLM53_DATA to aisim {AISIM_SHA[:8]}'s systems/data")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TABLE_SHA256:
        raise SystemExit(f"{path}: sha256 {digest}, not the pinned {TABLE_SHA256}")
    out = {kv: {"K": [], "y": [], "b": [], "x": [], "p": []} for kv in PATHS}
    for r in pq.read_table(path).to_pylist():
        if (r["model"], r["op_name"], r["num_heads"], r["gemm_type"]) != (SOURCE_MODEL, OP, HEADS, GEMM):
            continue
        kv = r["kv_cache_dtype"]
        if PATHS.get(kv) != r["kernel_source"]:
            raise SystemExit(f"unexpected row: kv {kv} on {r['kernel_source']}")
        b, x, p = int(r["batch_size"]), int(r["isl"]), int(r["step"])
        if not (TOPK < p + x <= MAX_CONTEXT):
            continue
        o = out[kv]
        o["K"].append(b * slots(p, x))
        o["y"].append(float(r["latency"]) * 1000.0)  # ms -> us
        o["b"].append(b)
        o["x"].append(x)
        o["p"].append(p)
    for kv, o in out.items():
        for k in o:
            o[k] = np.asarray(o[k], float)
        if len(o["y"]) != 984:
            raise SystemExit(f"{len(o['y'])} {kv} rows above the top-k, expected 984")
    return out


def ols(o: dict, sel=None) -> tuple[float, float]:
    K, y = (o["K"], o["y"]) if sel is None else (o["K"][sel], o["y"][sel])
    (a, b), *_ = np.linalg.lstsq(np.vstack([np.ones_like(K), K]).T, y, rcond=None)
    return float(a), float(b)


def run(data: Path) -> dict:
    r = rows(data)
    fit = {kv: ols(o) for kv, o in r.items()}
    scale = fit["fp8"][1] / fit["bfloat16"][1]
    err = {}
    for kv, (a, b) in fit.items():
        o = r[kv]
        err[kv] = math.exp(float(np.mean(np.abs(np.log((a + b * o["K"]) / o["y"])))))
    # Held out by prefix: fit on prefix <= 16,384, predict the scale's b on >= 32,768.
    held = {}
    for name, tr, te in (("prefix", lambda o: o["p"] <= 16384, lambda o: o["p"] >= 32768),
                         ("new tokens", lambda o: o["x"] <= 2048, lambda o: o["x"] >= 4096)):
        held[name] = (ols(r["fp8"], tr(r["fp8"]))[1] / ols(r["bfloat16"], tr(r["bfloat16"]))[1],
                      ols(r["fp8"], te(r["fp8"]))[1] / ols(r["bfloat16"], te(r["bfloat16"]))[1])
    # 95% interval: 200 bootstrap refits over cells (the same cell drawn for both caches), fixed seed.
    rng = np.random.RandomState(20261116)
    n = len(r["fp8"]["y"])
    order = {kv: np.lexsort((o["p"], o["x"], o["b"])) for kv, o in r.items()}
    bs = []
    for _ in range(200):
        i = rng.randint(0, n, n)
        bs.append(ols(r["fp8"], order["fp8"][i])[1] / ols(r["bfloat16"], order["bfloat16"][i])[1])
    o = r["fp8"]
    return {"scale": round(scale, 3), "fit": fit, "err": err, "held": held, "n": n,
            "ci": (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)),
            "range": {"tokens": (float((o["b"] * o["x"]).min()), float((o["b"] * o["x"]).max())),
                      "requests": (float(o["b"].min()), float(o["b"].max())),
                      "context_tokens": (float(o["p"].min()), float(o["p"].max()))}}


def entry(f: dict) -> list[str]:
    (a8, b8), (a16, b16) = f["fit"]["fp8"], f["fit"]["bfloat16"]
    h = f["held"]
    cite = (f"NVIDIA AISimulate (ai-dynamo/aisim {AISIM_SHA[:8]}) systems/data/{TABLE} (NVIDIA H200, sha256 "
            f"{TABLE_SHA256[:12]}), vLLM 0.30.0, {SOURCE_MODEL}, op {OP}, {HEADS} heads a rank, {GEMM} projections: "
            f"{f['n']} cells above the top-k on each of FLASHMLA_SPARSE (fp8 cache) and FLASH_ATTN_MLA_SPARSE (BF16 cache)")
    rationale = (
        f"{f['scale']}x: a sparse-MLA prefill over an fp8_ds_mla cache costs this many times the BF16 cache's time "
        f"per (query, 64-token key slot) on H200 at {HEADS} heads a rank, measured on GLM-5.3's attention geometry. "
        f"The kernel prices W at the 656 B cell as the BF16 record's W times this scale (kernel/prefill_forms.go "
        f"packedFP8Prefill); without it an fp8 cache was charged 656/1152 = 0.569 of the BF16 cost, an assumption "
        f"this measurement refutes. WHY DEARER: with fewer than 32 heads a rank vLLM runs an fp8_ds_mla prefill "
        f"through FlashMLA's FP8 decode kernel in one mixed batch, heads padded (vllm/v1/attention/backends/mla/"
        f"flashmla_sparse.py:66-75, :443-447 at v0.32.0). THE FIT: latency = a + b x K by ordinary least squares, K "
        f"= batch x sum over query positions p of 64 x ceil(min(p, 2048) / 64): fp8 a {a8:.1f} us, b {b8 * 1e6:.2f} ps "
        f"a slot ({f['err']['fp8']:.3f}x); BF16 a {a16:.1f} us, b {b16 * 1e6:.2f} ps a slot ({f['err']['bfloat16']:.3f}x). "
        f"The fp8 path's intercept is {a8 - a16:.0f} us above the BF16 one's; this entry does not carry it, and the "
        f"family's floor stays BF16's. HELD OUT: the scale fitted on prefix <= 16,384 is {h['prefix'][0]:.3f} and on "
        f">= 32,768 {h['prefix'][1]:.3f}; on new tokens <= 2,048 {h['new tokens'][0]:.3f} and >= 4,096 "
        f"{h['new tokens'][1]:.3f}. The 95% interval is from 200 bootstrap refits over cells. BRIDGE: the BF16 path's "
        f"{b16 * 1e6:.2f} ps a slot here against the family's fitted FlashInfer SM90 cost, 337.5 ps a slot at GLM-5.3's "
        f"record (scripts/fit_glm53_attention.py), {b16 * 1e6 / 337.5:.3f}x. Named substitutions: GLM-5.2's rows for "
        f"GLM-5.3 and GLM-5.3-Flash (Flash's BF16 record is 1024 B, so its ratio is moved further), vLLM 0.30.0's "
        f"kernels for the engine's pin, FLASH_ATTN_MLA_SPARSE's BF16 path as the reference for the family's FlashInfer "
        f"fit. Reproduce with scripts/fit_glm53_fp8_prefill_scale.py.")
    rg = f["range"]
    return [f"  - {NAME}:", f"      value: {f['scale']}", "      units: dimensionless",
            "      method: transferred", "      fitted: false",
            f"      scope: {{hardware: [h200], model: [{', '.join(MODELS)}]}}",
            '      transferred_from: "hardware=h200"',
            ('      transfer_rule: "measured on vLLM 0.30.0\'s FlashMLA sparse (fp8) and FlashAttention sparse (BF16) '
             'prefill paths on this chip at GLM-5.2\'s attention and used for the GLM-5.3 family at the engine\'s pin; '
             'no kernel hash shows the builds run the same kernels, and nothing is moved between chips"'),
            f"      ci95: [{f['ci'][0]}, {f['ci'][1]}]", "      fit_range:",
            *[f"        - {{dim: {d}, low: {lo}, high: {hi}}}" for d, (lo, hi) in rg.items()],
            "      sources:", f'        - {{kind: model, cite: "{cite}", role: primary}}',
            ('        - {kind: model, cite: "vLLM v0.32.0 v1/attention/backends/mla/flashmla_sparse.py '
             'FlashMLASparseImpl fp8_use_mixed_batch", role: supporting}'),
            "      rationale: >", "        " + rationale]


def render(text: str, e: list[str]) -> str:
    blocks = cb.split(text)
    i = cb.find(blocks, cb.name_of(e), cb.scope_of(e))
    if i is not None:
        blocks[i] = e
    else:
        at = next((j for j, b in enumerate(blocks) if b[0].startswith(MARKER)), None)
        if at is None:
            raise SystemExit(f"no {MARKER.strip()!r} line in {ATTN_PATH.name}")
        blocks.insert(at, e)
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AISIM_GLM53_DATA", ""),
                    help="aisim systems/data at AISIM_GLM53_SHA (default $AISIM_GLM53_DATA)")
    ap.add_argument("--check", action="store_true", help="exit 1 if the set would change; write nothing")
    args = ap.parse_args(argv[1:])
    if not args.data:
        raise SystemExit("set AISIM_GLM53_DATA (or --data) to aisim's systems/data directory")
    f = run(Path(args.data))
    print(f"scale {f['scale']} ci {f['ci']} fits {f['fit']} err {f['err']} held {f['held']}")
    a0 = ATTN_PATH.read_text(encoding="utf-8")
    a1 = render(a0, entry(f))
    if args.check:
        if a0 != a1:
            print(f"would change: {ATTN_PATH.name}", file=sys.stderr)
            return 1
        print("the set already matches")
        return 0
    ATTN_PATH.write_text(a1, encoding="utf-8")
    print(f"{ATTN_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
