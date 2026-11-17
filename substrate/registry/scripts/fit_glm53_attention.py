#!/usr/bin/env python3
"""GLM-5.3 and GLM-5.3-Flash on H200: the sparse-MLA PREFILL form, fitted on AISimulate's
GLM-5.3 attention-module table, net of what BLIS charges for the rest of that module.

    python scripts/fit_glm53_attention.py            # print the fit, rewrite the entries
    python scripts/fit_glm53_attention.py --check    # exit 1 if the set would change

Owns, scoped {hardware: [h200], model: [glm-5.3, glm-5.3-flash]} (one family for the two models,
which run the same kernel; not single-model entries, VAL-7), in cost-model-attention.yaml:

    attention_prefill_floor_sparse_mla        transferred  (a fit; see TRANSFERRED below)
    attention_prefill_rate_sparse_mla         transferred  (rate_basis: work)
    attention_prefill_per_request_sparse_mla  not_charged  (fitted free, no gain; see TERMS)
    attention_prefill_per_token_sparse_mla    not_charged

THE DATA. $AISIM_GLM53_DATA is the systems/data directory of ai-dynamo/aisim at AISIM_GLM53_SHA
(.github/workflows/rederive.yml), which holds h200_sxm/glm53_attention/vllm/0.31.0/
glm53_attention_module_perf.parquet; the writer refuses any other file (TABLE_SHA256). The table
times vLLM 0.31.0's whole GLM-5.3 attention module on one rank (Glm5NextMLAAttention: the
projections, the sparse indexer and its block scorer, FlashInferMLASparseSM90Impl, o_proj), with
GLM-5.3-Flash's geometry: at TP8, 8 query heads, kv_lora_rank 512, no rope dimension, an index
top-k of 2048, a BF16 KV cache, CUDA-graph replay over a real KV cache.

NET OF THE REST OF THE MODULE. The form prices one kernel, and the table times the module, so
the fit is of the measured module against (what BLIS charges for everything in the module except
the sparse-MLA prefill attention) + the form. The table is GPU kernel time only, with the output
all-reduce excluded (its collection_meta.yaml), so the rest is the device share (SM + HBM) of
BLIS's price of the module cut before that all-reduce. That first part is the generated file
registry/docs/glm53-attention-rest-h200-tp8.csv, written by
kernel/glm53_attention_rest_test.go (BLIS_WRITE_GLM53_REST=1), which also fails, in the rederive
job, when the kernel's price of the rest moves without the file being regenerated. About a
third of the measured module is the rest, so it cannot be dropped.

THE FORM is the kernel's (kernel/prefill_forms.go, sparsePrefillReadBytes): per layer, per rank,
per step holding prefill,
    t = floor + per_request x requests + per_token x query tokens + W / rate,
    W = sum over query positions p of 64 x ceil(min(p, index_topk) / 64) x record x heads / 64.
Only rows with prefix + new tokens > index_topk are fitted: there every query reads its top-k,
the regime a DSA layer's long prefills run in. Below the top-k the kernel takes a different
path (the MHA prefill route of FlashInferMLASparseSM90Impl) and the form over-prices it; the
error there is reported, not fitted.

TERMS. per_request and per_token are fitted free beside the floor and rate; a term is kept
only when it lowers the mean absolute log error by at least MIN_GAIN. Neither does on this
table, so both are written as not_charged, value 0, which keeps the kernel's four-member family
complete.

TRANSFERRED. The data ran vLLM 0.31.0's FlashInfer SM90 sparse-MLA kernel; no kernel hash shows
that build is the one the engine's pin runs, because the FLASHINFER_MLA_SPARSE_SM90 backend is a
gap in blis-vllm's kernel_impls.json (inference-sim/blis-substrate#149). So the fitted entries are
`transferred` (D5), with `fitted: false` as the schema requires, until that table lists the
kernel and the entries can be keyed to it. Two more named substitutions:
  - GLM-5.3-Flash's geometry stands for GLM-5.3's. At TP8 both run 8 heads a rank over a
    512-wide latent with a 2048 top-k; GLM-5.3's record is 1152 B (64 rope dims) against
    Flash's 1024 B. The rate (bytes of W a microsecond) is taken as the same, so the kernel,
    which counts each model's own record in W, charges GLM-5.3 1152/1024 of Flash's time per
    key slot: the extra 64 rope dims are read and multiplied like the latent's.
  - The table's cache is BF16. The kernel's W scales with the record, so an fp8 cache (656 B
    for GLM-5.3) is charged 656/1152 of the BF16 cost. That is an assumption, not a
    measurement.

UNITS. `latency` is milliseconds. The fit minimises the mean absolute log ratio, by a
deterministic coordinate search, so --check reproduces byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
REG = HERE.parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

ATTN_PATH = REG / "coefficients" / "cost-model-attention.yaml"
REST_PATH = REG / "docs" / "glm53-attention-rest-h200-tp8.csv"
CATALOG = Path(os.environ.get("BLIS_CATALOG", REG.parent / "catalog"))
TABLE = Path("h200_sxm/glm53_attention/vllm/0.31.0/glm53_attention_module_perf.parquet")
TABLE_SHA256 = "14821edd06a88d9ee67c88b810c1031d460abb54b89ef1f2ab3d4686a0f6d7ef"
AISIM_SHA = "6fdf6fb6b2e1c15b8f5cd1ba35fd2d50cdb2eb18"
LANE = "FlashInferMLASparseSM90Impl"
TP = 8
BLOCK = 64
MODELS = ("glm-5.3", "glm-5.3-flash")
MIN_GAIN = 0.005
MARKER = "  # BEGIN trtllm-gen prefill lane"


def geo(pred, y) -> tuple[float, float]:
    lg = np.log(np.asarray(pred, float) / np.asarray(y, float))
    return float(math.exp(np.mean(np.abs(lg)))), float(math.exp(np.mean(lg)))


def model_geometry(name: str) -> dict:
    """The DSA layer's per-rank geometry at TP8, from the catalog graph the kernel prices."""
    g = yaml.safe_load((CATALOG / "models" / name / "graph.yaml").read_text(encoding="utf-8"))
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"]
             if n.get("op") == "Attention" and n.get("kind") == "sparse_mla" and n.get("holds_engine_kv")]
    shapes = {(n["n_q"], n["kv_lora_rank"], n.get("qk_rope_head_dim", 0), n["index_topk"]) for n in nodes}
    if len(shapes) != 1:
        raise SystemExit(f"{name}: expected one sparse-MLA geometry, found {sorted(shapes)}")
    nq, lora, rope, topk = shapes.pop()
    return {"heads": nq // TP, "record": 2 * (lora + rope), "topk": topk, "lora": lora}


def table_rows(data: Path, flash: dict) -> dict:
    path = data / TABLE
    if not path.is_file():
        raise SystemExit(f"{path} is absent: set AISIM_GLM53_DATA to aisim {AISIM_SHA[:8]}'s systems/data")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TABLE_SHA256:
        raise SystemExit(f"{path}: sha256 {digest}, not the pinned {TABLE_SHA256}")
    rest = {}
    with REST_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rest[(int(r["batch"]), int(r["new_tokens"]), int(r["prefix"]))] = float(r["rest_us"])
    cols = {k: [] for k in ("b", "x", "p", "slots", "rest", "y")}
    for r in pq.read_table(path).to_pylist():
        g = json.loads(r["geometry"])
        if g["tp_size"] != TP or not g["is_context"]:
            continue
        if (LANE not in r["kernel_source"] or g["kv_cache_dtype"] != "bfloat16"
                or g["num_heads"] != flash["heads"] or g["kv_lora_rank"] != flash["lora"]
                or g["index_topk"] != flash["topk"] or str(r["used_cuda_graph"]) != "True"):
            raise SystemExit(f"unexpected row: {r['kernel_source']}, {g}")
        b, x, p = int(r["batch_size"]), int(r["x"]), int(r["prefix"])
        if (b, x, p) not in rest:
            raise SystemExit(f"{REST_PATH.name} has no row {b},{x},{p}: regenerate it")
        q = np.arange(p + 1, p + x + 1)
        cols["b"].append(b)
        cols["x"].append(x)
        cols["p"].append(p)
        cols["slots"].append(b * float((BLOCK * np.ceil(np.minimum(q, flash["topk"]) / BLOCK)).sum()))
        cols["rest"].append(rest[(b, x, p)])
        cols["y"].append(float(r["latency"]) * 1000.0)
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    if len(out["y"]) != 356:
        raise SystemExit(f"{len(out['y'])} TP8 prefill rows, expected 356")
    return out


def sub(r: dict, sel) -> dict:
    return {k: v[sel] for k, v in r.items()}


def design(r: dict) -> np.ndarray:
    """Columns: floor, per request, per query token, per (query, key slot)."""
    return np.stack([np.ones_like(r["y"]), r["b"], r["b"] * r["x"], r["slots"]], 1)


def fit(r: dict, terms: tuple[int, ...]) -> np.ndarray:
    """Non-negative theta over the given columns minimising mean |log((rest + X theta) / y)|:
    relative least squares to start, then a deterministic coordinate search in log space."""
    X, y, rest = design(r)[:, terms], r["y"], r["rest"]
    t0, *_ = np.linalg.lstsq(X / y[:, None], 1 - rest / y, rcond=None)
    scale = np.median(np.maximum(y - rest, 1e-3)) / np.maximum(np.median(X, axis=0), 1e-30)
    lt = np.log(np.where(t0 > 0, t0, 1e-2 * scale))

    def obj(v):
        return float(np.mean(np.abs(np.log((rest + X @ np.exp(v)) / y))))

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
    th = np.zeros(4)
    th[list(terms)] = np.exp(lt)
    return th


def pred(r: dict, th: np.ndarray) -> np.ndarray:
    return r["rest"] + design(r) @ th


def run(data: Path) -> dict:
    geo_by = {m: model_geometry(m) for m in MODELS}
    flash = geo_by["glm-5.3-flash"]
    if geo_by["glm-5.3"]["heads"] != flash["heads"] or geo_by["glm-5.3"]["topk"] != flash["topk"]:
        raise SystemExit("GLM-5.3's per-rank sparse geometry at TP8 is not GLM-5.3-Flash's")
    allr = table_rows(data, flash)
    above = allr["p"] + allr["x"] > flash["topk"]
    r = sub(allr, above)
    base = fit(r, (0, 3))
    err0 = np.mean(np.abs(np.log(pred(r, base) / r["y"])))
    gains = {}
    for name, terms in (("per_request", (0, 1, 3)), ("per_token", (0, 2, 3))):
        th = fit(r, terms)
        gains[name] = float(err0 - np.mean(np.abs(np.log(pred(r, th) / r["y"]))))
    if max(gains.values()) >= MIN_GAIN:
        raise SystemExit(f"a per-request or per-token term now earns its place ({gains}); "
                         "the writer would have to write it")
    floor = round(float(base[0]), 2)
    per_slot = float(base[3])
    # One rate for both models, fitted at Flash's record (the table's geometry).
    rate = int(round(flash["record"] * flash["heads"] / 64 / per_slot))
    rates = {m: rate for m in MODELS}
    # The values as written, back in the fit's terms.
    thr = np.array([floor, 0.0, 0.0, flash["record"] * flash["heads"] / 64 / rate])
    out = {"n": len(r["y"]), "floor": floor, "per_slot": per_slot, "rates": rates, "geo": geo_by,
           "gains": gains, "fit": geo(pred(r, thr), r["y"])}
    below = sub(allr, ~above)
    out["below"] = (len(below["y"]), *geo(pred(below, thr), below["y"]))
    holds = {}
    for name, tr, te in (("prefix", r["p"] <= 16384, r["p"] >= 32768),
                         ("new tokens", r["x"] <= 2048, r["x"] >= 4096)):
        th = fit(sub(r, tr), (0, 3))
        holds[name] = (int(te.sum()), *geo(pred(sub(r, te), th), r["y"][te]))
    out["holds"] = holds
    out["share"] = float(np.median((r["y"] - r["rest"]) / r["y"]))
    one = (allr["b"] == 1) & (allr["p"] == 0)
    small = one & (allr["x"] == allr["x"][one].min())
    if small.sum() != 1:
        raise SystemExit("no single smallest prefill row (batch 1, no prefix)")
    out["small"] = (int(allr["x"][small][0]), float(allr["y"][small][0]), float(allr["rest"][small][0]))
    out["by_batch"] = {int(b): geo(pred(sub(r, r["b"] == b), thr), r["y"][r["b"] == b])[1]
                       for b in sorted(set(r["b"]))}
    # 95% interval: 200 bootstrap refits over rows, fixed seed.
    rng = np.random.RandomState(20261116)
    fl, sl = [], []
    for _ in range(200):
        i = rng.randint(0, len(r["y"]), len(r["y"]))
        th = fit(sub(r, i), (0, 3))
        fl.append(th[0])
        sl.append(th[3])
    out["ci_floor"] = (round(float(np.percentile(fl, 2.5)), 2), round(float(np.percentile(fl, 97.5)), 2))
    out["ci_slot"] = (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)))
    out["range"] = {"tokens": (float((r["b"] * r["x"]).min()), float((r["b"] * r["x"]).max())),
                    "requests": (float(r["b"].min()), float(r["b"].max())),
                    "context_tokens": (float(r["p"].min()), float(r["p"].max()))}
    return out


def entries(f: dict) -> list[list[str]]:
    src_table = (f"NVIDIA AISimulate (ai-dynamo/aisim {AISIM_SHA[:8]}) systems/data/{TABLE} (NVIDIA H200, "
                 f"sha256 {TABLE_SHA256[:12]}), vLLM 0.31.0 Glm5NextMLAAttention with {LANE}, "
                 f"GLM-5.3-Flash geometry, {f['n']} TP8 prefill rows with prefix + new tokens above "
                 f"the top-k")
    src_rest = ("registry/docs/glm53-attention-rest-h200-tp8.csv (generated by "
                "kernel/glm53_attention_rest_test.go): BLIS's price of the rest of the module, subtracted")
    src_vllm = ("vLLM v0.31.0 v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py "
                "FlashInferMLASparseSM90Impl")
    rule = ("measured on vLLM 0.31.0's FlashInfer SM90 sparse-MLA prefill kernel on this chip and used for "
            "the engine's pin; no kernel hash shows the two builds run the same kernel (the "
            "FLASHINFER_MLA_SPARSE_SM90 backend is a gap in kernel_impls.json, "
            "inference-sim/blis-substrate#149), and nothing is moved between chips")
    rg = f["range"]
    fit_range = [f"        - {{dim: {d}, low: {lo}, high: {hi}}}" for d, (lo, hi) in rg.items()]
    h = f["holds"]
    errors = (
        f"Fitted on the {f['n']} TP8 rows whose prefix + new tokens exceed the top-k (2048), batch "
        f"{int(rg['requests'][0])}-{int(rg['requests'][1])}, prefix {int(rg['context_tokens'][0])}-"
        f"{int(rg['context_tokens'][1])}: the module is predicted at {f['fit'][0]:.3f}x, signed "
        f"{f['fit'][1]:.3f}. Signed by batch: "
        + ", ".join(f"{b}: {v:.2f}" for b, v in f["by_batch"].items())
        + f". HELD OUT: trained on prefix <= 16,384 and predicting >= 32,768, {h['prefix'][1]:.3f}x "
        f"(signed {h['prefix'][2]:.3f}, n={h['prefix'][0]}), the long-prefix under-price that the "
        f"scorer's pricing leaves (the scorer is in the rest of the module, priced on the part-wide "
        f"prefill pair); trained on new tokens <= 2,048 and predicting >= 4,096, "
        f"{h['new tokens'][1]:.3f}x (signed {h['new tokens'][2]:.3f}, n={h['new tokens'][0]}). AT OR "
        f"BELOW THE TOP-K ({f['below'][0]} rows, not fitted) the form gives {f['below'][1]:.3f}x, "
        f"signed {f['below'][2]:.3f}. The sparse attention is a median {f['share']:.0%} of the "
        f"measured module; the rest is BLIS's own price, so this fit inherits its error. THE FLOOR "
        f"IS NOT THE ATTENTION KERNEL'S ALONE: at the table's smallest prefill ({f['small'][0]} tokens, "
        f"one request, no prefix) the module measures {f['small'][1]:.1f} us of GPU time and BLIS's "
        f"device price of the rest of it is {f['small'][2]:.1f} us, so the floor also carries the "
        f"module's fixed device cost that the rest's forms leave out (the indexer's and scorer's small "
        f"kernels among them).")
    subst = (
        "Named substitutions: GLM-5.3-Flash's geometry for GLM-5.3 (8 heads a rank at TP8, a "
        "512-wide latent, top-k 2048; GLM-5.3's record is 1152 B against Flash's 1024 B, and the "
        "rate over W is taken as the same, so GLM-5.3 pays 1152/1024 of Flash's time a key slot); "
        "a BF16 cache, so an fp8 cache, whose "
        "record the kernel's W scales with, is charged its record's share of the BF16 cost, an "
        "assumption. TRANSFERRED, not measured, because blis-substrate#149 lists no hash for the "
        "kernel; fitted: false is the schema's, the number is a fit. The 95% interval is from 200 "
        "bootstrap refits over rows; rows of one shape family are not independent, so it is narrower "
        "than the data justify.")
    out = []
    for m in MODELS[1:]:
        g = f["geo"][m]
        g53 = f["geo"]["glm-5.3"]
        scope = f"{{hardware: [h200], model: [{', '.join(MODELS)}]}}"
        rate = f["rates"][m]
        c_lo, c_hi = f["ci_slot"]
        k = g["record"] * g["heads"] / 64
        ci_rate = (int(round(k / c_hi)), int(round(k / c_lo)))
        common = ["      transferred_from: \"hardware=h200\"", f"      transfer_rule: \"{rule}\""]

        def head(name, value, units, method, extra=()):
            ls = [f"  - {name}:", f"      value: {value}", f"      units: {units}",
                  f"      method: {method}", "      fitted: false"]
            return ls + list(extra) + [f"      scope: {scope}"]

        srcs = ["      sources:",
                f'        - {{kind: model, cite: "{src_table}", role: primary}}',
                f'        - {{kind: model, cite: "{src_rest}", role: supporting}}',
                f'        - {{kind: model, cite: "{src_vllm}", role: supporting}}']
        out.append(head("attention_prefill_floor_sparse_mla", f["floor"], "us_per_transfer", "transferred")
                   + common + [f"      ci95: [{f['ci_floor'][0]}, {f['ci_floor'][1]}]", "      fit_range:"]
                   + fit_range + srcs + ["      rationale: >", "        " + (
                       f"The per-call floor of the GLM-5.3 family's sparse-MLA PREFILL on H200, {f['floor']} us a layer a "
                       f"rank in a step with prefill. THE FORM: t = attention_prefill_floor_sparse_mla + "
                       f"attention_prefill_per_request_sparse_mla x prefill requests + "
                       f"attention_prefill_per_token_sparse_mla x prefill query tokens + W / "
                       f"attention_prefill_rate_sparse_mla, W = sum over prefill query positions p of 64 x "
                       f"ceil(min(p, {g['topk']}) / 64) x record x {g['heads']} heads / 64, the record "
                       f"{g53['record']} B for glm-5.3 and {g['record']} B for glm-5.3-flash at a BF16 cache. Before "
                       f"it, this layer's prefill was priced on the part-wide prefill pair over its FLOPs. "
                       + errors + " " + subst)])
        out.append(head("attention_prefill_rate_sparse_mla", float(rate), "bytes_per_us", "transferred",
                        ["      rate_basis: work"])
                   + common + [f"      ci95: [{float(ci_rate[0])}, {float(ci_rate[1])}]", "      fit_range:"]
                   + fit_range + srcs + ["      rationale: >", "        " + (
                       f"{rate:,} B/us over W, a WORK rate: W counts each query's read of its top-k "
                       f"records separately, and the queries of one prefill read overlapping keys, which "
                       f"the kernel serves from shared memory and L2 rather than HBM, so W is not HBM "
                       f"traffic and must not be rescaled by the part's bandwidth. "
                       f"{f['per_slot'] * 1e6:.1f} ps per (query, 64-token key slot) at glm-5.3-flash's "
                       f"{g['record']} B record x {g['heads']} heads / 64, where it was fitted; glm-5.3's "
                       f"{g53['record']} B record pays {g53['record'] / g['record']:.3f}x that. "
                       f"Every query past the top-k reads {g['topk']} keys, so a {g['topk']}-key read costs "
                       f"{g['topk'] * f['per_slot']:.3f} us a query. See attention_prefill_floor_sparse_mla "
                       f"(this scope) for the form, the errors and the substitutions.")])
        for name, units, gain in (("attention_prefill_per_request_sparse_mla", "us_per_request",
                                   f["gains"]["per_request"]),
                                  ("attention_prefill_per_token_sparse_mla", "us_per_token",
                                   f["gains"]["per_token"])):
            out.append(head(name, 0, units, "not_charged") + ["      rationale: >", "        " + (
                f"Not charged. Fitted free beside the floor and rate on the same {f['n']} rows, it lowers "
                f"the mean absolute log error by {gain:.4f}, under the {MIN_GAIN} a term must earn "
                f"(scripts/fit_glm53_attention.py), so the form keeps two terms; this entry completes "
                f"the four-member family the kernel reads.")])
    return out


def render(text: str, new: list[list[str]]) -> str:
    blocks = cb.split(text)
    for e in new:
        i = cb.find(blocks, cb.name_of(e), cb.scope_of(e))
        if i is not None:
            blocks[i] = e
            continue
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
    print(f"floor {f['floor']} us, per slot {f['per_slot']:.4e} us, rates {f['rates']}, fit "
          f"{f['fit'][0]:.3f}x signed {f['fit'][1]:.3f}, gains {f['gains']}, held out {f['holds']}, "
          f"below top-k {f['below']}, ci floor {f['ci_floor']} slot {f['ci_slot']}")
    a0 = ATTN_PATH.read_text(encoding="utf-8")
    a1 = render(a0, entries(f))
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
