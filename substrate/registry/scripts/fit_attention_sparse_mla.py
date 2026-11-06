#!/usr/bin/env python3
"""REJECTED. Fits a sparse-MLA decode rate that covers only half of deepseek-v4-pro.

Kept for the negative result, as `fit_attention_mla.py` and `probe_moe_roofline.py` are:
the fit is clean, physical and wrong for this cost model, and deleting it would leave the
next person to re-derive that.

THE FINDING. deepseek-v4-pro ALTERNATES two sparse geometries -- 30 csa128_moe layers
(window 128, compress_ratio 128) and 30 csa4_moe layers (index_topk 1024, compress_ratio
4) -- and their selected-token counts differ by 32x at a 1M context (8,319 against
262,912). The only attention-only table, dsv4_hca_attn_module_perf, states
compress_ratio 128 on every row, so this fit describes csa128_moe ALONE. Charging it to
all 60 layers applies a rate fitted on the small read to the large one.

The end-to-end check on the InferenceX corpus (573 points) refused it:

    deepseek-v4-pro TPOT    overall TPOT
    10.42%                  14.98%     neither (full context, part-wide rate)
     8.72%                  15.04%     sparse byte count only          <- shipped
    14.36%                  14.95%     sparse byte count + this rate   <- rejected

The byte count is the physics and earns 1.7 points on the affected model. This rate costs
5.6.

AND THIS DATASET CANNOT SUPPLY THE MISSING HALF. compress_ratio 4 appears only in the
dsv4_csa_* tables, and every one of those is a MODULE measurement (floors 61.4-80.6us
against a 13.5us attention floor), so fitting a csa4_moe rate from them would re-import
the projection double-charge that scripts/correct_mla_floor.py had to undo. A rate
covering half a model's layers is not a coefficient; it is a bias with a provenance
string.

WHAT SHIPPED INSTEAD. The fix is in blis-latency-kernel and needs no fitted constant:
`selectedKVTokens` bounds a sparse layer's decode read per LAYER and per REQUEST from the
catalog's own index_topk, window and compress_ratio. Sparse layers keep the part-wide
measured floor and rate, now charged against the right number of bytes. See
docs/methodology.md section 10.

TO MAKE A PER-KIND RATE FITTABLE: an attention-only table carrying compress_ratio 4 --
the csa4_moe geometry -- so both halves of the model are measured by the same kind of
measurement. The fitter below already handles the geometry and needs only the rows.

Original docstring follows.

Fit the sparse-MLA decode RATE from the one AISimulate table that measures the kernel.

    python scripts/fit_attention_sparse_mla.py --sku h200_sxm --chip h200
    python scripts/fit_attention_sparse_mla.py --all

WHY THIS WAS RECORDED AS IMPOSSIBLE. The kernel's new.go says `sparse_mla` is
"deliberately NOT mapped ... its byte count is a different function of the request and the
registry carries no fit for it", so a sparse_mla layer falls back to the full-attention
pair. That fallback is wrong by construction: all 61 layers of deepseek-v4-pro are priced
as if each read the whole KV cache.

`fit_attention_mla.py` recorded why the obvious tables cannot fix it, and
`correct_mla_floor.py` records what happens when that warning is ignored: AISimulate's MLA
and sparse tables are MODULE measurements covering the whole block, the catalog prices the
projections as separate GEMM nodes, and a floor fitted from a module table double-charges
them. Shipped once as `attention_decode_floor_mla`, it cost 2.45 points of end-to-end TPOT.

THE TABLE THAT IS NOT A MODULE MEASUREMENT. Of the thirteen tables in the
`sparse_attention` family, twelve floor between 37.7us and 1186us on h200 -- at or above
the 66.2us it costs to read the csa4_moe layer's 317.5 MB of projections at 4.80 TB/s, and
far above the 13.5us committed GQA attention floor. One does not:

    dsv4_hca_attn_module_perf.parquet    FLASHMLA_SPARSE_DSV4    9.6 us minimum

9.6us is BELOW this part's GQA attention floor and 7x below the projection read, so this
table measures the attention kernel rather than the block. 35,262 rows across all six
NVIDIA parts, and -- uniquely in this family -- a `compress_ratio` column, which is the
sparse geometry itself.

THE REGIME SPLIT. `isl > 1` rows are the CONTEXT path and `isl == 1` the DECODE path, which
the kernel prices with separate families. The split is not read off the column name: at
step 8192 on h200, batch 1 / isl 8192 costs 1368.9us and batch 2 / isl 4096 costs 1355.6us,
within 1% -- so cost is governed by batch*isl, the query-token count, which is what a
prefill is. This fitter takes `isl == 1` only. The decode grid holds batch*step ~ 1M, so
total KV tokens is swept across batch 1 to 1024 in 207 points.

THE BYTE COUNT, CHOSEN BY SEARCH RATHER THAN ASSUMED. Candidate forms were grid-searched
against (floor, rate) on three parts, and the discriminator is whether the optimum is
INTERIOR: a form that under-counts bytes forces the rate to the grid ceiling.

    form                                h200 err   rate as fraction of peak
    full context (today's fallback)      1.698     1.00  <- CLAMPED, every part
    topk=128 only                        1.309     0.07
    topk=128 + (step-128)/128            1.301     0.08  <- committed
    topk=1024 + (step-1024)/4            1.483     1.00  <- CLAMPED

The winner is exactly the geometry blis-catalog declares for this checkpoint's
`csa128_moe` layer (`window: 128`, `compress_ratio: 128`), resolved from the catalog rather
than tuned. And the full-context form clamping at 1.0 of datasheet peak on all three parts
is the quantitative statement that today's fallback cannot describe this kernel at all.

    selected = min(step, 128) + max(0, step - 128) / 128
    kv_bytes = batch * selected * 576          # 576 = d_h 512 + qk_rope 64, at fp8

ONLY THE RATE IS COMMITTED, AND THIS IS THE MLA LESSON APPLIED. Searching floor and rate
freely puts the floor at 12.6-14.4us on every part -- a band so tight it does not track the
committed part-wide floors at all (those span 9.5-14.5us and order differently: b300 is the
lowest at 9.5 while its free sparse floor is 12.6). PINNING the floor to each part's own
measured attention floor costs between 1.0% and 7.8% of error:

    chip          free floor  err      pinned  err      cost
    h100              13.6   1.317      14.5   1.337    1.016x
    h200              12.8   1.301      13.5   1.313    1.010x
    b200              13.6   1.259      10.5   1.314    1.044x
    b300              12.6   1.259       9.5   1.357    1.078x
    gb200-nvl72       14.4   1.242      11.0   1.306    1.052x
    gb300             14.2   1.243      11.0   1.300    1.045x

So the data does not demand a separate floor, and inventing one would add a parameter for
under 8% of fit while re-opening exactly the failure mode that cost 2.45 TPOT points on
MLA. The floor stays the part's measured attention floor; what this fitter commits is the
RATE, which the data does identify -- 0.08 to 0.12 of datasheet peak, interior on every
part, and an order of magnitude below the full-attention rates (0.52-0.88) because a sparse
read gathers scattered pages rather than streaming contiguous ones.

LANE. One lane (FLASHMLA_SPARSE_DSV4), one dtype pair (kv fp8, gemm fp8_block), so nothing
to pool and nothing to choose. Recorded because the MLA fit's 5x lane spread makes the
absence worth stating.

TWO CHECKPOINTS SHARE THE TABLE AND MUST NOT BE POOLED. vLLM 0.25.0 adds
DeepSeek-V4-Pro-FP8 beside DeepSeek-V4-Flash-FP8 and they differ in the axis the byte count
scales with -- Pro measures `num_heads: 128`, Flash `num_heads: 64` -- with zero decode
shapes in common, so pooling would concatenate two disjoint grids rather than average one
kernel. Only Pro is fitted: it is the checkpoint blis-catalog models
(`models/deepseek-v4-pro`, sparse_mla node `n_q: 128`, matching the measured head count)
and the one the InferenceX corpus scores.

UNITS. The `latency` column is MILLISECONDS, as everywhere in this dataset.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))

TABLE = "dsv4_hca_attn_module_perf.parquet"

# The collection this family is pinned to, per the same rule the other fitters follow: one
# framework and one version, so a difference between parts is the silicon rather than a
# software change. 0.25.0 rather than 0.24.0 because only 0.25.0 carries the Pro
# checkpoint this registry prices.
COLLECTION = "vllm/0.25.0"

# The checkpoint to fit. Named rather than "whatever is present", so a collection that
# drops it fails loudly instead of silently fitting the Flash variant's different geometry.
FIT_MODEL = "sgl-project/DeepSeek-V4-Pro-FP8"
MODEL_TO_CATALOG = {FIT_MODEL: "deepseek-v4-pro"}

PARTS = [
    ("h100_sxm", "h100"),
    ("h200_sxm", "h200"),
    ("b200_sxm", "b200"),
    ("b300_sxm", "b300"),
    ("gb200", "gb200-nvl72"),
    ("gb300", "gb300"),
]

# Fraction-of-peak grid for the rate. Fine near the bottom because the optimum is at
# 0.08-0.12: a coarse 0.01 grid would quantise the answer to within 12% of itself.
RATE_FRACTIONS = [x / 1000 for x in range(5, 1001)]      # 0.005 .. 1.000

# A fit over fewer rows than this is not worth stating.
MIN_POINTS = 150

DTYPE_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "bfloat16": 2.0, "float16": 2.0, "nvfp4": 0.5}


def sparse_geometry(catalog: Path, model_id: str) -> list[dict]:
    """Every sparse_mla node's geometry for a checkpoint, from the catalog.

    Resolved rather than assumed: the selected-token count is a stated property of the
    architecture, and taking it from anywhere else would defeat the point of the fit.
    """
    short = MODEL_TO_CATALOG.get(model_id)
    if short is None:
        raise SystemExit(
            f"no catalog mapping for {model_id!r}; add it to MODEL_TO_CATALOG rather "
            f"than guess the geometry")
    graph = yaml.safe_load((catalog / "models" / short / "graph.yaml").read_text())
    out = []
    for lk in graph["layer_kinds"]:
        for n in lk.get("nodes") or []:
            if n.get("op") == "Attention" and n.get("kind") == "sparse_mla":
                out.append({
                    "layer_kind": lk["id"],
                    "d_h": int(n["d_h"]),
                    "qk_rope_head_dim": int(n.get("qk_rope_head_dim", 0)),
                    "index_topk": n.get("index_topk"),
                    "window": n.get("window"),
                    "compress_ratio": n.get("compress_ratio"),
                })
    if not out:
        raise SystemExit(f"{short}: no sparse_mla Attention node in the catalog graph")
    return out


def selected_tokens(step: int, topk: int, compress_ratio: int | None) -> float:
    """Tokens the kernel reads for one query at this context length.

    Two tiers, which is what the measured curve shows: a SELECTED tier of `topk` tokens at
    full resolution, flat in context, and -- where the architecture compresses the
    remainder rather than discarding it -- a COMPRESSED tier of (step-topk)/ratio. Both
    bounded by the context, since a 16-token context cannot yield 128 selected tokens.
    """
    sel = float(min(step, topk))
    if compress_ratio and step > topk:
        sel += (step - topk) / compress_ratio
    return sel


def part_wide_floor(chip: str) -> float:
    """The committed attention_decode_floor for this chip, read from the registry.

    Read rather than tabulated here, so this fitter cannot drift from the value the kernel
    will actually pair with the rate it produces.
    """
    doc = yaml.safe_load(SET_PATH.read_text())
    for entry in doc["coefficients"]:
        (name, body), = entry.items()
        if name != "attention_decode_floor":
            continue
        if chip in ((body.get("scope") or {}).get("hardware") or []):
            return float(body["value"])
    raise SystemExit(
        f"{chip}: no committed attention_decode_floor scoped to it, so there is no "
        f"measured floor to pair this rate with")


def load(data: str, sku: str) -> list[dict]:
    """Decode rows (isl == 1) for the fitted checkpoint on one part."""
    out = []
    for p in sorted(glob.glob(
            os.path.join(data, sku, "sparse_attention", "*", "*", TABLE))):
        if COLLECTION not in p.replace(os.sep, "/"):
            continue
        t = pq.read_table(p).to_pydict()
        for i in range(len(t["latency"])):
            if (t["latency"][i] <= 0 or t["isl"][i] != 1
                    or t["model"][i] != FIT_MODEL):
                continue
            out.append({
                "lane": t["kernel_source"][i],
                "kv": t["kv_cache_dtype"][i],
                "heads": t["num_heads"][i],
                "batch": t["batch_size"][i],
                "step": t["step"][i],
                "compress_ratio": t["compress_ratio"][i],
                "us": t["latency"][i] * 1000.0,
            })
    return out


def log_error(points, floor: float, rate: float) -> float:
    """Geometric mean of the per-point ratio error, as the committed fits report it."""
    total = 0.0
    for kv_bytes, measured in points:
        pred = floor + kv_bytes / rate
        if pred <= 0 or measured <= 0:
            return math.inf
        total += abs(math.log(pred / measured))
    return math.exp(total / len(points))


def fit_part(data: str, catalog: Path, sku: str, chip: str, verbose: bool = True):
    rows = load(data, sku)
    if not rows:
        if verbose:
            print(f"{chip:14} no {FIT_MODEL} decode rows in {COLLECTION}",
                  file=sys.stderr)
        return None

    lanes = sorted({r["lane"] for r in rows})
    if len(lanes) != 1:
        raise SystemExit(f"{chip}: {len(lanes)} lanes, which must not be pooled: {lanes}")
    ratios = sorted({r["compress_ratio"] for r in rows})
    if len(ratios) != 1:
        raise SystemExit(f"{chip}: {len(ratios)} compress ratios in one fit: {ratios}")
    ratio = ratios[0]

    # Pair the measurement with the catalog layer declaring the SAME compression, so the
    # selected-token count is the architecture's rather than a tuned choice.
    geom = sparse_geometry(catalog, FIT_MODEL)
    match = [g for g in geom if g["compress_ratio"] == ratio]
    if not match:
        raise SystemExit(
            f"{chip}: table measures compress_ratio {ratio}, catalog declares "
            f"{[g['compress_ratio'] for g in geom]}; refusing to pair a measurement with "
            f"a geometry it does not describe")
    g = match[0]
    # index_topk where the architecture states one; otherwise the window is the selected
    # count, which is what a windowed sparse read means.
    topk = g["index_topk"] or g["window"]
    if not topk:
        raise SystemExit(f"{chip}: {g['layer_kind']} states neither index_topk nor window")
    width = g["d_h"] + g["qk_rope_head_dim"]

    pts = []
    for r in rows:
        eb = DTYPE_BYTES.get(r["kv"])
        if eb is None:
            raise SystemExit(f"{chip}: unknown kv_cache_dtype {r['kv']!r}")
        sel = selected_tokens(r["step"], topk, ratio)
        pts.append((r["batch"] * sel * width * eb, r["us"]))
    if len(pts) < MIN_POINTS:
        if verbose:
            print(f"{chip:14} only {len(pts)} decode points (need {MIN_POINTS})",
                  file=sys.stderr)
        return None

    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    if "BwPeakTBs" not in facts:
        raise SystemExit(f"{chip}: catalog states no BwPeakTBs")
    peak = float(facts["BwPeakTBs"]) * 1e6          # TB/s -> bytes per microsecond

    # The floor is NOT searched: it is this part's committed attention-kernel floor. See
    # the module docstring for the measured cost of pinning it (1.0%-7.8%) and why adding
    # a free parameter for that is the wrong trade.
    floor = part_wide_floor(chip)
    best = None
    for frac in RATE_FRACTIONS:
        e = log_error(pts, floor, peak * frac)
        if best is None or e < best[0]:
            best = (e, frac)
    err, frac = best
    rate = peak * frac

    # Regime split, reported rather than aggregated: the floor absorbs the small-byte end,
    # so a good overall number can hide a bad tail.
    cut = sorted(b for b, _ in pts)[len(pts) // 2]
    head = [(b, y) for b, y in pts if b < cut]
    tail = [(b, y) for b, y in pts if b >= cut]

    return {
        "sku": sku, "chip": chip, "coll": COLLECTION, "lane": lanes[0],
        "model": FIT_MODEL, "layer_kind": g["layer_kind"],
        "topk": topk, "ratio": ratio, "width": width,
        "n": len(pts), "floor": floor, "rate": int(round(rate)),
        "frac": round(frac, 3), "peak": round(peak / 1e6, 2),
        "err": round(err, 3),
        "head": round(log_error(head, floor, rate), 3), "headn": len(head),
        "tail": round(log_error(tail, floor, rate), 3), "tailn": len(tail),
    }


def report(f: dict) -> None:
    print(f"# {f['chip']} ({f['sku']}), {f['coll']}, kind=sparse_mla")
    print(f"#   model {f['model']}, catalog layer {f['layer_kind']}, lane {f['lane']}")
    print(f"#   selected = min(step, {f['topk']}) + max(0, step-{f['topk']})"
          f"/{f['ratio']}; latent width {f['width']} elements")
    print(f"#   floor PINNED to this part's attention_decode_floor "
          f"{f['floor']} us (not fitted; see the docstring)")
    print(f"attention_decode_rate_sparse_mla  {f['rate']:>9d} bytes_per_us "
          f"({f['frac']} of {f['peak']} TB/s)")
    print(f"#   {f['n']} decode points, geometric error {f['err']}x overall, "
          f"{f['head']}x head ({f['headn']}), {f['tail']}x tail ({f['tailn']})")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--sku")
    ap.add_argument("--chip")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv[1:])

    cat = Path(args.catalog)
    if args.all:
        ok = False
        for sku, chip in PARTS:
            f = fit_part(args.data, cat, sku, chip)
            if f:
                report(f)
                print()
                ok = True
        return 0 if ok else 1
    if not (args.sku and args.chip):
        ap.error("give --sku and --chip, or --all")
    f = fit_part(args.data, cat, args.sku, args.chip)
    if not f:
        return 1
    report(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
