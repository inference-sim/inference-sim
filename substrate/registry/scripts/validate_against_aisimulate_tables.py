#!/usr/bin/env python3
"""Check every fitted primitive against NVIDIA's measurement tables, per kind and per part.

    python scripts/validate_against_aisimulate_tables.py
    python scripts/validate_against_aisimulate_tables.py --check-known

# What this is for

A coefficient set can be internally valid, reproduce its own fit, and still be wrong about the
kernel it claims to describe. The only way to know is to price NVIDIA's own measured shapes
with the committed coefficients and compare. This script does that and reports the geometric
mean of predicted over measured -- above 1.0 the form is too expensive, below it too cheap.

It exists because two such findings were made by hand during this project and neither was
catchable by any existing check:

  - decode attention on h200 GQA prices 1.294x the measurement in the regime the evaluation
    corpus runs (context 512-8191, batch 4-256);
  - the MoE grouped-GEMM on b200 prices 0.625x the measurement against the lane a
    latency-sensitive engine picks.

Promoting them from hand measurements to a gate means the next such error is caught before it
reaches a coefficient, and that a re-fit which claims to improve a term can be shown to.

# Why a ratio and not a pass/fail

A closed-form primitive will not match a measured table exactly, and a threshold would either
pass everything or fail everything. What matters is the DIRECTION and the MAGNITUDE, and
whether they change when someone re-fits. So the script prints the ratio and, with
`--check-known`, asserts the two figures above still hold -- a gate that cannot reproduce a
known result cannot be trusted on a new one.

# The lane rule

Latencies for one shape differ by up to 5x between `kernel_source` lanes. The MoE check uses
the FASTEST lane because a latency-sensitive engine picks it; pooling lanes gives 0.309 where
the correct lane gives 0.625, which is the error this rule prevents.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

_spec = importlib.util.spec_from_file_location(
    "attention_table_map", Path(__file__).with_name("attention_table_map.py")
)
atm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(atm)

REGISTRY = Path(__file__).parent.parent
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))

# The regime the evaluation corpus actually runs. Reporting a ratio over every measured shape
# would average in contexts and batches no scored point visits, and the interesting question is
# whether the coefficients are right WHERE THEY ARE USED.
CORPUS_CONTEXT = (512, 8191)
CORPUS_BATCH = (4, 256)

# The GQA group sizes (query heads per KV head) the evaluation models run: minimax-m2.5 and
# granite at 6, gpt-oss and the llamas at 8, qwen3.5 and nemotron-3.5 at 16.
#
# This filter matters more than it looks. Pooled over EVERY group size the family-wide ratio is
# 0.78, which reads as a 22% systematic under-charge. Restricted to the groups the evaluation
# models occupy it is 1.064 -- essentially correct. The difference is group sizes 48 to 128,
# which in this catalog are MLA and sparse-MLA models (kimi-k3 at 96, the GLM-5 family at 64):
# a different architecture, priced by the GQA law because the kernel has no MLA law, and wrong
# by up to 10x there.
#
# So the pooled figure describes a coverage gap, not a mis-scaled coefficient, and reporting it
# alone would have justified a re-fit that made the evaluation models worse.
EVAL_GROUP_SIZES = (6.0, 8.0, 16.0)


def coefficients(chip: str) -> dict[str, float]:
    """Every committed cost-model coefficient scoped to one chip."""
    out: dict[str, float] = {}
    for path in sorted((REGISTRY / "coefficients").glob("cost-model-*.yaml")):
        doc = yaml.safe_load(path.read_text())
        for entry in doc["coefficients"]:
            (name, body), = entry.items()
            hardware = body["scope"].get("hardware")
            if hardware is None or chip in hardware:
                out[name] = body["value"]
    return out


def geo(ratios: list[float]) -> float:
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def check_attention(data: str, sku: str, chip: str, kind: str,
                    in_corpus_regime: bool,
                    geometry: tuple[int, int, int] | None = None,
                    kv_dtype: str | None = None,
                    eval_groups: bool = False) -> dict | None:
    """Price the measured decode-attention shapes for one kind with the committed terms."""
    coeff = coefficients(chip)
    suffix = f"_{kind}"
    floor = coeff.get("attention_decode_floor" + suffix,
                      coeff.get("attention_decode_floor"))
    rate = coeff.get("attention_decode_rate" + suffix,
                     coeff.get("attention_decode_rate"))
    if not floor or not rate:
        return {"status": "no committed terms"}

    rows = atm.rows_for_kind(data, sku, kind)
    if not rows:
        return {"status": "no rows"}
    lane = max({r["lane"] for r in rows},
               key=lambda l: sum(1 for r in rows if r["lane"] == l))
    rows = [r for r in rows if r["lane"] == lane and r["kv_heads"] and r["head_dim"]]
    if in_corpus_regime:
        rows = [r for r in rows
                if CORPUS_CONTEXT[0] <= r["context"] <= CORPUS_CONTEXT[1]
                and CORPUS_BATCH[0] <= r["batch"] <= CORPUS_BATCH[1]]
    if geometry is not None:
        heads, kv_heads, head_dim = geometry
        rows = [r for r in rows if r["heads"] == heads and r["kv_heads"] == kv_heads
                and r["head_dim"] == head_dim]
    # The KV cache dtype halves or doubles the bytes per token. `kv_bytes` accounts for it, so
    # pooling dtypes is not WRONG -- but the two populations carry different error, and the
    # evaluation corpus runs fp8 throughout, so a figure meant to describe the corpus must
    # filter to it. Pooling them gave 1.156 where the fp8 rows alone give 1.294.
    if kv_dtype is not None:
        rows = [r for r in rows if r["kv_cache_dtype"] == kv_dtype]
    if eval_groups:
        rows = [r for r in rows if r["kv_heads"]
                and r["heads"] / r["kv_heads"] in EVAL_GROUP_SIZES]
    if not rows:
        return {"status": "no rows in the requested regime", "lane": lane}

    ratios = []
    for r in rows:
        width = 1.0 if r["kv_cache_dtype"] == "fp8" else 2.0
        positions = r["context"]
        if r.get("window") and r["window"] > 0:
            positions = min(positions, r["window"])
        kv = r["batch"] * positions * 2 * r["kv_heads"] * r["head_dim"] * width
        pred = floor * 1e-6 + kv / (rate * 1e6)
        ratios.append(pred / (r["latency_ms"] * 1e-3))
    return {"status": "ok", "ratio": geo(ratios), "n": len(ratios), "lane": lane,
            "floor": floor, "rate": rate}


def check_moe(data: str, sku: str, chip: str, catalog: Path, geometry: dict) -> dict | None:
    """Price the measured MoE grouped-GEMM shapes with the committed GEMM envelope.

    The kernel has no MoE-specific coefficient: it prices the grouped GEMM from the same
    efficiency ramp and bandwidth the dense GEMMs use, with the expert count coming from
    ExpertsTouched. This check is therefore of that composition, not of a single number.
    """
    coeff = coefficients(chip)
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    dtype = geometry["dtype"]
    peak_field = {"nvfp4": "TFlopsNVFP4", "fp8": "TFlopsFP8", "bf16": "TFlopsPeak"}[dtype]
    peak = facts[peak_field] * 1e12
    bw = facts["BwPeakTBs"] * 1e12 * coeff["hbm_derate"]
    eps = coeff[f"gemm_eps_max_{dtype}"]
    m_half = coeff[f"gemm_m_half_{dtype}"]
    width = {"nvfp4": 0.5, "fp8": 1.0, "bf16": 2.0}[dtype]

    paths = sorted(Path(data, sku, "moe").glob("**/moe_perf.parquet"))
    pinned = [p for p in paths if "trtllm/1.3.0rc20" in str(p)]
    if not pinned:
        return {"status": "no pinned moe collection"}
    t = pq.read_table(pinned[0]).to_pydict()
    n = len(t["latency"])

    sel = [
        i for i in range(n)
        if t["hidden_size"][i] == geometry["hidden"]
        and t["inter_size"][i] == geometry["inter"]
        and t["num_experts"][i] == geometry["experts"]
        and t["topk"][i] == geometry["topk"]
        and t["moe_tp_size"][i] == geometry["tp"]
        and t["moe_ep_size"][i] == 1
        and t["moe_dtype"][i] == dtype
        and t["distribution"][i] == "balanced"
        and t["latency"][i] > 0
    ]
    if not sel:
        return {"status": "no rows at this geometry"}

    # The FASTEST lane: a latency-sensitive engine picks it. Pooling the lanes here gives
    # 0.309 against 0.625 for the fast lane, which is the mistake this rule prevents.
    lanes = {t["kernel_source"][i] for i in sel}
    best_lane, best_median = None, None
    for l in lanes:
        lat = sorted(t["latency"][i] for i in sel if t["kernel_source"][i] == l)
        med = lat[len(lat) // 2]
        if best_median is None or med < best_median:
            best_lane, best_median = l, med
    sel = [i for i in sel if t["kernel_source"][i] == best_lane]

    H, I, E, k, tp = (geometry["hidden"], geometry["inter"], geometry["experts"],
                      geometry["topk"], geometry["tp"])
    ratios = []
    for i in sel:
        tok = t["num_tokens"][i]
        flops = 2 * tok * k * 3 * I * H / tp
        eff = eps * tok / (tok + m_half)
        touched = E * (1 - ((E - k) / E) ** tok)
        by_bytes = touched * 3 * I * H * width / tp / bw
        pred = max(flops / (peak * eff), by_bytes)
        ratios.append(pred / (t["latency"][i] * 1e-3))
    return {"status": "ok", "ratio": geo(ratios), "n": len(ratios), "lane": best_lane}


# The two findings this gate was built to preserve, measured by hand before it existed.
KNOWN = [
    # The recorded 1.294 was measured on ONE geometry -- 12 query heads, 2 KV heads, head
    # dimension 128, which is minimax-m2.5's per-rank attention shape at tp=4 -- and on 26
    # points. Averaged over EVERY geometry in the family the ratio is 0.795 on 3,003 points.
    # Both numbers are right; they answer different questions, and conflating them would have
    # justified a re-fit in the wrong direction.
    #
    # The gate asserts the family-wide figure, because a coefficient scoped to a part applies
    # to every geometry on it. The per-geometry figure is kept below as a separate entry so the
    # spread between the two stays visible: a form that is 0.80 on average and 1.29 on one
    # shape is not uniformly mis-scaled, it has geometry-dependent error, which is scatter and
    # not bias.
    {"what": "decode attention, h200 gqa, corpus regime, all geometries",
     "check": ("attention", "h200_sxm", "h200", "gqa", True),
     "kv_dtype": "fp8", "ratio": 0.795, "tol": 0.03},
    {"what": "decode attention, h200 gqa, corpus regime, minimax per-rank geometry",
     "check": ("attention", "h200_sxm", "h200", "gqa", True),
     "geometry": (12, 2, 128), "kv_dtype": "fp8", "ratio": 1.294, "tol": 0.02},
    {"what": "decode attention, h200 gqa, corpus regime, evaluation group sizes",
     "check": ("attention", "h200_sxm", "h200", "gqa", True),
     "kv_dtype": "fp8", "eval_groups": True, "ratio": 1.064, "tol": 0.03},
    {"what": "MoE grouped-GEMM, b200, minimax geometry, fastest lane",
     "check": ("moe", "b200_sxm", "b200",
               {"hidden": 3072, "inter": 1536, "experts": 256, "topk": 8, "tp": 4,
                "dtype": "nvfp4"}, None),
     "ratio": 0.625, "tol": 0.02},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=atm.DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check-known", action="store_true",
                    help="assert the two hand-measured findings still reproduce")
    args = ap.parse_args()

    if args.check_known:
        failures = []
        for entry in KNOWN:
            family = entry["check"][0]
            if family == "attention":
                _, sku, chip, kind, regime = entry["check"][:5]
                got = check_attention(args.data, sku, chip, kind, regime,
                                      entry.get("geometry"), entry.get("kv_dtype"),
                                      entry.get("eval_groups", False))
            else:
                _, sku, chip, geometry, _ = entry["check"]
                got = check_moe(args.data, sku, chip, Path(args.catalog), geometry)
            if got is None or got["status"] != "ok":
                failures.append(f"{entry['what']}: {got and got['status']}")
                print(f"FAIL {entry['what']}: {got and got['status']}")
                continue
            ok = abs(got["ratio"] - entry["ratio"]) <= entry["tol"]
            print(f"{'OK  ' if ok else 'FAIL'} {entry['what']}: "
                  f"{got['ratio']:.3f} against a recorded {entry['ratio']:.3f} "
                  f"(n={got['n']:,}, lane {got['lane']})")
            if not ok:
                failures.append(
                    f"{entry['what']}: {got['ratio']:.3f} against {entry['ratio']:.3f}")
        print()
        if failures:
            print(f"FAIL: {len(failures)} known finding(s) no longer reproduce")
            for f in failures:
                print(f"  {f}")
            return 1
        print("PASS: both hand-measured findings reproduce, so the gate is trustworthy")
        return 0

    print("Committed coefficients priced against NVIDIA's measured tables")
    print("ratio = geometric mean of predicted / measured; >1 too expensive, <1 too cheap")
    print(f"corpus regime: context {CORPUS_CONTEXT}, batch {CORPUS_BATCH}\n")
    print(f"{'part':8s} {'kind':12s} {'regime':8s} {'ratio':>7s} {'n':>8s}  lane")
    for sku, chip in sorted(atm.SKUS.items()):
        if not Path(args.data, sku).is_dir():
            continue
        for kind in atm.KINDS:
            for regime, groups, label in ((True, True, "eval"), (True, False, "corpus"),
                                          (False, False, "all")):
                r = check_attention(args.data, sku, chip, kind, regime, None, "fp8", groups)
                if r is None or r["status"] != "ok":
                    if label == "corpus":
                        print(f"{chip:8s} {kind:12s} {label:8s} {r['status']}")
                    continue
                print(f"{chip:8s} {kind:12s} {label:8s} {r['ratio']:7.3f} {r['n']:8,d}  "
                      f"{r['lane']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
