#!/usr/bin/env python3
"""Fit the grouped (routed-expert) GEMM ramp of NVFP4 MoE on Blackwell: FlashInfer TRT-LLM.

    python scripts/fit_nvfp4_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_nvfp4_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS (inference-sim/blis-registry#51, #60 gap 1). On a part with a native
FP4 peak, blis-latency-kernel priced an NVFP4 MoE's routed experts with the DENSE NVFP4 ramp
(`gemm_*_nvfp4`), which is fitted on vLLM's dense linear kernel
(FlashInferCuteDslNvFp4LinearKernel), not on the fused-MoE kernel the experts run. The
companion kernel change reads `gemm_{eps_max,m_half}_nvfp4_grouped` for a grouped NVFP4 GEMM
on such a part first, and falls back to `nvfp4` with a recorded assumption.

WHICH KERNEL. vLLM v0.31.0 picks an NVFP4 MoE backend in this order: FLASHINFER_TRTLLM,
FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN
(model_executor/layers/fused_moe/oracle/nvfp4.py:194-203), and for FLASHINFER_TRTLLM prefers
TrtLlmNvFp4ExpertsMonolithic over the Modular form (nvfp4.py:80-90). So the fit is on
`..._flashinfer_trtllm_trtllmnvfp4expertsmonolithic`.

THE QUANTITY, and the composition it is conditioned on. Exactly the form
fit_marlin_grouped_ramp.py fits, at a different peak and byte count:

    t = T*top_k*6*h*i / (NVFP4_peak * eff(T))  +  touched(T) * 3*h*i*B / (BwPeak * hbm_derate)
    eff(T) = eps_max * T / (T + M_half),   touched(T) = E * (1 - ((E - top_k)/E)^T)

the routed-expert term blis-latency-kernel composes (step.go: the SM and HBM halves summed,
the ramp evaluated at the step's token count, compute at the chip's NVFP4 peak `TFlopsNVFP4`,
memory as ExpertsTouched experts' stored bytes at BwPeak x hbm_derate). B is the STORED
bytes per NVFP4 weight, 0.5 + 1/16 = 0.5625 (one FP8 E4M3 scale per 16 values; blis-schemas
StoredBytes for DTypeNVFP4). Fitted at moe_tp = moe_ep = 1 on balanced routing, so imbalance
and the tensor-parallel slice are 1 here and are the consumer's to apply.

THE DATA. NVIDIA AISimulate `systems/data/<sku>/moe/vllm/0.30.0/moe_perf.parquet` at
ai-dynamo/aisimulate 8ae6b65801c4993febb3c827a0e68518b20b379e (2026-10-29; the registry's
other families re-derive at 2f0f4cd, which carries no 0.30.0 MoE table, so this family needs
its own checkout). Read from $AISIMULATE_NVFP4_DATA, else $AISIMULATE_DATA. It writes no entry
(see THE GATE), so neither the rederive workflow nor the writer tests run it. moe_dtype nvfp4, distribution balanced,
moe_tp 1, moe_ep 1, 16 expert geometries x 27 token counts (1-16,384). `latency` is
MILLISECONDS.

WHY vLLM 0.30.0 MEASURES v0.31.0, AND WHERE IT MAY NOT. The kernel is FlashInfer's, and the
FlashInfer pin moves: flashinfer-python 0.6.18.post1 at vLLM v0.30.0, 0.7.0.post1 at v0.31.0
(requirements/cuda.txt). That diff was not audited, so held-out C below scores the fit on the
one v0.31.0 table AISimulate carries (a ModelOpt NVFP4 checkpoint, GLM-5.3-Flash geometry,
the same Monolithic kernel), on the parts that have it (gb200, b300, gb300; not b200).

THE SPLIT AND RULES, declared in this docstring before the fit below was run. An exploratory
fit of the same form on all 16 geometries had already been read (held-out medians 14-17%),
so no pass band is claimed as pre-registered. The landing rule is relative and declared here:
an entry is written only where its held-out median |rel err| is below that of the dense
`gemm_*_nvfp4` ramp the kernel uses today, scored under the same composition on the same rows.
  * EXCLUSION (physical): a geometry with any row faster than reading its touched experts'
    stored weights at the datasheet BwPeak, with no derate, is impossible as labelled and is
    excluded whole, because its row set is then not the stated geometry. Found on exploration:
    (hidden 7168, inter 2048, top-8 of 256) on every part, up to 1.8x faster than the read.
  * held-out A, geometry: of the remaining geometries sorted by (hidden, inter, top_k,
    experts), every second one (odd index) is held out; the rest are fitted.
  * held-out B, kernel: TrtLlmNvFp4ExpertsModular, vLLM's second choice on the same backend.
  * held-out C, version: the v0.31.0 table, where present.
  * reported: median and 90th-percentile |rel err|, and the signed median.

THE GRID. eps_max over 0.001-1.000 in 0.001 steps, M_half over 1-8192 tokens; a fit pinned to
the grid's edge is refused.

THE GATE, AS DECIDED (run at aisimulate 8ae6b658): PARKED, nothing written. On held-out A the
grouped fit is no better than the dense nvfp4 ramp on any part (median |rel err|, grouped vs
dense): gb200-nvl72 16.9% vs 15.2%, b200 16.2% vs 15.4%, b300 14.4% vs 12.9%, gb300 13.8% vs
12.6%. B and C agree (C: gb200 12.0% vs 8.2%, b300 8.5% vs 6.7%, gb300 9.8% vs 6.5%). Fitted
values, for the record: gb200-nvl72 0.549 / 122, b200 0.522 / 112, b300 0.390 / 126, gb300
0.389 / 140, against the committed dense nvfp4 pairs gb200-nvl72 0.732 / 199, b200 0.689 /
174, b300 0.501 / 192, gb300 0.526 / 227. The routed-expert error is dominated by geometry (small hidden sizes under-predicted by
about 27%, the largest over-predicted), which a ramp over tokens alone cannot carry, so a
second token ramp buys nothing at the operator level. The kernel's fallback to the dense
nvfp4 ramp therefore stands, and the +10% / +20% kimi-k2.5-nvfp4 TPOT residual #51 records is
not explained by the ramp. The script is kept so the evidence re-runs; it writes an entry
only for a part whose held-out A beats the dense ramp.
"""

from __future__ import annotations

import argparse
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

SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
AISIMULATE_SHA = "8ae6b65801c4993febb3c827a0e68518b20b379e"
DEFAULT_DATA = (os.environ.get("AISIMULATE_NVFP4_DATA") or os.environ.get("AISIMULATE_DATA")
                or "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = (os.environ.get("BLIS_CATALOG")
                   or str(HERE.parents[1] / "catalog"))

COLLECTION = "vllm/0.30.0"
HOLDOUT_VERSION = "vllm/0.31.0"
PREFIX = "vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_"
FIT_KERNEL = PREFIX + "trtllmnvfp4expertsmonolithic"
MODULAR_KERNEL = PREFIX + "trtllmnvfp4expertsmodular"
V031_KERNEL = "vllm_modeloptnvfp4fusedmoe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic"
BYTES_PER_WEIGHT = 0.5 + 1 / 16

# (AISimulate sku, catalog chip). gb300 is included: it has the same tables.
PARTS = [("gb200", "gb200-nvl72"), ("b200_sxm", "b200"), ("b300_sxm", "b300"),
         ("gb300", "gb300")]
NAMES = ("gemm_eps_max_nvfp4_grouped", "gemm_m_half_nvfp4_grouped")

EPS_GRID = np.round(np.arange(0.001, 1.0005, 0.001), 3)
MHALF_GRID = np.arange(1, 8193)
GEOM = ("hidden_size", "inter_size", "topk", "num_experts")


def chip_facts(catalog: Path, chip: str) -> tuple[float, float]:
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts["TFlopsNVFP4"]) * 1e12, float(facts["BwPeakTBs"]) * 1e12


def committed(name: str, chip: str) -> float:
    doc = yaml.safe_load(SET_PATH.read_text())
    hits = [body["value"] for e in doc["coefficients"] for n, body in e.items()
            if n == name and body["scope"] == {"hardware": [chip]}]
    if len(hits) != 1:
        raise SystemExit(f"{chip}: expected one part-wide {name}, found {len(hits)}")
    return float(hits[0])


def table(data: Path, sku: str, collection: str) -> Path:
    return data / sku / "moe" / collection / "moe_perf.parquet"


def load(data: Path, sku: str, collection: str, kernel: str) -> list[dict]:
    path = table(data, sku, collection)
    if not path.is_file():
        return []
    t = pq.read_table(path).to_pydict()
    rows = []
    for i in range(len(t["latency"])):
        if (t["kernel_source"][i] != kernel or t["moe_dtype"][i] != "nvfp4"
                or t["distribution"][i] != "balanced" or t["moe_tp_size"][i] != 1
                or t["moe_ep_size"][i] != 1):
            continue
        if not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0):
            raise SystemExit(f"{sku}: {kernel} row {i} has latency {t['latency'][i]}")
        rows.append({k: t[k][i] for k in ("num_tokens",) + GEOM} | {"s": t["latency"][i] * 1e-3})
    return rows


def geom(r: dict) -> tuple:
    return tuple(r[k] for k in GEOM)


def read_seconds(r: dict, bw: float) -> float:
    h, i, k, e, T = r["hidden_size"], r["inter_size"], r["topk"], r["num_experts"], r["num_tokens"]
    return e * (1 - ((e - k) / e) ** T) * 3 * h * i * BYTES_PER_WEIGHT / bw


def impossible(rows: list[dict], bw_peak: float) -> set:
    return {geom(r) for r in rows if r["s"] < read_seconds(r, bw_peak)}


def terms(rows: list[dict], bw: float):
    T = np.array([r["num_tokens"] for r in rows], float)
    F = np.array([r["num_tokens"] * r["topk"] * 6 * r["hidden_size"] * r["inter_size"]
                  for r in rows], float)
    H = np.array([read_seconds(r, bw) for r in rows], float)
    S = np.array([r["s"] for r in rows], float)
    return T, F, H, S


def rel(T, F, H, S, peak, eps, mh) -> np.ndarray:
    return (F / (peak * eps * T / (T + mh)) + H) / S - 1


def fit(T, F, H, S, peak) -> tuple[float, int, float]:
    mh = MHALF_GRID[:, None].astype(float)
    best = (math.inf, 0.0, 0)
    for eps in EPS_GRID:
        pred = F[None, :] / (peak * eps * T[None, :] / (T[None, :] + mh)) + H[None, :]
        err = np.mean(np.abs(np.log(pred / S[None, :])), axis=1)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(eps), int(MHALF_GRID[j]))
    err, eps, mh = math.exp(best[0]), best[1], best[2]
    if eps in (EPS_GRID[0], EPS_GRID[-1]) or mh in (MHALF_GRID[0], MHALF_GRID[-1]):
        raise SystemExit(f"fit is not interior: eps_max={eps}, m_half={mh}")
    return eps, mh, err


def score(rows, bw, peak, eps, mh) -> dict | None:
    if not rows:
        return None
    r = rel(*terms(rows, bw), peak, eps, mh)
    return {"n": len(r), "med": float(np.median(np.abs(r))),
            "p90": float(np.percentile(np.abs(r), 90)), "signed": float(np.median(r))}


def fit_part(data: Path, catalog: Path, sku: str, chip: str) -> dict:
    peak, bw_peak = chip_facts(catalog, chip)
    bw = bw_peak * committed("hbm_derate", chip)
    rows = load(data, sku, COLLECTION, FIT_KERNEL)
    if len(rows) < 100:
        raise SystemExit(f"{chip}: {table(data, sku, COLLECTION)} carries {len(rows)} "
                         f"{FIT_KERNEL} rows; the pinned AISimulate {AISIMULATE_SHA[:8]} "
                         f"collection is required (AISIMULATE_NVFP4_DATA)")
    bad = impossible(rows, bw_peak)
    kept = sorted({geom(r) for r in rows} - bad)
    held = set(kept[1::2])
    train = [r for r in rows if geom(r) in set(kept) - held]
    hold_a = [r for r in rows if geom(r) in held]
    eps, mh, err = fit(*terms(train, bw), peak)
    dense = (committed("gemm_eps_max_nvfp4", chip), committed("gemm_m_half_nvfp4", chip))
    modular = [r for r in load(data, sku, COLLECTION, MODULAR_KERNEL) if geom(r) not in bad]
    v031 = load(data, sku, HOLDOUT_VERSION, V031_KERNEL)
    out = {"sku": sku, "chip": chip, "eps": eps, "mh": mh, "err": err, "peak_tf": peak / 1e12,
           "n": len(train), "n_geom": len(kept) - len(held), "excluded": sorted(bad),
           "held_geoms": sorted(held), "dense": dense, "scores": {}}
    for label, sel in (("fit", train), ("A", hold_a), ("B", modular), ("C", v031)):
        out["scores"][label] = (score(sel, bw, peak, eps, mh), score(sel, bw, peak, *dense))
    a, a_dense = out["scores"]["A"]
    out["lands"] = a["med"] < a_dense["med"]
    return out


def _s(x: dict | None) -> str:
    if x is None:
        return "absent"
    return (f"median {x['med']:.1%}, p90 {x['p90']:.1%}, signed {x['signed']:+.1%} "
            f"(n {x['n']})")


def entry_lines(name: str, f: dict) -> list[str]:
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet "
            f"(aisimulate {AISIMULATE_SHA[:8]}), {FIT_KERNEL}, moe_dtype nvfp4, balanced, "
            f"moe_tp 1 / moe_ep 1, {f['n']} rows over {f['n_geom']} expert geometries")
    sc = f["scores"]
    held = "; ".join(f"{h}x{i} top-{k} of {e}" for h, i, k, e in f["held_geoms"])
    if name.startswith("gemm_eps_max"):
        value, units = f"{f['eps']:.3f}", "dimensionless"
        rat = (
            f"Asymptotic fraction of this part's dense NVFP4 peak ({f['peak_tf']:g} TFLOPS) that "
            f"vLLM's FlashInfer TRT-LLM NVFP4 fused-MoE kernel (TrtLlmNvFp4ExpertsMonolithic, "
            f"v0.31.0's first choice for NVFP4 MoE on SM100, oracle/nvfp4.py:80-90, 194-203) "
            f"reaches. Fitted with gemm_m_half_nvfp4_grouped under the kernel's routed-expert "
            f"composition: the SM half over the routed FLOPs (tokens x top_k x 6 x hidden x "
            f"inter at tp = ep = 1) at the NVFP4 peak, summed with ExpertsTouched experts' "
            f"stored bytes (0.5625 B/weight, E4M3 scale per 16 included) at BwPeak x "
            f"hbm_derate. Geometric error {f['err']:.3f}x on the fit rows ({_s(sc['fit'][0])}). "
            f"Held out A, every second geometry ({held}): {_s(sc['A'][0])}, against the dense "
            f"nvfp4 ramp this replaces ({f['dense'][0]} / {f['dense'][1]:g}) at "
            f"{_s(sc['A'][1])}. Held out B, the Modular kernel: {_s(sc['B'][0])} (dense ramp "
            f"{_s(sc['B'][1])}). Held out C, AISimulate's vllm/0.31.0 ModelOpt NVFP4 table: "
            f"{_s(sc['C'][0])}. Excluded as physically impossible (faster than its own weight "
            f"read at datasheet BwPeak): {', '.join(str(g) for g in f['excluded']) or 'none'}. "
            f"FlashInfer moves 0.6.18.post1 (vLLM 0.30.0) to 0.7.0.post1 (v0.31.0), not audited; "
            f"held-out C is the in-version check. The error is shape-dependent (small "
            f"hidden sizes are under-predicted), which a single ramp over tokens cannot "
            f"carry. Applies to the routed-expert term on a part with a native FP4 peak "
            f"only. Reproduce with scripts/fit_nvfp4_grouped_ramp.py.")
    else:
        value, units = str(f["mh"]), "tokens"
        rat = (f"The step token count at which the NVFP4 grouped (FlashInfer TRT-LLM fused-MoE) "
               f"ramp reaches half its asymptote, fitted jointly with gemm_eps_max_nvfp4_grouped "
               f"over {f['n']} rows; the kernel evaluates it at the step's token count, as it "
               f"does every ramp. Reproduce with scripts/fit_nvfp4_grouped_ramp.py.")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{f['chip']}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def render(text: str, fits: dict[str, dict]) -> str:
    blocks = cb.split(text)
    owned = {cb.scope_key(f"{{hardware: [{c}]}}") for c in fits}
    for b in blocks:
        if cb.name_of(b) in NAMES and cb.scope_key(cb.scope_of(b) or "{}") not in owned:
            raise SystemExit(f"{cb.name_of(b)} {cb.scope_of(b)} is not a part this writer "
                             f"maintains")
    for chip, f in fits.items():
        scope = f"{{hardware: [{chip}]}}"
        for name in NAMES:
            cb.upsert(blocks, entry_lines(name, f),
                      after=lambda b: (cb.name_of(b) or "").startswith("gemm_")
                      and cb.scope_key(cb.scope_of(b) or "{}") == cb.scope_key(scope))
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])
    fits = {}
    for sku, chip in PARTS:
        f = fit_part(Path(args.data), Path(args.catalog), sku, chip)
        print(f"{chip:12} eps_max={f['eps']:.3f} m_half={f['mh']:4d} fit {f['err']:.3f}x "
              f"(n={f['n']}); excluded {f['excluded']}")
        for k in ("fit", "A", "B", "C"):
            new, old = f["scores"][k]
            print(f"{'':12} {k:3} grouped {_s(new)} | dense nvfp4 {_s(old)}")
        print(f"{'':12} {'lands' if f['lands'] else 'NOT written: held-out A no better than the dense ramp'}")
        if f["lands"]:
            fits[chip] = f
    before = SET_PATH.read_text(encoding="utf-8")
    after = render(before, fits)
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
