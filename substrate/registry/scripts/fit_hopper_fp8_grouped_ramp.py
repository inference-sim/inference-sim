#!/usr/bin/env python3
"""Fit the grouped-GEMM ramp and floor of each vLLM block-FP8 MoE backend on Hopper.

    python scripts/fit_hopper_fp8_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_hopper_fp8_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS (inference-sim/blis-substrate#150). Block-FP8 routed experts on SM90
run the grouped kernel the engine's MoE backend selects. vLLM v0.33.0 picks Triton's fused
experts (TritonExperts) at expert-parallel width 1 and FlashInfer's CUTLASS experts
(FlashInferExperts) above it, unless moe_backend says otherwise
(model_executor/layers/fused_moe/oracle/fp8.py:79-130@4ba787ec). Before this family the cost
model priced both on the dense FP8 linear ramp (gemm_*_fp8, CUTLASS scaled-mm), with the expert
work charged whole to every tensor-parallel rank and no floor. Scored on the same rows that
puts FlashInfer's CUTLASS experts at 0.62x of their measured time on h200 (0.70x on h100), and
Triton's at a geometric error of 1.38x (1.34x).

THE QUANTITY. Per backend and part, a ramp eff(M) = eps_max * M / (M + M_half) as a fraction
of the catalog's dense FP8 peak (TFlopsFP8), and a per-layer device floor F, CONDITIONED ON THE
COMPOSITION the kernel applies to a routed-expert term on these ramps (kernel/step.go, the
routed SM and HBM halves SUMMED, the floor added to the SM half):

    t = F + T*top_k/ep*6*h*(i/tp) / (FP8_peak * eff(T))  +  touched(T) * 3*h*(i/tp)*B / (BwPeak * hbm_derate)
    touched(T) = (E/ep) * (1 - ((E - top_k)/E)^T)        # price.ExpertsTouched

with M = T, the step's token count. The FLOPs are the rank's OWN: each tensor-parallel rank
computes inter / tp of each expert and each expert-parallel rank its share of the routed
tokens, and the kernel divides by the slice on these ramps (gemmFit.rankOwn). B is the stored
bytes per weight of a 128 x 128 block-scaled FP8 expert, 1 + 4 / 16384.

THE DATA. AISimulate `moe/vllm/0.30.0/moe_perf.parquet`, moe_dtype fp8_block, balanced
routing, every expert geometry the sweep carries (19-20 of them, among them MiniMax-M2.5's
3072 / 1536 / top-8 of 256 and GLM-5.3's 6144 / 2048 / top-8 of 256): Triton on the
expert-parallel-width-1 rows at moe_tp 1 to 8, FlashInfer CUTLASS on moe_tp 1 rows at moe_ep 2
to 16, 1 to 16,384 tokens. Held out: the 0.31.0 collection, a different build on GLM-5.3-Flash's
geometry (4096 / 2048 / top-8 of 288). Balanced rows, because the kernel applies the routing
imbalance (moe_routing_imbalance_median) on top of the ramp.

WHY 0.30.0. It is the newest collection with more than one geometry; the 0.31.0 tables carry
one, used as the hold-out. Neither is v0.33.0, which BLIS models; the hold-out's error is the
evidence the kernels did not move between builds.

FORM LIMITS, REPORTED, NOT TUNED AWAY. Triton's latency at decode-sized steps tracks the weights
of the distinct experts it reads, at an effective bandwidth that is not constant: on
MiniMax-M2.5's geometry at moe_tp 8 it falls from about 2.1 TB/s at 64 tokens to 1.4 TB/s at 128,
where the kernel's per-M configuration changes, so the time jumps between 64 and 128 tokens
(189 to 318 us) while the experts read rise by an eighth. The jumps do not sit on one token-block
boundary across geometries and widths, so no step function is fitted; a smooth ramp under-prices
the rows above each jump and over-prices those below. The fit's error is that limit.

INTERVALS (VAL-10). `ci95` is the 2.5 and 97.5 percentiles of 200 bootstrap refits that
resample rows, each refit searched on a local grid around the full fit (floor step 1 us,
eps_max step 0.01, M_half over the fit's grid within a factor of 4), widened by half a step either
side. Rows of one geometry are correlated, so it is narrower than the data would justify.

LANE. vLLM. Units: `latency` is MILLISECONDS in every AISimulate parquet.
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
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA", "/tmp/aisim/python/aisim/src/aisim_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG") or str(HERE.parents[1] / "catalog")

COLLECTION = "vllm/0.30.0"
HOLDOUT = "vllm/0.31.0"
DTYPE = "fp8_block"
BYTES_PER_WEIGHT = 1 + 4 / (128 * 128)
HARNESS = ("AISimulate collector/vllm/collect_moe.py at ai-dynamo/aisim "
           "6fdf6fb6b2e1c15b8f5cd1ba35fd2d50cdb2eb18: the MoE module's forward, timed by CUDA events "
           "over CUDA-graph replays on the autotuned tactic (collector/helper.py benchmark_with_power)")
PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100")]

# backend word -> (AISimulate kernel_source, which rows, vLLM experts class, when v0.33.0 runs it)
BACKENDS = {
    "triton": ("vllm_fp8moe_triton_tritonexperts", lambda tp, ep: ep == 1 and tp <= 8,
               "TritonExperts", "moe_tp 1-8 at moe_ep 1",
               "vLLM v0.33.0 runs it by default on SM90 at expert-parallel width 1 "
               "(fused_moe/oracle/fp8.py:122-130@4ba787ec), and wherever moe_backend is triton"),
    "flashinfer_cutlass": ("vllm_fp8moe_flashinfer_cutlass_flashinferexperts",
                           lambda tp, ep: tp == 1 and 2 <= ep <= 16,
                           "FlashInferExperts", "moe_ep 2-16 at moe_tp 1",
                           "vLLM v0.33.0 runs it by default on SM90 above expert-parallel "
                           "width 1 (fused_moe/oracle/fp8.py:122-130@4ba787ec), and wherever "
                           "moe_backend is flashinfer_cutlass"),
}

EPS_GRID = np.round(np.arange(0.01, 1.0001, 0.01), 2)
MHALF_GRID = np.unique(np.round(np.geomspace(1, 4096, 160))).astype(float)
FLOOR_GRID = np.arange(0.0, 80e-6, 1e-6)


def chip_facts(catalog: Path, chip: str) -> tuple[float, float]:
    """(dense FP8 peak in FLOP/s, datasheet HBM bandwidth in B/s), from the catalog."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts["TFlopsFP8"]) * 1e12, float(facts["BwPeakTBs"]) * 1e12


def committed(name: str, chip: str) -> float:
    doc = yaml.safe_load(SET_PATH.read_text())
    hits = [body["value"] for e in doc["coefficients"] for n, body in e.items()
            if n == name and body["scope"] == {"hardware": [chip]}]
    if len(hits) != 1:
        raise SystemExit(f"{chip}: expected one part-wide {name}, found {len(hits)}")
    return float(hits[0])


def load(data: Path, sku: str, collection: str, kernel: str, keep) -> dict[str, np.ndarray]:
    path = data / sku / "moe" / collection / "moe_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"{sku}: {path} is absent; the pinned collection is required")
    t = pq.read_table(path).to_pydict()
    idx = [i for i in range(len(t["latency"]))
           if t["kernel_source"][i] == kernel and t["moe_dtype"][i] == DTYPE
           and t["distribution"][i] == "balanced" and keep(t["moe_tp_size"][i], t["moe_ep_size"][i])]
    for i in idx:
        if not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0):
            raise SystemExit(f"{sku}: {kernel} row {i} has latency {t['latency'][i]}")
    cols = ("num_tokens", "hidden_size", "inter_size", "topk", "num_experts", "moe_tp_size", "moe_ep_size")
    out = {c: np.asarray([t[c][i] for i in idx], dtype=float) for c in cols}
    out["s"] = np.asarray([t["latency"][i] * 1e-3 for i in idx], dtype=float)
    return out


def terms(r: dict[str, np.ndarray], bw: float) -> tuple[np.ndarray, ...]:
    """Per row: tokens, the rank's routed FLOPs, the kernel's memory-half seconds, measured seconds."""
    T, h, i = r["num_tokens"], r["hidden_size"], r["inter_size"] / r["moe_tp_size"]
    k, e, ep = r["topk"], r["num_experts"], r["moe_ep_size"]
    F = T * k / ep * 6 * h * i
    touched = (e / ep) * (1 - ((e - k) / e) ** T)
    return T, F, touched * 3 * h * i * BYTES_PER_WEIGHT / bw, r["s"]


def predict(T, F, H, peak, eps, mh, floor):
    return floor + F / (peak * eps * T / (T + mh)) + H


def errors(pred, S) -> tuple[float, float]:
    r = np.log(pred) - np.log(S)
    return math.exp(float(np.mean(np.abs(r)))), math.exp(float(np.mean(r)))


def fit(T, F, H, S, peak) -> tuple[float, float, float, float]:
    """Exhaustive grid over (floor, eps_max, M_half), minimising the geometric error."""
    best = (math.inf, 0.0, 0.0, 0.0)
    for floor in FLOOR_GRID:
        for eps in EPS_GRID:
            pred = floor + F[None, :] / (peak * eps * T[None, :] / (T[None, :] + MHALF_GRID[:, None])) + H[None, :]
            err = np.mean(np.abs(np.log(pred / S[None, :])), axis=1)
            j = int(np.argmin(err))
            if err[j] < best[0]:
                best = (float(err[j]), float(eps), float(MHALF_GRID[j]), float(floor))
    err, eps, mh, floor = best
    if eps in (EPS_GRID[0], EPS_GRID[-1]) or floor == FLOOR_GRID[-1] or mh == MHALF_GRID[-1]:
        raise SystemExit(f"fit is not interior: eps_max={eps}, m_half={mh}, floor={floor}")
    return eps, mh, floor, math.exp(err)


def bootstrap(T, F, H, S, peak, eps, mh, floor, n=200, seed=150) -> dict[str, tuple[float, float]]:
    """2.5 and 97.5 percentiles of refits on resampled rows, on a local grid around the fit."""
    rng = np.random.default_rng(seed)
    eg = np.round(np.arange(max(0.01, eps - 0.08), eps + 0.0801, 0.01), 2)
    mg = MHALF_GRID[(MHALF_GRID >= mh / 4) & (MHALF_GRID <= mh * 4)]
    fg = np.arange(max(0.0, floor - 8e-6), floor + 8.01e-6, 1e-6)
    out = {"eps": [], "mh": [], "floor": []}
    for _ in range(n):
        i = rng.integers(0, len(S), len(S))
        t, f, h, y = T[i], F[i], H[i], S[i]
        best = (math.inf, 0, 0, 0)
        for fl in fg:
            for e in eg:
                pred = fl + f[None, :] / (peak * e * t[None, :] / (t[None, :] + mg[:, None])) + h[None, :]
                err = np.mean(np.abs(np.log(pred / y[None, :])), axis=1)
                j = int(np.argmin(err))
                if err[j] < best[0]:
                    best = (err[j], e, mg[j], fl)
        out["eps"].append(best[1]); out["mh"].append(best[2]); out["floor"].append(best[3])
    lo, hi = {}, {}
    q = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}
    return {"eps": (round(q["eps"][0] - 0.005, 3), round(q["eps"][1] + 0.005, 3)),
            "mh": (max(1.0, math.floor(q["mh"][0])), math.ceil(q["mh"][1])),
            "floor": (round(q["floor"][0] * 1e6 - 0.5, 1), round(q["floor"][1] * 1e6 + 0.5, 1))}


def fit_part(data: Path, catalog: Path, sku: str, chip: str, backend: str) -> dict:
    kernel, keep, _, rows, _ = BACKENDS[backend]
    peak, bw_peak = chip_facts(catalog, chip)
    bw = bw_peak * committed("hbm_derate", chip)
    train = load(data, sku, COLLECTION, kernel, keep)
    n = len(train["s"])
    if n < 500:
        raise SystemExit(f"{chip} {backend}: only {n} training rows")
    geoms = len({tuple(x) for x in np.stack([train[c] for c in ("hidden_size", "inter_size", "num_experts", "topk")], 1)})
    T, F, H, S = terms(train, bw)
    eps, mh, floor, err = fit(T, F, H, S, peak)
    _, signed = errors(predict(T, F, H, peak, eps, mh, floor), S)
    ci = bootstrap(T, F, H, S, peak, eps, mh, floor)
    # The ranks the rows span (moe_tp, moe_ep) are stated in the citation; fit_range has no
    # dimension for them.
    span = {"tokens": (float(T.min()), float(T.max()))}
    hold = load(data, sku, HOLDOUT, kernel, keep)
    if len(hold["s"]) < 50:
        raise SystemExit(f"{chip} {backend}: the {HOLDOUT} hold-out is missing ({len(hold['s'])} rows)")
    ht = terms(hold, bw)
    ho = errors(predict(ht[0], ht[1], ht[2], peak, eps, mh, floor), ht[3])
    # What the kernel priced these rows at before this family: the dense FP8 ramp, the whole
    # expert's FLOPs on every tensor-parallel rank, no floor.
    whole = dict(train)
    whole["moe_tp_size"] = np.ones_like(train["moe_tp_size"])
    Tw, Fw, _, _ = terms(whole, bw)
    before = errors(predict(T, Fw, H, peak, committed("gemm_eps_max_fp8", chip),
                            committed("gemm_m_half_fp8", chip), 0.0), S)
    return {"sku": sku, "chip": chip, "backend": backend, "eps": eps, "mh": int(mh),
            "floor_us": round(floor * 1e6, 1), "err": err, "signed": signed, "n": n,
            "geoms": geoms, "rows": rows, "holdout": ho, "holdout_n": len(hold["s"]),
            "before": before, "peak_tf": peak / 1e12, "ci": ci, "span": span}


def entry_lines(name: str, f: dict) -> list[str]:
    kernel, _, cls, rows, when = BACKENDS[f["backend"]]
    b = f["backend"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet, "
            f"{kernel}, moe_dtype {DTYPE}, balanced, {rows}, {f['n']} rows over {f['geoms']} "
            f"expert geometries")
    common = (f"vLLM's {cls} ({b}) grouped GEMM on block-FP8 experts; {when}. Fitted jointly "
              f"(gemm_eps_max_fp8_grouped_{b}, gemm_m_half_fp8_grouped_{b}, gemm_floor_fp8_grouped_{b}) "
              f"under the kernel's routed-expert composition: the floor plus the rank's OWN routed "
              f"FLOPs (tokens x top_k / ep x 6 x hidden x inter / tp) at the dense FP8 peak "
              f"({f['peak_tf']:g} TFLOPS) times the ramp, summed with the distinct experts' "
              f"stored bytes (1 + 4/16384 B/weight) at BwPeak x hbm_derate. Geometric error "
              f"{f['err']:.3f}x (signed {f['signed']:.3f}); the dense FP8 ramp the kernel priced "
              f"these rows on before scores {f['before'][0]:.3f}x (signed {f['before'][1]:.3f}). "
              f"The interval is a bootstrap of row resamples (see the writer's INTERVALS). "
              f"Held out on the {HOLDOUT} collection ({f['holdout_n']} rows, GLM-5.3-Flash's "
              f"geometry) at {f['holdout'][0]:.3f}x, signed {f['holdout'][1]:.3f}. A change to the "
              f"composition -- the floor's place, sum versus max, the ramp's argument, the "
              f"tensor-parallel division -- needs a refit. Reproduce with "
              f"scripts/fit_hopper_fp8_grouped_ramp.py.")
    if name.startswith("gemm_eps_max"):
        value, units = f"{f['eps']:.2f}", "dimensionless"
        rat = f"Asymptotic fraction of the part's dense FP8 peak that {common}"
    elif name.startswith("gemm_m_half"):
        value, units = str(f["mh"]), "tokens"
        lim = (" It sits at the grid's lower edge: the rows identify no knee above one token, "
               "and the floor carries the low-token time." if f["mh"] <= 1 else "")
        rat = f"The step token count at which the ramp reaches half its asymptote.{lim} {common}"
    else:
        value, units = f"{f['floor_us']:g}", "us_per_transfer"
        rat = (f"The per-layer device time of the grouped experts however few tokens reach "
               f"them, which the ramp's work term alone cannot carry. {common}")
    key = "eps" if name.startswith("gemm_eps_max") else "mh" if name.startswith("gemm_m_half") else "floor"
    lo, hi = f["ci"][key]
    fmt = (lambda v: f"{v:.3f}") if key == "eps" else (lambda v: f"{v:g}")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{f['chip']}]}}",
        "      measures:",
        "        time: device",
        "        execution: captured",
        f'        harness: "{HARNESS}"',
        f"      ci95: [{fmt(lo)}, {fmt(hi)}]",
        "      fit_range:",
        *[f"        - {{dim: {d}, low: {a:.1f}, high: {b:.1f}}}" for d, (a, b) in f["span"].items()],
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def names(backend: str) -> tuple[str, ...]:
    return tuple(f"gemm_{p}_fp8_grouped_{backend}" for p in ("eps_max", "m_half", "floor"))


def render(text: str, fits: list[dict]) -> str:
    blocks = cb.split(text)
    owned = {(n, cb.scope_key(f"{{hardware: [{f['chip']}]}}")) for f in fits for n in names(f["backend"])}
    family = {n for b in BACKENDS for n in names(b)}
    for blk in blocks:
        n = cb.name_of(blk)
        if n in family and (n, cb.scope_key(cb.scope_of(blk) or "{}")) not in owned:
            raise SystemExit(f"{n} {cb.scope_of(blk)} is not an entry this writer maintains")
    for f in fits:
        scope = f"{{hardware: [{f['chip']}]}}"
        for n in names(f["backend"]):
            cb.upsert(blocks, entry_lines(n, f),
                      after=lambda b: (cb.name_of(b) or "").startswith("gemm_")
                      and cb.scope_key(cb.scope_of(b) or "{}") == cb.scope_key(scope))
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true", help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])
    fits = []
    for sku, chip in PARTS:
        for backend in BACKENDS:
            f = fit_part(Path(args.data), Path(args.catalog), sku, chip, backend)
            fits.append(f)
            print(f"{chip:5} {backend:18} eps_max={f['eps']:.2f} m_half={f['mh']:4d} floor={f['floor_us']:5.1f}us "
                  f"train {f['err']:.3f}x signed {f['signed']:.3f} (n={f['n']}, {f['geoms']} geometries)  "
                  f"before {f['before'][0]:.3f}x signed {f['before'][1]:.3f}  hold-out {f['holdout'][0]:.3f}x "
                  f"signed {f['holdout'][1]:.3f} (n={f['holdout_n']})")
    old = SET_PATH.read_text()
    new = render(old, fits)
    if args.check:
        if new != old:
            print(f"{SET_PATH.name} is out of date; run without --check")
            return 1
        return 0
    SET_PATH.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
