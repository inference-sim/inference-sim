#!/usr/bin/env python3
"""Fit the grouped-GEMM ramp and floor of vLLM's block-FP8 MoE on Blackwell (FlashInfer TRT-LLM).

    python scripts/fit_blackwell_fp8_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_blackwell_fp8_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS (inference-sim/blis-substrate#112, #121). Block-FP8 routed experts on a
Blackwell part had no grouped ramp: the kernel priced them on the dense FP8 linear ramp (gemm_*_fp8),
the whole expert's FLOPs on every tensor-parallel rank, with no floor. With the tensor-parallel
division (#112) that composition prices MiniMax-M2.5's routed layer at 7 us for one token on B200
against 28 us in AISimulate's table of the kernel, and 61 against 100 us at 16 tokens: 1.6-4x low
below 32 tokens. This family is the kernel's own ramp and floor, in the form
fit_hopper_fp8_grouped_ramp.py fits for Hopper (same composition, same grids):

    t = F + T*top_k*6*h*(i/tp) / (FP8_peak * eps_max * T / (T + M_half))
          + touched(T) * 3*h*(i/tp)*B / (BwPeak * hbm_derate)

THE DATA. AISimulate `moe/vllm/0.24.0/moe_perf.parquet` of each part, kernel_source
vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic (the only block-FP8 kernel that
collection measures; vLLM's FP8 MoE oracle selects FlashInfer TRT-LLM on SM100 for block-scaled
weights), moe_dtype fp8_block, balanced routing, moe_ep 1, 1 to 4,096 tokens, every moe_tp the
sweep carries, 17 expert geometries (MiniMax-M2.5's 3072 / 1536 / top-8 of 256 among them).
Balanced rows, because the kernel applies the routing imbalance on top of the ramp.

SPLIT, declared before the fit. Geometries sorted by (hidden, inter, top_k, experts); every
second one (odd index) is held out and the rest are fitted. Also held out, never fitted: the same
kernel's rows in the 0.30.0 collection (a later build) and, on B300, the 0.31.0 collection.
The landing rule: a part's entries are written only if its held-out geometric error is below
that of the dense fp8_block ramp under the same composition on the same rows and the fit is
interior to the grid (the knee M_half sits at the grid's lower edge, as Hopper's do: the rows
identify no knee above one token and the floor carries the low-token time).

GRADED ASSUMED, STATED. kernel_impls.json of the engine release lists no sm_100a build for the FP8
MoE backends (only sm_90a Triton and FlashInfer CUTLASS), so these entries cannot be keyed to a
code hash (VAL-8) and a fit to a kernel's table that is not keyed to it is not `measured`: they are
`assumed`, with the fit's interval and range recorded, and are regraded `measured` and keyed when
the engine release lists the build. An unkeyed entry applies to every implementation without having
been shown to match. The kernel reads them only where the plan
states no experts kernel (the engine's default selection); a plan that states Triton or CUTLASS
experts on a Blackwell part falls back to the fp8 ramp with an assumption recorded.

INTERVALS (VAL-10). `ci95` is fit_hopper_fp8_grouped_ramp.py's bootstrap of 200 row resamples.
LANE. vLLM. Units: `latency` is MILLISECONDS in every AISimulate parquet.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

_spec = importlib.util.spec_from_file_location("hopper_fp8", HERE / "fit_hopper_fp8_grouped_ramp.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

SET_PATH = h.SET_PATH
DEFAULT_DATA = h.DEFAULT_DATA
DEFAULT_CATALOG = h.DEFAULT_CATALOG
COLLECTION = "vllm/0.24.0"
LATER = ("vllm/0.30.0", "vllm/0.31.0")
KERNEL = "vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic"
SUFFIX = "fp8_grouped_trtllm"
MAX_TOKENS = 4096
PARTS = [("b200_sxm", "b200"), ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300")]
HARNESS = h.HARNESS
NAMES = tuple(f"gemm_{p}_{SUFFIX}" for p in ("eps_max", "m_half", "floor"))


def load(data: Path, sku: str, collection: str) -> dict[str, np.ndarray]:
    r = h.load(data, sku, collection, KERNEL, lambda tp, ep: ep == 1)
    keep = r["num_tokens"] <= MAX_TOKENS
    return {k: v[keep] for k, v in r.items()}


def geometry_split(r: dict[str, np.ndarray]) -> np.ndarray:
    """True for rows of a held-out geometry: every second one in sorted order."""
    key = list(zip(r["hidden_size"], r["inter_size"], r["topk"], r["num_experts"]))
    held = {g for j, g in enumerate(sorted(set(key))) if j % 2 == 1}
    return np.array([g in held for g in key])


def sub(r: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in r.items()}


def fit_part(data: Path, catalog: Path, sku: str, chip: str) -> dict:
    peak, bw_peak = h.chip_facts(catalog, chip)
    bw = bw_peak * h.committed("hbm_derate", chip)
    allrows = load(data, sku, COLLECTION)
    isheld = geometry_split(allrows)
    train, held = sub(allrows, ~isheld), sub(allrows, isheld)
    if len(train["s"]) < 500:
        raise SystemExit(f"{chip}: only {len(train['s'])} training rows")
    T, F, H, S = h.terms(train, bw)
    eps, mh, floor, err = h.fit(T, F, H, S, peak)
    _, signed = h.errors(h.predict(T, F, H, peak, eps, mh, floor), S)
    ci = h.bootstrap(T, F, H, S, peak, eps, mh, floor)
    de, dm = h.committed("gemm_eps_max_fp8_block", chip), h.committed("gemm_m_half_fp8_block", chip)
    ht = h.terms(held, bw)
    ho = h.errors(h.predict(ht[0], ht[1], ht[2], peak, eps, mh, floor), ht[3])
    ho_dense = h.errors(h.predict(ht[0], ht[1], ht[2], peak, de, dm, 0.0), ht[3])
    later = {}
    for coll in LATER:
        try:
            r = load(data, sku, coll)
        except SystemExit:
            continue
        if len(r["s"]) < 50:
            continue
        t = h.terms(r, bw)
        later[coll] = (len(r["s"]), h.errors(h.predict(t[0], t[1], t[2], peak, eps, mh, floor), t[3]),
                       h.errors(h.predict(t[0], t[1], t[2], peak, de, dm, 0.0), t[3]))
    geoms = len(set(zip(allrows["hidden_size"], allrows["inter_size"], allrows["topk"], allrows["num_experts"])))
    return {"sku": sku, "chip": chip, "eps": eps, "mh": int(mh), "floor_us": round(floor * 1e6, 1),
            "err": err, "signed": signed, "n": len(S), "geoms": geoms, "ci": ci, "peak_tf": peak / 1e12,
            "held": ho, "held_dense": ho_dense, "held_n": len(ht[3]), "later": later,
            "ok": ho[0] < ho_dense[0] and eps not in (h.EPS_GRID[0], h.EPS_GRID[-1]) and floor != h.FLOOR_GRID[-1],
            "span": {"tokens": (float(T.min()), float(T.max()))}}


def entry_lines(name: str, f: dict) -> list[str]:
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet, {KERNEL}, "
            f"moe_dtype fp8_block, balanced, moe_ep 1, 1-{MAX_TOKENS} tokens, {f['n']} training rows over the even half of {f['geoms']} "
            f"expert geometries, 1-{MAX_TOKENS} tokens")
    common = (f"vLLM's FlashInfer TRT-LLM grouped GEMM on block-FP8 experts, which AISimulate's collector measures "
              f"as vLLM's block-FP8 MoE on SM100. Fitted jointly (gemm_eps_max_{SUFFIX}, gemm_m_half_{SUFFIX}, "
              f"gemm_floor_{SUFFIX}) under the kernel's routed-expert composition: the floor plus the rank's OWN routed "
              f"FLOPs (tokens x top_k / ep x 6 x hidden x inter / tp) at the dense FP8 peak ({f['peak_tf']:g} TFLOPS) "
              f"times the ramp, summed with the distinct experts' stored bytes (1 + 4/16384 B/weight) at BwPeak x "
              f"hbm_derate. Held-out geometries (every second one): geometric error {f['held'][0]:.3f}x "
              f"(signed {f['held'][1]:.3f}) against the dense fp8_block ramp's {f['held_dense'][0]:.3f}x "
              f"(signed {f['held_dense'][1]:.3f}) under the same composition. "
              + "".join(f"The {c} collection ({n} rows of the same kernel, not fitted): {e[0]:.3f}x against the dense "
                        f"ramp's {d[0]:.3f}x. " for c, (n, e, d) in f["later"].items())
              + "GRADED ASSUMED, NOT MEASURED: the engine release's kernel_impls.json lists no sm_100a build for the FP8 MoE backends, so "
                "the entry cannot name a code hash (VAL-8); it is regraded measured and keyed when the build is listed. It applies to every implementation without having been "
                "shown to match; the kernel reads it only where the plan states no experts kernel. A change to the "
                f"composition -- the floor's place, sum versus max, the ramp's argument, the tensor-parallel division "
                f"-- needs a refit. Reproduce with scripts/fit_blackwell_fp8_grouped_ramp.py.")
    key = {"gemm_eps_max": "eps", "gemm_m_half": "mh", "gemm_floor": "floor"}[name.rsplit("_" + SUFFIX, 1)[0]]
    if key == "eps":
        value, units, rat = f"{f['eps']:.2f}", "dimensionless", f"Asymptotic fraction of the part's dense FP8 peak that {common}"
    elif key == "mh":
        value, units = str(f["mh"]), "tokens"
        lim = (" It sits at the grid's lower edge: the rows identify no knee above one token, and the floor carries "
               "the low-token time." if f["mh"] <= 1 else "")
        rat = f"The step token count at which the ramp reaches half its asymptote.{lim} {common}"
    else:
        value, units = f"{f['floor_us']:g}", "us_per_transfer"
        rat = (f"The per-layer device time of the grouped experts however few tokens reach them, which the ramp's work "
               f"term alone cannot carry. {common}")
    lo, hi = f["ci"][key]
    fmt = (lambda v: f"{v:.3f}") if key == "eps" else (lambda v: f"{v:g}")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: assumed",
        "      fitted: false",
        f"      scope: {{hardware: [{f['chip']}]}}",
        f"      ci95: [{fmt(lo)}, {fmt(hi)}]",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def render(text: str, fits: list[dict]) -> str:
    blocks = cb.split(text)
    owned = {(n, cb.scope_key(f"{{hardware: [{f['chip']}]}}")) for f in fits for n in NAMES}
    for blk in blocks:
        n = cb.name_of(blk)
        if n in NAMES and (n, cb.scope_key(cb.scope_of(blk) or "{}")) not in owned:
            raise SystemExit(f"{n} {cb.scope_of(blk)} is not an entry this writer maintains")
    for f in fits:
        scope = f"{{hardware: [{f['chip']}]}}"
        for n in NAMES:
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
        f = fit_part(Path(args.data), Path(args.catalog), sku, chip)
        print(f"{chip:5} eps_max={f['eps']:.2f} m_half={f['mh']:3d} floor={f['floor_us']:5.1f}us train {f['err']:.3f}x "
              f"held-out {f['held'][0]:.3f}x (dense {f['held_dense'][0]:.3f}x, n={f['held_n']}) "
              + " ".join(f"{c}: {e[0]:.3f}x (dense {d[0]:.3f}x, n={n})" for c, (n, e, d) in f["later"].items())
              + ("" if f["ok"] else "  REFUSED: the landing rule is not met"))
        if not f["ok"]:
            return 1
        fits.append(f)
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
