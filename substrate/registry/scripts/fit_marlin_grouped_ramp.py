#!/usr/bin/env python3
"""Fit the GEMM ramp of vLLM's Marlin weight-only grouped GEMM (MXFP4 W4A16 experts).

    python scripts/fit_marlin_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_marlin_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS (inference-sim/blis-registry#38). On SM90 vLLM v0.31.0 runs MXFP4
routed experts through Marlin:
  * an OCP-MXFP4 checkpoint (DeepSeek-V4.1-Flash, `Mxfp4MoEMethod`): the backend order is
    FlashInfer-TRTLLM, DeepGEMM, Marlin, and the first two require SM100
    (`model_executor/layers/fused_moe/oracle/mxfp4.py`, `_get_priority_backends`);
  * a compressed-tensors MXFP4 checkpoint (`CompressedTensorsW4A4Mxfp4MoEMethod`, Kimi-K3):
    `CutlassExpertsMxfp4` when `_supports_current_device()` -- SM100, SM110 or SM120 --
    and `MarlinExperts` otherwise (`compressed_tensors_moe_w4a4_mxfp4.py`).
So on h200 and h100 (the same SM90 GH100 die) both resolve to Marlin by default. On Blackwell stock v0.31.0 does not select
Marlin, but Marlin is still what some measured Blackwell deployments ran: the Kimi-K3
preview build (`vllm 0.1.dev19262`), whose CUTLASS activation whitelist rejected SiTU, served
those experts on MarlinExperts W4A16 on every SM (AISimulate `collector/vllm/collect_moe.py`),
and the FPM dataset's own runtime-dtype annotation records that build's GB300 runs as
`moe_quant_mode: w4a16_mxfp4` (`fpm_identity_correction`, policy
`runtime-dtype-annotation-v1`, in the Kimi-K3 gb300 `0.1.dev19262/tep8` metadata). That run is
the only end-to-end evidence for the KDA pair (blis-registry#41), so gb300 carries the family too.

A ramp here is a property of (kernel, part), not a claim about which kernel a deployment
selects: a consumer applies it only where the deployment's MoE backend IS Marlin W4A16, and
must not apply it to a stock v0.31.0 Blackwell deployment, which runs CutlassExpertsMxfp4
(W4A4) or a TRTLLM/DeepGEMM backend.

Marlin is WEIGHT-ONLY: the four-bit weights are dequantized in registers and multiplied
against BF16 activations with `mma.sync` (`csrc/libtorch_stable/quantization/marlin/
marlin_mma.h`), so its compute rate is a fraction of the part's BF16 peak, not of an FP8 or
FP4 one. No committed ramp describes it: bf16, fp8, fp8_block and nvfp4 are all dense
tensor-core paths.

THE QUANTITY. `eff(M) = eps_max * M / (M + M_half)`, as a fraction of the catalog's dense
BF16 peak (`TFlopsPeak`), CONDITIONED ON THE COMPOSITION blis-latency-kernel applies to a
routed-expert term (`kernel.go`, the routed SM and HBM halves are SUMMED):

    t = T*top_k*6*h*i / (BF16_peak * eff(T))  +  touched(T) * 3*h*i*B / (BwPeak * hbm_derate)
    touched(T) = E * (1 - ((E - top_k)/E)^T)          # price.ExpertsTouched

with M = T, the step's token count, as the kernel evaluates every ramp. The FLOPs are the
rank's OWN work: vLLM shards each expert's intermediate width across tensor-parallel ranks
(`intermediate_size_per_partition`), so at moe_tp > 1 a rank computes inter / tp of each
expert. A consumer that charges a rank the whole expert's FLOPs prices this ramp tp times
too slow; blis-latency-kernel#44 divides by tp on the weight-only ramps for that reason,
keeping its pure-TP over-charge only on the native ones (blis-latency-kernel#47). B is the
STORED
bytes per weight: 0.5 for the e2m1 values plus one E8M0 scale per 32, 0.53125. So the ramp
is the SM half that remains once the kernel's own memory half is charged, which is the
only way a fitted ramp composes back to the measured latency inside the kernel.

THE DATA. AISimulate `moe/vllm/0.25.0/moe_perf.parquet`, kernel_source
`vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu`, moe_dtype
`w4a16_mxfp4`: Kimi-K3's expert geometry (hidden 3584, inter 3072, top-16 of 896) with
SiTU's elementwise epilogue approximated as SiLU. The epilogue is an elementwise pass
between the two grouped GEMMs, so the approximation leaves the GEMMs the ramp describes
untouched. Fitted on balanced routing at moe_tp = moe_ep = 1, 27 token counts from 1 to
16,384.

WHY vLLM 0.25.0 MEASURES THE v0.31.0 KERNEL. Between the two tags the Marlin MoE template
(`csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h`) changes only by removing
the runtime activation-ordering path (#54809): `has_act_order` was `group_blocks == 0`, a
compile-time constant, and MXFP4's group-32 instantiation never took that branch.

HELD OUT, AND WHAT IT SHOWS.
  * A DIFFERENT GEOMETRY AND WEIGHT TYPE. The same sweep carries vLLM's int4 Marlin MoE
    (`compressedtensorswna16marlinmoe_marlin_marlinexperts`, hidden 7168, inter 2048,
    top-8 of 384), collected with the collector's default group size of 128
    (`collect_moe.py`, `module_config.get("group_size", 128)`; the Kimi-K2.5 case passes
    none). Same Marlin template, different expert shape, so its error is the evidence that
    the asymptote is the kernel's rather than one shape's. Its geometry is asserted.
  * EXPERT AND TENSOR PARALLELISM ARE A FORM LIMIT, NOT THIS FIT'S. The error grows with
    moe_ep and moe_tp, because the kernel's routed term evaluates the ramp at the STEP's
    token count and carries no per-launch floor. `form_check` prints the same growth for
    the committed BF16 ramp on vLLM's BF16 Triton experts, so it is a property of the
    composition and is reported, not tuned away.

LANE. vLLM, the engine BLIS prices. Units: `latency` is MILLISECONDS in every AISimulate
parquet.
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
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
# The repository's own catalog, so the writer runs in CI; BLIS_CATALOG overrides it
# with another catalog directory.
DEFAULT_CATALOG = (os.environ.get("BLIS_CATALOG")
                   or str(HERE.parents[1] / "catalog"))

COLLECTION = "vllm/0.25.0"
FIT_KERNEL = "vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu"
FIT_DTYPE = "w4a16_mxfp4"
HOLDOUT_KERNEL = "vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts"
HOLDOUT_DTYPE = "int4_wo"
HOLDOUT_GEOMETRY = (7168, 2048, 8, 384)
# The BF16 baseline the form check scores the committed BF16 ramp against.
FORM_KERNEL = "vllm_unquantizedfusedmoe_triton_tritonexperts"
FORM_DTYPE = "bfloat16"

# Stored bytes per weight. MXFP4: e2m1 nibbles plus one E8M0 byte per 32 values (OCP MX
# block size). int4 holdout: nibbles plus one BF16 scale per group of 128. BF16: 2.
BYTES_PER_WEIGHT = {FIT_DTYPE: 0.5 + 1 / 32, HOLDOUT_DTYPE: 0.5 + 2 / 128, FORM_DTYPE: 2.0}

# The parts this family is committed for, and why each (see the module docstring): h200
# and h100, where stock v0.31.0 selects Marlin for MXFP4 experts; gb300, where the Kimi-K3
# FPM run the KDA validation (blis-registry#41) scores against ran Marlin W4A16.
PARTS = [("h200_sxm", "h200"), ("gb300", "gb300"), ("h100_sxm", "h100"),
         ("b200_sxm", "b200"), ("b300_sxm", "b300")]

# The landing gate for a part added after the first three (inference-sim/blis-registry#51
# part 2: b200, b300), declared before their fits were run: the held-out int4 geometric error
# must not exceed 1.30x, the worst the committed parts reach (gb300 1.266x). A part over it
# is printed and not written.
LATE_PARTS = {"b200", "b300"}
HOLDOUT_GATE = 1.30

# A part measured beside a sibling on the same die, whose latencies are compared row for
# row so the rationale can say what a difference in the fitted ramp is and is not.
SIBLING = {"h100": "h200"}

# Where Marlin is the default on the part, and where it is not.
WHEN = {
    "h200": ("On SM90 v0.31.0 runs MXFP4 routed experts on Marlin (Mxfp4MoEMethod's TRTLLM "
             "and DeepGEMM backends, and the compressed-tensors CutlassExpertsMxfp4, all "
             "require SM100+)."),
    "h100": ("On SM90 v0.31.0 runs MXFP4 routed experts on Marlin (Mxfp4MoEMethod's TRTLLM "
             "and DeepGEMM backends, and the compressed-tensors CutlassExpertsMxfp4, all "
             "require SM100+)."),
    "b200": ("On this part stock v0.31.0 does NOT select Marlin for MXFP4 experts "
             "(CutlassExpertsMxfp4, TRTLLM or DeepGEMM), but it DOES run Kimi-K2.5's WNA16 int4 "
             "experts on Marlin on Blackwell (int_wna16.py:132-141; VLLM_USE_FLASHINFER_MOE_INT4 "
             "defaults to 0, envs.py:221), so apply this where the deployment's MoE backend is "
             "Marlin W4A16 (blis-registry#51 part 2)."),
    "b300": ("On this part stock v0.31.0 does NOT select Marlin for MXFP4 experts "
             "(CutlassExpertsMxfp4, TRTLLM or DeepGEMM), but it DOES run Kimi-K2.5's WNA16 int4 "
             "experts on Marlin on Blackwell (int_wna16.py:132-141; VLLM_USE_FLASHINFER_MOE_INT4 "
             "defaults to 0, envs.py:221), so apply this where the deployment's MoE backend is "
             "Marlin W4A16 (blis-registry#51 part 2)."),
    "gb300": ("On this part stock v0.31.0 does NOT select Marlin -- CutlassExpertsMxfp4 "
              "(W4A4) for a compressed-tensors MXFP4 checkpoint, TRTLLM or DeepGEMM for OCP "
              "MXFP4 -- so apply this only where the deployment's MoE backend is Marlin W4A16: "
              "the Kimi-K3 preview build vllm 0.1.dev19262, which the FPM dataset's "
              "runtime-dtype annotation records as moe_quant_mode w4a16_mxfp4 on gb300."),
}

NAMES = ("gemm_eps_max_w4a16_mxfp4_grouped", "gemm_m_half_w4a16_mxfp4_grouped")

EPS_GRID = np.round(np.arange(0.001, 1.0005, 0.001), 3)
MHALF_GRID = np.arange(1, 4097)


def chip_facts(catalog: Path, chip: str) -> tuple[float, float]:
    """(BF16 dense peak in FLOP/s, datasheet HBM bandwidth in B/s), from the catalog."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts["TFlopsPeak"]) * 1e12, float(facts["BwPeakTBs"]) * 1e12


def committed(name: str, chip: str) -> float:
    """The committed part-wide entry for this chip -- scoped to it alone, no model."""
    doc = yaml.safe_load(SET_PATH.read_text())
    hits = [body["value"] for e in doc["coefficients"] for n, body in e.items()
            if n == name and body["scope"] == {"hardware": [chip]}]
    if len(hits) != 1:
        raise SystemExit(f"{chip}: expected one part-wide {name}, found {len(hits)}")
    return float(hits[0])


def load(data: Path, sku: str, kernel: str, dtype: str) -> list[dict]:
    path = data / sku / "moe" / COLLECTION / "moe_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"{sku}: {path} is absent; the pinned collection is required")
    t = pq.read_table(path).to_pydict()
    rows = []
    for i in range(len(t["latency"])):
        if (t["kernel_source"][i] != kernel or t["moe_dtype"][i] != dtype
                or t["distribution"][i] != "balanced"):
            continue
        if not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0):
            raise SystemExit(f"{sku}: {kernel} row {i} has latency {t['latency'][i]}")
        rows.append({k: t[k][i] for k in (
            "num_tokens", "hidden_size", "inter_size", "topk", "num_experts",
            "moe_tp_size", "moe_ep_size")} | {"s": t["latency"][i] * 1e-3})
    return rows


def geometry(rows: list[dict]) -> tuple[int, int, int, int]:
    geoms = {(r["hidden_size"], r["inter_size"], r["topk"], r["num_experts"]) for r in rows}
    if len(geoms) != 1:
        raise SystemExit(f"expected one expert geometry, got {sorted(geoms)}")
    return geoms.pop()


def terms(rows: list[dict], dtype: str, bw: float) -> tuple[np.ndarray, ...]:
    """Per row: tokens, routed FLOPs, the kernel's memory-half seconds, measured seconds."""
    T, F, H, S = [], [], [], []
    for r in rows:
        tp, ep = r["moe_tp_size"], r["moe_ep_size"]
        h, i = r["hidden_size"], r["inter_size"] / tp
        k, e = r["topk"], r["num_experts"]
        tokens = r["num_tokens"]
        T.append(tokens)
        F.append(tokens * k / ep * 6 * h * i)
        touched = (e / ep) * (1 - ((e - k) / e) ** tokens)
        H.append(touched * 3 * h * i * BYTES_PER_WEIGHT[dtype] / bw)
        S.append(r["s"])
    return tuple(np.asarray(x, dtype=float) for x in (T, F, H, S))


def errors(T, F, H, S, peak, eps, mh) -> tuple[float, float]:
    """(geometric mean |error|, geometric mean signed ratio), as the committed fits report."""
    r = np.log(F / (peak * eps * T / (T + mh)) + H) - np.log(S)
    return math.exp(float(np.mean(np.abs(r)))), math.exp(float(np.mean(r)))


def fit(T, F, H, S, peak) -> tuple[float, int, float]:
    """Exhaustive grid over (eps_max, M_half), minimising the geometric error."""
    mh = MHALF_GRID[:, None].astype(float)
    best = (math.inf, 0.0, 0)
    for eps in EPS_GRID:
        pred = F[None, :] / (peak * eps * T[None, :] / (T[None, :] + mh)) + H[None, :]
        err = np.mean(np.abs(np.log(pred / S[None, :])), axis=1)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(eps), int(MHALF_GRID[j]))
    err, eps, mh = math.exp(best[0]), best[1], best[2]
    # A fit pinned to its grid's edge is the grid's answer, not the data's.
    if not math.isfinite(err) or eps in (EPS_GRID[0], EPS_GRID[-1]) \
            or mh in (MHALF_GRID[0], MHALF_GRID[-1]):
        raise SystemExit(f"fit is not interior: eps_max={eps}, m_half={mh}, err={err}")
    return eps, mh, err


def form_check(data: Path, sku: str, peak: float, bw: float, chip: str) -> dict:
    """The committed BF16 ramp, scored on vLLM's BF16 Triton experts across EP widths."""
    eps, mh = committed("gemm_eps_max_bf16", chip), committed("gemm_m_half_bf16", chip)
    by_geom: dict = {}
    for r in load(data, sku, FORM_KERNEL, FORM_DTYPE):
        by_geom.setdefault((r["hidden_size"], r["inter_size"], r["topk"],
                            r["num_experts"]), []).append(r)
    geom = min(by_geom)  # deterministic: the smallest geometry the sweep carries
    out = {}
    for ep in (1, 8, 32):
        cell = [r for r in by_geom[geom] if r["moe_tp_size"] == 1 and r["moe_ep_size"] == ep]
        if cell:
            out[ep] = errors(*terms(cell, FORM_DTYPE, bw), peak, eps, mh)[0]
    return {"geom": geom, "by_ep": out}


def fit_part(data: Path, catalog: Path, sku: str, chip: str) -> dict:
    peak, bw_peak = chip_facts(catalog, chip)
    bw = bw_peak * committed("hbm_derate", chip)
    train = [r for r in load(data, sku, FIT_KERNEL, FIT_DTYPE)
             if r["moe_tp_size"] == 1 and r["moe_ep_size"] == 1]
    if len(train) < 20:
        raise SystemExit(f"{chip}: only {len(train)} training rows")
    geom = geometry(train)
    T, F, H, S = terms(train, FIT_DTYPE, bw)
    eps, mh, err = fit(T, F, H, S, peak)
    _, signed = errors(T, F, H, S, peak, eps, mh)

    hold = [r for r in load(data, sku, HOLDOUT_KERNEL, HOLDOUT_DTYPE)
            if r["moe_tp_size"] == 1 and r["moe_ep_size"] == 1]
    if len(hold) < 20 or geometry(hold) != HOLDOUT_GEOMETRY:
        raise SystemExit(f"{chip}: the int4 Marlin holdout is missing or has moved "
                         f"({len(hold)} rows); refusing to commit an unvalidated ramp")
    ho = errors(*terms(hold, HOLDOUT_DTYPE, bw), peak, eps, mh)
    # Sensitivity to the byte count: what the fit becomes at payload-only MXFP4 bytes,
    # which is what a consumer that omits E8M0 scales charges.
    saved = BYTES_PER_WEIGHT[FIT_DTYPE]
    BYTES_PER_WEIGHT[FIT_DTYPE] = 0.5
    try:
        eps05, mh05, _ = fit(*terms(train, FIT_DTYPE, bw), peak)
    finally:
        BYTES_PER_WEIGHT[FIT_DTYPE] = saved
    return {"eps05": eps05, "mh05": mh05,"sku": sku, "chip": chip, "eps": eps, "mh": mh, "err": err, "signed": signed,
            "n": len(train), "geom": geom, "peak_tf": peak / 1e12, "bw_tb": bw_peak / 1e12,
            "latency": {r["num_tokens"]: r["s"] for r in train},
            "holdout": ho, "holdout_n": len(hold),
            "form": form_check(data, sku, peak, bw, chip)}


def sibling_note(f: dict) -> str:
    """For a part fitted beside a same-die sibling, what the two ramps' difference is."""
    s = f.get("sibling")
    if s is None:
        return ""
    ratios = [f["latency"][t] / s["latency"][t] for t in f["latency"]]
    return (f"THE SAME KERNEL AS ON {s['chip'].upper()}, ON THE SAME DIE: at the same "
            f"{len(ratios)} token counts this part's latency is {min(ratios):.2f}-"
            f"{max(ratios):.2f}x {s['chip']}'s, against the {s['bw_tb'] / f['bw_tb']:.2f}x a "
            f"kernel bound on its weight read would show at {f['bw_tb']:g} against "
            f"{s['bw_tb']:g} TB/s. So the ramp differs from {s['chip']}'s "
            f"({s['eps']:.3f} / {s['mh']}) because the memory half charged here is larger "
            f"at this part's lower BwPeak, and the ramp carries what it leaves -- not because "
            f"the compute is faster; the pair holds only with this part's own memory half. ")


def entry_lines(name: str, f: dict) -> list[str]:
    h, i, k, e = f["geom"]
    hh, hi, hk, he = HOLDOUT_GEOMETRY
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet, "
            f"{FIT_KERNEL}, moe_dtype {FIT_DTYPE}, balanced, moe_tp 1 / moe_ep 1, "
            f"{f['n']} token counts (hidden {h}, inter {i}, top-{k} of {e})")
    if name.startswith("gemm_eps_max"):
        value, units = f"{f['eps']:.3f}", "dimensionless"
        rat = (f"Asymptotic fraction of this part's dense BF16 peak ({f['peak_tf']:g} "
               f"TFLOPS) that vLLM's Marlin MXFP4 W4A16 grouped GEMM reaches. "
               f"{WHEN[f['chip']]} Marlin dequantizes the four-bit weights in registers and "
               f"multiplies against BF16 activations with mma.sync, so its rate is a "
               f"fraction of the BF16 peak, not of an FP8 or FP4 one. Fitted with "
               f"gemm_m_half_w4a16_mxfp4_grouped under the kernel's routed-expert "
               f"composition -- the SM half, over the rank's OWN routed FLOPs "
               f"(tokens x top_k / ep x 6 x hidden x inter / tp; fitted at tp = ep = 1), "
               f"summed with the distinct experts' stored weight "
               f"bytes (0.53125 B/weight, E8M0 scales included) at BwPeak x hbm_derate -- so "
               f"the ramp is what remains once the kernel's own memory half is charged. "
               f"Geometric error {f['err']:.3f}x (signed {f['signed']:.3f}); held out on "
               f"vLLM's int4 Marlin MoE (hidden {hh}, inter {hi}, top-{hk} of {he}, group "
               f"128; {f['holdout_n']} token counts) at {f['holdout'][0]:.3f}x, signed "
               f"{f['holdout'][1]:.3f}. {sibling_note(f)}COUPLED TO THE CONSUMER'S ROUTED TERM, as checked "
               f"against blis-latency-kernel#44 (483f6aa): compute and memory SUMMED; the "
               f"ramp evaluated at the step's token count; compute at the BF16 peak; memory "
               f"as ExpertsTouched experts' stored bytes at BwPeak x hbm_derate; imbalance and "
               f"SM derate 1 (balanced rows, no concurrent fetch); FLOPs the rank's own "
               f"(#44 divides routed FLOPs by tp on this ramp). #44 also scales routed FLOPs "
               f"and bytes by Marlin's round-up of inter / tp to a multiple of 128 (V4.1 at "
               f"TP8: 288 -> 384); fitted at tp 1, where that pad is 1, the fit has not checked "
               f"it -- it is the consumer's addition. A "
               f"change to any of those -- sum versus max, the ramp's argument, the byte "
               f"count -- needs a refit. The bytes are blis-schemas v0.3.0's StoredBytes for "
               f"MXFP4 (0.5 + 1/32), which #44 charges; a consumer charging payload only "
               f"(0.5, blis-latency-kernel main e9cf217) would need {f['eps05']:.3f} / "
               f"{f['mh05']}. vLLM's Python side changed since 0.25.0 too: #48144 sets the M "
               f"used for Marlin's block-size choice to ceil(M x local / global experts) under "
               f"EP -- inert at ep 1, where this is fitted, but the 0.25.0 rows at ep > 1 "
               f"may not describe v0.31.0 wherever it changes the chosen block size. Applies to the routed-expert term only; a "
               f"dense MXFP4 linear runs a different Marlin kernel and is not measured. "
               f"Reproduce with scripts/fit_marlin_grouped_ramp.py.")
    else:
        value, units = str(f["mh"]), "tokens"
        rat = (f"The step token count at which the Marlin MXFP4 W4A16 grouped ramp reaches "
               f"half its asymptote, fitted jointly with gemm_eps_max_w4a16_mxfp4_grouped "
               f"over {f['n']} token counts; the kernel evaluates it at the step's token "
               f"count, as it does every ramp. Reproduce with "
               f"scripts/fit_marlin_grouped_ramp.py.")
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
    """Replace each owned (name, chip) entry in place; insert a missing one after the
    chip's last gemm_* entry, so the family sits with the part's other ramps. An entry of
    this family on a part outside PARTS is refused rather than left unmaintained."""
    blocks = cb.split(text)
    owned = {cb.scope_key(f"{{hardware: [{chip}]}}") for chip in fits}
    for b in blocks:
        if cb.name_of(b) in NAMES and cb.scope_key(cb.scope_of(b) or "{}") not in owned:
            raise SystemExit(f"{cb.name_of(b)} {cb.scope_of(b)} is not a part this writer "
                             f"maintains; remove it or add the part to PARTS")
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
        if chip in LATE_PARTS and f["holdout"][0] > HOLDOUT_GATE:
            print(f"{chip:8} NOT written: held-out int4 {f['holdout'][0]:.3f}x exceeds the "
                  f"declared {HOLDOUT_GATE}x gate")
            continue
        fits[chip] = f
        form = ", ".join(f"ep {ep} {e:.2f}x" for ep, e in f["form"]["by_ep"].items())
        print(f"{chip:8} eps_max={f['eps']:.3f} m_half={f['mh']:4d}  "
              f"train {f['err']:.3f}x signed {f['signed']:.3f} (n={f['n']})  holdout int4 "
              f"{f['holdout'][0]:.3f}x signed {f['holdout'][1]:.3f} (n={f['holdout_n']})")
        print(f"{'':8} form check, committed BF16 ramp on BF16 Triton experts "
              f"{f['form']['geom']}: {form}")

    for chip, sib in SIBLING.items():
        fits[chip]["sibling"] = fits[sib]

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
