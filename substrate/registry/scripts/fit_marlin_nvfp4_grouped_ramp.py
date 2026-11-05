#!/usr/bin/env python3
"""The Marlin NVFP4 W4A16 grouped ramp on Hopper, and whether the byte convention matters
(inference-sim/blis-registry#51 part 3).

    python scripts/fit_marlin_nvfp4_grouped_ramp.py --data <dir> [--check]
    (or set OPERATORX_DATA; <dir> holds moe_MiniMax-M3.json)

THE QUESTION. gemm_*_w4a16_mxfp4_grouped on h100/h200 (fit_marlin_grouped_ramp.py) is fitted on
vLLM's Marlin MXFP4 W4A16 experts: E8M0 scales per 32 weights, 0.53125 stored bytes per weight.
An NVFP4 checkpoint on Hopper runs the same Marlin MoE template with FP8 E4M3 scales per 16
weights, 0.5625 B/weight, plus twice the scale reads in the dequant. blis-latency-kernel
(formatFit, muse-spark/pp-and-fabric) reads `gemm_*_w4a16_nvfp4_grouped` for that case and
falls back to the MXFP4 ramp. Does the convention change the price, and does an NVFP4 ramp
of its own price NVFP4 Marlin better?

THE DATA. InferenceX OperatorX, op moe, workload MiniMax-M3:
https://inferencex.semianalysis.com/api/v1/operatorx/compare?op=moe&workload=moe:MiniMax-M3
(response sha256 in SOURCE_SHA below; runs 36292796863 on h100 and 36292840239 on h200,
InferenceX 8f03d279200eedfae08027615fdeca9f43b81cc1, image vllm/vllm-openai:nightly-cd10ed6f9f6b
37a8ace9cf380007e66fe12ec0c3, experimental/operatorx/containers.toml). Rows: MarlinExperts,
w1/w2 e2m1 with e4m3 scales per 16 (NVFP4), tp 1, hidden 6144, inter 3072, top-4 of 128, one
shared expert in MXFP8 (e4m3, ue8m0 per 32), natural routing; CUDA-graph rows at
T 1-512, eager rows at T 768-65536. Latency is the whole MoE block (router, routed experts,
shared expert) through vLLM's own FusedMoE (operatorx/runners/common/vllm/moe.py).

WHY THE NIGHTLY MEASURES v0.31.0. cd10ed6 is 765 commits behind v0.31.0 and 0 ahead. On the
Marlin MoE path: csrc/libtorch_stable/moe/marlin_moe_wna16/{kernel.h, marlin_template.h,
ops.cu, generate_kernels.py} and quantization/marlin/{dequant.h, marlin_mma.h,
marlin_dtypes.cuh} are byte-identical; quantization/marlin/marlin.cuh and moe/moe_wna16_utils.h
only lose helpers no Marlin MoE source calls (cp_async4_ca_pred, float22num2); fused_moe/
experts/marlin_moe.py changes docstrings and takes its workspace from get_marlin_workspace,
which allocates the same SMs x MARLIN_MAX_BLOCKS_PER_SM (4) locks (marlin_utils.py).

THE BLOCK MODEL, fixed (no parameter fitted in it), as blis-latency-kernel composes a MoE
layer (step.go): the shared expert is a dense term, max(SM, HBM), priced on the committed
gemm_*_w8a16_mxfp8 ramp (MXFP8 on SM90 runs Marlin W8A16) at 1.03125 B/weight; the routed
experts are SM + HBM, the ramp at the BF16 peak over tokens x top_k x 6 x h x i FLOPs plus
ExpertsTouched experts' stored bytes at BwPeak x hbm_derate; block = shared + routed. The
router GEMM (fp32, 6144 x 128) is omitted (below 0.1% of the routed FLOPs). Imbalance 1:
the rows route naturally on random inputs, which this model does not price.

CANDIDATES, and THE SPLIT, declared here before any of them was scored. Exploration had read
one aggregate: candidate A on the h200 graph rows, with the shared expert mispriced, at a
-15% signed median. Nothing else.
  A  today's price: gemm_*_w4a16_mxfp4_grouped, routed bytes 0.5625 (NVFP4 StoredBytes).
  B  the convention ignored: the same ramp, routed bytes 0.53125.
  C  the MXFP4 ramp refitted at 0.5625 B/weight on its own AISimulate rows (as committed
     beside it: fit_marlin_grouped_ramp.py eps05/mh05 machinery), so it composes with the
     NVFP4 byte count.
  D  a ramp fitted on these NVFP4 rows themselves, through the block model.
  Split: rows sorted by tokens; every second token count (odd index) is held out, per part;
  D fits the rest. A, B, C fit nothing here and are scored on the same held-out rows.
  Band: the convention matters if, on held-out rows, |price(A) / price(B) - 1| has a median
  above 5% (one third of the 15% band the collective fits use). D lands as
  gemm_*_w4a16_nvfp4_grouped only if its held-out median |rel err| beats A's by at least 2
  points on both parts; otherwise the MXFP4 fallback stands and this is parked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
CATALOG = Path(os.environ.get("BLIS_CATALOG") or HERE.parents[1] / "catalog")
URL = "https://inferencex.semianalysis.com/api/v1/operatorx/compare?op=moe&workload=moe:MiniMax-M3"
SOURCE_SHA = "ea8fb4078168319c05f347e89a646ced9b7c879f276f8b5b9936066f0b097d89"
RUNS = {"h100": "36292796863", "h200": "36292840239"}
GEOM = (6144, 3072, 4, 128)          # hidden, inter, top_k, experts
SHARED_INTER = 3072
B_NVFP4, B_MXFP4, B_MXFP8 = 0.5 + 1 / 16, 0.5 + 1 / 32, 1 + 1 / 32
NAMES = ("gemm_eps_max_w4a16_nvfp4_grouped", "gemm_m_half_w4a16_nvfp4_grouped")
CONVENTION_BAND, LAND_MARGIN = 0.05, 0.02
EPS_GRID = np.round(np.arange(0.001, 1.0005, 0.001), 3)
MH_GRID = np.arange(1, 4097)


def committed(name: str, chip: str) -> float:
    doc = yaml.safe_load(SET_PATH.read_text())
    hits = [b["value"] for e in doc["coefficients"] for n, b in e.items()
            if n == name and b["scope"] == {"hardware": [chip]}]
    if len(hits) != 1:
        raise SystemExit(f"{chip}: expected one part-wide {name}, found {len(hits)}")
    return float(hits[0])


def rows(path: Path, chip: str) -> list[tuple[int, bool, float]]:
    d = json.loads(path.read_text())
    m = d["measurements"][chip]
    out = []
    for i, c in enumerate(d["cases"]):
        k = json.loads(c["key"])
        w, e = k["experts"]["w1"], k["experts"]
        if m["status"][i] != "ok" or d["kernels"][m["kernel"][i]] != "MarlinExperts":
            continue
        if (w["dtype"], w["scale"]["dtype"], w["scale"]["group"][1]) != ("e2m1", "e4m3", 16):
            continue
        if k.get("parallel") or m["runId"][i] != RUNS[chip]:
            continue
        if (k["hidden"], e["inter"], e["top_k"], e["num"]) != GEOM:
            raise SystemExit(f"{chip}: geometry moved: {k['hidden'], e['inter'], e['top_k'], e['num']}")
        sh = k["shared"]
        if (sh["count"], sh["inter"], sh["w1"]["dtype"], sh["w1"]["scale"]["dtype"]) != (
                1, SHARED_INTER, "e4m3", "ue8m0"):
            raise SystemExit(f"{chip}: shared expert moved: {sh}")
        out.append((k["tokens"], bool(m["cudaGraph"][i]), m["latencyUs"][i] * 1e-6))
    return sorted(out)


class Part:
    def __init__(self, chip: str):
        hw = yaml.safe_load((CATALOG / "hardware" / f"{chip}.yaml").read_text())
        self.peak = float(hw["TFlopsPeak"]) * 1e12
        self.bw = float(hw["BwPeakTBs"]) * 1e12 * committed("hbm_derate", chip)
        self.sh = (committed("gemm_eps_max_w8a16_mxfp8", chip),
                   committed("gemm_m_half_w8a16_mxfp8", chip))
        self.mx = (committed("gemm_eps_max_w4a16_mxfp4_grouped", chip),
                   committed("gemm_m_half_w4a16_mxfp4_grouped", chip))

    def price(self, T, eps, mh, b):
        h, i, k, e = GEOM
        T = np.asarray(T, float)
        eff = lambda ep, m: ep * T / (T + m)
        sh = np.maximum(T * 6 * h * SHARED_INTER / (self.peak * eff(*self.sh)),
                        3 * h * SHARED_INTER * B_MXFP8 / self.bw)
        touched = e * (1 - ((e - k) / e) ** T)
        routed = T * k * 6 * h * i / (self.peak * eff(eps, mh)) + touched * 3 * h * i * b / self.bw
        return sh + routed


def fit(part: Part, T, S) -> tuple[float, int]:
    best = (math.inf, 0.0, 0)
    for eps in EPS_GRID:
        pred = np.array([part.price(T, eps, m, B_NVFP4) for m in MH_GRID])
        err = np.mean(np.abs(np.log(pred / S[None, :])), axis=1)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(eps), int(MH_GRID[j]))
    if best[1] in (EPS_GRID[0], EPS_GRID[-1]) or best[2] in (MH_GRID[0], MH_GRID[-1]):
        raise SystemExit(f"fit not interior: {best}")
    return best[1], best[2]


def refit_mxfp4_at_nvfp4_bytes(chip: str) -> tuple[float, int]:
    """Candidate C: the committed MXFP4 ramp's own fit, re-run at 0.5625 B/weight."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fmg", HERE / "fit_marlin_grouped_ramp.py")
    fmg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmg)
    sku = {"h100": "h100_sxm", "h200": "h200_sxm"}[chip]
    data = Path(os.environ.get("AISIMULATE_DATA", fmg.DEFAULT_DATA))
    hw_peak, bw_peak = fmg.chip_facts(CATALOG, chip)
    bw = bw_peak * committed("hbm_derate", chip)
    train = [r for r in fmg.load(data, sku, fmg.FIT_KERNEL, fmg.FIT_DTYPE)
             if r["moe_tp_size"] == 1 and r["moe_ep_size"] == 1]
    fmg.BYTES_PER_WEIGHT[fmg.FIT_DTYPE] = B_NVFP4
    eps, mh, _ = fmg.fit(*fmg.terms(train, fmg.FIT_DTYPE, bw), hw_peak)
    return eps, mh


def stats(p, y):
    r = p / y - 1
    return {"med": float(np.median(np.abs(r))), "signed": float(np.median(r)),
            "max": float(np.max(np.abs(r))), "n": len(r)}


def run_part(path: Path, chip: str, with_c: bool) -> dict:
    part = Part(chip)
    rs = rows(path, chip)
    if len(rs) < 24:
        raise SystemExit(f"{chip}: {len(rs)} NVFP4 Marlin rows")
    T = np.array([t for t, _, _ in rs], float)
    S = np.array([s for _, _, s in rs])
    held = np.arange(len(rs)) % 2 == 1
    out = {"chip": chip, "n": len(rs), "n_fit": int((~held).sum()), "n_held": int(held.sum())}
    pa = part.price(T, *part.mx, B_NVFP4)
    pb = part.price(T, *part.mx, B_MXFP4)
    out["convention"] = float(np.median(np.abs(pa[held] / pb[held] - 1)))
    out["convention_max"] = float(np.max(np.abs(pa[held] / pb[held] - 1)))
    out["A"], out["B"] = stats(pa[held], S[held]), stats(pb[held], S[held])
    if with_c:
        c = refit_mxfp4_at_nvfp4_bytes(chip)
        out["C_ramp"] = c
        out["C"] = stats(part.price(T[held], *c, B_NVFP4), S[held])
    eps, mh = fit(part, T[~held], S[~held])
    out["D_ramp"] = (eps, mh)
    out["D_fit"] = stats(part.price(T[~held], eps, mh, B_NVFP4), S[~held])
    out["D"] = stats(part.price(T[held], eps, mh, B_NVFP4), S[held])
    out["mx"] = part.mx
    return out


def _s(x):
    return f"median {x['med']:.1%}, signed {x['signed']:+.1%}, max {x['max']:.1%} (n {x['n']})"


def entry_lines(name, f):
    eps, mh = f["D_ramp"]
    cite = (f"InferenceX OperatorX {URL} (response sha256 {SOURCE_SHA}), run {RUNS[f['chip']]}, "
            f"MarlinExperts NVFP4 (e2m1, e4m3 scale per 16), tp 1, MiniMax-M3 block hidden 6144 "
            f"inter 3072 top-4 of 128 with one MXFP8 shared expert, {f['n']} token counts")
    if name.startswith("gemm_eps_max"):
        val, units = f"{eps:.3f}", "dimensionless"
        rat = (f"Asymptotic fraction of the dense BF16 peak vLLM's Marlin NVFP4 W4A16 grouped GEMM "
               f"reaches on this part (FP8 E4M3 scales per 16, 0.5625 stored B/weight). Fitted "
               f"with gemm_m_half_w4a16_nvfp4_grouped through the kernel's MoE composition: a "
               f"shared expert priced as a dense max(SM, HBM) on gemm_*_w8a16_mxfp8 at 1.03125 "
               f"B/weight plus the routed SM + HBM sum, on every second token count; held out "
               f"on the rest: {_s(f['D'])}, against the MXFP4 ramp the kernel otherwise falls "
               f"back to ({f['mx'][0]} / {f['mx'][1]:g}) at {_s(f['A'])}. The vLLM nightly "
               f"cd10ed6 these rows ran has v0.31.0's Marlin MoE kernel byte for byte "
               f"(scripts/fit_marlin_nvfp4_grouped_ramp.py states the diff). One geometry, "
               f"natural routing, priced at imbalance 1. Reproduce with "
               f"scripts/fit_marlin_nvfp4_grouped_ramp.py.")
    else:
        val, units = str(mh), "tokens"
        rat = (f"The token count at which the Marlin NVFP4 W4A16 grouped ramp reaches half its "
               f"asymptote, fitted jointly with gemm_eps_max_w4a16_nvfp4_grouped. Reproduce with "
               f"scripts/fit_marlin_nvfp4_grouped_ramp.py.")
    return [f"  - {name}:", f"      value: {val}", f"      units: {units}",
            "      method: measured", "      fitted: true",
            f"      scope: {{hardware: [{f['chip']}]}}", "      sources:",
            f'        - {{kind: model, cite: "{cite}", role: primary}}',
            "      rationale: >", f"        {rat}"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("OPERATORX_DATA"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-refit", action="store_true",
                    help="skip candidate C (it needs the AISimulate tree)")
    args = ap.parse_args(argv[1:])
    if not args.data:
        print("no OperatorX data: pass --data or set OPERATORX_DATA", file=sys.stderr)
        return 2
    path = Path(args.data) / "moe_MiniMax-M3.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        print(f"refusing: {path} is not the cited response (sha256 {SOURCE_SHA[:12]})",
              file=sys.stderr)
        return 3
    fits = {c: run_part(path, c, not args.no_refit) for c in ("h100", "h200")}
    for c, f in fits.items():
        print(f"{c}: {f['n']} rows, fit {f['n_fit']}, held {f['n_held']}")
        print(f"  convention |A/B - 1| on held rows: median {f['convention']:.1%}, max "
              f"{f['convention_max']:.1%} (band {CONVENTION_BAND:.0%})")
        print(f"  A mxfp4 ramp @0.5625 : {_s(f['A'])}")
        print(f"  B mxfp4 ramp @0.53125: {_s(f['B'])}")
        if "C" in f:
            print(f"  C refit @0.5625 {f['C_ramp'][0]:.3f}/{f['C_ramp'][1]}: {_s(f['C'])}")
        print(f"  D nvfp4 fit {f['D_ramp'][0]:.3f}/{f['D_ramp'][1]}: fit {_s(f['D_fit'])}; "
              f"held {_s(f['D'])}")
    lands = all(f["D"]["med"] <= f["A"]["med"] - LAND_MARGIN for f in fits.values())
    print("D lands" if lands else "D does not beat A by the declared margin on both parts: parked")
    before = SET_PATH.read_text(encoding="utf-8")
    blocks = cb.split(before)
    if lands:
        for c, f in fits.items():
            for n in NAMES:
                cb.upsert(blocks, entry_lines(n, f),
                          after=lambda b, c=c: (cb.name_of(b) or "").startswith("gemm_")
                          and cb.scope_key(cb.scope_of(b) or "{}") == cb.scope_key(
                              f"{{hardware: [{c}]}}"))
    after = cb.join(blocks)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
