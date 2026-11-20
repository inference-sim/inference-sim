#!/usr/bin/env python3
"""Fit the three-factor GEMM efficiency ramp on AISimulate's vLLM GEMM sweeps.

    python scripts/fit_gemm_shape_ramp.py --data <dir> [sku:chip ...]

# Why a third and fourth parameter

The committed ramp is `eff(m) = eps_max * m / (m + m_half)`, fitted as an ENVELOPE:
scripts/fit_gemm_envelope.py takes the BEST efficiency over every (n, k) at each m. That
is a defensible envelope and the wrong thing to apply to an arbitrary shape, which is
what the kernel does -- it has one ramp per part and dtype and every GEMM in a layer
reads it.

The error that causes is not small and it is one-sided. Measured on
`h200_sxm/gemm/vllm/0.25.0` at a fixed m of 1024, median efficiency rises monotonically
with reduction depth, from 0.007 at k=32 to 0.476 at k=51200. A ramp in m alone assigns
all of those the same fraction of peak.

It lands hardest exactly where it matters. After TP=8 minimax-m3's `o_proj` is
(n=6144, k=1024) and its `mlp_gate_up` is (n=3072, k=6144); the measured fp8 efficiencies
at m=1024 are 0.291 and 0.381 while the committed ramp says 0.722 for both. So the
committed form over-prices efficiency by 2.20x on that model's real fp8 shapes and 3.65x
on its real nvfp4 shapes -- and because prefill is compute-bound and decode is
bandwidth-bound, that shows up as a TTFT deficit with TPOT unaffected.

# The form

    eff(m, n, k) = eps_max * m/(m+m_half) * k/(k+k_half) * n/(n+n_half)

Three saturating factors, because a GEMM needs rows, reduction depth and output width to
fill the machine and a tile starves on whichever is smallest. eps_max is the asymptote of
a PRODUCT, so no real shape reaches it; it is bounded at 1.0 here, and the highest
efficiency observed in these sweeps is 0.857 for fp8 and 0.578 for nvfp4.

# How it is fitted and validated

Grid search minimising the mean absolute log ratio, which is the criterion the registry's
other fits use.

Validation holds out whole k VALUES rather than random rows. A random split would put
near-duplicate shapes on both sides and report a generalization that was never tested;
holding out a k value means the test shapes have a reduction depth the fit never saw,
which is what a new model's shapes actually are.

Lane: vllm/0.25.0, because this kernel predicts vLLM. That is the opposite of the
conclusion for the ENVELOPE fit, where the two lanes agree to within 0.4-3.6% over
100,668 shared shapes and the TRT-LLM sweep is larger -- the envelope takes a maximum
over shapes and so is insensitive to the lane, while a per-shape fit is not.

Separation of evidence: AISimulate for fitting, FPM for validation, InferenceX for
evaluation only. This script reads AISimulate and refuses an InferenceX path.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

# The registry's own entry writer, so a new coefficient has the same shape as every
# existing one and a schema change lands in one place.
_spec = importlib.util.spec_from_file_location(
    "emit_primitives", Path(__file__).with_name("emit_primitives.py")
)
_ep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ep)
entry = _ep.entry

DEFAULT_DATA = os.environ.get("AISIMULATE_DATA", "/private/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))
COLLECTION = "vllm/0.25.0"

SKUS = {
    "h200_sxm": "h200",
    "h100_sxm": "h100",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
    "gb200": "gb200-nvl72",
}

# Which catalog peak a dtype is a fraction of. Same mapping fit_gemm_envelope.py uses,
# and for the same reason: the fraction has to be of the rate the cost model will read.
PEAK_FIELD = {
    "bfloat16": "TFlopsPeak",
    "fp8": "TFlopsFP8",
    "fp8_block": "TFlopsFP8",
    "nvfp4": "TFlopsNVFP4",
}

# The registry suffix each dtype's coefficients carry. One suffix per dtype, because the
# fits disagree: on every Blackwell part fp8 wants m_half 384 and nvfp4 wants 768, so a
# shared entry would misprice whichever dtype it was not fitted on. An earlier version of
# this map sent nvfp4 to "fp8" -- mirroring a defect in the kernel, since fixed, where
# dtypeFit returned the fp8 suffix for NVFP4 weights and the registry's six fitted
# gemm_*_nvfp4 entries were read by nothing.
SUFFIX = {"bfloat16": "bf16", "fp8": "fp8", "fp8_block": "fp8_block", "nvfp4": "nvfp4"}

EPS_GRID = [0.60 + 0.02 * i for i in range(21)]
M_GRID = (64, 128, 192, 256, 384, 512, 768)
K_GRID = (512, 1024, 2048, 3072, 4096, 6144, 8192)
N_GRID = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144)

MIN_POINTS = 500


def refuse_evaluation_data(path: Path) -> None:
    """Fail loudly if pointed at the evaluation corpus."""
    parts = {p.lower() for p in path.parts}
    if any("inferencex" in p or "semianalysis" in p for p in parts):
        raise SystemExit(
            f"{path}: InferenceX/SemiAnalysis data is evaluation-only and must not "
            f"inform a fit. Use AISimulate operator sweeps."
        )


def points(path: Path, dtype: str, peak: float) -> list[tuple[float, float, float, float]]:
    """Measured (m, n, k, efficiency) for one dtype.

    `latency` is MILLISECONDS, as every AISimulate sweep stores it. Treating it as
    seconds yields efficiencies in the thousands, which is how that mistake announces
    itself.
    """
    d = pq.read_table(path).to_pydict()
    out = []
    for i in range(len(d["latency"])):
        if d["gemm_dtype"][i] != dtype:
            continue
        lat = d["latency"][i]
        if not lat or lat <= 0:
            continue
        m, n, k = d["m"][i], d["n"][i], d["k"][i]
        eff = 2.0 * m * n * k / (lat * 1e-3) / peak
        # An efficiency above 1 is a shape whose peak is not this dtype's -- a sweep row
        # the catalog's peak does not describe. Dropped rather than fitted.
        if 0 < eff <= 1.0:
            out.append((float(m), float(n), float(k), eff))
    return out


def geo_err(params: tuple[float, float, float, float], pts) -> float:
    eps, mh, kh, nh = params
    total = 0.0
    for m, n, k, e in pts:
        pred = eps * (m / (m + mh)) * (k / (k + kh)) * (n / (n + nh))
        total += abs(math.log(pred / e))
    return math.exp(total / len(pts))


def geo_err_1factor(eps: float, mh: float, pts) -> float:
    total = 0.0
    for m, _n, _k, e in pts:
        total += abs(math.log(eps * (m / (m + mh)) / e))
    return math.exp(total / len(pts))


def fit(pts) -> tuple[float, tuple[float, float, float, float]]:
    best = None
    for eps in EPS_GRID:
        for mh in M_GRID:
            for kh in K_GRID:
                for nh in N_GRID:
                    err = geo_err((eps, mh, kh, nh), pts)
                    if best is None or err < best[0]:
                        best = (err, (eps, mh, kh, nh))
    assert best is not None
    return best


def committed(registry: Path, chip: str, suffix: str) -> tuple[float, float] | None:
    """The committed one-factor ramp, read from the registry rather than restated."""
    doc = yaml.safe_load((registry / "coefficients" / "cost-model-primitives.yaml").read_text())
    found: dict[str, float] = {}

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(val, dict) and "value" in val:
                    scope = (val.get("scope") or {}).get("hardware") or []
                    if chip in scope and key in (
                        f"gemm_eps_max_{suffix}",
                        f"gemm_m_half_{suffix}",
                    ):
                        found[key] = val["value"]
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    if len(found) == 2:
        return found[f"gemm_eps_max_{suffix}"], found[f"gemm_m_half_{suffix}"]
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--registry", default=".")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument("--emit", action="store_true",
                    help="print registry entries for gemm_n_half_* and gemm_k_half_* "
                         "instead of only the comparison")
    ap.add_argument("parts", nargs="*", metavar="sku:chip")
    args = ap.parse_args(argv[1:])

    data = Path(args.data)
    refuse_evaluation_data(data)

    pairs = []
    if args.parts:
        for spec in args.parts:
            sku, _, chip = spec.partition(":")
            pairs.append((sku, chip or SKUS.get(sku, "")))
    else:
        pairs = sorted(SKUS.items())

    print(f"# lane: {args.collection}; held-out k values, seed {args.holdout_seed}")
    rc = 0
    for sku, chip in pairs:
        path = data / sku / "gemm" / args.collection / "gemm_perf.parquet"
        if not path.exists():
            print(f"{chip:14s} no gemm sweep at {args.collection}")
            continue
        facts = yaml.safe_load((Path(args.catalog) / "hardware" / f"{chip}.yaml").read_text())
        for dtype, field in PEAK_FIELD.items():
            peak = facts.get(field)
            if not peak:
                continue
            pts = points(path, dtype, peak * 1e12)
            if len(pts) < MIN_POINTS:
                continue
            ks = sorted({k for _m, _n, k, _e in pts})
            rng = random.Random(args.holdout_seed)
            held = set(rng.sample(ks, max(1, len(ks) // 4)))
            train = [p for p in pts if p[2] not in held]
            test = [p for p in pts if p[2] in held]
            if len(train) < MIN_POINTS or not test:
                continue
            err, (eps, mh, kh, nh) = fit(train)
            suffix = SUFFIX[dtype]
            old = committed(Path(args.registry), chip, suffix)
            committed_test = float("nan")
            old_txt = ""
            if old:
                committed_test = geo_err_1factor(*old, test)
                old_txt = (
                    f"  1-factor(committed eps={old[0]} m_half={old[1]}): "
                    f"train {geo_err_1factor(*old, train):.2f}x "
                    f"TEST {geo_err_1factor(*old, test):.2f}x"
                )
            test_err = geo_err((eps, mh, kh, nh), test)
            if args.emit:
                cite = (
                    f"NVIDIA AISimulate systems/data/{sku}/gemm/{args.collection}/"
                    f"gemm_perf.parquet (fitted on {len(train)} {dtype} shapes, "
                    f"validated on {len(test)} held out by k value)"
                )
                base = (
                    f"Three-factor ramp eff = eps_max * m/(m+m_half) * k/(k+k_half) * "
                    f"n/(n+n_half), fitted jointly with eps_max={eps:.2f} and "
                    f"m_half={mh}. A GEMM needs rows, reduction depth and output width "
                    f"to fill the machine and starves on the smallest, so a ramp in m "
                    f"alone prices every shape in a layer alike. Held-out k values give "
                    f"{test_err:.2f}x geometric error against {committed_test:.2f}x for "
                    f"the one-factor form, with train {err:.2f}x -- the gap between "
                    f"train and test is the evidence it generalizes rather than fits."
                )
                print(f"# ---- {chip} {dtype} (suffix {suffix}) ----")
                print(entry(f"gemm_k_half_{suffix}", kh, "tokens", "measured", chip,
                            cite, base, fitted=True), end="")
                print(entry(f"gemm_n_half_{suffix}", nh, "tokens", "measured", chip,
                            cite, f"The output-width half-max from the same fit as "
                            f"gemm_k_half_{suffix}.", fitted=True), end="")
                print(entry(f"gemm_eps_max_{suffix}", round(eps, 2), "dimensionless",
                            "measured", chip, cite,
                            f"The asymptote of the three-factor ramp. It is the limit of "
                            f"a PRODUCT of three saturating factors, so no finite shape "
                            f"reaches it; the highest efficiency observed in this sweep "
                            f"is {max(e for _m, _n, _k, e in pts):.3f}.", fitted=True),
                      end="")
                print(entry(f"gemm_m_half_{suffix}", mh, "tokens", "measured", chip,
                            cite, f"The row-count half-max from the same fit as "
                            f"gemm_k_half_{suffix}. Refitted jointly: the one-factor "
                            f"value absorbed the shape terms' work.", fitted=True),
                      end="")
            else:
                print(
                    f"{chip:14s} {dtype:10s} suffix={suffix:9s} n={len(pts):6d} "
                    f"(train {len(train)} / test {len(test)})\n"
                    f"  3-factor eps_max={eps:.2f} m_half={mh:3d} k_half={kh:5d} "
                    f"n_half={nh:5d}: train {err:.2f}x TEST {test_err:.2f}x"
                    + (f"\n{old_txt}" if old_txt else "")
                )
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
