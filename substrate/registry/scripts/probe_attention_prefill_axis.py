#!/usr/bin/env python3
"""PROBE, not a fit that ships: the efficiency KEY for prefill attention is the wrong shape.

The kernel prices prefill attention as

    latency = floor + causal_flops / (peak * eff(KEY) * work_scale),  eff(m) = eps*m/(m+m_half)

with KEY = batch_size * isl, the step's total scheduled tokens. But eff() is the dense-GEMM ramp,
fitted on MATMUL ROWS (gemm_eps_max_bf16 / gemm_m_half_bf16), so one request of 2048 tokens and
eight requests of 256 receive identical attention efficiency although their attention shapes
differ entirely. AISimulate's interpolation design states the governing fact: curvature belongs to
the AXIS, not the table — context attention is ~seq^2 along seq and ~linear along batch and heads.

WHAT THIS MEASURES. Four candidate keys, each refitting its OWN (floor, work_scale) on the same
rows over the same grids as scripts/fit_attention_prefill.py, scored on held-out shapes. Keying on
isl alone wins on every part by 5.2% to 10.6%, with work_scale barely moving (0.48 to 0.48 on
h200) — a better key, not a rescaling.

WHY, measured. At fixed isl and head count, measured latency scales as batch^0.730 (isl=512),
batch^0.843 (2048) and batch^0.943 (8192): sub-linear, approaching linear as sequences grow.
causal_flops already scales linearly in batch, so carrying batch in the efficiency key as well
makes efficiency RISE with batch and partially cancels the over-count — an accidental
approximation of sub-linearity with the wrong functional form, confounded across two axes.

WHY NOTHING SHIPPED. Neither key fixes the trend. Signed residuals run from +28% over-prediction
at batch 1 to -80% under at batch 256 on the current key, and -73.7% on the best alternative; no
single scalar key removes it, because the form needs a batch term with its own exponent. And the
regime barely occurs: instrumenting the simulator over the vLLM corpus (8,186,921 steps) finds
98.33% of steps carry NO prefill request and 0.336% carry two or more, which bounds the effect on
step time across a scoring run at -0.0388%. One 2,877-token prefill fills the token budget, so
vLLM's scheduler essentially never co-schedules prefills; the parquet sweeps batch 1 to 256
uniformly and real serving traffic does not.

Two lessons worth keeping: a defect can be real, large, and irrelevant to the workload that
matters, and the frequency of the regime is part of the evidence rather than an afterthought.

GUARD. The b*isl arm, fitted on all rows, must reproduce the committed registry entry
(attention_prefill_floor 26.5, attention_prefill_work_scale 0.48 on 55,096 h200 rows). A harness
that cannot reproduce the published fit from the published data is not measuring the same thing,
and the script says so rather than printing arms that look like evidence.

Usage:
    python scripts/probe_attention_prefill_axis.py
    python scripts/probe_attention_prefill_axis.py --parts h200_sxm h100_sxm
    python scripts/probe_attention_prefill_axis.py --residuals h200_sxm
"""

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

DEFAULT_DATA = os.environ.get("AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))
DEFAULT_REGISTRY = "."
COLLECTION = "trtllm/1.3.0rc20"

SKUS = {
    "h200_sxm": "h200",
    "h100_sxm": "h100",
    "l40s": "l40s",
    "gb200": "gb200-nvl72",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
}

# The same grids scripts/fit_attention_prefill.py searches.
FLOOR_GRID = [x * 0.5 for x in range(2, 80)]
SCALE_GRID = [x * 0.02 for x in range(5, 101)]

# What coefficients/cost-model-attention.yaml publishes, as the reproduce guard.
COMMITTED = {"h200": (26.5, 0.48, 55096)}

KEYS = {
    "b*isl (current)": lambda b, isl: b * isl,
    "isl only": lambda b, isl: isl,
    "sqrt(b)*isl": lambda b, isl: math.sqrt(b) * isl,
    "b*sqrt(isl)": lambda b, isl: b * math.sqrt(isl),
}


def envelope(registry: Path) -> dict[str, tuple[float, float]]:
    """The fitted bf16 GEMM ramp per chip, from the registry that owns it."""
    doc = yaml.safe_load(
        (registry / "coefficients" / "cost-model-primitives.yaml").read_text()
    )
    out: dict[str, dict[str, float]] = {}
    for entry in doc["coefficients"]:
        (name, body), = entry.items()
        if name not in ("gemm_eps_max_bf16", "gemm_m_half_bf16"):
            continue
        for chip in body["scope"]["hardware"]:
            out.setdefault(chip, {})[name] = body["value"]
    return {
        chip: (v["gemm_eps_max_bf16"], v["gemm_m_half_bf16"])
        for chip, v in out.items()
        if len(v) == 2
    }


def rows(data: Path, sku: str):
    """Full-attention context rows: (batch, isl, heads, head_dim, latency_seconds)."""
    path = data / sku / "attention" / COLLECTION / "context_attention_perf.parquet"
    if not path.exists():
        return None
    d = pq.read_table(path).to_pydict()
    return [
        (d["batch_size"][i], d["isl"][i], d["num_heads"][i], d["head_dim"][i],
         d["latency"][i] * 1e-3)
        for i in range(len(d["latency"]))
        if d["window_size"][i] == 0 and d["latency"][i] > 0
    ]


def split(rs):
    """Deterministic 70/30 split on the SHAPE, so a shape never straddles the split and the
    partition does not depend on row order."""
    train, test = [], []
    for r in rs:
        key = f"{r[0]:.0f}|{r[1]:.0f}|{r[2]:.0f}|{r[3]:.0f}".encode()
        (test if hashlib.sha256(key).digest()[0] % 10 >= 7 else train).append(r)
    return train, test


def fit(pts):
    """Grid-search (floor, work_scale) minimising the mean absolute log ratio."""
    best = None
    for floor in FLOOR_GRID:
        for scale in SCALE_GRID:
            err = sum(abs(math.log((floor * 1e-6 + w / scale) / lat)) for w, lat in pts)
            err = math.exp(err / len(pts))
            if best is None or err < best[0]:
                best = (err, floor, scale)
    return best


def work(rs, keyf, eps, m_half, peak):
    out = []
    for b, isl, nq, dh, lat in rs:
        flops = 2.0 * 2.0 * b * nq * dh * isl * isl / 2.0  # causal: half the naive count
        m = keyf(b, isl)
        eff = eps * m / (m + m_half)
        if eff > 0:
            out.append((flops / (peak * eff), lat))
    return out


def score(pts, floor, scale):
    return math.exp(
        sum(abs(math.log((floor * 1e-6 + w / scale) / lat)) for w, lat in pts) / len(pts)
    )


def residuals(rs, eps, m_half, peak):
    """Signed residuals by BATCH for the current key and the best alternative, which is what
    shows that neither removes the trend."""
    arms = (("b*isl (current)", KEYS["b*isl (current)"], 26.5, 0.48),
            ("isl only", KEYS["isl only"], 24.5, 0.48))
    batches = sorted({r[0] for r in rs})
    print(f"\n{'batch':>6} {'n':>6}  " + "  ".join(f"{n:>24}" for n, _, _, _ in arms))
    print(f"{'':6} {'':6}  " + "  ".join(f"{'median signed / geo-err':>24}" for _ in arms))
    for bz in batches:
        sel = [r for r in rs if r[0] == bz]
        cells = []
        for _, keyf, floor, scale in arms:
            rel = []
            for b, isl, nq, dh, lat in sel:
                flops = 2.0 * 2.0 * b * nq * dh * isl * isl / 2.0
                m = keyf(b, isl)
                pred = floor * 1e-6 + flops / (peak * (eps * m / (m + m_half))) / scale
                rel.append(pred / lat)
            rel.sort()
            med = rel[len(rel) // 2]
            geo = math.exp(sum(abs(math.log(x)) for x in rel) / len(rel))
            cells.append(f"{(med - 1) * 100:+9.1f}% {geo:9.3f}x")
        print(f"{bz:6.0f} {len(sel):6d}  " + "  ".join(f"{c:>24}" for c in cells))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--parts", nargs="*", default=sorted(SKUS))
    ap.add_argument("--residuals", metavar="SKU",
                    help="print signed residuals by batch for one part")
    args = ap.parse_args(argv[1:])

    data, catalog, registry = Path(args.data), Path(args.catalog), Path(args.registry)
    env = envelope(registry)
    rc = 0
    parts = [args.residuals] if args.residuals else args.parts

    for sku in parts:
        chip = SKUS[sku]
        rs = rows(data, sku)
        if not rs:
            print(f"{sku:14s} no context-attention rows")
            continue
        eps, m_half = env[chip]
        peak = yaml.safe_load(
            (catalog / "hardware" / f"{chip}.yaml").read_text()
        )["TFlopsPeak"] * 1e12

        if args.residuals:
            print(f"=== {sku} ({chip})  n={len(rs)}")
            residuals(rs, eps, m_half, peak)
            continue

        train, test = split(rs)
        print(f"\n=== {sku} ({chip})  n={len(rs)}  eps_max={eps} m_half={m_half} "
              f"peak={peak / 1e12:.0f}TF  train={len(train)} test={len(test)}")

        # GUARD: the current key on ALL rows must reproduce the committed entry.
        _, floor_all, scale_all = fit(work(rs, KEYS["b*isl (current)"], eps, m_half, peak))
        if chip in COMMITTED:
            cf, cs, cn = COMMITTED[chip]
            ok = len(rs) == cn and abs(floor_all - cf) < 1e-9 and abs(scale_all - cs) < 1e-9
            print(f"  reproduce: n {len(rs)} vs {cn}, floor {floor_all} vs {cf}, "
                  f"scale {scale_all:.2f} vs {cs} -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print("  GUARD FAILED — the arms below are not evidence.")
                rc = 1

        print(f"  {'key':18s} {'train':>9s} {'TEST':>9s} {'floor':>7s} {'scale':>6s} "
              f"{'vs current':>12s}")
        base = None
        for name, keyf in KEYS.items():
            _, floor, scale = fit(work(train, keyf, eps, m_half, peak))
            etr = score(work(train, keyf, eps, m_half, peak), floor, scale)
            ete = score(work(test, keyf, eps, m_half, peak), floor, scale)
            if base is None:
                base = ete
            delta = "" if ete == base else f"{(ete / base - 1) * 100:+11.1f}%"
            print(f"  {name:18s} {etr:8.4f}x {ete:8.4f}x {floor:7.1f} {scale:6.2f} "
                  f"{delta:>12s}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
