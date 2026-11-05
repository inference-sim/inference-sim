#!/usr/bin/env python3
"""Fit the prefill-attention primitive from AISimulate's context-attention sweeps.

The decode fit (scripts/fit_attention.py) covers the generation phase. Prefill is a different
regime — compute-bound and quadratic in prompt length rather than bandwidth-bound in context
— so it needs its own constants, and the cost model's FLOPs form is wrong here too: 12.9x on
L40S to 32.4x on GB200 over roughly 55,000 full-attention points per part.

The form keeps the efficiency ramp, which is the right shape for a compute-bound kernel, and
adds the two things the FLOPs view omits:

    latency = floor + causal_flops / (peak * eff(tokens) * work_scale)

where causal_flops halves the naive count because attention attends only to earlier
positions, and work_scale is the fraction of the dense-GEMM ramp an attention kernel reaches.
That scale lands at 0.38 to 0.48 across parts — an attention kernel gets under half the
efficiency a well-shaped matmul does at the same token count, which is why reusing the dense
ramp unmodified under-predicts.

Both parameters are fitted per part by minimising the mean absolute log ratio.

Only FULL attention is fitted; the sweeps' sliding-window rows read a bounded number of
positions and would fit a different curve.

The bf16 envelope this fit rests on is READ FROM THE REGISTRY rather than restated here. It
was previously a literal table in this file, duplicating values that
coefficients/cost-model-primitives.yaml already held; two copies of one number is how a
re-fit on one side goes unnoticed on the other.

Usage:
    python scripts/fit_attention_prefill.py [--data <dir>] [--registry <dir>]
        [--catalog <dir>] [sku:chip ...]

With no SKU arguments every part that has both a context-attention sweep and a bf16 envelope
in the registry is fitted.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

DEFAULT_DATA = os.environ.get("AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))
DEFAULT_REGISTRY = "."
# The lane the sweep was measured in. This kernel predicts vLLM, so vllm/0.25.0 is
# the lane its prefill coefficients belong on. The TRT-LLM default is retained only
# because the first committed fit used it; passing --collection vllm/0.25.0
# reproduces the values the registry now carries for the five parts that have a vLLM
# context sweep.
#
# The lanes are not interchangeable. Over ~38,000 shapes shared between them, vLLM
# prefill attention runs at 0.79-0.81x of TRT-LLM's on every part, and the gap is
# concentrated at short prompts rather than spread across the ramp: the ratio is
# 0.505x at isl=1, rising to a ~0.87x plateau for isl >= 128. That is the floor, and
# the raw sweeps say so without any fit -- at batch 1, isl 1, full attention, the
# median is 16.50us against 11.46us on H200 and 15.17us against 8.98us on B200.
#
# L40S has no vLLM context-attention sweep, so its prefill pair stays on TRT-LLM and
# says so in its own citation.
DEFAULT_COLLECTION = "trtllm/1.3.0rc20"

# AISimulate SKU directory -> catalog chip name.
SKUS = {
    "h200_sxm": "h200",
    "h100_sxm": "h100",
    "l40s": "l40s",
    "gb200": "gb200-nvl72",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
    "a100_sxm": "a100-sxm",
    "gb300": "gb300",
}

FLOOR_GRID = [x * 0.5 for x in range(2, 80)]
SCALE_GRID = [x * 0.02 for x in range(5, 101)]


def envelope(registry: Path) -> dict[str, tuple[float, float]]:
    """The fitted bf16 GEMM ramp per chip, from the registry that owns it."""
    path = registry / "coefficients" / "cost-model-primitives.yaml"
    doc = yaml.safe_load(path.read_text())
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


def fit(data: Path, catalog: Path, sku: str, chip: str,
        env: tuple[float, float], collection: str) -> str | None:
    path = data / sku / "attention" / collection / "context_attention_perf.parquet"
    if not path.exists():
        return f"{chip:14s} no context-attention sweep at {collection}"
    d = pq.read_table(path).to_pydict()
    hw = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = hw["TFlopsPeak"] * 1e12
    eps, m_half = env

    pts = []
    for i in range(len(d["latency"])):
        # window_size is absent from the older collections -- vLLM 0.14.0 on A100
        # predates the sliding-window sweep -- and a missing column means every row is
        # full attention. Read through a default so those collections fit rather than
        # failing on the key; a collection that DOES sweep windows still has its
        # windowed rows excluded.
        if (d.get("window_size", [0] * len(d["latency"]))[i] != 0
                or d["latency"][i] <= 0):
            continue
        b, isl = d["batch_size"][i], d["isl"][i]
        nq, dh = d["num_heads"][i], d["head_dim"][i]
        tok = b * isl
        # Causal: each query attends to earlier positions only, so half the naive count.
        flops = 2.0 * 2.0 * b * nq * dh * isl * isl / 2.0
        eff = eps * tok / (tok + m_half)
        if eff <= 0:
            continue
        pts.append((flops / (peak * eff), d["latency"][i] * 1e-3))
    if not pts:
        return f"{chip:14s} no full-attention rows"

    best = None
    for floor in FLOOR_GRID:
        for scale in SCALE_GRID:
            err = math.exp(
                sum(abs(math.log((floor * 1e-6 + w / scale) / lat)) for w, lat in pts)
                / len(pts)
            )
            if best is None or err < best[0]:
                best = (err, floor, scale)
    naive = math.exp(
        sum(abs(math.log(max(w, 1e-12) / lat)) for w, lat in pts) / len(pts)
    )
    return (f"{chip:14s} n={len(pts):6d} floor={best[1]:5.1f}us "
            f"work_scale={best[2]:.2f} geo-err {best[0]:.3f}x  (FLOPs-only {naive:.1f}x)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="framework/version lane under <sku>/attention/ "
                         f"(default {DEFAULT_COLLECTION!r})")
    ap.add_argument("parts", nargs="*", metavar="sku:chip")
    args = ap.parse_args(argv[1:])

    env = envelope(Path(args.registry))
    if args.parts:
        pairs = []
        for spec in args.parts:
            sku, _, chip = spec.partition(":")
            pairs.append((sku, chip or SKUS.get(sku, "")))
    else:
        pairs = sorted(SKUS.items())

    print(f"# lane: {args.collection}")
    rc = 0
    for sku, chip in pairs:
        if not chip:
            print(f"error: {sku} has no chip mapping; add it to SKUS", file=sys.stderr)
            rc = 1
            continue
        if chip not in env:
            print(f"error: {chip} has no bf16 envelope in the registry; run "
                  f"emit_primitives.py first", file=sys.stderr)
            rc = 1
            continue
        print(fit(Path(args.data), Path(args.catalog), sku, chip, env[chip],
                  args.collection))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
