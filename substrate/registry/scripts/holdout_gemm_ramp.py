#!/usr/bin/env python3
"""Hold out whole reduction depths, and whole token counts, from the GEMM ramp fit.

WHY THIS SCRIPT EXISTS. The GEMM efficiency envelope (`gemm_eps_max_*`, `gemm_m_half_*`)
prices every dense matmul in every layer, so it is the coefficient family with the widest
reach and the one whose generalization matters most. `fit_gemm_envelope.py` reports an RMS
residual on the rows it fitted. That is not a generalization test.

§4 of docs/methodology.md asks for splits along a dimension generalization has to cross,
not at random, because a random row split puts near-duplicate shapes on both sides. Two
such dimensions exist in the sweep and this script tests both:

  * K, the reduction depth. A held-out K is an arithmetic intensity the fit never saw.
  * M, the token count, which is the ramp's own independent variable. Holding out the
    LARGEST M values is the harder and more honest test, because `eps_max` is an
    asymptote: a fit that never saw a large M must extrapolate to claim one.

The second fold is the reason the script exists rather than reusing the existing
`fit_gemm_shape_ramp.py` K-holdout. A two-parameter curve can fit a ramp's knee and still
be wrong about its ceiling, and the ceiling is what `eps_max` ships.

WHAT IT DOES NOT DO. It never writes a coefficient. It reads AISimulate only — FPM and
InferenceX play no part here, because this is a per-kernel fit and AISimulate is the only
source that isolates one kernel.

Usage:
    python scripts/holdout_gemm_ramp.py <data>/h200_sxm/gemm/vllm/0.25.0
    python scripts/holdout_gemm_ramp.py <data>/h200_sxm/gemm/vllm/0.25.0 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fge = _load("fit_gemm_envelope")

DEFAULT_CATALOG = os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))


def fit_points(points: dict[int, float]) -> tuple[float, int]:
    """Delegate to fit_gemm_envelope.fit_ramp so the fold and the shipped fit agree."""
    eps, m_half, _rms = fge.fit_ramp(points)
    return eps, m_half


def rms(points: dict[int, float], eps: float, m_half: int) -> float:
    return math.sqrt(
        sum((eps * m / (m + m_half) - y) ** 2 for m, y in points.items()) / len(points)
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("collection", type=Path)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--dtype", help="restrict to one gemm_dtype")
    ap.add_argument("--top-m", type=int, default=3,
                    help="how many of the largest M values to hold out (default 3)")
    args = ap.parse_args(argv[1:])

    rows = fge.read_rows(args.collection, "gemm_perf")
    if not rows:
        print(f"{args.collection}: no usable GEMM rows", file=sys.stderr)
        return 1
    chip = fge.chip_for(args.collection)
    if chip is None:
        print(f"{args.collection}: path maps to no catalog chip", file=sys.stderr)
        return 1
    peaks = fge.peaks_from_catalog(Path(args.catalog), chip)

    dtypes = sorted({r["gemm_dtype"] for r in rows})
    if args.dtype:
        dtypes = [d for d in dtypes if d == args.dtype]
    print(f"# {args.collection}")
    print(f"# catalog chip {chip}")

    for dt in dtypes:
        peak = peaks.get(dt)
        if not peak:
            continue
        sel = [r for r in rows if r["gemm_dtype"] == dt]
        # The envelope is the best efficiency observed at each M, which is what
        # fit_gemm_envelope fits; reuse its own reducer so the two cannot disagree.
        env = fge.envelope(sel, dt, peak)
        if len(env) < 8:
            continue
        eps_all, mh_all = fit_points(env)
        print(f"\n=== {dt}  M values={len(env)}  rows={len(sel)}")
        print(f"  all M        eps_max={eps_all:.3f} m_half={mh_all:3}  "
              f"rms={rms(env, eps_all, mh_all):.4f}")

        # --- Fold 1: hold out the largest M values (extrapolate the ceiling) ---
        ms = sorted(env)
        held = ms[-args.top_m:]
        train = {m: env[m] for m in ms if m not in held}
        if len(train) >= 8:
            eps, mh = fit_points(train)
            pred = {m: eps * m / (m + mh) for m in held}
            errs = [abs(pred[m] - env[m]) for m in held]
            print(f"  hold out top-{args.top_m} M {held}:")
            print(f"    train eps_max={eps:.3f} m_half={mh:3}   "
                  f"shipped-from-all={eps_all:.3f}  delta={eps - eps_all:+.3f}")
            for m in held:
                print(f"      M={m:6} predicted={pred[m]:.3f} measured={env[m]:.3f} "
                      f"err={pred[m] - env[m]:+.3f}")
            print(f"    max |err| on held-out M: {max(errs):.4f}")

        # --- Fold 2: hold out whole K values (unseen arithmetic intensity) ---
        by_k = collections.defaultdict(list)
        for r in sel:
            by_k[int(r["k"])].append(r)
        ks = sorted(by_k)
        if len(ks) >= 3:
            deltas = []
            for k in ks:
                tr = [r for r in sel if int(r["k"]) != k]
                te = [r for r in sel if int(r["k"]) == k]
                env_tr = fge.envelope(tr, dt, peak)
                env_te = fge.envelope(te, dt, peak)
                if len(env_tr) < 8 or not env_te:
                    continue
                eps, mh = fit_points(env_tr)
                e = rms(env_te, eps, mh)
                deltas.append((k, eps, mh, e, len(te)))
            if deltas:
                print(f"  hold out each K ({len(deltas)} folds):")
                for k, eps, mh, e, n in deltas:
                    print(f"      K={k:6} n={n:6} train eps_max={eps:.3f} "
                          f"m_half={mh:3}  held-out rms={e:.4f}")
                sp = [d[1] for d in deltas]
                print(f"    eps_max across K folds: min={min(sp):.3f} max={max(sp):.3f} "
                      f"spread={max(sp) - min(sp):.3f}  "
                      f"worst held-out rms={max(d[3] for d in deltas):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
