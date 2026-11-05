#!/usr/bin/env python3
"""Write the direct corpus in harness.LoadCorpus's own format, so the existing scorer
reads it with no new scoring code.

# What this file carries, and what it deliberately omits

`metricscore` computes mape from the absolutes file -- predicted over measured at every
point, the anchor included -- and shape from this file's relatives, each side divided by
its own value at the sweep's lowest concurrency. So a corpus file needs the relatives, and
they are computable here because the absolutes are known: a point's relative is its
absolute over the absolute at the anchor.

The AISimulate and AIC prediction fields are written as 0, which the scorer reads as "not
published". That is a finding rather than an omission.

An artifact prediction exists per (model, gpu, precision, framework, workload, tp,
concurrency). This corpus keys a deployment additionally on the container image and on what
the run's args line states, because both change the measurement. Only 38 of the 3,345
points map one-to-one onto an artifact prediction; the rest share one across up to 14
distinct deployments. On deepseek-v4-pro h200 fp8 8k1k tp8 at concurrency 64, fourteen
deployments spanning 2.49x in measured TPOT -- differing in image and in expert- and
attention-parallel width -- all collapse onto a single artifact key. An AISimulate column
built on that reuse would compare one prediction against fourteen different deployments,
which is the conflation this corpus exists to avoid.

So this table is the kernel against InferenceX's own measurement, and the estimator
comparison stays in the artifact-based tables, where every arm predicts the same
artifact-defined deployment.

Usage:
    python scripts/gen_direct_corpus.py \
        -c testdata/measurements/inferencex_direct.json \
        -o testdata/measurements/inferencex_direct_corpus.json
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--agreeing-only", action="store_true",
                    help="keep only points whose repeats agree on every "
                         "per-point setting: the strictest comparable set")
    args = ap.parse_args()

    src = json.loads(Path(args.corpus).read_text())
    sweeps = []
    dropped = 0
    for sw in src["sweeps"]:
        pts = sw["points"]
        if args.agreeing_only:
            keep = [p for p in pts
                    if not any(isinstance(v, list) for v in p["passed"].values())]
            dropped += len(pts) - len(keep)
            pts = keep
        if len(pts) < src["min_points_per_sweep"]:
            continue
        a_tpot = pts[0]["measured_tpot_us"]
        a_ttft = pts[0]["measured_ttft_us"]
        if a_tpot <= 0 or a_ttft <= 0:
            continue
        out_pts = []
        for p in pts:
            out_pts.append({
                "concurrency": p["concurrency"],
                "measured_tpot_relative": round(p["measured_tpot_us"] / a_tpot, 6),
                "measured_ttft_relative": round(p["measured_ttft_us"] / a_ttft, 6),
                # No published prediction exists for these deployments; see the module
                # docstring. Zero is how the scorer reads "not published".
                "aisimulate_tpot_relative": 0.0,
                "aisimulate_ttft_relative": 0.0,
                "aic_tpot_relative": 0.0,
                "aic_ttft_relative": 0.0,
                "status": "success",
            })
        sweeps.append({
            "scenario": sw["scenario"],
            "model": sw["model"],
            "gpu": sw["gpu"],
            "workload": sw["workload"],
            "label": sw["label"],
            "framework": sw["framework"],
            "precision": sw["precision"],
            "serving": sw["serving"],
            "spec_method": sw["spec_method"],
            "parallelism": sw["parallelism"],
            # What the run's own args line passed, which gen_aisimulate_scenarios.py
            # writes into the deployment rather than a held-fixed value.
            "stated_config": sw["stated_config"],
            "points": out_pts,
        })

    doc = {
        "source": src["source"],
        "measurement_source": "SemiAnalysis InferenceX",
        "what": ("relatives for shape, computed from this corpus's own absolutes; mape "
                 "comes from the absolutes file, as it does for the artifact corpus"),
        "published_arms": ("none: only 38 of 3,345 points map one-to-one onto an "
                           "artifact prediction, so an AISimulate or AIC column "
                           "here would reuse one prediction across up to 14 "
                           "one prediction across up to 14 distinct deployments"),
        "scope": src["scope"],
        "sweeps": sweeps,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    npts = sum(len(s["points"]) for s in sweeps)
    print(f"{len(sweeps)} sweeps, {npts} points -> {args.out}")
    if args.agreeing_only:
        print(f"  dropped {dropped} point(s) whose repeats disagreed on a setting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
