#!/usr/bin/env python3
"""Extract a direct BLIS-vs-InferenceX corpus: absolute latencies, configuration from
each
run's own server log, and no dependence on NVIDIA's accuracy artifact.

# Why this exists alongside extract_aisimulate_e2e.py

That extractor reads NVIDIA's published artifact, which carries relatives only and
covers
the nine models and four chips NVIDIA chose to publish. Two consequences follow. A mean
ABSOLUTE percentage error needs the anchor latency, which the artifact withholds. And a
deployment NVIDIA did not publish cannot be scored at all, even where InferenceX
measured
it -- MiniMax-M3 and DeepSeek-V4-Pro on Hopper being the cases in hand.

This extractor reads the InferenceX dump directly. Ground truth is the dump's own
`mean_tpot` and `mean_ttft` in microseconds, so mape needs no anchor; and every
deployment
InferenceX measured is reachable, not only the published subset.

# What identifies a deployment, and why the server log decides it

A row's (model, hardware, decode_tp, isl, osl, concurrency) does NOT identify a
deployment.
Over 630 such cells in this dump the measured TPOT spans a median of 1.384x and up to
18.43x, which reads as measurement noise and is not: the rows are different deployments.
Five MiniMax-M3 h200 tp8 rows at concurrency 4 span 4,862 to 10,085 us because two serve
Inferact/MiniMax-M3-EAGLE3 with eagle3 speculation at 3 draft tokens and three serve the
plain MXFP8 checkpoint.

Grouping additionally by the configuration the run's own `non-default args` line states
takes that median spread to 1.016x and resolves 849 distinct deployments. So a sweep
here is
keyed by the stated configuration, and rows whose logs disagree are never pooled. The
1.6%
residual is run-to-run noise, concentrated at concurrency 1 to 4 where a handful of
requests
makes the mean noisy.

# Scope, and what is excluded

  vLLM only            BLIS models vLLM; no other engine prints a vLLM args line.
  NVIDIA parts only    blis-registry carries no coefficients for MI300X/MI325X/MI355X,
  so
                       scoring there would mean fitting a new hardware family.
  non-disaggregated    A P/D deployment is not one step-time function.
  single_turn          agentic_traces rows carry NULL isl and osl and percentile
                       distributions rather than a fixed workload.
  has a server log     Without one the configuration is unknown, which is the whole
  point.
  >= 3 concurrencies   Two points pin a level rather than a curvature.

Speculative arms are KEPT and marked, not dropped: `num_speculative_tokens` changes
tokens
per step, so a consumer that scores ITL as a step time must divide by the accepted
tokens
per step. Dropping them would discard 436 of 1,673 points.

Usage:
    python scripts/extract_inferencex_direct.py -o
    testdata/measurements/inferencex_direct.json
"""

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_aisimulate_e2e import CHIPS, MODELS  # noqa: E402
from extract_inferencex_engine_settings import parse_args_line  # noqa: E402

PSQL = "psql"
DB = "inferencex"

# The fields of the stated configuration that define a DEPLOYMENT rather than a tuning
# choice. Two rows agreeing on all of these are the same engine serving the same model.
DEPLOYMENT_FIELDS = (
    "model",
    "block_size",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "enable_expert_parallel",
    "num_speculative_tokens",
    "speculative_method",
    "draft_model",
)

# Settings the harness varies WITH the sweep, so they identify a point rather than a
# deployment. Keying on them fragments every sweep into singletons: this harness sets
# max_num_seqs equal to the client concurrency on 953 rows, and the capture size tracks
# the concurrency on 3,938 of the 3,967 rows that state one, so an earlier version
# of this key produced 473 one-point groups out of 495 and dropped kimi-k2.5 and two
# thirds of gpt-oss-120b entirely. They are recorded per point instead.
PER_POINT_FIELDS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "max_cudagraph_capture_size",
)

MIN_POINTS = 3


def query(sql: str) -> list[list[str]]:
    out = subprocess.run([PSQL, "-d", DB, "-t", "-A", "-F", "\t", "-c", sql],
                         capture_output=True, text=True, check=True)
    return [line.split("\t") for line in out.stdout.split("\n") if line.strip()]


def rows() -> list[dict]:
    """Every candidate row, with its log's stated configuration attached."""
    sql = """
    select c.model, c.hardware, c.framework, c.precision, c.spec_method,
           c.decode_tp, c.decode_ep, c.decode_dp_attention,
           b.isl, b.osl, b.conc,
           (b.metrics->>'mean_tpot')::numeric * 1e6,
           (b.metrics->>'mean_ttft')::numeric * 1e6,
           b.date, b.config_id, coalesce(b.image, ''),
           -- The args line, extracted and flattened IN SQL. A whole server log runs
           -- 50 KB to 335 KB and carries tabs, newlines and carriage returns, any of
           -- which splits a row in psql's delimited output: shipping the log whole
           -- truncated every row at its first control character, leaving no args line
           -- to parse and producing an empty corpus with no error reported. Only the
           -- args line is needed, so only it crosses.
           regexp_replace(
             substring(sl.server_log from position('non-default args' in sl.server_log)
                       for 8000),
             '[\r\n\t]+', ' ', 'g')
    from benchmark_results b
      join configs c on c.id = b.config_id
      join server_logs sl on sl.id = b.server_log_id
    where b.benchmark_type = 'single_turn'
      and c.framework = 'vllm'
      and c.disagg = false and c.is_multinode = false
      and b.error is null
      and b.metrics ? 'mean_tpot' and b.metrics ? 'mean_ttft'
      and b.isl is not null and b.osl is not null
      and position('non-default args' in sl.server_log) > 0
    """
    out = []
    for r in query(sql):
        if len(r) < 17:
            continue
        passed = parse_args_line(r[16])
        if not passed:
            continue
        out.append({
            "slug": r[0], "gpu_raw": r[1], "framework": r[2], "precision": r[3],
            "spec_method": r[4], "tp": int(r[5]), "ep": int(r[6]),
            "dp_attention": r[7] == "t",
            "isl": int(r[8]), "osl": int(r[9]), "conc": int(r[10]),
            "tpot_us": float(r[11]), "ttft_us": float(r[12]),
            "date": r[13], "config_id": int(r[14]), "image": r[15],
            "passed": passed,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    skipped: collections.Counter = collections.Counter()
    sweeps: dict[tuple, dict] = {}
    for r in rows():
        model = MODELS.get(r["slug"])
        if model is None:
            skipped["model not in catalog"] += 1
            continue
        chip = CHIPS.get(r["gpu_raw"])
        if chip is None:
            skipped[f"no coefficients for {r['gpu_raw']}"] += 1
            continue
        conf = tuple(str(r["passed"].get(f)) for f in DEPLOYMENT_FIELDS)
        # The container image is part of the deployment, not metadata. On config 1776 at
        # 8k1k one stated configuration across three dates reads 4,289 to 22,389 us at
        # concurrency 1: three nightlies cluster and the 2026-08-04 one is 5x slower.
        # That is a build regression, so pooling dates averages two engines. Keying on
        # it takes the median within-point spread from 1.037x to 1.006x, the worst from
        # 8.79x to 3.13x, and raises the sweep count because builds separate correctly.
        key = (model, chip, r["precision"], r["tp"], r["ep"], r["dp_attention"],
               r["isl"], r["osl"], conf, r["image"])
        sw = sweeps.setdefault(key, {
            "model": model,
            "gpu": chip,
            "framework": "vllm",
            "precision": r["precision"],
            "workload": f"{r['isl']}:{r['osl']}",
            "label": f"{r['isl'] // 1024 or 1}k{r['osl'] // 1024 or 1}k",
            "serving": "aggregated",
            "spec_method": r["spec_method"],
            "parallelism": {"tp_size": r["tp"], "pp_size": 1,
                            "attention_dp_size": 2 if r["dp_attention"] else 1,
                            "moe_ep_size": r["ep"],
                            "moe_tp_size": 1 if r["ep"] > 1 else r["tp"]},
            "stated_config": {f: r["passed"].get(f) for f in DEPLOYMENT_FIELDS
                              if r["passed"].get(f) is not None},
            "image": r["image"],
            "config_ids": set(),
            "dates": set(),
            "points": {},
        })
        sw["config_ids"].add(r["config_id"])
        sw["dates"].add(r["date"])
        # A concurrency seen twice under one stated configuration is a repeat of the
        # same
        # deployment, so the two are averaged and the count recorded rather than one
        # being
        # chosen. Choosing would need a run-selection policy this corpus deliberately
        # avoids.
        sw["points"].setdefault(r["conc"], []).append(
            (r["tpot_us"], r["ttft_us"],
             {f: r["passed"].get(f) for f in PER_POINT_FIELDS
              if r["passed"].get(f) is not None}))

    out = []
    for key, sw in sweeps.items():
        if len(sw["points"]) < MIN_POINTS:
            skipped["fewer than three concurrencies"] += len(sw["points"])
            continue
        pts = []
        for conc in sorted(sw["points"]):
            reps = sw["points"][conc]
            per_point: dict = {}
            for _, _, stated in reps:
                for k_, v_ in stated.items():
                    per_point.setdefault(k_, set()).add(v_)
            pts.append({
                "concurrency": conc,
                "measured_tpot_us": round(sum(p[0] for p in reps) / len(reps), 3),
                "measured_ttft_us": round(sum(p[1] for p in reps) / len(reps), 3),
                "repeats": len(reps),
                "tpot_spread": round(max(p[0] for p in reps) / min(p[0] for p in reps),
                4)
                if min(p[0] for p in reps) > 0 else None,
                # One value where the repeats agree, the sorted set where they do not,
                # so a disagreement stays visible rather than being averaged away.
                "passed": {k_: (sorted(v_)[0] if len(v_) == 1 else sorted(v_))
                           for k_, v_ in sorted(per_point.items())},
            })
        p = sw["parallelism"]
        tag = f"tp{p['tp_size']}"
        if p["moe_ep_size"] > 1:
            tag += f"-ep{p['moe_ep_size']}"
        if p["attention_dp_size"] > 1:
            tag += f"-dp{p['attention_dp_size']}"
        if sw["spec_method"] != "none":
            tag += f"-{sw['spec_method']}"
        # One scenario file per distinct stated configuration. The index disambiguates
        # two
        # deployments that share model, chip, precision and parallelism but state
        # different
        # settings -- without it they would share a file and one would be scored wrong.
        base = f"{sw['model']}-{sw['gpu']}-{sw['precision']}-vllm-{tag}"
        sw["scenario"] = f"{base}.yaml"
        sw["points"] = pts
        sw["config_ids"] = sorted(sw["config_ids"])
        sw["dates"] = sorted(sw["dates"])
        out.append(sw)

    by_base: collections.Counter = collections.Counter()
    for sw in out:
        by_base[sw["scenario"]] += 1
    seen: collections.Counter = collections.Counter()
    for sw in sorted(out, key=lambda s: (s["scenario"], s["workload"])):
        if by_base[sw["scenario"]] > 1:
            stem = sw["scenario"][:-5]
            seen[stem] += 1
            sw["scenario"] = f"{stem}-cfg{seen[stem]}.yaml"

    out.sort(key=lambda s: (s["model"], s["gpu"], s["scenario"], s["workload"]))
    doc = {
        "source": "SemiAnalysis InferenceX database dump",
        "ground_truth": "mean_tpot and mean_ttft in microseconds, from the dump itself",
        "what": "absolute latencies with each run's own stated configuration",
        "independent_of": "NVIDIA's e2e accuracy artifact; no relatives, no anchor",
        "deployment_key": list(DEPLOYMENT_FIELDS),
        "per_point_fields": list(PER_POINT_FIELDS),
        "min_points_per_sweep": MIN_POINTS,
        "scope": {
            "framework": "vllm",
            "serving": "aggregated",
            "benchmark_type": "single_turn",
            "hardware": sorted({s["gpu"] for s in out}),
            "models": sorted({s["model"] for s in out}),
            "speculative_arms": "kept and marked, not dropped",
        },
        "repeat_agreement": (
            "Where a point's repeated runs agree on every per-point setting, their "
            "measured TPOT agrees to a median 1.000x over 551 points, 95% within 5%. "
            "Where they disagree on one, the median is 1.211x over 243 points, which "
            "is the settings differing rather than noise: max_num_seqs and the capture "
            "size both change performance. Each point records what it ran, with a list "
            "where its runs disagree, so a consumer can restrict to the agreeing set. "
            "The worst remaining cases are all at concurrency 1 or 2, where a few "
            "requests make the mean noisy."),
        "skipped": dict(skipped),
        "sweeps": out,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    npts = sum(len(s["points"]) for s in out)
    print(f"{len(out)} sweeps, {npts} points, "
          f"{len({s['model'] for s in out})} models -> {args.out}")
    for why, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n:6d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
