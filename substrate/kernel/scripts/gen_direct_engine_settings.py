#!/usr/bin/env python3
"""Turn the direct corpus's stated configuration into the engine-settings file the
scoring harness already consumes.

# Why this exists rather than widening the scenario generator

A scenario file describes ONE deployment, so it can carry a setting only if that setting
is
constant across the sweep. In this corpus three of the four per-point fields are not:
max_num_seqs varies within 80 of the 117 sweeps that state it, and
max_cudagraph_capture_size within 58 of 285. Writing either into the scenario would
describe a deployment that half the points did not run.

The harness already solves this. `harness.LoadEngineSettings` reads a per-sweep record,
keyed (scenario, label), whose `by_concurrency` block carries what each point passed and
configures each
point as its run was rather than from the scenario's defaults. That is the mechanism the
measured-configuration tier uses; this script writes the same file for the direct
corpus,
so no scoring code changes.

# What this fixes, concretely

Without it the generic scenario defaults stand in for stated values, and they disagree.
On
minimax-m3-h200-fp8-vllm-tp4 the log states block_size 128 and max_model_len 2304 where
the
scenario writes 16 and 32768. Block size sets the KV page granularity, so scoring that
sweep from the default measures a different deployment -- which is the opposite of the
apples-to-apples comparison this corpus exists to make.

Usage:
    python scripts/gen_direct_engine_settings.py \
        -c testdata/measurements/inferencex_direct.json \
        -o testdata/measurements/inferencex_direct_settings.json
"""

import argparse
import json
from pathlib import Path

# The PassedSettings fields the harness reads, and where each is stated. A field in the
# deployment key is constant across the sweep by construction; the rest vary per point.
DEPLOYMENT_LEVEL = ("block_size", "enable_prefix_caching", "gpu_memory_utilization")
PER_POINT = ("max_num_seqs", "max_num_batched_tokens", "max_model_len")


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v in ("True", "False"):
        return v == "True"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    settings = []
    for sw in corpus["sweeps"]:
        dep = sw["stated_config"]
        by_conc = {}
        for p in sw["points"]:
            passed = {
                "block_size": as_int(dep.get("block_size")),
                "enable_prefix_caching": as_bool(dep.get("enable_prefix_caching")),
                "gpu_memory_utilization": as_float(dep.get("gpu_memory_utilization")),
            }
            for f in PER_POINT:
                v = p["passed"].get(f)
                # A list means this point's repeated runs disagreed on the field. Taking
                # either would describe one run and score the mean of both, so the field
                # is
                # left unstated and the scenario's default applies -- recorded in
                # `disagreeing_fields` so the count is visible rather than silent.
                passed[f] = as_int(v) if not isinstance(v, list) else None
            by_conc[str(p["concurrency"])] = {
                "passed": passed,
                # The dump publishes no resolved-engine lines for these rows, so the
                # resolved block is empty rather than filled with the request values.
                # The
                # harness treats a zero KV pool as "not measured" and derives it.
                "resolved": {"resolved_gpu_kv_tokens": 0,
                             "resolved_max_concurrency": 0.0,
                             "engine_version": ""},
            }
        settings.append({
            "scenario": sw["scenario"],
            "label": sw["label"],
            "framework": sw["framework"],
            "gpu": sw["gpu"],
            "source_run_date": sw["dates"][-1] if sw.get("dates") else "",
            "source_config_id": sw["config_ids"][0] if sw.get("config_ids") else 0,
            "by_concurrency": by_conc,
        })

    disagreeing = sum(
        1 for sw in corpus["sweeps"] for p in sw["points"]
        for f in PER_POINT if isinstance(p["passed"].get(f), list))
    doc = {
        "source": corpus["source"],
        "source_url": "",
        "release_tag": "",
        "run_selection": (
            "Not a run selection. Every row sharing a stated configuration is kept and "
            "averaged, so no policy chooses between runs; the settings here are what "
            "that configuration states."),
        "disagreeing_fields": disagreeing,
        "settings": settings,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    npts = sum(len(s["by_concurrency"]) for s in settings)
    print(f"{len(settings)} sweeps, {npts} concurrency points -> {args.out}")
    print(f"  per-point fields left unstated because repeats disagreed: {disagreeing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
