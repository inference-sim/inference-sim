#!/usr/bin/env python3
"""Extract the scoreable per-point corpus from NVIDIA AISimulate's e2e accuracy artifact.

The artifact is published by the `E2E Accuracy Matrix` workflow in ai-dynamo/aisimulate as
`e2e-accuracy-web`, and carries a nested summary:

    models -> workloads -> gpus -> topologies -> points

Each point holds `measured`, `aic` and `aisimulate` blocks of TTFT/TPOT relatives. A topology
is one deployment at one workload; its points are a concurrency sweep.

# Provenance, stated because it is mixed

The predictions and the snapshot are NVIDIA's. The `measured_*` ground truth is NOT: the
artifact's own `measurement_source` is "SemiAnalysis InferenceX", with a release URL. This
project's REGISTRY is single-sourced from AISimulate and contains no InferenceX data -- that
constraint is about coefficients, the numbers the model is built from. This file is an
evaluation set, a different role: it is what the model is scored against, never fitted to. The
distinction is recorded here so a later reader does not mistake one for the other.

# Why relatives, and what that costs

Absolute latencies are not disclosed, so every value is normalised to its sweep's lowest
concurrency. That is why no simulator is needed to score it -- a ratio of step times at one
fixed deployment is a pure function of the kernel -- and also why it measures shape only.

# Selection

A topology is emitted when it has at least three successful points. Two points carry a single
ratio, which pins a level rather than a curvature -- scoring a shape against it measures almost
nothing and dilutes the reported figure with sweeps that cannot disagree. Aggregated serving with no speculation is the whole of the reachable set
here; both are asserted rather than assumed, so a future artifact that adds disaggregated or
speculative arms fails loudly instead of being scored under the wrong deployment.

Usage:

    python scripts/extract_aisimulate_e2e.py summary.json -o testdata/measurements/aisimulate_e2e.json
"""

import argparse
import collections
import json
import sys

# Artifact model slug -> blis-catalog model name. A slug absent here is skipped: the catalog
# has no graph for it, so it cannot be priced.
MODELS = {
    "gptoss120b": "gpt-oss-120b",
    "minimaxm2.5": "minimax-m2.5",
    "qwen3.5": "qwen3.5-397b-a17b",
    "glm5": "glm-5",
    # The snapshot measures meta-llama/Meta-Llama-3.1-70B; the catalog carries the Instruct
    # variant. The mapping is VERIFIED rather than assumed on the grounds that instruction
    # tuning changes no shape: the base config is gated on HuggingFace, so it was read from
    # two independent mirrors (unsloth/Meta-Llama-3.1-70B and
    # NousResearch/Meta-Llama-3.1-70B) and compared field by field against the committed
    # Instruct config. All thirteen shape fields agree -- 80 layers, 8192 hidden, 28672
    # intermediate, 64 q heads over 8 kv, 128256 vocab, bfloat16, rope_theta 500000,
    # 131072 positions, silu, rms_norm_eps 1e-05, tie_word_embeddings false,
    # LlamaForCausalLM, no head_dim -- with zero differences.
    #
    # It is the corpus's largest model at 408 points and its only DENSE one; the other four
    # are all MoE, so without it the comparison says nothing about dense architectures.
    "llama70b": "llama-3.1-70b-instruct",
    # Added to blis-catalog this cycle, each with its graph DERIVED by the catalog's own
    # scripts/derive_graph.py rather than written: deepseek-v3 prices as MLA plus routed
    # experts with a three-layer dense prologue, and kimi-k2.5 as the same family -- vLLM
    # instantiates its text decoder as DeepseekV2ForCausalLM over config.text_config, and the
    # text_config names DeepseekV3ForCausalLM itself.
    #
    # DeepSeek-V4-Pro and MiniMax-M3 are deliberately absent. Both carry a learned
    # block-sparse indexer (index_n_heads / sparse_attention_config), which the schema names
    # as sparse_mla but blis-registry has no coefficients for; the kernel would fall back to
    # the generic attention rate and misprice them silently.
    "dsr1": "deepseek-v3",
    "kimik2.5": "kimi-k2.5",
    # MiniMax-M3 is MiniMaxM3SparseForConditionalGeneration, a vision-language checkpoint
    # whose text decoder the artifact measures. Its attention is block-sparse: the read is
    # bounded by sparse_topk_blocks * sparse_block_size and a separate indexer scores which
    # blocks to read. The catalog derives it from the published config at revision
    # f0e1c1e0, with every shape cross-checked against the safetensors headers.
    "minimaxm3": "minimax-m3",
    # DeepSeek-V4-Pro: compressed sparse attention at two ratios with a top-k indexer.
    # The catalog derives it from the published config at revision b5968e91, with every
    # bound read from vLLM's own compressor and sparse_mla rather than from field names.
    "dsv4": "deepseek-v4-pro",
}

# Artifact GPU slug -> blis-catalog hardware name.
CHIPS = {"h100": "h100", "h200": "h200", "b200": "b200", "b300": "b300"}

MIN_POINTS = 3


def scenario_name(model: str, gpu: str, precision: str, framework: str, par: dict) -> str:
    """One file per distinct deployment, named from the fields that define it.

    A DENSE model has no experts, and the snapshot records that as moe_ep_size null rather
    than 1: every one of llama70b's 81 topologies carries None where an MoE model carries an
    integer. None is not 1 -- a dense deployment has no expert group at all, where ep=1 means
    one expert group that is not sharded -- so it is read as absent rather than coerced.
    """
    ep = par["moe_ep_size"]
    parts = [model, gpu, precision, framework, f"tp{par['tp_size']}"]
    if ep is not None and ep > 1:
        parts.append(f"ep{ep}")
    if par["attention_dp_size"] > 1:
        parts.append(f"dp{par['attention_dp_size']}")
    return "-".join(parts) + ".yaml"


def extract(summary: dict) -> dict:
    sweeps = []
    skipped = collections.Counter()
    for model in summary["models"]:
        name = MODELS.get(model["model"])
        if name is None:
            skipped["model not in catalog"] += 1
            continue
        for workload in model["workloads"]:
            for gpu in workload["gpus"]:
                chip = CHIPS.get(gpu["gpu"])
                if chip is None:
                    skipped["gpu not in catalog"] += 1
                    continue
                for topology in gpu["topologies"]:
                    serving = topology["serving"]
                    spec = topology.get("spec_method") or "none"
                    if serving != "aggregated" or spec != "none":
                        # Not reachable by a step-time ratio at one deployment.
                        skipped[f"serving={serving} spec={spec}"] += 1
                        continue
                    par = topology["parallelism"]
                    if par["pp_size"] != 1:
                        skipped["pp_size>1"] += 1
                        continue
                    points = [point_record(p) for p in topology["points"]
                              if p.get("status") == "success"]
                    if len(points) < MIN_POINTS:
                        skipped["fewer than three successful points"] += 1
                        continue
                    points.sort(key=lambda p: p["concurrency"])
                    sweeps.append(
                        {
                            "scenario": scenario_name(
                                name, chip, topology["precision"],
                                topology["framework"], par,
                            ),
                            "model": name,
                            "gpu": chip,
                            "workload": workload["identity"],
                            "label": workload["label"],
                            "framework": topology["framework"],
                            "precision": topology["precision"],
                            "serving": serving,
                            "spec_method": spec,
                            "parallelism": par,
                            "topology_id": topology["id"],
                            "points": points,
                        }
                    )
    sweeps.sort(key=lambda s: (s["model"], s["gpu"], s["scenario"], s["label"]))
    return {
        "source": "NVIDIA AISimulate e2e accuracy artifact (e2e-accuracy-web)",
        "measurement_source": summary["snapshot"].get("measurement_source"),
        "measurement_source_url": summary["snapshot"].get("measurement_source_url"),
        "snapshot": summary["snapshot"],
        "scope": summary["scope"],
        # Carries all four published figures per estimator. The *_shape_error_pct ones are
        # comparable to a normalised-curve score; the *_mape_pct ones are absolute and cannot be
        # recomputed from this corpus, which ships no absolute latencies.
        "aisimulate_totals": summary["totals"].get("aisimulate", {}),
        "aic_totals": summary["totals"].get("aic", {}),
        "selection": {
            "models": sorted(set(MODELS.values())),
            "min_points_per_sweep": MIN_POINTS,
            "serving": "aggregated",
            "spec_method": "none",
            "skipped": dict(skipped),
        },
        "sweeps": sweeps,
    }


# The artifact's own field names, kept verbatim so a reader can trace a value back.
#
# TTFT is carried alongside TPOT, and AIC alongside AISIMULATE, because the snapshot publishes
# four accuracy figures and an earlier revision of this extractor kept one. TTFT is where the
# baseline is weakest -- 22.82% shape against 10.05% for TPOT over the whole snapshot -- so
# discarding it discarded the interesting half of the comparison.
#
# AIC and AISIMULATE are the same repository (snapshot.aic_source names ai-dynamo/aisimulate at a
# pinned commit): AIC is the analytic configurator path, AISimulate the full simulator. Both are
# kept so a score can be read against either.
#
# Only RELATIVES exist. Every point in the artifact carries exactly tpot_relative and
# ttft_relative per estimator, and no absolute latency field appears anywhere in it -- verified by
# enumerating every key of every point. So a mean ABSOLUTE percentage error cannot be computed
# against this corpus at all; the mape figures the snapshot publishes come from rows it does not
# ship. Shape error is the only per-point metric this file can support, which is a property of
# the data rather than a choice.
ESTIMATORS = ("aisimulate", "aic")
METRICS = ("tpot", "ttft")


def point_record(p: dict) -> dict:
    """One concurrency point: the measured relatives and every estimator's prediction."""
    rec = {"concurrency": p["concurrency"], "status": p["status"]}
    for metric in METRICS:
        rec[f"measured_{metric}_relative"] = p["measured"][f"{metric}_relative"]
    for est in ESTIMATORS:
        block = p.get(est)
        if not isinstance(block, dict):
            continue
        for metric in METRICS:
            rel = block.get(f"{metric}_relative")
            if rel is None:
                continue
            rec[f"{est}_{metric}_relative"] = rel
            err = block.get(f"{metric}_error_pct")
            if err is not None:
                rec[f"{est}_{metric}_error_pct"] = err
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="summary.json from the e2e-accuracy-web artifact")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    with open(args.summary) as fh:
        summary = json.load(fh)
    corpus = extract(summary)

    scenarios = {s["scenario"] for s in corpus["sweeps"]}
    points = sum(len(s["points"]) for s in corpus["sweeps"])
    with open(args.out, "w") as fh:
        json.dump(corpus, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"{len(corpus['sweeps'])} sweeps, {points} points, "
          f"{len(scenarios)} distinct deployments -> {args.out}")
    for reason, n in sorted(corpus["selection"]["skipped"].items()):
        print(f"  skipped {n:4d}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
