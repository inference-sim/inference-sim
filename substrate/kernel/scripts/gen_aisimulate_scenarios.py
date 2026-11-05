#!/usr/bin/env python3
"""Generate one scenario+deployment file per deployment in the AISimulate e2e corpus.

# Why generated rather than committed by hand

There are 46 of them, and every field that matters is already stated by the corpus's own
topology record: model, gpu, precision, framework and the parallelism block. Transcribing
those by hand is how a scenario comes to describe a deployment other than the one it scores --
the previous corpus had two sweeps pointing at a single scenario file, so one of them was
scored against the wrong parallelism. Generating from the corpus makes that class of error
impossible: the file and the points it is scored against come from one record.

# Two documents, one file

blis-schemas v0.2.0 splits the immutable problem (Scenario: model, cluster inventory,
coefficient and engine-version references) from the tunable configuration applied to it
(Deployment: pools, each a parallelism layout with its own engine settings). Each file
written here holds both, `---` separated, rather than a scenario and a sibling
deployment file.

One file because a measurement row addresses a deployment by a single filename --
`{"scenario": "deepseek-v3-b200-fp4-sglang-tp4.yaml", ...}` -- and there are over three
thousand such rows across testdata/measurements. Splitting the pair would rewrite every
one of them to say nothing it does not already say. harness.LoadBundle reads the pair.

# What the corpus states, and what it does not

STATED, and written here: tp_size, pp_size, attention_dp_size, moe_ep_size, moe_tp_size,
framework, precision, serving, gpu, model.

NOT stated by the artifact corpus, and therefore set here: max_num_seqs, max_model_len,
block_size, gpu_memory_utilization, cudagraph_mode. A sweep carrying `stated_config` (the
InferenceX direct corpus: what the run's own args line passed) supplies
gpu_memory_utilization, block_size, enable_prefix_caching and the speculative config, which
are then written as stated. gpu_memory_utilization not stated is vLLM v0.31.0's default,
0.92 (vllm/config/cache.py:103); prefix caching and speculation not stated are left out, so
v0.31.0's defaults apply (caching on, no drafting). A stated drafter the catalog graph does
not describe is not written (see UNPRICED_SPECULATORS), and the file's header says so.
The rest are NOT vLLM's defaults, and the written header says how each differs: block_size
16 is only the nominal default, which a backend may raise (vllm/platforms/interface.py),
and a DSA sparse-MLA model is stated at 64, the size vLLM picks for it and a multiple of the
only kernel block its indexer runs (see DSA_MODELS);
max_num_seqs 256 is the default only below 70 GiB or on an A100
(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py); cudagraph_mode's default is
FULL_AND_PIECEWISE (vllm/config/compilation.py) -- all at v0.29.0 and v0.31.0. They are held fixed so committed scores stay comparable,
and the comparison scores the RATIO of step times across concurrency at a fixed deployment,
which is what a setting held constant affects least. Where a value could NOT be held it is
derived from the workload instead -- max_model_len must admit the longest sequence the
workload runs, or the scenario would describe a deployment that cannot serve its own points.

# Expert parallelism

The rule this asserts: with EP on, each rank holds whole experts and moe_tp_size is 1; with
EP off, experts are sliced tensor-parallel and moe_tp_size equals tp_size. That is vLLM's
rule only at attention_dp_size 1 -- with EP off and dp above one vLLM slices experts over
dp x tp ranks (flatten_tp_across_dp_and_pcp, vllm/model_executor/layers/fused_moe/config.py at
v0.31.0) -- and it is the corpus's own convention, which states both fields; this asserts
the corpus is consistent with it rather than assuming it.

Usage:

    python scripts/gen_aisimulate_scenarios.py testdata/measurements/aisimulate_e2e.json \
        -o testdata/aisimulate
"""

import argparse
import collections
import json
import os
import sys

COEFFICIENTS = (
    "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, "
    "cost-model-attention, cost-model-recurrent, cost-model-memory]"
)

# Artifact precision -> the engine's quantization name. bf16 means the checkpoint is served
# unquantised, which the schema spells as an absent quantization rather than a name.
# int4 is a weight-only integer format the checkpoint declares in its own
# quantization_config, which the graph already read: servedDType returns the CHECKPOINT
# width for compressed-tensors rather than a flag width, so stating a flag here would add
# nothing and risk disagreeing with the graph. Kimi-K2.5 is the case in hand --
# compressed-tensors at group_size 32, which the catalog derives as weight_dtype int4.
QUANT = {"fp4": "nvfp4", "fp8": "fp8", "bf16": None, "int4": None}

# The DSA sparse-MLA models in the corpus. Their indexer runs on 64-token kernel blocks
# (vllm/v1/attention/backends/mla/indexer.py:203-204 at v0.31.0), so vLLM picks block_size 64
# for them when none is stated and refuses a stated size that is not a multiple of 64. They
# are stated at 64; every other model at 16.
DSA_MODELS = {"glm-5"}

# The DeepSeek-V4 models in the corpus. Their sparse attention and indexer backends accept
# exactly one kernel block, 256 on every part (vllm/models/deepseek_v4/sparse_mla.py:59-61,
# vllm/v1/attention/backends/mla/indexer.py:259-262 at v0.31.0), so vLLM picks it when none
# is stated and refuses a stated size it does not accept. They are stated at it.
V4_MODELS = {"deepseek-v4-pro"}


def v4_block_size(gpu: str) -> int:
    return 256


# vLLM v0.31.0's CacheConfig.gpu_memory_utilization default (vllm/config/cache.py:103).
GPU_MEMORY_UTILIZATION_DEFAULT = 0.92

# Stated drafters the catalog graph does not describe: the kernel refuses a deployment that
# drafts with a method other than its graph's speculator (liftDraft, draft.go). MiniMax-M3's
# graph states its own MTP head; EAGLE3 runs a separate draft checkpoint.
UNPRICED_SPECULATORS = {("minimax-m3", "eagle3")}

# A KV cache dtype is not stated by the corpus. fp8 KV is the default for the fp8 and fp4
# arms in these frameworks; bf16 serving keeps an unquantised cache.
CACHE_DTYPE = {"fp4": "fp8", "fp8": "fp8", "bf16": "auto", "int4": "auto"}

GPUS_PER_NODE = 8


def context_length(workloads: set[str]) -> int:
    """The smallest power-of-two window that admits every workload in this deployment.

    A workload identity is "<isl>:<osl>"; a request occupies isl+osl tokens at its longest.
    Rounded up to a power of two because that is how these engines are configured, and kept
    at least 32768 so a short workload does not produce an implausibly tight window.
    """
    longest = max(sum(int(p) for p in w.split(":")) for w in workloads)
    window = 32768
    while window < longest:
        window *= 2
    return window


def emit(dep: dict, workloads: set[str], labels: set[str]) -> str:
    par = dep["parallelism"]
    tp, ep, dp = par["tp_size"], par["moe_ep_size"], par["attention_dp_size"]
    # A dense model records moe_ep_size as null: it has no expert group, which is a different
    # thing from one unsharded group (ep=1). Expert parallelism is off either way.
    expert_parallel = ep is not None and ep > 1
    ranks = tp * dp
    nodes = max(1, -(-ranks // GPUS_PER_NODE))
    quant = QUANT[dep["precision"]]
    max_len = context_length(workloads)
    name = dep["scenario"][: -len(".yaml")]
    block_size = 64 if dep["model"] in DSA_MODELS else 16
    if dep["model"] in V4_MODELS:
        block_size = v4_block_size(dep["gpu"])
    stated = dep.get("stated_config", {})
    block_size = stated.get("block_size", block_size)
    gmu = stated.get("gpu_memory_utilization", GPU_MEMORY_UTILIZATION_DEFAULT)
    spec = stated.get("speculative_method")
    unpriced_spec = (dep["model"], spec) in UNPRICED_SPECULATORS

    lines = [
        f"# {dep['model']} on {dep['gpu']}, {dep['precision']} under {dep['framework']},",
        f"# as NVIDIA's AISimulate end-to-end accuracy artifact records this deployment.",
        "#",
        "# GENERATED by scripts/gen_aisimulate_scenarios.py from the corpus's own topology",
        "# record. Re-run after a corpus update rather than editing here.",
        "#",
        f"# Topology states: tp_size {tp}, pp_size {par['pp_size']}, attention_dp_size {dp},",
        f"# moe_ep_size {ep}, moe_tp_size {par['moe_tp_size']}. Workloads scored at this",
        f"# deployment: {', '.join(sorted(labels))}.",
        "#",
        f"# gpu_memory_utilization {gmu}: "
        + ("stated by the run." if "gpu_memory_utilization" in stated else
           "not stated, so vLLM v0.31.0's default (vllm/config/cache.py:103)."),
        f"# block_size {block_size}: "
        + ("stated by the run." if "block_size" in stated else "not stated; see below."),
    ]
    if "enable_prefix_caching" in stated:
        lines.append("# enable_prefix_caching: stated by the run.")
    if spec and unpriced_spec:
        lines += [f"# speculative: the run drafts with {spec} x"
                  f"{stated['num_speculative_tokens']}, NOT written: the catalog graph",
                  "# describes no such drafter, so the kernel would refuse the file."]
    elif spec:
        lines.append("# speculative: stated by the run.")
    lines += [
        "#",
        "# max_num_seqs and cudagraph_mode are not in the corpus, nor is an unstated",
        "# block_size, so they are set here -- and not to vLLM's defaults. block_size 16 is its",
        "# nominal default, which a backend may raise;",
        "# max_num_seqs 256 is its default only below 70 GiB or on an A100; and its cudagraph_mode",
        "# default is FULL_AND_PIECEWISE, not PIECEWISE (all at v0.29.0 and v0.31.0). They are held",
        "# fixed so committed scores stay comparable; moving them to the engine's defaults is a",
        "# re-scoring decision. The comparison scores a ratio across concurrency at a fixed",
        "# deployment, which is what a setting held constant affects least. max_model_len is",
        "# derived from the workload instead, since a window shorter than isl+osl would describe a",
        "# deployment that cannot serve its own points.",
    ]
    if "block_size" in stated:
        pass  # stated by the run; the rule notes below do not apply
    elif dep["model"] in V4_MODELS:
        lines += [
            "#",
            f"# block_size is {block_size} here, not 16: this is a DeepSeek-V4 model, whose sparse",
            f"# attention and indexer accept only {block_size}-token kernel blocks: vLLM v0.31.0",
            "# picks that size when none is stated and refuses a stated size it does not accept",
            "# (vllm/models/deepseek_v4/sparse_mla.py:59-61).",
        ]
    elif block_size != 16:
        lines += [
            "#",
            f"# block_size is {block_size} here, not 16: this is a DSA sparse-MLA model, whose indexer",
            "# runs on 64-token kernel blocks: vLLM v0.31.0 picks 64 for it when none is stated and",
            "# refuses a stated size that is not a multiple of 64",
            "# (vllm/v1/attention/backends/mla/indexer.py:203-204).",
        ]
    lines += [
        "kind: Scenario",
        f"name: {name}",
        'engine_version: "0.29.0"',
        "",
        f"model: {dep['model']}",
        f"coefficients: {COEFFICIENTS}",
        "",
        # The hardware inventory lives inside the cluster: it is what the problem is
        # handed and a deployment cannot change it.
        "cluster:",
        f"  hardware: {dep['gpu']}",
    ]
    if nodes > 1:
        lines.append("  fabric: ib-400g")
    lines += [
        f"  nodes: {nodes}",
        f"  gpus_per_node: {GPUS_PER_NODE}",
        # The tunable half, as its own document. Same file because a measurement row
        # addresses a deployment by one filename; same name because there is exactly one
        # deployment per scenario here.
        "---",
        "kind: Deployment",
        f"name: {name}",
        "",
        "pools:",
        "  - role: colocated",
        f"    nodes: {nodes}",
        "    parallel:",
        f"      tp: {tp}",
        f"      pp: {par['pp_size']}",
        f"      dp: {dp}",
        f"      enable_expert_parallel: {str(expert_parallel).lower()}",
        "    engine:",
    ]
    if quant is not None:
        lines.append(f"      quantization: {quant}")
    lines += [
        f"      cache_dtype: {CACHE_DTYPE[dep['precision']]}",
        f"      block_size: {block_size}",
        "      max_num_batched_tokens: 8192",
        "      max_num_seqs: 256",
        f"      max_model_len: {max_len}",
        "      cudagraph_mode: PIECEWISE",
        f"      gpu_memory_utilization: {gmu}",
    ]
    if "enable_prefix_caching" in stated:
        pc = str(stated["enable_prefix_caching"]).lower()
        lines.append(f"      enable_prefix_caching: {pc}")
    if spec and not unpriced_spec:
        lines += [
            "      speculative:",
            f"        method: {stated['speculative_method']}",
            f"        num_spec_tokens: {stated['num_speculative_tokens']}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("-o", "--out", required=True, help="scenario directory")
    ap.add_argument("--check", action="store_true",
                    help="verify the files on disk match what would be generated")
    args = ap.parse_args()

    with open(args.corpus) as fh:
        corpus = json.load(fh)

    deps: dict[str, dict] = {}
    workloads = collections.defaultdict(set)
    labels = collections.defaultdict(set)
    for s in corpus["sweeps"]:
        name = s["scenario"]
        par = s["parallelism"]
        ep, moe_tp = par["moe_ep_size"], par["moe_tp_size"]
        # vLLM's expert-sharding rule, asserted rather than assumed. It governs how experts
        # are sharded, so it says nothing about a DENSE model, which has none: the snapshot
        # records both fields as null there and the rule is skipped rather than given a
        # default that would make it pass for the wrong reason.
        if ep is None and moe_tp is None:
            pass
        elif ep is None or moe_tp is None:
            return fail(f"{name}: moe_ep_size {ep!r} with moe_tp_size {moe_tp!r}; a "
                        f"deployment states both for an MoE model and neither for a dense "
                        f"one, so one present and one absent is a corpus error")
        elif ep > 1 and moe_tp != 1:
            return fail(f"{name}: moe_ep_size {ep} with moe_tp_size {moe_tp}; "
                        f"EP holds whole experts so moe_tp_size must be 1")
        elif ep == 1 and moe_tp != par["tp_size"]:
            return fail(f"{name}: EP off but moe_tp_size {moe_tp} != tp_size "
                        f"{par['tp_size']}; sliced experts span the TP group")
        prior = deps.get(name)
        if prior is not None and prior["parallelism"] != par:
            return fail(f"{name}: two deployments share one scenario name")
        deps[name] = s
        workloads[name].add(s["workload"])
        labels[name].add(s["label"])

    os.makedirs(args.out, exist_ok=True)
    stale = []
    for name, dep in sorted(deps.items()):
        text = emit(dep, workloads[name], labels[name])
        path = os.path.join(args.out, name)
        if args.check:
            try:
                with open(path) as fh:
                    current = fh.read()
            except FileNotFoundError:
                stale.append(f"{name} (missing)")
                continue
            if current != text:
                stale.append(name)
        else:
            with open(path, "w") as fh:
                fh.write(text)

    if args.check:
        if stale:
            print(f"{len(stale)} scenario(s) stale or missing:", file=sys.stderr)
            for n in stale:
                print(f"  {n}", file=sys.stderr)
            return 1
        print(f"checked {len(deps)} scenario(s): all current")
        return 0
    print(f"wrote {len(deps)} scenario(s) to {args.out}")
    return 0


def fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
