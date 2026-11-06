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

Every value written is stated by the corpus (design X19: a generator never writes an engine
default). The artifact corpus's topology record states model, gpu, precision, framework and
the parallelism block; a sweep carrying `stated_config` (the InferenceX direct corpus: what the
run's own args line passed) also states block_size, gpu_memory_utilization,
enable_prefix_caching and the speculative config. Those are written as stated, and nothing
else is: an unstated setting is left out, so the engine resolves it as vLLM does
(`blis fields` lists each field and its rule). In particular no block_size, cache dtype,
batch caps or CUDA-graph mode is written unless the run stated it.

Two values are not engine settings but the deployment's shape, and are derived rather than
stated: the cluster's node count (from the rank count and eight GPUs per node) and, for a
multi-node deployment, its fabric (ib-400g, the catalog's InfiniBand entry). max_model_len is
the third: vLLM 0.32's port requires it stated (an unstated window is the model's or the
memory's, which the caller resolves), and the corpus does not state it, so it is the smallest
power of two, at least 32768, that admits the longest isl+osl the deployment is scored at.
A window shorter than that would describe a deployment that cannot serve its own points.

Every file names engine {kind: vllm, version: "0.32.0"}, the release inference-sim runs
(release plan §3.3). The kernel encodes vLLM 0.31.0's behaviour whatever release a fixture
names until the boundary transition (M2), and scripts/kernel/priceprobe reports
what this regeneration moved.

# Expert parallelism

The rule this asserts: with EP on, each rank holds whole experts and moe_tp_size is 1; with
EP off, experts are sliced tensor-parallel and moe_tp_size equals tp_size. That is vLLM's
rule only at attention_dp_size 1 -- with EP off and dp above one vLLM slices experts over
dp x tp ranks (flatten_tp_across_dp_and_pcp, vllm/model_executor/layers/fused_moe/config.py at
v0.31.0) -- and it is the corpus's own convention, which states both fields; this asserts
the corpus is consistent with it rather than assuming it.

Usage:

    python3 scripts/kernel/gen_aisimulate_scenarios.py <corpora>/aisimulate_e2e.json \
        -o testdata/kernel/aisimulate
    python3 scripts/kernel/gen_aisimulate_scenarios.py <corpora>/inferencex_direct.json \
        -o testdata/kernel/direct
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

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

# Stated drafters the catalog graph does not describe: the kernel refuses a deployment that
# drafts with a method other than its graph's speculator (liftDraft, draft.go). MiniMax-M3's
# graph states its own MTP head; EAGLE3 runs a separate draft checkpoint.
UNPRICED_SPECULATORS = {("minimax-m3", "eagle3")}

# The blis-substrate catalog whose hardware/ entries say which chips fix their own packaging:
# $BLIS_CATALOG, else the catalog/ of a blis-substrate checkout beside this repository.
# --catalog names another.
DEFAULT_CATALOG = Path(os.environ.get("BLIS_CATALOG") or
                       Path(__file__).resolve().parents[3] / "blis-substrate" / "catalog")


def catalog_packaging(chip: str, catalog: Path) -> dict[str, int]:
    """The GPUsPerNode and GPUsPerRack that chip's catalog entry fixes, by key; {} if none.

    A chip whose product fixes its packaging (GB200 NVL72's 4 per tray and 72 per rack) owns
    those figures, and a scenario for it states neither (design SUB-3; scenario.Cluster.Packaging
    resolves them). Read as plain top-level `Key: integer` lines, so the script needs no YAML
    library; a missing entry is an error, not "fixes nothing".
    """
    path = catalog / "hardware" / f"{chip}.yaml"
    try:
        text = path.read_text()
    except OSError as e:
        raise SystemExit(f"error: cannot read the catalog entry for {chip!r}: {e}")
    out = {}
    for key in ("GPUsPerNode", "GPUsPerRack"):
        m = re.search(rf"^{key}:\s*(\d+)\s*$", text, re.M)
        if m and int(m.group(1)) > 0:
            out[key] = int(m.group(1))
    return out


def node_size(dep: dict, catalog: Path) -> tuple[int, bool]:
    """(GPUs per node, whether the scenario states it) for one deployment row.

    A chip whose catalog entry fixes GPUsPerNode owns it: the scenario omits the field. For any
    other chip the row states the node size, and a row that does not is refused: no node size is
    invented here (X19). No chip in the current corpora fixes its own packaging.
    """
    fixed = catalog_packaging(dep["gpu"], catalog)
    if "GPUsPerNode" in fixed:
        return fixed["GPUsPerNode"], False
    if "gpus_per_node" in dep:
        n = dep["gpus_per_node"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise SystemExit(f"error: {dep['scenario']}: gpus_per_node {n!r} is not a positive integer")
        return n, True
    if dep["gpu"] not in HGX_8GPU_CHIPS:
        raise SystemExit(
            f"error: {dep['scenario']}: the catalog entry for {dep['gpu']!r} does not fix "
            f"GPUsPerNode, the row states no gpus_per_node, and {dep['gpu']!r} is not one of the "
            f"8-GPU HGX chips this script assumes ({', '.join(sorted(HGX_8GPU_CHIPS))}); "
            f"no node size is invented (X19)")
    return DEFAULT_NODE_SIZE, True


# The node size of a row that states none, for a chip whose catalog entry leaves it open.
#
# The corpora state no node size: no AISimulate or InferenceX row has a gpus_per_node. This
# script assumes 8 for the chips in HGX_8GPU_CHIPS, and every fixture it writes says so in a
# comment, so a reader sees the value was not recorded. The assumption is a fact about the
# cluster the sweep ran on, which the scenario owns where the catalog leaves it open, not an
# engine default. What supports it, exactly:
#   - b200, b300: the catalog entries name the part as the 8-GPU HGX baseboard
#     (blis-substrate catalog/hardware/b200.yaml: "8-GPU HGX B200 baseboard"; b300.yaml: "HGX
#     B300 system total of 62 TB/s divided by 8 GPUs").
#   - h100, h200: the catalog entries say SXM (SXM5, NVLink 4) and fix no node size; the
#     catalog's storage-device note speaks of "a DGX/HGX H100 node" (catalog/devices/
#     storage.yaml). Neither document states 8 for them; the 8 is NVIDIA's HGX H100/H200
#     board, which the corpora's tp8 and ep8 sweeps on one node also presuppose.
#   - a100-80, a100-sxm: the same HGX A100 8-GPU board; no corpus row uses them.
# A chip outside the list is refused: a future corpus row for another chip cannot silently get 8.
DEFAULT_NODE_SIZE = 8
HGX_8GPU_CHIPS = frozenset({"h100", "h200", "b200", "b300", "a100-80", "a100-sxm"})

# The engine release every generated file names: the one inference-sim runs (LIF-6).
ENGINE_KIND = "vllm"
ENGINE_VERSION = "0.32.0"

# The fields a run's stated_config may carry, in the order the header names them.
STATED_FIELDS = ("block_size", "gpu_memory_utilization", "enable_prefix_caching",
                 "speculative_method", "num_speculative_tokens")


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


def emit(dep: dict, workloads: set[str], labels: set[str], catalog: Path = DEFAULT_CATALOG) -> str:
    par = dep["parallelism"]
    tp, ep, dp = par["tp_size"], par["moe_ep_size"], par["attention_dp_size"]
    # A dense model records moe_ep_size as null: it has no expert group, which is a different
    # thing from one unsharded group (ep=1). Expert parallelism is off either way.
    expert_parallel = ep is not None and ep > 1
    ranks = tp * dp
    per_node, states_node_size = node_size(dep, catalog)
    nodes = max(1, -(-ranks // per_node))
    quant = QUANT[dep["precision"]]
    max_len = context_length(workloads)
    name = dep["scenario"][: -len(".yaml")]
    stated = dep.get("stated_config", {})
    spec = stated.get("speculative_method")
    unpriced_spec = (dep["model"], spec) in UNPRICED_SPECULATORS

    lines = [
        f"# {dep['model']} on {dep['gpu']}, {dep['precision']} under {dep['framework']},",
        f"# as NVIDIA's AISimulate end-to-end accuracy artifact records this deployment.",
        "#",
        "# GENERATED by scripts/kernel/gen_aisimulate_scenarios.py from the corpus's own",
        "# record. Re-run after a corpus update rather than editing here.",
        "#",
        f"# Topology states: tp_size {tp}, pp_size {par['pp_size']}, attention_dp_size {dp},",
        f"# moe_ep_size {ep}, moe_tp_size {par['moe_tp_size']}. Workloads scored at this",
        f"# deployment: {', '.join(sorted(labels))}.",
        "#",
        "# Only stated values are written (design X19); the engine resolves every other",
        "# setting as vLLM 0.32.0 does. Stated by the run: "
        + (", ".join(k for k in STATED_FIELDS if k in stated) or "none beyond the topology") + ".",
        f"# max_model_len {max_len} is derived from the workload: the engine requires it stated",
        "# and the corpus does not state it.",
    ]
    if spec and unpriced_spec:
        lines += [f"# speculative: the run drafts with {spec} x"
                  f"{stated['num_speculative_tokens']}, NOT written: the catalog graph",
                  "# describes no such drafter, so the kernel would refuse the file."]
    lines += [
        "kind: Scenario",
        f"name: {name}",
        "engine:",
        f"  kind: {ENGINE_KIND}",
        f'  version: "{ENGINE_VERSION}"',
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
    ]
    if states_node_size and "gpus_per_node" not in dep:
        lines.append("  # gpus_per_node: 8 assumed: the corpus states none; every chip in it is an 8-GPU HGX board.")
    if states_node_size:
        # Only a chip whose entry leaves the node size open has it stated here. One that fixes
        # it (and its rack) is left to the catalog: Cluster.Packaging fills both in.
        lines.append(f"  gpus_per_node: {per_node}")
    lines += [
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
    if "block_size" in stated:
        lines.append(f"      block_size: {stated['block_size']}")
    lines.append(f"      max_model_len: {max_len}")
    if "gpu_memory_utilization" in stated:
        lines.append(f"      gpu_memory_utilization: {stated['gpu_memory_utilization']}")
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
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG,
                    help="blis-substrate catalog whose hardware/ entries fix packaging")
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
        text = emit(dep, workloads[name], labels[name], args.catalog)
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
