# Point BLIS at the catalog

A run of BLIS uses the catalog embedded in the blis-substrate release the simulator was built
with. There is nothing to clone or install: a released `blis` binary carries the exact catalog
it was tested with.

## Where a run's catalog comes from

Each run reads one catalog, from one of two sources (design §3.3):

The embedded copy
:   When no catalog is named, the run uses the data compiled into the substrate release.

A path
:   `--catalog <dir>`, or the `BLIS_CATALOG` environment variable, names a directory with the
    same layout as `catalog/` in blis-substrate: a working tree, a copy of it, or an exported
    revision. The run uses that directory instead of the embedded copy.

To add entries to the catalog in use rather than replace it, `--catalog-overlay <dir>` names a
directory of additions. An overlay entry is a whole model directory (`models/<name>/`), or a
whole hardware, device, network or workload file. An entry with the same identity as an
existing one replaces it; any other entry is added. The combined catalog is then validated in
full. An overlay may not contain ledger files.

The catalog flags and the registry flags (`--registry`, `--registry-overlay`) are independent.
A run may, for example, use the embedded catalog with a candidate registry.

Whatever its source, the catalog is validated when it is loaded, by the same validators CI
runs, including the checks between catalog and registry. A model runs only if it is in the
catalog in use. A missing model is refused with the catalog's source (the embedded release, or
a path) and the relative path its entry belongs at. No run creates or changes catalog files,
and no run fetches anything from the network.

## How a run names its entries

A run is described by a *scenario file*. It holds two YAML documents. The `Scenario` names the
model, the chip, the fabric and the coefficient sets. The `Deployment` states, for each *pool*,
its role, its node count, its parallelism and its engine settings. A pool is one serving
engine. A colocated pool runs prefill and decode together; a disaggregated deployment has
separate prefill and decode pools. You supply the scenario; the catalog supplies the facts its
names refer to, and the [registry](../../registry/index.md) supplies the coefficients.

```sh
blis run --scenario path/to/deepseek-v3-h200-fp8-sglang-tp8.yaml
```

An abridged scenario file:

```yaml
kind: Scenario
name: deepseek-v3-h200-fp8-sglang-tp8
engine_version: "0.29.0"           # the engine rules pack to validate against
model: deepseek-v3                 # models/deepseek-v3/graph.yaml
coefficients: [cost-model-primitives, cost-model-collectives, ...]
cluster:
  hardware: h200                   # hardware/h200.yaml
  nodes: 1
  gpus_per_node: 8
  # fabric: ib-400g                # networks/ib-400g.yaml; for a multi-node cluster
---
kind: Deployment
name: deepseek-v3-h200-fp8-sglang-tp8
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1, enable_expert_parallel: false}
    engine: {quantization: fp8, ...}
```

Every name is a path under the catalog root, used exactly as written:

| The scenario names | The cost model reads |
| :--- | :--- |
| `model: deepseek-v3` | `models/deepseek-v3/graph.yaml` |
| `cluster.hardware: h200` | `hardware/h200.yaml` |
| `cluster.fabric: ib-400g`, when present | `networks/ib-400g.yaml` |
| (always) | `devices/storage.yaml` |

The cost model prices the model's [model graph](../concepts/model-graphs.md). No runtime code
reads a model's `config.json`; every model value a run needs is a field of the graph (SUB-3).
Each file is loaded strictly, so a misspelled key is an error rather than a silent zero.

inference-sim reads catalog data only through blis-substrate's loader, so every file a run
reads, including a workload shape named with `--workload` and the storage tiers that KV-cache
offload names, comes from the catalog in use and is validated the same way.

## What a result records

Every run computes a content hash of the catalog it used: the SHA-256 of a sorted list of every
data file's SHA-256 and its path. The Result document records, for the catalog and for the
registry:

- its source: `embedded` or `path`;
- its content hash;
- the hash of each overlay, and the hash of the combined data.

It also records the build the simulator came from: a tagged release, or `unreleased`. A run
uses released data only when the simulator is a tagged release, the catalog and registry hashes
equal that release's embedded hashes, and no overlay is used. Every other run is experimental.
Only released-data results are published (X8).
