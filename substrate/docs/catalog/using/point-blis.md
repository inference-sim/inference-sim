# Point BLIS at the catalog

Clone a release and set `BLIS_CATALOG` to the clone's root:

```sh
git clone --branch <release> --depth 1 https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog
```

Replace `<release>` with the release your version of BLIS declares compatible; [Releases and pinning](releases.md) says where that declaration is.

## How BLIS finds the catalog

BLIS takes the catalog root from the `--catalog` flag or, failing that, from the `BLIS_CATALOG` environment variable. When both are set to different paths, the flag wins and BLIS logs a warning. When neither is set, the run is refused. There is no default location and no search path.

A relative path is resolved against the current working directory. The path names the clone root, not `models/` or a single file: every namespace is a subdirectory of that root.

## How a run names its entries

On the kernel backend, a run is described by a *scenario file*. It holds two YAML documents. The `Scenario` names the model, the chip, the fabric and the coefficient sets. The `Deployment` states, for each *pool*, its role, its node count, its parallelism and its engine settings. A pool is one serving engine. A colocated pool runs prefill and decode together; a disaggregated deployment has separate prefill and decode pools. You supply the scenario; the catalog supplies the facts its names refer to, and [blis-registry](https://github.com/inference-sim/blis-registry) supplies the coefficients.

```sh
blis run --latency-model blis-latency-kernel \
  --registry ~/blis-registry \
  --scenarios ~/scenarios --scenario deepseek-v3-h200-fp8-sglang-tp8.yaml
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

| The scenario names | The kernel reads |
| :--- | :--- |
| `model: deepseek-v3` | `models/deepseek-v3/graph.yaml` |
| `cluster.hardware: h200` | `hardware/h200.yaml` |
| `cluster.fabric: ib-400g`, when present | `networks/ib-400g.yaml` |
| (always) | `devices/storage.yaml` |

The kernel prices the model's [model graph](../concepts/model-graphs.md), not its `config.json`. Each file is loaded strictly, so a misspelled key is an error rather than a silent zero. A name with no entry is an error that names the path it looked for; BLIS does not fetch a missing model from Hugging Face. BLIS runs only models the catalog has an entry for.

## Which program reads which file

On the kernel backend, inference-sim itself also reads three kinds of catalog file, with its own strict readers rather than through blis-schemas:

| inference-sim reads | When |
| :--- | :--- |
| `workloads/<name>.yaml` | a run names a workload shape with `--workload` (a replay workload is bound through a scenario instead) |
| `devices/storage.yaml` | KV-cache offload names a storage tier |
| `models/<name>/config.json` | a disaggregated (prefill/decode) run |

Those readers expect the formats of the catalog release inference-sim declares compatible, which can be older than the formats on this site. Clone that release; a newer one can fail at startup. [Releases and pinning](releases.md) explains why.

## Record which catalog a result came from

A result can be reproduced only if you know which catalog produced it. Record the clone's commit, and whether the clone differs from it, beside each result:

```sh
git -C "$BLIS_CATALOG" describe --tags --always    # 0.2.1, or 0.2.1-3-g1a2b3c4 after it
git -C "$BLIS_CATALOG" status --porcelain          # no output means no edits
```

Any output from the second command, including a file git does not track yet, means the catalog has been edited and the run is an experiment. A release tag with no edits names data anyone can fetch.
