# Scenarios and their inputs

<p class="lede">A kernel is built from a scenario file and the catalog and registry files the
scenario names. This page shows a scenario, says where each of its names leads, and describes
where the catalog and registry it reads come from.</p>

## A scenario file

A scenario file holds two YAML documents. The first, the *scenario*, fixes the problem: the
model, the hardware, the coefficient sets and the engine version. The second, the
*deployment*, is the configuration chosen for it: one or more pools of GPUs, each with its
parallelism and its engine settings. They are kept apart because a search over
configurations varies the deployment while the scenario stays fixed.

```yaml
--8<-- "docs/contracts/examples/scenario.yaml"
---
--8<-- "docs/contracts/examples/deployment.yaml"
```

`contracts/spec` defines both documents, and the [contracts reference](../../contracts/reference/scenario.md)
documents every field. The kernel reads the two from one file with `LoadBundle`, which rejects an unknown key as strictly as its other
loaders do, and validates them before building anything.

## Where the names lead

| In the scenario | Read from |
|---|---|
| `model: llama-3.1-8b-instruct` | `<catalog>/models/llama-3.1-8b-instruct/graph.yaml`: the model as a graph of operations |
| `cluster.hardware: h100` | `<catalog>/hardware/h100.yaml`: the chip's peak rates, memory and SM count |
| `cluster.fabric`, if present | `<catalog>/networks/<fabric>.yaml`: the network between nodes |
| (always) | `<catalog>/devices/storage.yaml`: the storage tiers offload can use |
| `coefficients: [...]` | `<registry>/coefficients/<set>.yaml`, one file per set |
| `engine_version` | The engine rules for that version of vLLM, from the caller's `Repos.Rules` (the vLLM rules packs in [blis-vllm](https://github.com/inference-sim/blis-vllm)) |

A deployment that spans nodes should name its fabric, because collectives that cross nodes
are priced from it. Of the coefficient sets a scenario may name, three are required:
without `cost-model-primitives`, `cost-model-collectives` or `cost-model-memory`, `New`
refuses. A missing attention, recurrent or host-overheads set leaves the terms it covers
unpriced, and the kernel records each missing coefficient the deployment needed as an
assumption.

`Repos` gives the three roots: where the scenario files are, the catalog checkout, and the
registry checkout. It also carries the engine rules, `Rules`, which the caller supplies: the
kernel looks no rules up itself, so `Open` refuses a `Repos` without them.
[Price a batch from Go](go.md) shows it in use.

## In this repository

The catalog and the registry are this repository's own `catalog/` and `registry/`, so the
cost model's tests always read the data it is released with. To test against a candidate
instead, set `BLIS_CATALOG` or `BLIS_REGISTRY`:

```sh
go test ./kernel/...
BLIS_REGISTRY=/path/to/candidate/registry go test ./kernel/...
```

The scenario files the cost model was once tested against carry engine settings, so they
need an engine to lower them; they live in inference-sim, under `testdata/kernel/`.
