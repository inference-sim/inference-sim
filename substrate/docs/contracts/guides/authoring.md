# Write a scenario and a deployment

<p class="lede">A run is described by two documents: a scenario stating the problem and a
deployment stating the choices made against it. This guide writes one of each for a single
node, then extends them to a disaggregated deployment across four.</p>

Every example on this page is a file in
[`docs/examples/`](https://github.com/inference-sim/blis-substrate/tree/main/docs/contracts/examples),
and the test suite validates each one.

## A scenario

```yaml title="scenario.yaml"
--8<-- "docs/contracts/examples/scenario.yaml"
```

`model`
:   A directory under `models/` in the catalog.

`coefficients`
:   Coefficient sets in the registry, named by file. The five above are those inference-sim's
    scenarios name. The cost model reads them in the order listed: where two sets both supply a coefficient that applies to the
    deployment, the later set wins.

`engine_version`
:   The engine version the run describes. Version-specific rules run only when it matches a
    release the engine's rules know exactly, so write `"0.29.0"` and not `"0.29"`. Quote it so that
    YAML reads it as a string.

`workload`
:   The traffic, here the catalog's `chatbot` shape. Omit it for a question about capacity
    alone.

`cluster`
:   The hardware the problem is given. `hardware` names a file under `hardware/` in the
    catalog; every node carries that chip.

## A deployment

```yaml title="deployment.yaml"
--8<-- "docs/contracts/examples/deployment.yaml"
```

One pool serves both prefill and decode (`colocated`) on the cluster's single node. Its
layout is four data-parallel replicas, each split two ways by tensor parallelism. That uses
all eight GPUs: an engine needs `pp × tp × pcp × dp` GPUs, one per rank, and they must fit
in its pool's nodes.

Engine settings are optional. Each one omitted is left to the engine's default, as in a
real launch.

## Check them

There is no command for scenarios yet. Validating them takes a few lines of Go, explained
in [Validate documents from Go](go.md):

```go
--8<-- "contracts/example_test.go:validate"
```

```text title="Output"
ok: true rules applied: 0.29.0
```

Pass the model graph and the chip as well, as above, when you have them. Some rules need
the model; whether experts divide evenly across ranks, for instance, depends on how many
there are.

## Across nodes, disaggregated

Prefill and decode can run on separate pools, with KV cache moving from one to the other.
This scenario gives DeepSeek-V3 four H200 nodes:

```yaml title="scenario-disaggregated.yaml"
--8<-- "docs/contracts/examples/scenario-disaggregated.yaml"
```

A cluster of more than one node must name a `fabric`, because the cost of anything that
crosses a node boundary comes from the fabric, not the chip. `storage` lists the storage
classes the cluster provides.

```yaml title="deployment-disaggregated.yaml"
--8<-- "docs/contracts/examples/deployment-disaggregated.yaml"
```

Four things change.

1. There are two pools, `prefill` and `decode`. Neither may appear without the other, and
   neither may appear beside a `colocated` pool.
2. Their node counts, 1 and 3, sum to the cluster's 4. Pools always account for every node.
3. `pd_transfer` is present, as a disaggregated deployment requires, and names the KV
   connector that moves KV between the pools.
4. The offload tier uses `cpu_dram`, one of the storage classes the cluster lists.

Each engine in either pool is `dp 8` with `tp 1`: eight ranks, one node. The decode pool's
three nodes therefore run three such engines. Each pool enables expert parallelism. The
expert-parallel width is not written down: it is `tp × pcp × dp`, which is 8, and
DeepSeek-V3's 256 experts divide evenly across it. [Parallel layouts](../design/layouts.md)
explains why the width is derived.

## Traffic from a trace

To use captured traffic instead of a distribution, bind the workload to a trace file:

```yaml title="trace-binding.yaml"
--8<-- "docs/contracts/examples/trace-binding.yaml"
```

The requests stay in the CSV file that `data` names. A `workload` takes `shape` or
`trace`, never both. The keys are in [Workload](../reference/workload.md#trace).

## Problems you are likely to see

| Message | What to do |
|---|---|
| `pool node counts sum to 3 but the cluster declares 4` | Make the pools' `nodes` add up to `cluster.nodes`. |
| `needs 32 GPUs (pp 1 x tp 8 x pcp 1 x dp 4, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8` | Shrink the layout or give the pool more nodes. |
| `required for a 4-node cluster: cross-node cost resolves from the fabric, not the chip` | Add `cluster.fabric`. |
| `a prefill pool with no decode pool has nowhere to send KV` | Add the decode pool, or make the pool `colocated`. |
| `required for a disaggregated deployment`, at `pd_transfer` | Add a `pd_transfer` block naming a connector. |
| `"cpu_dram" is not in the cluster storage inventory [nvme_gen4]` | List the class under `cluster.storage`, or use one that is listed. |
| `a workload is a shape or a trace, not both; ...` | Keep one of `workload.shape` and `workload.trace`. |
| `required when a deployment is present: ...`, at `scenario` | Validate the deployment together with its scenario. |

Every key and every check is listed in [Scenario](../reference/scenario.md) and
[Deployment](../reference/deployment.md).
