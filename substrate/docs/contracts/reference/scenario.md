# Scenario

<p class="lede">The problem a run is given: a model, the cluster it may use, the traffic it
serves, the coefficient sets that price it, and the engine release it runs. Everything in
it is fixed for the run. It names catalog and registry entries rather than containing
them.</p>

```yaml
--8<-- "docs/contracts/examples/scenario-disaggregated.yaml"
```

## Keys

<!-- fields: scenario.Scenario -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `Scenario`. |
| `name` | string | yes | The scenario's identity. |
| `model` | string | yes | A model directory under `models/` in the catalog. |
| `coefficients` | list of strings | yes | Coefficient sets in the registry, named by file, at least one. How a cost model combines them is under [Selecting an entry](coefficients.md#selecting-an-entry). |
| `engine` | mapping | yes, or `engine_version` | The engine that runs the scenario, below. The simulator chooses the engine by it and refuses a release it does not run. |
| `engine_version` | string | no | Deprecated; replaced by `engine`, and removed in M4. A scenario may state it alone, or beside `engine` with the same version. |
| `workload` | mapping | no | The traffic, as a shape, a trace or a replay. See [Workload](workload.md). Omitted for a question about capacity alone. |
| `routing` | mapping | no | Pins expert-parallel routing imbalance for the run, by a routing-profile set in the registry or an inline value. See [Scenario pin](routing.md#scenario-pin). Omitted, the workload and the registry's default sets decide, else evenly loaded. |
| `acceptance` | mapping | no | Pins speculative acceptance for the run, by a profile set in the registry or an inline value. See [Scenario pin](acceptance.md#scenario-pin). Omitted, the workload and the registry decide. |
| `cluster` | mapping | yes | The hardware available, below. |

### `engine`

<!-- fields: scenario.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The engine, as its repository registers it: `vllm` or `sglang`. |
| `version` | string | yes | The upstream release the engine models, such as `0.32.0`. Selects the [rules pack](rules.md). |

### `cluster`

<!-- fields: scenario.Cluster -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `hardware` | string | yes | A chip under `hardware/` in the catalog. Every node carries it. |
| `fabric` | string | if `nodes` > 1 | A fabric under `networks/` in the catalog. Cross-node cost comes from it. |
| `nodes` | integer | yes | Node count, at least 1. |
| `gpus_per_node` | integer | unless the chip states it | GPUs per node, at least 1. Required when the chip's entry states no `GPUsPerNode`; otherwise omit it or state the entry's value. See [Packaging](#packaging). |
| `gpus_per_rack` | integer | no | GPUs per rack, an NVLink domain that spans nodes. Only a chip's entry states a rack: omit it or state the entry's `GPUsPerRack`. See [Packaging](#packaging). |
| `storage` | list of strings | no | Storage classes from the catalog's `devices/storage.yaml` that the cluster provides. If any are listed, a deployment's offload tiers must use them. |

### Packaging

A cluster's packaging is the GPUs in one node and in one rack. A rack is an NVLink domain
that spans nodes, priced at the chip's `IntraRackBwGBps`. Two documents own the two values:

| Value | Owner when the chip's product fixes it | Owner otherwise |
|---|---|---|
| GPUs per node | the chip's [catalog entry](hardware.md), `GPUsPerNode` (GB200 NVL72: 4) | the scenario, `gpus_per_node` (an HGX H100 server: 8) |
| GPUs per rack | the chip's catalog entry, `GPUsPerRack` (GB200 NVL72: 72) | nobody: the chip has no rack tier |

`Cluster.Packaging(chip)` resolves the two against the chip's entry, and every reader
calls it: the cost model, the engines' placement, the deployment-fit checks and the
Result's [`input.packaging`](result.md#input). A field the scenario omits takes the
entry's value. A field it restates unchanged is accepted. Any other value is refused, with a
message naming the field, both values and the entry. The resolved packaging records each
value's source, `stated` or `catalog`.

A scenario's `gpus_per_rack` on a chip whose entry states no `GPUsPerRack` is refused. Such
a chip has no rack tier at all: the entry may not state `IntraRackBwGBps` without
`GPUsPerRack`, and the cost model prices a rack only from the entry. A different NVLink
domain is a different product, and needs its own catalog entry.

## Checks

A scenario is invalid if

- `kind` is not `Scenario`, or `name` or `model` is empty;
- it states neither `engine` nor `engine_version`; `engine` lacks `kind` or `version`; or
  both are stated and `engine_version` differs from `engine.version`;
- `coefficients` is empty;
- `cluster.hardware` is empty, `nodes` is less than 1, or `gpus_per_node` or `gpus_per_rack`
  is negative;
- `gpus_per_rack` and `gpus_per_node` are both set and the first is not a multiple of the
  second;
- the cluster has more than one node and names no `fabric`;
- a `storage` entry is empty or appears twice;
- `workload` is present and is not a valid [binding](workload.md#binding).
- `acceptance` is present and is not a valid [pin](acceptance.md#scenario-pin).

Whether the names resolve is checked by `ValidateAgainstCatalog`, not by `Validate`; see
[Validation](../concepts/validation.md#do-the-names-resolve).

The packaging checks need the chip's entry, so `Validate` does not run them. They run in
`Cluster.Packaging`, which `ValidateAgainstCatalog`, the bundle validator given a `Chip`, and
the kernel call. A cluster is refused there if

- `gpus_per_node` is stated and differs from the entry's `GPUsPerNode`;
- `gpus_per_rack` is stated and differs from the entry's `GPUsPerRack`;
- `gpus_per_node` is omitted and the entry states no `GPUsPerNode`;
- `gpus_per_rack` is stated and the entry states no `GPUsPerRack`.

A bundle with a deployment and no `Chip` is refused when the scenario omits `gpus_per_node`,
since the deployment's fit to the cluster could not be checked.

Source: [`spec/scenario`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/scenario).
