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
| `engine_version` | string | yes | The exact release string of the engine the run describes, such as `0.29.0`. Selects the [rules pack](rules.md). |
| `workload` | mapping | no | The traffic, as a shape, a trace or a replay. See [Workload](workload.md). Omitted for a question about capacity alone. |
| `routing` | mapping | no | Pins expert-parallel routing imbalance for the run, by a routing-profile set in the registry or an inline value. See [Scenario pin](routing.md#scenario-pin). Omitted, the workload and the registry's default sets decide, else evenly loaded. |
| `acceptance` | mapping | no | Pins speculative acceptance for the run, by a profile set in the registry or an inline value. See [Scenario pin](acceptance.md#scenario-pin). Omitted, the workload and the registry decide. |
| `cluster` | mapping | yes | The hardware available, below. |

### `cluster`

<!-- fields: scenario.Cluster -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `hardware` | string | yes | A chip under `hardware/` in the catalog. Every node carries it. |
| `fabric` | string | if `nodes` > 1 | A fabric under `networks/` in the catalog. Cross-node cost comes from it. |
| `nodes` | integer | yes | Node count, at least 1. |
| `gpus_per_node` | integer | yes | GPUs per node, at least 1. |
| `gpus_per_rack` | integer | no | GPUs per rack, where a rack boundary is a different link from a node boundary. A multiple of `gpus_per_node`. |
| `storage` | list of strings | no | Storage classes from the catalog's `devices/storage.yaml` that the cluster provides. If any are listed, a deployment's offload tiers must use them. |

## Checks

A scenario is invalid if

- `kind` is not `Scenario`, or `name`, `model` or `engine_version` is empty;
- `coefficients` is empty;
- `cluster.hardware` is empty, or `nodes` or `gpus_per_node` is less than 1;
- `gpus_per_rack` is set and is not a multiple of `gpus_per_node`;
- the cluster has more than one node and names no `fabric`;
- a `storage` entry is empty or appears twice;
- `workload` is present and is not a valid [binding](workload.md#binding).
- `acceptance` is present and is not a valid [pin](acceptance.md#scenario-pin).

Whether the names resolve is checked by `ValidateAgainstCatalog`, not by `Validate`; see
[Validation](../concepts/validation.md#do-the-names-resolve).

Source: [`spec/scenario`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/scenario).
