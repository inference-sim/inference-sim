# Deployment

<p class="lede">The choices made against a scenario: how its cluster is divided into pools,
each pool's parallel layout and engine settings, the KV offload tiers, and how KV moves
from prefill to decode. A deployment is validated with its scenario, because most of what
makes a layout feasible depends on the cluster.</p>

```yaml
--8<-- "docs/contracts/examples/deployment-disaggregated.yaml"
```

## Keys

<!-- fields: deployment.Deployment -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `Deployment`. |
| `name` | string | yes | The deployment's identity. |
| `pools` | list | yes | At least one pool, below. |
| `offload` | mapping | no | KV offload tiers below HBM. |
| `pd_transfer` | mapping | if any pool is `prefill` | How KV moves from prefill to decode. |
| `router` | mapping | no | The inputs of the llm-d router that routes the deployment's requests. Absent: the router's built-in configuration over every pod. |

## Pools

<!-- fields: deployment.Pool -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | `colocated`, `prefill` or `decode`. |
| `nodes` | integer | yes | Nodes in the pool, at least 1. |
| `parallel` | mapping | yes | The parallel layout of each engine in the pool. |
| `engine` | mapping | no | The engine's settings for the pool, held unopened; the engine model opens and checks it. See below. |
| `labels` | mapping | no | Kubernetes labels on the pool's model-server pods, as its pod template sets them. The router selects and filters endpoints by them. `llm-d.ai/role` is derived from `role` (`colocated` -> `prefill-decode`, `prefill` -> `prefill`, `decode` -> `decode`); given here, it must have that value. |

A pool may run several engines with the same layout and settings. Its `nodes` is the
pool's whole extent; one engine may occupy only part of it.

Pools of role `prefill` and `decode` come together or not at all, and neither may appear
beside a `colocated` pool.

### `parallel`

<!-- fields: deployment.Parallelism -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tp` | integer | yes | Tensor-parallel width, at least 1. |
| `pp` | integer | yes | Pipeline-parallel width, at least 1. |
| `dp` | integer | yes | Data-parallel width, at least 1. |
| `dp_local` | integer | no | Data-parallel replicas resident on one node. Must divide `gpus_per_node` and not exceed `dp`. Zero means the launch decides. |
| `data_parallel_load_balancing` | string | no | How the data-parallel ranks are exposed, which decides what a router chooses between. One of the [load-balancing modes](vocabularies.md#data-parallel-load-balancing). Omitted, `internal`. `hybrid` and `external` need `dp` > 1. The rules pack refuses `external` on a model without experts and `hybrid` without `dp_local`, and names the mode that runs when the engine resolves `hybrid` to another. `--api-server-count` is not part of it: it sets how many frontend processes share an endpoint, not which ranks a router can reach. |
| `pcp` | integer | no | Prefill-context-parallel width. Zero or 1 means off. |
| `dcp` | integer | no | Decode-context-parallel width. Reuses the tensor-parallel ranks; see below. Zero or 1 means off. |
| `enable_expert_parallel` | boolean | no | Shard MoE experts across ranks. Default false. |

There is no key for the expert-parallel width. It is derived: `tp × pcp × dp` when expert
parallelism is enabled, and 1 otherwise. A key for it would let a document state a width
no engine would build. [Parallel layouts](../design/layouts.md) explains this and the
checks below in more depth.

### `engine`

A pool's `engine` block is the engine's own settings, and this schema holds it unopened: it
loads the block as written and passes it through unchanged. The engine model that a
scenario's `engine_version` selects opens it, and its rules check it. A validator passes
those rules in (`Bundle.Rules`), and they report a problem in the block at its path below
`deployment.pools[i].engine`. For vLLM the keys, their defaults and their checks are
blis-vllm's (`rules/engineconfig`, with its reference in blis-vllm's `docs/engine-config.md`).

## Offload

<!-- fields: deployment.Offload -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tiers` | list | yes | At least one tier, below. |
| `eviction_policy` | string | no | KV offload cache policy. Omitted means `lru`. |
| `prefetch_depth` | integer | no | How many steps ahead a fetch is issued. |
| `connector` | string | no | The KV connector that implements offload. |
| `spec` | string | no | The offloading spec, such as `CPUOffloadingSpec`. |
| `connector_options` | mapping | no | The connector's own options, held unopened for the engine model; see below. |

### `offload.connector_options`

The offload connector's own options. This schema holds them unopened, as it does a pool's
`engine` block, and the engine model's rules check them (`Bundle.Rules`). A problem in them
is reported at its path below `deployment.offload.connector_options`.

### `offload.tiers`

<!-- fields: deployment.Tier -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tier` | string | yes | The storage class this tier uses. If the scenario's cluster lists `storage`, it must be one of those. |
| `bytes` | integer | yes | Capacity of the tier in bytes, at least 1. |

## PD transfer

<!-- fields: deployment.PDTransfer -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `connector` | string | no | The KV connector that moves KV from prefill to decode. |

There is no bandwidth key. Transfer bandwidth comes from the scenario's fabric and a
registry coefficient, so it cannot be set here.


## `router`

<!-- fields: deployment.Router -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `epp_flags` | list | no | The router's (EPP's) command-line flags, verbatim, one list entry per flag (`--name=value`). They select the pool with `--pool-name` or `--endpoint-selector`. A Helm chart's `epp.flags` map is rendered to this list (`name: value` becomes `--name=value`). |
| `endpoint_picker_config` | string | no | The `EndpointPickerConfig` file, as the router reads it with `--config-file`; a blank path is an error. Naming the configuration here and in `epp_flags` is an error. |
| `manifests` | list | no | Kubernetes manifest files. The router reads their `InferencePool`, `InferenceObjective` and `InferenceModelRewrite` objects; other kinds are ignored. Objectives and rewrites take effect only for an `InferencePool` (`--pool-name`); with a standalone pool (`--endpoint-selector`) the router does not reconcile them, so a consumer may refuse them there. |

Paths are relative to the directory of the file that holds the deployment. The block names
the router's own inputs and nothing else: scorers, weights, admission and flow control are
configured in those files, exactly as for a real deployment, so the same files configure the
real router and a simulated one. There is no deployment key for a routing decision. The router here is the Endpoint Picker. The
multimodal Coordinator pipeline is a separate llm-d component and is not named by this block; a
simulator that cannot host it refuses it at startup.

These checks are syntactic. A deployment that passes them is well formed, not necessarily
routable by a given consumer: a simulator refuses, at its startup and with the reason, router
flags, plugins or manifests it cannot run faithfully.

`pools[].labels` are a different kind of input from the three above. In a real deployment the
labels live in each pool's pod template; a deployment here has no pod spec, so the labels restate
them, so that an `InferencePool` selector or a label filter has the pods' labels to match.

## Checks

### On its own

A deployment is invalid if

- `kind` is not `Deployment`, `name` is empty, or `pools` is empty;
- a pool's `role` is not recognized, or its `nodes` is less than 1;
- `tp`, `pp` or `dp` is less than 1, `pcp` or `dcp` is negative, `dp_local` is negative,
  or `dp_local` exceeds `dp`;
- `dcp` does not partition the group it reuses. With `pcp` at most 1, `tp` must be
  divisible by `dcp`. With `pcp` above 1, `dcp` must be 1, `pcp` or `tp × pcp`;
- a `prefill` pool has no `decode` pool, a `decode` pool has no `prefill` pool, or a
  `colocated` pool appears beside either;
- a `prefill` pool is present and `pd_transfer` is absent;
- `offload` is present with no tiers, a tier is unnamed or named twice, a tier's `bytes`
  is below 1, or `prefetch_depth` is negative;

It draws a warning if expert parallelism is enabled but the derived width is 1. The checks of
a pool's `engine` block are its engine model's.

### Against its scenario's cluster

These run when the bundle holds both documents. A deployment without its scenario is
itself reported, because these checks would otherwise be skipped.

- The pools' `nodes` must sum to the cluster's `nodes`. A deployment lays out the whole
  cluster.
- A pool's `dp_local`, when set, must divide `gpus_per_node`.
- An engine needs one GPU per rank, `pp × tp × pcp × dp` in all, and they must fit in the
  pool's nodes. Decode-context and expert parallelism add no ranks.
- With `dp_local` set, the local replicas must fit on one node in the way the engine places
  them. The check refuses only layouts that no launch could place. In the rare case that
  its bounded search cannot decide, it reports a warning that the layout's fit is
  unverified.
- Each offload tier must be one of the cluster's `storage` classes, if the cluster lists any.

Engine-release checks, such as whether a backend name exists or whether prefill-context
parallelism may be combined with data parallelism, are in the [rules pack](rules.md).

Source: [`spec/deployment`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/deployment).
