# Admission Control

Admission and flow control in a cluster simulation are the llm-d router's. BLIS runs the
router's own admission controller and flow-control layer (see [Routing](routing.md)); it has no
admission logic of its own. This page covers how the router admits, queues and sheds simulated
requests, how BLIS SLO classes reach it, and how its outcomes are counted. Single-instance runs
have no router and admit every request.

The mapping from each removed BLIS admission and flow-control flag to its llm-d configuration
is the table in [Routing: From BLIS's former routing features](routing.md#from-bliss-former-routing-features).

## Request priority: SLO classes and InferenceObjectives

Each request carries an SLO class (`slo_class` on its workload client). The router receives it
as the `x-llm-d-inference-objective` header and the request's tenant as
`x-llm-d-inference-fairness-id`. The router turns the objective name into a priority by looking
up the `InferenceObjective` of that name; a request whose objective it does not know gets
priority 0. A priority below 0 makes a request **sheddable**.

The router reconciles `InferenceObjective` objects only for an `InferencePool`, so priorities
apply when the EPP flags select a pool with `--pool-name` and the manifests (`--epp-manifests`)
carry the pool and its objectives. With the default standalone pool (`--endpoint-selector`),
every request has priority 0 and nothing is sheddable.
`examples/llm-d/inferencepool.yaml` holds a pool over every simulated pod and one objective per
BLIS SLO class, at the same priorities as BLIS's engine SLO priority map; select it with
`examples/llm-d/pool-flags.yaml`:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry --num-instances 2 --workload-spec my-workload.yaml \
  --epp-flags examples/llm-d/pool-flags.yaml --epp-manifests examples/llm-d/inferencepool.yaml
```

| SLO class | Objective priority | Sheddable |
|---|---|---|
| `critical` | 4 | No |
| `standard` | 3 | No |
| `batch` | -1 | Yes |
| `sheddable` | -2 | Yes |
| `background` | -3 | Yes |

The router reconciles both served `InferenceObjective` versions, `llm-d.ai/v1` (`spec.poolRefs`) and `llm-d.ai/v1alpha2` (`spec.poolRef`), with their CRD defaults applied (see [Routing](routing.md#the-routers-inputs)).

## Legacy admission (flow control off)

With the `flowControl` feature gate off (the default), the router's legacy admission
controller checks each request before scheduling: a sheddable request is rejected when the
saturation detector reports the pool saturated (saturation >= 1). Non-sheddable requests are
never rejected by it. The saturation detector is `utilization-detector` unless the
configuration names another (`concurrency-detector`, `max-saturation-detector`); the
utilization detector's thresholds are its `queueDepthThreshold` (default 5 waiting requests)
and `kvCacheUtilThreshold` (default 0.8) parameters. It reads the metrics the router scrapes
from the simulated engines, so its view is as old as the last scrape
(`--refresh-metrics-interval`).

An admitter plugin (`probabilistic-admitter`) can reject requests in addition.

## Flow control

With `featureGates: [flowControl]` in the EndpointPickerConfig, the router queues requests
instead of shedding them. Requests are grouped into priority bands (one per objective
priority, from the reconciled InferenceObjectives) and, within a band, into flows by fairness
ID. A request is dispatched when the saturation detector reports capacity; otherwise it waits.
The `flowControl` block configures:

| Setting | What it does |
|---|---|
| `maxRequests`, `maxBytes` | Global capacity. A request arriving at a full queue is rejected. |
| `priorityBands[]` (`priority`, `maxRequests`, `maxBytes`, `defaultRequestTTL`, `fairnessPolicyRef`, `orderingPolicyRef`) | Per-band capacity, queue-wait budget and policies. `defaultPriorityBand` and `defaultNegativePriorityBand` set the bands not listed. |
| `defaultRequestTTL` | Queue-wait budget (default 60s). A request that cannot dispatch within it is shed. `0s` disables it. |
| `saturationDetector.pluginRef` | The detector that gates dispatch. |
| `usageLimitPolicyPluginRef` | `static-usage-limit-policy`, `soft-reflective-ceiling-policy` |
| `bandSelectionPolicyPluginRef` | `strict-band-selection-policy`, `priority-holdback-policy` |

Fairness policies are `global-strict-fairness-policy` and `round-robin-fairness-policy`;
ordering policies are `fcfs-ordering-policy`, `edf-ordering-policy` and
`slo-deadline-ordering-policy`. The dispatch tick (1 ms) and the expiry sweep (1 s) are the
router's own. In-flight eviction (`enableEviction`) is refused at startup.

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
flowControl:
  maxRequests: "200"
  defaultRequestTTL: 10s
  defaultPriorityBand:
    fairnessPolicyRef: round-robin-fairness-policy
plugins:
- type: round-robin-fairness-policy
- type: queue-scorer
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
```

## Outcomes and INV-1

Every request the router does not dispatch lands in one bucket of request conservation
([INV-1](../contributing/standards/invariants.md#inv-1-request-conservation)):

| Router outcome | Bucket | Anomaly counter |
|---|---|---|
| Legacy admission shed a sheddable request, or an admitter rejected it | `rejected_requests` | `Rejected Requests (Admission)` |
| No endpoint can serve it | `routing_rejections` | `Rejected Requests (Routing)` |
| Flow control rejected it: queues at capacity | `gateway_queue_rejected` | `Gateway Queue Rejected` |
| It waited in flow control past its TTL | `gateway_expired` | `Gateway Expired (TTL)` |
| It is still queued in flow control at the end of the run | `gateway_queue_depth` | -- |

Admission rejections and TTL expiries are also counted per SLO class (`Shed (<class>)`). See
[Results](results.md) for the output format.

## Engine-level priority

Separately from the router, each instance's scheduler uses an engine priority derived from the
SLO class through `SLOPriorityMap`: `critical` 4, `standard` 3 (also for an empty or unknown
class), `batch` -1, `sheddable` -2, `background` -3. Priority scheduling and priority
preemption use it (see [Scheduling & Priority](scheduling.md)). Override classes in the
policy bundle (`--policy-config`), in the range -100 to 100; unlisted classes keep their
defaults:

```yaml
priority:
  slo_priorities:
    batch: 0       # raise batch above the other negative classes
    critical: 10
```

`priority.slo_priorities` affects only the engines. The router's priorities come from the
InferenceObjectives, so to change both, change both.

## Further reading

- [Routing](routing.md) -- the router's inputs, supported plugins and the feature mapping table
- [Cluster Simulation](cluster.md) -- the pipeline around the router
- [Results](results.md) -- anomaly counters and per-SLO output
