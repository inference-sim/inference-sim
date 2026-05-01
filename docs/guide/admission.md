# Admission Control

Admission control is the first gate in the cluster pipeline. It decides whether to accept or reject incoming requests before they reach the routing stage. Admission only applies in cluster mode (`--num-instances` > 1) -- single-instance simulations skip directly to the wait queue.

```bash
# Rate-limit a 4-instance cluster with token bucket admission
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 500 --num-requests 2000 \
  --admission-policy token-bucket \
  --token-bucket-capacity 10000 --token-bucket-refill-rate 1000
```

## Available Policies

| Policy | Flag Value | Behavior |
|--------|-----------|----------|
| **Always-admit** | `--admission-policy always-admit` (default) | Accepts all requests unconditionally. No filtering. |
| **Token-bucket** | `--admission-policy token-bucket` | Rate-limiting. Each request consumes tokens equal to its input token count. Tokens refill at a constant rate. Rejects when the bucket is empty. |
| **Tier-shed** | `--admission-policy tier-shed` | SLO-aware shedding. Under overload, rejects requests whose SLO tier priority is below `tier_shed_min_priority`. See [SLO Tier Priorities](#slo-tier-priorities) below. |
| **GAIE-legacy** | `--admission-policy gaie-legacy` | Saturation-based shedding matching production llm-d/GAIE behavior. Non-sheddable requests always pass; sheddable requests (priority < 0) rejected when pool-average saturation >= 1.0. See [GAIE-Legacy Admission](#gaie-legacy-admission) below. |
| **Reject-all** | `--admission-policy reject-all` | Rejects all requests unconditionally. Pathological template for testing. |

## Token Bucket Mechanics

The token bucket policy controls throughput by treating each request's input token count as a cost:

| Flag | Description | Default |
|------|-------------|---------|
| `--token-bucket-capacity` | Maximum number of tokens the bucket can hold | 10000 |
| `--token-bucket-refill-rate` | Tokens added per second of simulation time | 1000 |

How it works:

1. **Bucket starts full.** At initialization, the bucket holds `capacity` tokens.
2. **Refill is continuous.** On each admission decision, the bucket refills proportionally to elapsed simulation time: `refill = elapsed_microseconds * refill_rate / 1,000,000`.
3. **Cost per request = number of input tokens.** A request with 512 input tokens costs 512 tokens from the bucket.
4. **Admission check.** If `current_tokens >= cost`, the request is admitted and the cost is subtracted. Otherwise the request is rejected with reason `"insufficient tokens"`.
5. **Capacity cap.** Tokens never accumulate beyond `capacity`, even after long idle periods.

!!! example "Sizing the bucket"
    With `--token-bucket-capacity 10000 --token-bucket-refill-rate 1000` and requests averaging 512 input tokens, the sustained admission rate is roughly `1000 / 512 ~ 1.95 req/s`. The bucket's capacity of 10000 tokens allows a burst of up to `10000 / 512 ~ 19` requests before rate-limiting kicks in.

Rejected requests are counted in the output anomaly counters (`Rejected Requests`) and in the full pipeline conservation formula (`num_requests == injected_requests + rejected_requests`), but they never enter the routing stage or any instance queue. Every rejection — regardless of admission policy — is also recorded in the per-SLO-class `ShedByTier` counter, so you can see which request classes are being rejected (e.g., `{"batch": 12, "sheddable": 8}`).

## When to Use Admission Control

- **Overload protection.** When the arrival rate significantly exceeds service capacity, unbounded queues grow without limit. Admission shedding keeps queue depth manageable.
- **Cost control.** Limit total token throughput to match a token budget or downstream rate limit.
- **Graceful degradation.** Shed excess load to protect latency for admitted requests. Under extreme overload, routing distributes load and scheduling orders within instances, but neither can reduce total queue depth — admission is the lever that can. The **tier-shed** policy provides SLO-aware shedding — rejecting lower-priority classes (batch, sheddable, background) before higher-priority ones (critical, standard).
- **Testing rejection paths.** The `reject-all` policy verifies that rejection counting, trace recording, and conservation invariants hold when no requests are admitted.

!!! tip "Admission is the third lever"
    Routing distributes load across instances. Scheduling orders requests within each instance. But when total arrival rate exceeds total service capacity, neither routing nor scheduling can reduce the queue -- they can only redistribute it. Admission control is the mechanism that actually reduces inbound volume.

## SLO Tier Priorities

Every request carries an `SLOClass` label that determines its priority throughout the admission pipeline. Priorities follow the GAIE (Gateway API Inference Extension) convention where **negative priority means sheddable**.

### Default Priorities

| SLO Class | Priority | Sheddable? |
|-----------|----------|------------|
| `critical` | 4 | No |
| `standard` | 3 | No (also the default for empty/unknown classes) |
| `batch` | -1 | Yes |
| `sheddable` | -2 | Yes |
| `background` | -3 | Yes |

The sheddable/non-sheddable boundary is `priority < 0`. This matches the `IsSheddable` contract from llm-d's gateway implementation.

### Customizing Priorities

Override specific priorities via the `slo_priorities` field in a policy bundle YAML:

```yaml
admission:
  policy: "tier-shed"
  tier_shed_min_priority: 3
  slo_priorities:
    batch: 0       # promote batch to non-sheddable
    critical: 10   # widen the gap between critical and standard
```

Unspecified classes retain their GAIE defaults. The `slo_priorities` map merges on top of defaults — you only need to specify the classes you want to change.

### Where Priorities Are Used

Priorities affect three components:

1. **Tier-shed admission** (`sim/admission.go`): Under overload, rejects requests with `Priority(class) < MinAdmitPriority`. With the default `tier_shed_min_priority: 3`, this admits critical (4) and standard (3), while rejecting batch (-1), sheddable (-2), and background (-3).

2. **Tenant budget enforcement** (`sim/cluster/cluster_event.go`): When a tenant exceeds their capacity budget, only sheddable requests (`IsSheddable = priority < 0`) are shed. Critical and standard traffic is always protected regardless of budget.

3. **Gateway queue dispatch** (`sim/cluster/gateway_queue.go`): In `priority` dispatch mode, higher-priority requests are dequeued first. When the queue is at capacity, the lowest-priority request is evicted.

## Tier-Shed Admission

The `tier-shed` policy sheds lower-priority SLO tiers under cluster overload. It activates when the maximum per-instance in-flight load exceeds `tier_shed_threshold`:

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 500 --num-requests 2000 \
  --admission-policy tier-shed \
  --policy-config policies.yaml
```

Where `policies.yaml` contains:

```yaml
admission:
  policy: "tier-shed"
  tier_shed_threshold: 0        # 0 = shed at any load level
  tier_shed_min_priority: 3     # admit standard(3)+critical(4), shed the rest
```

Under overload, any request with `Priority(class) < tier_shed_min_priority` is rejected. Under normal load (below threshold), all requests are admitted regardless of priority.

!!! tip "Choosing tier_shed_min_priority"
    - `3` (default): Admits critical and standard. Sheds batch, sheddable, background.
    - `0`: Admits all non-sheddable classes (priority >= 0). Sheds only negative-priority classes.
    - `-3`: Admits everything (effectively disables tier-shed). Useful when you want tenant budget enforcement but not tier-level shedding.

## GAIE-Legacy Admission

The `gaie-legacy` policy replicates the saturation-based admission behavior from production llm-d's [Gateway API Inference Extension (GAIE)](https://github.com/kubernetes-sigs/gateway-api-inference-extension). It uses a two-tier decision tree:

1. **Non-sheddable requests** (priority >= 0: critical, standard) are **always admitted**, regardless of cluster saturation.
2. **Sheddable requests** (priority < 0: batch, sheddable, background) are rejected when pool-average saturation >= 1.0.

### Saturation Formula

The saturation formula averages per-instance utilization ratios across the cluster, taking the most constrained resource (compute queue or memory) for each instance:

```
saturation = avg across instances of max(queueDepth / qdThreshold, kvUtilization / kvThreshold)
```

This matches the production GAIE implementation in `gateway-api-inference-extension/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go:115-137`.

When saturation >= 1.0, the cluster is considered overloaded and sheddable traffic is rejected. When saturation < 1.0, all requests pass.

### Configuration

Configure via `--policy-config` YAML (no CLI flags for thresholds, consistent with tier-shed):

```yaml
admission:
  policy: "gaie-legacy"
  gaie_qd_threshold: 5     # queue depth threshold per instance (default: 5)
  gaie_kv_threshold: 0.8   # KV cache utilization threshold, in (0, 1.0] (default: 0.8)
```

| YAML Field | Type | Default | Description |
|------------|------|---------|-------------|
| `gaie_qd_threshold` | float | 5 | Per-instance queue depth at which the QD component reaches 1.0. Must be > 0. |
| `gaie_kv_threshold` | float | 0.8 | Per-instance KV cache utilization at which the KV component reaches 1.0. Must be in (0, 1.0]. |

### Default Justification

Both defaults come directly from the GAIE production source code:

- **`gaie_qd_threshold = 5`**: From `DefaultQueueDepthThreshold` in `saturationdetector/utilization/config.go:31`. Represents the "ideal" queue capacity for a single endpoint — at 5 queued requests per instance, the compute resource is considered at capacity.
- **`gaie_kv_threshold = 0.8`**: From `DefaultKVCacheUtilThreshold` in `saturationdetector/utilization/config.go:33`. At 80% KV cache utilization, the memory resource is considered at capacity, leaving 20% headroom for continuous batching dynamics.

### Edge Cases

!!! note "Empty cluster"
    When there are no instance snapshots (e.g., all instances are still loading), saturation defaults to 1.0 — a conservative choice matching GAIE's behavior where stale or missing metrics are treated as fully saturated (`detector.go:116-118`). Non-sheddable requests still pass; sheddable requests are rejected.

!!! note "Stale metrics"
    GAIE production treats stale per-pod metrics (older than `MetricsStalenessThreshold`, default 200ms) as score=1.0. BLIS does not model per-snapshot staleness — signal freshness is controlled globally via `--snapshot-refresh-interval` (INV-7). This is a deliberate simplification: BLIS controls the simulator clock, so signal freshness is deterministic.

!!! tip "Choosing thresholds"
    The default thresholds (QD=5, KV=0.8) match production llm-d. Lower `gaie_qd_threshold` makes the policy more aggressive about shedding under queue buildup. Lower `gaie_kv_threshold` makes it more sensitive to KV cache pressure. Both thresholds follow the same validation as GAIE: `QD > 0` (strictly positive), `KV in (0, 1.0]`. Extreme values (e.g., `gaie_qd_threshold: 0.001`) are accepted as long as they pass validation — BLIS does not clamp or warn. Use the GAIE defaults unless you have a specific reason to change them.

### Comparison with Tier-Shed

| Aspect | `tier-shed` | `gaie-legacy` |
|--------|-------------|---------------|
| **Signal** | Max per-instance effective load (QueueDepth + BatchSize) | Pool-average saturation (QD and KV ratios) |
| **Granularity** | Configurable priority threshold (`tier_shed_min_priority`) | Binary: sheddable (priority < 0) vs non-sheddable |
| **Activation** | When any instance exceeds `tier_shed_threshold` | When pool-average saturation >= 1.0 |
| **Production parity** | BLIS-specific | Matches llm-d/GAIE |

## Flow Control Admission

When `--flow-control` is enabled, the `FlowControlAdmission` policy replaces the configured
admission policy. In this mode, admission and queuing are a single step -- the queue IS the
admission decision, matching llm-d's `FlowControlAdmissionController`.

### How It Works

1. Incoming request is enqueued into a per-priority-band, per-flow queue
2. Each unique (TenantID, Priority) pair gets its own FIFO queue within a priority band
3. Dispatch iterates bands highest-priority first (strict priority across bands)
4. Within a band, the request with the earliest arrival (lowest sequence ID) is dispatched first (global-strict fairness)
5. Saturation gating: dispatch only when cluster saturation < 1.0
6. Completion-triggered dispatch: each completion frees capacity and tries to dispatch from the queue

### Per-Band Capacity

| Flag | Description | Default |
|------|-------------|---------|
| `--per-band-capacity` | Max requests per priority band (0=unlimited) | 0 |

When a band reaches its capacity limit, incoming requests for that band are rejected unless a sheddable
entry within the band can be evicted. The global `--max-gateway-queue-depth` limit applies across all bands.

### Example

```bash
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 \
  --per-band-capacity 100 --max-gateway-queue-depth 500
```

### Comparison with Legacy Admission

| Aspect | Legacy (AlwaysAdmit, TierShed, etc.) | FlowControlAdmission |
|--------|--------------------------------------|---------------------|
| Admission | Separate from queuing | Queue IS admission |
| Queue structure | Single heap | Per-priority-band, per-flow |
| Dispatch order | `--dispatch-order` (fifo/priority) | Always priority across bands |
| Capacity | Global only | Per-band + global |

## Pipeline Latency

The `--admission-latency` and `--routing-latency` flags model real network and processing overhead between gateway and backend (gRPC hops, service mesh serialization, queue dispatch). These are pipeline concerns that affect both admission and routing stages. See [Cluster Simulation](cluster.md#admission-and-routing-latency) for details on configuring pipeline latency.

## Further Reading

- [Cluster Simulation](cluster.md) -- full pipeline overview
- [Routing Policies](routing.md) -- the next stage after admission
- [Cluster Architecture](../concepts/architecture.md#admission-pipeline) -- architectural details
