# BLIS Evolutionary Policy Optimization: Macro-Level Implementation Plan (v2)

**Date:** 2026-02-11
**Revision:** v2.3 (incorporates mock study findings + online routing architecture)
**Status:** Draft
**Target:** Multi-replica cluster simulation with pluggable policies
**Based on:** [Design Document](2026-02-06-evolutionary-policy-optimization-design.md)

---

## Revision Notes (v2.3)

This revision incorporates feedback from external review (Perplexity, Gemini, GPT-4o), scaffolding cleanup, and mock study findings:

**v2 changes (Perplexity):**
1. **Earlier research-ready checkpoint** — Metrics/fitness evaluation moved up to enable policy research after Phase 2
2. **Deferred tiered KV** — Single-tier KV sufficient for initial policy research; tiered offload/reload moves to Phase 4
3. **Mock study checkpoint** — Added after PR 3 to validate interfaces before freeze
4. **Reordered for fastest research loop** — Research-ready in ~4 weeks vs ~6 weeks

**v2.1 changes (Gemini + GPT-4o):**
5. **Interface extension point** — Added `Extended` map to `InstanceSnapshot` for Phase 4+ observables; clarified "freeze" means no breaking changes
6. **Policy lifecycle clarification** — New section F.6 documents that policy instances persist and may maintain internal state
7. **Edge case workloads** — Expanded workload generator scope to include bursty/diurnal arrivals and multi-tenant fairness scenarios
8. **Parallel trace collection** — Added note that real-world trace collection can begin during Phase 2 as separate workstream

**v2.2 changes (scaffolding cleanup):**
9. **Merged PR 3+4** — DeploymentConfig now introduced with ClusterSimulator (no scaffolding)
10. **Merged PR 12+13** — WorkloadSpec types introduced with generator (no scaffolding)
11. **Consolidated pathological templates** — Moved from policy PRs to anomaly validation PR (no test infrastructure in feature PRs)
12. **Merged PR 10+11** — RawMetrics and anomaly validation combined into single PR
13. **Reduced total PRs** — 24 → 21 PRs with no scaffolding or dead code

**v2.3 changes (mock study findings, 2026-02-13):**
14. **Online routing architecture** — ClusterSimulator restructured: routing decisions happen at arrival time during the event loop, not pre-dispatched (mock study proved pre-dispatch breaks load-aware policies)
15. **Cluster-level event queue** — New event types (ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent) with configurable per-event latency to model real control plane delays
16. **InstanceSnapshot staleness model** — Immutable value type with Timestamp; SnapshotProvider interface + ObservabilityConfig control per-field refresh (immediate/periodic/on-demand)
17. **Control plane / data plane separation** — ClusterSimulator orchestrates a ControlPlane (cluster event queue, policies, SnapshotProvider) and DataPlane (InstanceSimulators with their own event queues)
18. **PR 4 expanded** — Now includes cluster event infrastructure + AdmissionPolicy (combined); PRs 5-7 depend on PR 4 (no longer parallel with it)
19. **InstanceSimulator observation methods** — QueueDepth(), BatchSize(), KVUtilization(), FreeKVBlocks() added in PR 4 (mock study identified 4 observable gaps: queue depth, batch size, KV utilization, prefix cache state)

---

## A) Executive Summary

This plan transforms BLIS from a single-instance LLM inference simulator into a multi-replica cluster simulator with pluggable control policies.

**BLIS remains a standalone tool.** Users can:
1. Run cluster simulations directly via CLI for capacity planning
2. Use parameterized policies to experiment with routing/scheduling strategies
3. Analyze results via metrics and decision traces

**Optional framework integration.** Evolutionary frameworks (OpenEvolve, GEPA) can wrap BLIS as an evaluator—BLIS provides adapters to make this easier, but does not depend on these frameworks.

**Coefficient learning is out of scope.** BLIS consumes pre-trained coefficients (alpha/beta for latency estimation). A separate instrumentation effort collects real vLLM data to train these coefficients.

**Implementation:**
- 6 phases, 21 PRs
- **Research-ready checkpoint after Phase 2** (~5 weeks; PR 4 expanded with cluster event infrastructure)
- Each PR is CLI-exercisable immediately after merge — no scaffolding
- Estimated 10-11 weeks with 3-4 developers

---

## B) Repository Recon Summary

### B.1 Package Structure

| Package | SLOC | Responsibility |
|---------|------|----------------|
| `sim/` | 2051 | Core single-instance discrete-event simulator |
| `cmd/` | 373 | CLI interface (Cobra), configuration loading |
| `main.go` | 12 | Entry point |

### B.2 Core Data Structures

**Simulator** (`sim/simulator.go:72-113`):
- Event queue (min-heap), clock (`int64` ticks), wait queue, KV cache, running batch
- Single `*rand.Rand`, alpha/beta coefficients for latency estimation

**Request** (`sim/request.go:18-34`):
- Lifecycle: `queued → running → completed`
- Tracks `ProgressIndex`, TTFT, ITL, arrival/completion times

**Events** (`sim/event.go`):
- Interface: `Timestamp() int64`, `Execute(*Simulator)`
- Types: `ArrivalEvent`, `QueuedEvent`, `ScheduledEvent`, `StepEvent`, `RequestLeftEvent`, `PreemptionEvent`

**KVCache** (`sim/kvcache.go`):
- Block-based with LRU eviction (models vLLM's PagedAttention)
- Prefix caching via SHA256 hash matching

### B.3 Existing Assets We Can Leverage

| Asset | Location | Reuse in Plan |
|-------|----------|---------------|
| Basic workload generation | `sim/workload_config.go` | Sufficient for Phase 2 research |
| Metrics collection | `sim/metrics.go` | Foundation for RawMetrics |
| KV cache with prefix caching | `sim/kvcache.go` | Single-tier sufficient initially |

### B.4 Hardcoded Behaviors Requiring Extraction

| Behavior | Location | Target Interface |
|----------|----------|------------------|
| FIFO batch formation | `simulator.go:341-365` | `InstanceScheduler` |
| LIFO preemption | `simulator.go:248-277` | `InstanceScheduler` |
| Single-tier KV | `kvcache.go` | Tiered `KVCacheState` (Phase 4) |
| All-admit | implicit | `AdmissionPolicy` |

### B.5 Extension Strategy

- **Composition over modification**: `ClusterSimulator` wraps `InstanceSimulator` instances
- **Interface extraction**: Pull hardcoded behaviors into pluggable interfaces
- **Backward compatibility**: `--num-instances 1` produces identical results to current

### B.6 CLI Entrypoints and Flag Surface

**Current CLI structure** (`cmd/root.go`):

```
simulation_worker run
  --model STRING           # Required: model identifier
  --seed INT               # RNG seed (default: time-based)
  --workload STRING        # "distribution" | "traces"
  --workload-traces-filepath STRING
  --rate FLOAT             # requests/sec
  --max-prompts INT        # total requests
  --prompt-tokens INT      # avg input tokens
  --output-tokens INT      # avg output tokens
  --model-config-folder STRING  # enables roofline mode
  --hardware-config STRING
  --hardware STRING        # GPU type
  --tp INT                 # tensor parallelism
  --results-path STRING    # output JSON path
```

**Flags this plan will add:**
- `--num-instances INT` (PR 3)
- `--admission-policy STRING` (PR 4)
- `--admission-latency INT` (PR 4, microseconds, default 0)
- `--routing-latency INT` (PR 4, microseconds, default 0)
- `--priority-policy STRING` (PR 5)
- `--routing-policy STRING` (PR 6)
- `--scheduler STRING` (PR 7)
- `--policy-config STRING` (PR 8)
- `--fitness-weights STRING` (PR 9)

### B.7 Configuration Flow

```
CLI flags → cmd/root.go → SimulatorConfig → Simulator.Run()
                              ↓
                    defaults.yaml (alpha/beta coefficients)
                              ↓
                    model_configs/*.json (HuggingFace configs)
```

**Key observation:** Configuration is currently flat (no nested YAML). PolicyBundle (PR 8) introduces hierarchical config.

### B.8 Areas of Coupling and Fragility

| Area | Risk | Mitigation |
|------|------|------------|
| `Simulator` struct has 17 fields | Adding more fields increases constructor complexity | Use options pattern in new code |
| Event execution mutates `Simulator` directly | Hard to intercept for cluster coordination | `InstanceSimulator` wrapper provides interception point |
| Metrics collection tightly coupled to `Simulator` | `RawMetrics` needs different aggregation for cluster | New `ClusterMetrics` aggregates per-instance metrics |
| Single RNG shared across all randomness | Changing workload affects scheduler randomness | `PartitionedRNG` (PR 1) isolates subsystems |

### B.9 Open Uncertainties

| Uncertainty | Impact | Resolution |
|-------------|--------|------------|
| Will existing alpha/beta coefficients generalize to multi-instance? | Latency estimates may diverge | **RESOLVED (mock study):** Coefficients work; primary issue was dispatch architecture, not coefficients |
| ~~Is round-robin sufficient as default routing?~~ | ~~May mask policy bugs~~ | **RESOLVED:** Round-robin is a valid default; `--routing-policy` adds alternatives. Under-saturation masks all policy differences — contention workloads required (validates PR 9 pathological templates + PR 10 workload generator) |
| How will CLI flag explosion be managed? | UX degradation with 20+ flags | `--policy-config YAML` consolidates policy flags (PR 8) |

---

## C) High-Level Objectives + Non-Goals

### Objectives

1. **Multi-replica simulation** with shared clock and coordinated events
2. **Deterministic execution** — same seeds produce bit-for-bit identical results
3. **Pluggable policies** — admission, priority, routing, scheduling, auto-scaling
4. **Rich observability** — decision traces, counterfactual analysis
5. **Research-ready checkpoint** — enable policy experiments early (end of Phase 2)
6. **Tiered KV cache** — GPU + CPU with offload/reload latency modeling (Phase 4)
7. **P/D disaggregation** — separate prefill and decode pools (Phase 4)
8. **Framework adapters** — optional conveniences for GEPA/OpenEvolve

### Non-Goals

- **Coefficient training** — BLIS consumes coefficients; training is a separate effort
- **Sim-to-real validation** — deferred to instrumentation workstream
- **Production deployment** — research/planning tool only
- **GPU execution** — CPU-only simulation
- **Arbitrary code execution** — no Starlark/WASM; frameworks handle code generation

### Compatibility Guarantees

- Existing `./simulation_worker run` commands unchanged
- Single-instance mode identical to current implementation
- Output JSON schema: additions only, no breaking changes

---

## D) Modeling Decisions and Simplifications

This section documents how BLIS models real systems and where we intentionally simplify.

### D.1 KV Cache Model

**Based on:** vLLM PagedAttention [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023]

| Real System | BLIS Model | Simplification |
|-------------|------------|----------------|
| Variable block sizes | Fixed `BlockSizeTokens` | Simpler allocation math |
| GPU memory fragmentation | Perfect packing | Overly optimistic; acceptable for relative comparisons |
| Async block operations | Synchronous within step | No intra-step concurrency |

**Prefix Caching:** Models vLLM's automatic prefix caching. Hash computed over token sequence; matching prefix reuses existing blocks.

**Phased Implementation:**
- **Phase 2:** Single-tier KV (existing `sim/kvcache.go`) — sufficient for routing/scheduling research
- **Phase 4:** Multi-tier with offload/reload — fidelity enhancement

### D.2 Latency Estimation

**Based on:** Empirical coefficients from instrumented vLLM deployments.

| Component | Model | Coefficients |
|-----------|-------|--------------|
| Queueing delay | `α₀ + α₁ * input_len` | Learned from trace data |
| Step time | `β₀ + β₁ * cache_miss_tokens + β₂ * decode_tokens` | Learned from busy-loop instrumentation |
| Roofline (alternative) | FLOPs / peak_throughput | Analytical, for new hardware |

**Limitation:** Coefficients are hardware/model/TP-specific. Generalization requires re-training.

### D.3 Multi-Tier KV Cache (Phase 4)

**Based on:** vLLM CPU offloading and research systems like FlexGen [Sheng et al., 2023].

| Tier | Latency Model | Capacity Model |
|------|---------------|----------------|
| GPU | `access_latency = 0` | `gpu_blocks` parameter |
| CPU | `access_latency = transfer_time(blocks)` | `cpu_blocks` parameter |
| Storage | Deferred | — |

**Transfer model:** `time = base_latency + blocks * block_size / bandwidth`

### D.4 Prefill-Decode Disaggregation (Phase 4)

**Based on:** DistServe [Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024] and Splitwise [Patel et al., 2024].

| Real System | BLIS Model | Simplification |
|-------------|------------|----------------|
| RDMA KV transfer | Latency + bandwidth model | No queue depth modeling |
| Speculative decode | Not modeled | Out of scope |
| Multi-stage pipeline | Two-stage (P → D) only | Sufficient for policy research |

**Ownership model:** Single ownership; blocks transfer from prefill to decode instance, not duplicated.

### D.5 Router and Scheduling

**Based on:** llm-d router architecture concepts.

| Component | BLIS Model | Notes |
|-----------|------------|-------|
| Admission | ADMIT / REJECT / DELAY | Rate limiting, tenant quotas |
| Priority | Numeric score | Enables SLO differentiation |
| Routing | Filter → Score → Select | Prefix-aware, load-aware |
| Instance scheduling | Batch formation + preemption | FCFS default, pluggable |

**Routing policy freedom:** Individual routing policy implementations are free to maintain their own internal state (e.g., predicted cache state for prefix-aware routing). The router provides observable metrics and instance snapshots; policies decide what additional tracking they need.

### D.5.1 Control Plane Latency Model (New in v2.3)

Policy decisions incur latency in real systems. BLIS models this as configurable per-event delays:

| Decision Point | Real-World Latency | BLIS Default | Configurable |
|---|---|---|---|
| Admission | 10-50μs | 0 (instantaneous) | `--admission-latency` |
| Routing | 50-200μs | 0 (instantaneous) | `--routing-latency` |
| Scheduling | 0 (instance-local) | 0 | N/A (alpha model handles this) |

With zero-latency defaults, the pipeline collapses to "route immediately on arrival" — backward compatible with v2.2 behavior.

### D.5.2 InstanceSnapshot Staleness Model (New in v2.3)

InstanceSnapshot is an immutable value type captured at a specific clock time. A SnapshotProvider controls when snapshots are refreshed.

| Update Mode | Fields (default) | Real-World Analog |
|---|---|---|
| Immediate | QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, InFlightRequests | Load balancer connection tracking |
| Periodic | CacheHitRate (10ms), RecentTTFT (100ms), RecentTPOT (100ms) | Prometheus scrape, xDS push |
| On-demand | Prefix cache state (Extended map) | Health check probe |

Default: core load fields immediate; statistical fields periodic. Researchers configure staleness via ObservabilityConfig to study robustness (e.g., making KVUtilization periodic to model Prometheus scrape delay).

### D.6 Auto-Scaling

**Based on:** Kubernetes HPA patterns with LLM-specific considerations.

| Aspect | BLIS Model |
|--------|------------|
| Trigger | Periodic or event-based (queue depth, SLO) |
| Actuation | Provisioning delay + model load time + warmup |
| Scale-down | Drain policy (wait, immediate, redirect) |
| Cost modeling | Replica-seconds accumulated; configurable cost-per-replica |

**Simplification:** No predictive scaling; threshold-based default policy.

### D.7 Observable Signals (Policy Input Contract)

Policies operate under **architectural locality constraints**: they may only consume signals naturally available at their control point. This section enumerates what each policy type can observe.

**Admission Policy Observables:**
- Request: `InputTokens`, `TenantID`, `PrefixHash`, `SLOClass`
- Tenant state: `RequestCount`, `ActiveRequests`, `RecentRate`
- Global: `InFlightRequests`, `RequestRate`

**Priority Policy Observables:**
- Same as admission, plus admission decision

**Routing Policy Observables:**
- Request: as above, plus `PriorityScore`
- Per-instance: `QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`, `InFlightRequests`, `CacheHitRate`, `RecentTTFT`, `RecentTPOT`, `EstimatedWaitTime`
- Global: aggregate metrics across instances

**Instance Scheduler Observables:**
- Local only: `WaitQueue`, `RunningBatch`, `KVCacheState`
- Request metadata: `PriorityScore`, `ArrivalTime`, `InputTokens`

**AutoScale Policy Observables:**
- Deployment: `CurrentReplicas`, `MinReplicas`, `MaxReplicas`
- Aggregate metrics: `AvgQueueDepth`, `AvgUtilization`, `RequestRate`, `SLOAttainment`
- History: `TimeSinceLastScale`, recent load samples

**Non-observables (internal state):**
- Exact KV block placement (only utilization % is visible)
- Other tenants' request contents
- Future arrivals
- Internal scheduler decisions on other instances

### D.8 Failure Mode Detection

The research agenda identifies specific failure modes that policies should suppress. BLIS detects these as **anomalies** in the decision trace.

| Failure Mode | Detection Method | Metric |
|--------------|------------------|--------|
| **Priority inversion** | Higher-priority request scheduled after lower-priority | `PriorityInversionCount` |
| **Head-of-line blocking** | Request waits while lower-priority requests in same queue complete | `HOLBlockingEvents` |
| **Scale oscillation** | UP followed by DOWN (or vice versa) within cooldown window | `ScaleOscillationCount` |
| **KV thrashing** | Offload immediately followed by reload for same block | `KVThrashingRate` |

These are detected during simulation and reported in `RawMetrics` and `DecisionTrace`.

---

## E) Determinism Guarantees

Evolutionary optimization requires bit-for-bit reproducible simulations. This section specifies how BLIS achieves determinism.

### E.1 Sources of Non-Determinism and Mitigations

| Source | Risk | Mitigation |
|--------|------|------------|
| **Go map iteration** | Non-deterministic order | Use slices with explicit sorting, or `OrderedMap` |
| **Goroutines** | Scheduling non-determinism | Single-threaded event loop; no goroutines in hot path |
| **`time.Now()`** | Wall-clock dependency | Forbidden in simulation; use simulated `Clock` only |
| **Floating-point ordering** | Accumulated error differences | Use `int64` ticks for time; careful operation ordering |
| **External I/O** | File system, network | Policies are pure functions; no I/O during simulation |
| **Uninitialized memory** | Random values | Go zero-initializes; explicit initialization in structs |

### E.2 Partitioned RNG Design

```go
// Each subsystem gets a deterministically-derived RNG
type PartitionedRNG struct {
    masterSeed int64
    subsystems map[string]*rand.Rand // lazily created
}

func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
    if rng, ok := p.subsystems[name]; ok {
        return rng
    }
    // Derive seed deterministically: masterSeed XOR hash(name)
    derivedSeed := p.masterSeed ^ int64(fnv1a(name))
    p.subsystems[name] = rand.New(rand.NewSource(derivedSeed))
    return p.subsystems[name]
}
```

**Subsystems:** `"workload"`, `"router"`, `"instance_0"`, `"instance_1"`, ...

**Isolation guarantee:** Changing draws in one subsystem does not affect others.

### E.3 Event Ordering Rules

**Note:** This is the target ordering for the v2.3 cluster control plane. The current implementation (PR 3) uses timestamp-only ordering within instances; cluster-level tie-breaking uses lowest instance index. PR 4 will implement the full ordering below.

```go
// Events are ordered by: (1) timestamp, (2) type priority, (3) event ID
func (a Event) Less(b Event) bool {
    if a.Timestamp() != b.Timestamp() {
        return a.Timestamp() < b.Timestamp()
    }
    if a.TypePriority() != b.TypePriority() {
        return a.TypePriority() < b.TypePriority()
    }
    return a.ID() < b.ID() // deterministic tie-breaker
}

// Type priorities (lower = processed first)
// Cluster-level events (New in v2.3) precede instance-level events
const (
    PriorityClusterArrival  = 0  // Cluster-level request arrival
    PriorityAdmission       = 1  // Admission decision
    PriorityRouting         = 2  // Routing decision
    PriorityArrival         = 3  // Instance-level arrival (injected after routing)
    PriorityStep            = 4
    PriorityCompletion      = 5
    PriorityScaleCheck      = 6
    PrioritySnapshotRefresh = 7  // Periodic snapshot updates
)
```

### E.4 Tie-Breaking Rules

| Situation | Rule |
|-----------|------|
| Routing: equal scores | Select instance with lowest `InstanceID` (lexicographic) |
| Scheduling: equal priority | FIFO by `Request.ArrivalTime`, then `Request.ID` |
| Eviction: equal LRU time | Evict block with lowest `BlockID` |
| Preemption: equal candidates | Preempt request with highest `Request.ID` (LIFO) |

### E.5 Determinism Verification

```bash
# CI test: run 100 times, verify identical output
for i in {1..100}; do
  ./simulation_worker run --seed 42 --results-path /tmp/run_$i.json
done
md5sum /tmp/run_*.json | awk '{print $1}' | sort -u | wc -l
# Expected output: 1 (all identical)
```

---

## F) Policy Expressiveness

Without Starlark/WASM, users configure policies through **parameterized templates**.

### F.1 Policy Configuration Model

```yaml
# policies.yaml
admission:
  type: "token-bucket"
  params:
    bucket_size: 1000
    refill_rate: 100  # tokens/sec
    per_tenant: true

priority:
  type: "slo-based"
  params:
    realtime_score: 100
    batch_score: 10
    default_score: 50

routing:
  type: "weighted-scoring"
  params:
    cache_affinity_weight: 0.6
    load_balance_weight: 0.3
    queue_depth_weight: 0.1
    prefix_match_bonus: 0.5

scheduler:
  type: "priority-fcfs"
  params:
    respect_priority: true
    preemption_enabled: true
    preemption_policy: "lowest-priority"

autoscale:
  type: "threshold"
  params:
    scale_up_threshold: 0.8    # utilization
    scale_down_threshold: 0.3
    cooldown_seconds: 60
```

### F.2 Built-in Policy Templates

**Core templates (PR 4-7):**

| Policy Type | Templates Available |
|-------------|---------------------|
| **Admission** | `always-admit`, `token-bucket`, `rate-limit`, `tenant-quota` |
| **Priority** | `constant`, `slo-based`, `tenant-priority`, `deadline-aware` |
| **Routing** | `round-robin`, `least-loaded`, `weighted-scoring`, `prefix-affinity` |
| **Scheduler** | `fcfs`, `priority-fcfs`, `sjf` (shortest job first) |

**Pathological templates (PR 9, with anomaly detection):**

| Policy Type | Pathological Template |
|-------------|----------------------|
| **Admission** | `reject-all` |
| **Priority** | `inverted-slo` |
| **Routing** | `always-busiest` |
| **Scheduler** | `reverse-priority` |

**AutoScale templates (PR 11):**

| Policy Type | Templates Available |
|-------------|---------------------|
| **AutoScale** | `threshold`, `target-utilization`, `queue-depth`, `oscillator`* |

*\* Pathological template for scale oscillation testing*

### F.3 Pathological Templates (Consolidated in PR 9)

Pathological templates are introduced alongside anomaly detection in PR 9, not scattered across policy PRs. This ensures:
1. No test infrastructure in feature PRs
2. Templates are immediately testable when added
3. Anomaly detection and validation are cohesive

| Template | Purpose | Expected Anomalies |
|----------|---------|-------------------|
| `reject-all` | Admission baseline | 100% rejection rate |
| `inverted-slo` | Priority inversion testing | High `PriorityInversionCount` |
| `always-busiest` | Load imbalance testing | High `HOLBlockingEvents`, poor tail latency |
| `reverse-priority` | Scheduler fairness testing | High `PriorityInversionCount` |
| `oscillator` | Scale stability testing (PR 11) | High `ScaleOscillationCount` |

### F.4 Evolutionary Search Space

Frameworks like OpenEvolve explore the parameter space:

```python
# OpenEvolve candidate representation
candidate = {
    "routing.cache_affinity_weight": 0.7,
    "routing.load_balance_weight": 0.2,
    "routing.queue_depth_weight": 0.1,
    "admission.bucket_size": 500,
    "autoscale.scale_up_threshold": 0.75,
    # ... etc
}
```

BLIS evaluates candidates by loading parameters into built-in templates.

### F.5 Extensibility Path

For policies that can't be expressed as parameter combinations:
1. **Add new template** to BLIS (requires PR)
2. **External policy service** — BLIS calls out to user-provided HTTP endpoint (future)

### F.6 Policy Instance Lifecycle (New in v2.1)

Policy structs are **instantiated once per simulation** from the `PolicyBundle` and **persist for the entire run**. Internal state is allowed and expected for stateful policies.

**Determinism constraint:** State updates must depend only on method inputs (`clock`, `req`, `state`), never on wall-clock time (`time.Now()`) or external I/O.

**Example: TokenBucket with internal state**

```go
type TokenBucketPolicy struct {
    // Configuration (from YAML)
    bucketSize   int
    refillRate   float64  // tokens per tick

    // Internal state (persists across Decide() calls)
    tokens       float64
    lastRefillTs int64
}

func (t *TokenBucketPolicy) Decide(req *Request, state *RouterState, clock int64) AdmissionDecision {
    // Refill tokens based on simulated time elapsed (deterministic)
    elapsed := clock - t.lastRefillTs
    t.tokens = min(float64(t.bucketSize), t.tokens + float64(elapsed) * t.refillRate)
    t.lastRefillTs = clock

    // Consume token if available
    if t.tokens >= 1.0 {
        t.tokens -= 1.0
        return AdmissionDecision{Action: Admit, Reason: "token available"}
    }
    return AdmissionDecision{Action: Reject, Reason: "bucket empty"}
}
```

**Lifecycle guarantees:**
- Policy is instantiated once when `PolicyBundle` is loaded
- Same instance is used for all calls during the simulation
- State is NOT persisted across simulation runs (each run starts fresh)
- State is NOT shared across policy types (each policy has its own instance)

---

## G) Concrete Interface Definitions

### G.1 Core Policy Interfaces

```go
// === Admission ===

type AdmissionDecision struct {
    Action AdmissionAction // ADMIT, REJECT, DELAY
    Delay  time.Duration   // if Action == DELAY
    Reason string          // for tracing
}

type AdmissionAction string

const (
    Admit  AdmissionAction = "admit"
    Reject AdmissionAction = "reject"
    Delay  AdmissionAction = "delay"
)

type AdmissionPolicy interface {
    Decide(req *Request, state *RouterState, clock int64) AdmissionDecision
}

// === Priority ===

type PriorityDecision struct {
    Score float64
    Hints map[string]any // optional scheduling hints
}

type PriorityPolicy interface {
    Compute(req *Request, state *RouterState, clock int64) PriorityDecision
}

// === Routing ===

type RoutingDecision struct {
    // For monolithic architecture (PR 6+)
    TargetInstance InstanceID

    // For disaggregated P/D (PR 15+, zero-valued until then)
    // Forward-compatible: defined in PR 6, exercised in PR 15
    PrefillInstance InstanceID
    DecodeInstance  InstanceID

    // Observability
    Reason     string
    Scores     map[InstanceID]float64
    Candidates []CandidateScore // top-k for counterfactual
}

type CandidateScore struct {
    Instance  InstanceID
    Score     float64
    Breakdown map[string]float64
}

type RoutingPolicy interface {
    Route(req *Request, priority PriorityDecision, snapshots []InstanceSnapshot,
          state *RouterState, clock int64) RoutingDecision
}

// === Instance Scheduler ===

type BatchDecision struct {
    ToSchedule       []*Request
    ChunkSizes       map[RequestID]int // optional prefill chunking
    PreemptionVictim *Request          // nil if no preemption needed
}

type SchedulerContext struct {
    WaitQueue      []*Request
    RunningBatch   []*Request
    KVCache        *KVCacheSnapshot
    Clock          int64
    MaxBatchSize   int
    MaxTokenBudget int
}

type InstanceScheduler interface {
    MakeBatch(ctx SchedulerContext) BatchDecision
    OnRequestArrival(req *Request, ctx SchedulerContext)
}

// === Auto-Scale ===

type ScaleDecision struct {
    Action         ScaleAction // NONE, UP, DOWN
    TargetReplicas int
    Reason         string
}

type ScaleAction string

const (
    ScaleNone ScaleAction = "none"
    ScaleUp   ScaleAction = "up"
    ScaleDown ScaleAction = "down"
)

type AutoScaleContext struct {
    ConfigID           string
    CurrentReplicas    int
    MinReplicas        int
    MaxReplicas        int
    Metrics            DeploymentMetrics
    Clock              int64
    TimeSinceLastScale time.Duration
}

type AutoScalePolicy interface {
    Evaluate(ctx AutoScaleContext) ScaleDecision
}
```

### G.2 State Structures

```go
type RouterState struct {
    // WARNING: Maps are for O(1) lookup only. Policies must NOT iterate over
    // PerTenant or Instances maps directly (non-deterministic order).
    // Use provided accessor methods that return sorted slices.
    PerTenant  map[TenantID]*TenantState
    Global     GlobalMetrics
    Instances  map[InstanceID]*InstanceSnapshot
    Clock      int64
}

// Deterministic accessors (return sorted slices)
func (r *RouterState) TenantIDs() []TenantID          // sorted
func (r *RouterState) InstanceIDs() []InstanceID      // sorted
func (r *RouterState) AllSnapshots() []InstanceSnapshot // sorted by ID

type TenantState struct {
    RequestCount   int
    ActiveRequests int
    RecentRate     float64 // requests/sec over sliding window
    SLOClass       string
}

type InstanceSnapshot struct {
    // Core fields (frozen after PR 8 - no removals or type changes)
    ID                InstanceID
    Timestamp         int64    // clock time when snapshot was captured (New in v2.3)
    PoolType          PoolType // MONOLITHIC (PR 4+), PREFILL/DECODE (PR 15+)
    QueueDepth        int
    BatchSize         int
    KVUtilization     float64  // GPU tier utilization (0.0-1.0)
    FreeKVBlocks      int64    // TotalBlocks - UsedBlockCnt (New in v2.3)
    InFlightRequests  int
    RecentTTFT        float64
    RecentTPOT        float64
    CacheHitRate      float64  // recent cache hit rate (observable)
    EstimatedWaitTime float64

    // Extension point for Phase 4+ observables (New in v2.1)
    // Policies should handle missing keys gracefully (use default or ignore).
    // Known keys added in Phase 4:
    //   "CPUKVUtilization"   - CPU tier utilization (0.0-1.0)
    //   "TransferQueueDepth" - pending offload/reload operations
    //   "PendingOffloads"    - blocks queued for GPU→CPU transfer
    //   "PendingReloads"     - blocks queued for CPU→GPU transfer
    Extended map[string]float64
}

// SnapshotProvider controls when/how snapshots are refreshed. (New in v2.3)
type SnapshotProvider interface {
    // Snapshot returns the current (possibly cached) snapshot for an instance.
    Snapshot(id InstanceID, clock int64) InstanceSnapshot
    // RefreshAll rebuilds all snapshots that are stale per ObservabilityConfig.
    RefreshAll(clock int64)
}

type UpdateMode int
const (
    Immediate UpdateMode = iota // Re-read on every access
    Periodic                     // Re-read at fixed intervals
    OnDemand                     // Re-read only when explicitly requested
)

type FieldConfig struct {
    Mode     UpdateMode
    Interval int64 // ticks between updates (for Periodic mode)
}

// ObservabilityConfig controls per-field refresh behavior.
// Fields not listed here (InFlightRequests, FreeKVBlocks, EstimatedWaitTime)
// are always Immediate and not independently configurable.
type ObservabilityConfig struct {
    QueueDepth       FieldConfig // default: Immediate
    BatchSize        FieldConfig // default: Immediate
    KVUtilization    FieldConfig // default: Immediate (configurable to Periodic)
    CacheHitRate     FieldConfig // default: Periodic(10000 ticks)
    RecentTTFT       FieldConfig // default: Periodic(100000 ticks)
    RecentTPOT       FieldConfig // default: Periodic(100000 ticks)
}
```

### G.3 Evaluation Result

```go
type EvaluationResult struct {
    // Fitness scores for evolutionary optimization
    Fitness map[string]float64

    // Raw metrics for analysis
    Metrics *RawMetrics

    // Decision trace (verbosity-controlled)
    Trace *SimulationTrace

    // Summary for LLM reflection
    Summary *TraceSummary

    // Metadata
    PolicyID    string
    WorkloadID  string
    SimDuration int64
    WallTime    time.Duration
}

type RawMetrics struct {
    // Latency distributions (per SLO class)
    TTFT map[string]*Distribution
    TPOT map[string]*Distribution
    E2E  map[string]*Distribution

    // Throughput
    RequestsPerSec float64
    TokensPerSec   float64

    // SLO attainment
    SLOAttainment map[string]float64 // class -> fraction meeting target

    // Fairness
    JainFairnessIndex float64

    // Efficiency
    CacheHitRate    float64  // single value for Phase 2; per-tier in Phase 4
    PreemptionRate  float64

    // Scale events and cost
    ScaleUpCount       int
    ScaleDownCount     int
    TotalReplicaSeconds float64 // for cost modeling
    ScaleOscillations  int      // UP→DOWN or DOWN→UP within cooldown

    // Failure mode detection
    PriorityInversions int     // higher-priority scheduled after lower
    HOLBlockingEvents  int     // head-of-line blocking detected
}
```

---

## H) Architectural Evolution

### Before → After

```
CURRENT                              TARGET (Phase 2 Research-Ready, v2.3)
───────                              ──────────────────────────────────────
┌─────────────┐                      ┌──────────────────────────────────────────────────────┐
│  Simulator  │                      │                  ClusterSimulator                      │
│  (single)   │                      │                                                        │
│             │                      │  ┌──────────────── Control Plane ──────────────────┐  │
│ - WaitQueue │                      │  │                                                  │  │
│ - KVCache   │      ────────►       │  │  ClusterEventQueue ──── Admission → Routing      │  │
│ - Batch     │                      │  │  SnapshotProvider  ──── ObservabilityConfig       │  │
│ - EventQ    │                      │  │  PolicyBundle      ──── RouterState               │  │
│ - Clock     │                      │  │  PartitionedRNG                                   │  │
│ - RNG       │                      │  │  RawMetrics ◄──── Fitness eval                    │  │
└─────────────┘                      │  │                                                  │  │
                                     │  └──────────────────────────────────────────────────┘  │
                                     │                          │ inject request               │
                                     │                          ▼                              │
                                     │  ┌──────────────── Data Plane ─────────────────────┐  │
                                     │  │                                                  │  │
                                     │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │
                                     │  │  │Instance 0│  │Instance 1│  │Instance 2│ ...   │  │
                                     │  │  │-EventQ   │  │-EventQ   │  │-EventQ   │      │  │
                                     │  │  │-WaitQ    │  │-WaitQ    │  │-WaitQ    │      │  │
                                     │  │  │-KVCache  │  │-KVCache  │  │-KVCache  │      │  │
                                     │  │  │-Batch    │  │-Batch    │  │-Batch    │      │  │
                                     │  │  │-Scheduler│  │-Scheduler│  │-Scheduler│      │  │
                                     │  │  └──────────┘  └──────────┘  └──────────┘      │  │
                                     │  │                                                  │  │
                                     │  └──────────────────────────────────────────────────┘  │
                                     │                                                        │
                                     │  Main Loop: pick earliest event across ALL queues      │
                                     │    cluster event → execute (may inject into instance)  │
                                     │    instance event → delegate to instance                │
                                     └──────────────────────────────────────────────────────┘

TARGET (Phase 4 Full Fidelity)
──────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ClusterSimulator                                   │
│                                                                             │
│  PolicyBundle ─────────────────────────────────────────────────────────┐    │
│  RouterState                                                           │    │
│  EventHeap (ordered)                                                   │    │
│  PartitionedRNG                                                        │    │
│  AutoScaler ◄──── Scale decisions                                      │    │
│  DecisionTrace ◄──── Counterfactual analysis                           │    │
│                                                                        │    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐           │    │
│  │     Prefill Pool        │    │      Decode Pool        │           │    │
│  │  ┌────────┐ ┌────────┐  │    │  ┌────────┐ ┌────────┐  │           │    │
│  │  │Inst P0 │ │Inst P1 │  │───►│  │Inst D0 │ │Inst D1 │  │           │    │
│  │  │TieredKV│ │TieredKV│  │ KV │  │TieredKV│ │TieredKV│  │           │    │
│  │  └────────┘ └────────┘  │xfer│  └────────┘ └────────┘  │           │    │
│  └─────────────────────────┘    └─────────────────────────┘           │    │
│                                                                        │    │
└────────────────────────────────────────────────────────────────────────┘    │
```

### Package Structure

```
sim/
├── simulator.go          # Existing (minimal changes)
├── request.go            # Existing (extended with TenantID, Priority)
├── event.go              # Existing (kept for single-instance compat)
├── kvcache.go            # Existing (single-tier, sufficient for Phase 2)
├── metrics.go            # Existing (foundation for RawMetrics)
├── rng.go                # NEW: PartitionedRNG
├── cluster/
│   ├── cluster.go        # ClusterSimulator (restructured Run() with control/data plane)
│   ├── instance.go       # InstanceSimulator wrapper (+ observation methods: QueueDepth, BatchSize, KVUtilization, FreeKVBlocks)
│   ├── cluster_event.go  # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent (New in v2.3)
│   ├── snapshot.go       # InstanceSnapshot, SnapshotProvider, ObservabilityConfig (New in v2.3)
│   ├── event.go          # Instance-level cluster event types
│   ├── deployment.go     # DeploymentConfig, ReplicaPool
│   ├── router_state.go   # RouterState, TenantState
│   └── metrics.go        # ClusterMetrics, RawMetrics, FitnessFunction
├── policy/
│   ├── admission.go      # AdmissionPolicy interface + templates
│   ├── priority.go       # PriorityPolicy interface + templates
│   ├── routing.go        # RoutingPolicy interface + templates
│   ├── scheduler.go      # InstanceScheduler interface + templates
│   ├── autoscale.go      # AutoScalePolicy interface + templates
│   └── bundle.go         # PolicyBundle, loading from YAML
├── kv/                   # Phase 4
│   ├── tiered.go         # TieredKVCache
│   └── transfer.go       # Offload/reload, P/D transfer
├── workload/             # Phase 3 (optional enhancement)
│   ├── spec.go           # WorkloadSpec, TenantSpec
│   ├── generator.go      # Workload generation
│   └── arrival.go        # ArrivalPattern implementations
├── trace/                # Phase 4
│   ├── trace.go          # SimulationTrace, DecisionTrace
│   ├── record.go         # RoutingRecord, ScaleRecord, etc.
│   └── summary.go        # TraceSummary, summarization
└── adapter/              # Phase 5
    ├── gepa.go           # BLISGEPAAdapter
    └── openevolve.go     # BLISEvaluator
```

### What Remains Unchanged

The following components are **not modified** by this plan:

| Component | Location | Reason |
|-----------|----------|--------|
| Core event loop logic | `sim/simulator.go:Run()` | Wrapped by `InstanceSimulator`, not replaced |
| Request state machine | `sim/request.go` | Extended with `TenantID`, `Priority` fields but lifecycle unchanged |
| KV cache block management | `sim/kvcache.go` | Single-tier behavior preserved; tiered extension is additive |
| Latency estimation | `sim/roofline_step.go` | Alpha/beta coefficients consumed as-is |
| Workload generation | `sim/workload_config.go` | Preserved for `--workload distribution`; new generator is additive |
| Output JSON schema | Results output | Additions only, no field removals or type changes |
| Existing CLI flags | `cmd/root.go` | All existing flags continue to work identically |

**Backward compatibility guarantee:** `./simulation_worker run --model X` (without new flags) produces identical output to current implementation.

---

## I) PR Plan (Restructured for Research-First, No Scaffolding)

### Phase 1: Foundation (Sequential, 3 PRs)

These PRs must be done in order—each depends on the previous.

---

#### PR 1: PartitionedRNG and SimulationKey

| Aspect | Details |
|--------|---------|
| **Title** | `feat(sim): Add PartitionedRNG for deterministic multi-subsystem simulation` |
| **Motivation** | Determinism requires isolated RNG per subsystem |
| **In Scope** | `PartitionedRNG`, `SimulationKey`, refactor `Simulator.randomNumberGenerator` |
| **Out of Scope** | Multi-replica, policies |
| **Files Changed** | New: `sim/rng.go` (~100 LOC). Modified: `sim/simulator.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --seed 42` (unchanged, proves compatibility) |
| **Tests** | Unit: subsystem isolation. Integration: determinism verification (100 runs identical) |
| **No Dead Code** | `PartitionedRNG` immediately used by `Simulator` |
| **LOC Estimate** | ~120 |
| **Architectural Impact** | Replaces single RNG with partitioned RNG; enables future multi-instance isolation |
| **Behavioral Guarantees** | Existing behavior unchanged; same seed produces same output |
| **API Surface Changes** | Internal only; no public API changes |
| **README Changes** | None |
| **Risks + Mitigations** | Risk: Hash collision in subsystem derivation. Mitigation: Use FNV-1a with subsystem name. |
| **Why Independently Reviewable** | Self-contained RNG refactor; no dependencies on other PRs |

---

#### PR 2: InstanceSimulator Wrapper

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add InstanceSimulator wrapper` |
| **Motivation** | Composable unit for multi-replica |
| **In Scope** | `sim/cluster/` package, `InstanceSimulator`, `InstanceID` type |
| **Out of Scope** | `ClusterSimulator`, policies |
| **Files Changed** | New: `sim/cluster/instance.go` (~150 LOC). Modified: `cmd/root.go` (~20 LOC to route through wrapper) |
| **CLI** | `./simulation_worker run --model X --seed 42` (routes through `InstanceSimulator`, behavior identical) |
| **Tests** | `InstanceSimulator.Step()` produces identical results to `Simulator.Step()` |
| **No Dead Code** | CLI routes through wrapper; all code exercised |
| **LOC Estimate** | ~170 |
| **Architectural Impact** | Introduces composition layer; `Simulator` becomes internal to `InstanceSimulator` |
| **Behavioral Guarantees** | Bit-for-bit identical output to PR 1 |
| **API Surface Changes** | Internal only; CLI unchanged |
| **README Changes** | None |
| **Risks + Mitigations** | Risk: Wrapper overhead. Mitigation: Benchmark in PR 3 validates no regression. |
| **Why Independently Reviewable** | Clean wrapper pattern; tests prove equivalence to unwrapped version |

---

#### PR 3: ClusterSimulator with DeploymentConfig

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add ClusterSimulator with multi-instance event loop` |
| **Motivation** | Run N instances with shared clock |
| **In Scope** | `ClusterSimulator`, `DeploymentConfig`, `ReplicaPool`, `EventHeap` with ordering, `--num-instances` flag, basic round-robin dispatch |
| **Out of Scope** | Policy interfaces (temporary hardcoded dispatch), P/D disaggregation |
| **Files Changed** | New: `sim/cluster/cluster.go` (~300 LOC), `sim/cluster/deployment.go` (~100 LOC), `sim/cluster/event.go` (~150 LOC). Modified: `cmd/root.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --rate 20` |
| **Tests** | `--num-instances 1` identical to PR 2; deterministic replay with N>1; benchmark |
| **Benchmark** | `BenchmarkClusterSimulator_10K` added here |
| **No Dead Code** | `--num-instances` flag exercises all paths; `DeploymentConfig` used by `ClusterSimulator` |
| **LOC Estimate** | ~600 |
| **Architectural Impact** | Major: introduces cluster abstraction; shared clock across instances |
| **Behavioral Guarantees** | `--num-instances 1` identical to PR 2; N>1 distributes requests round-robin |
| **API Surface Changes** | New CLI flag: `--num-instances INT` |
| **README Changes** | Add "Multi-Instance Mode" section with example |
| **Risks + Mitigations** | Risk: Event ordering non-determinism. Mitigation: Explicit ordering rules (timestamp, type priority, event ID). |
| **Why Independently Reviewable** | Delivers complete multi-instance feature; usable immediately for capacity planning |

**✅ CHECKPOINT: Mock Study (COMPLETED 2026-02-13)**

Findings documented in `docs/plans/2026-02-13-mock-study-findings.md`:
1. Pre-dispatch routing breaks load-aware policies → PR 4 restructured with online routing
2. Four observable gaps identified → InstanceSimulator observation methods added in PR 4
3. InstanceSnapshot + SnapshotProvider architecture validated
4. Policy differentiation requires contention workloads (validates PR 9 pathological templates + PR 10 workload generator)

---

### Phase 2: Policy Interfaces + Metrics (6 PRs, PR 4 first then 5-7 parallel)

After PR 3 and mock study, PR 4 must land first (cluster event infrastructure). PRs 5-7 can then be developed **in parallel**.

---

#### PR 4: Cluster Control Plane + AdmissionPolicy

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add cluster event infrastructure, SnapshotProvider, and AdmissionPolicy` |
| **Motivation** | Mock study proved pre-dispatch routing breaks load-aware policies. Cluster needs event-driven control plane with online routing. Admission is the simplest policy to exercise the pipeline. |
| **In Scope** | Cluster-level event queue (`ClusterArrivalEvent`, `AdmissionDecisionEvent`, `RoutingDecisionEvent`), `InstanceSnapshot` with `Timestamp`, `SnapshotProvider` + `CachedSnapshotProvider`, `ObservabilityConfig`, `InstanceSimulator` observation methods (`QueueDepth()`, `BatchSize()`, `KVUtilization()`, `FreeKVBlocks()`), restructured `ClusterSimulator.Run()`, `AdmissionPolicy` interface, `AlwaysAdmit` + `TokenBucket` templates, configurable per-event latency |
| **Out of Scope** | Routing policy templates (PR 6), pathological templates (PR 9) |
| **Files Changed** | New: `sim/cluster/cluster_event.go` (~150 LOC), `sim/cluster/snapshot.go` (~200 LOC), `sim/policy/admission.go` (~150 LOC). Modified: `sim/cluster/cluster.go` (~200 LOC restructure), `sim/cluster/instance.go` (~30 LOC observation methods), `sim/request.go` (~5 LOC TenantID), `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --admission-policy always-admit` (default; `token-bucket` available but parameterized via `--policy-config` in PR 8) |
| **Tests** | Unit: SnapshotProvider refresh, event ordering, admission templates. Integration: online round-robin matches old pre-dispatch (backward compat). |
| **No Dead Code** | All code exercised via CLI flags and event pipeline |
| **LOC Estimate** | ~450 |
| **Architectural Impact** | Major: introduces control plane / data plane separation, cluster event queue, online routing. This is the architectural pivot point. |
| **Behavioral Guarantees** | `always-admit` (default) + round-robin (default) preserves PR 3 behavior exactly. Zero-latency defaults mean no observable change. |
| **API Surface Changes** | New CLI flags: `--admission-policy`, `--admission-latency`, `--routing-latency` |
| **README Changes** | Add "Cluster Control Plane" and "Admission Policies" sections |
| **Risks + Mitigations** | Risk: Larger PR. Mitigation: Infrastructure and admission are cohesive — infrastructure without a policy exercising it would be dead code. |
| **Why Independently Reviewable** | Complete control plane feature with admission policy exercising the pipeline; default preserves existing behavior |

---

#### PR 5: PriorityPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add PriorityPolicy with Constant and SLOBased` |
| **Motivation** | Enable request prioritization for scheduling |
| **Depends On** | PR 4 |
| **In Scope** | `PriorityPolicy` interface, `ConstantPriority`, `SLOBasedPriority`, `TenantPriority`, `DeadlineAware` templates |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/policy/priority.go` (~120 LOC). Modified: `cmd/root.go` (~15 LOC), `sim/request.go` (~10 LOC for Priority field) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --priority-policy slo-based` |
| **Tests** | Unit: each template. Integration: priority ordering in scheduling |
| **Parallel With** | PR 6, PR 7 |
| **No Dead Code** | All templates exercisable via `--priority-policy` flag |
| **LOC Estimate** | ~145 |
| **Architectural Impact** | Adds Priority field to Request; routing/scheduling can use priority scores |
| **Behavioral Guarantees** | `constant` (default) assigns equal priority; priority affects scheduling order |
| **API Surface Changes** | New CLI flag: `--priority-policy` |
| **README Changes** | Add "Priority Policies" section |
| **Risks + Mitigations** | Risk: Priority affecting determinism. Mitigation: Tie-breaking rules documented. |
| **Why Independently Reviewable** | Complete priority feature; default preserves existing behavior |

---

#### PR 6: RoutingPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RoutingPolicy with RoundRobin and WeightedScoring` |
| **Motivation** | Enable intelligent request routing across instances |
| **Depends On** | PR 4 (InstanceSnapshot and SnapshotProvider already exist; this PR adds routing policy templates that consume snapshots) |
| **In Scope** | `RoutingPolicy` interface, `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity` templates |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/policy/routing.go` (~200 LOC). Modified: `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --routing-policy weighted --routing-cache-weight 0.6` |
| **Tests** | Unit: each template. Integration: load distribution across instances |
| **Parallel With** | PR 5, PR 7 |
| **No Dead Code** | All templates exercisable via `--routing-policy` flag |
| **LOC Estimate** | ~220 |
| **Architectural Impact** | Replaces hardcoded round-robin with pluggable routing; consumes `InstanceSnapshot` from PR 4 |
| **Behavioral Guarantees** | `round-robin` (default) matches PR 3 behavior; weighted scoring respects weights |
| **API Surface Changes** | New CLI flags: `--routing-policy`, `--routing-cache-weight`, `--routing-load-weight` |
| **README Changes** | Add "Routing Policies" section |
| **Risks + Mitigations** | Risk: Prefix affinity cache misses. Mitigation: Fallback to least-loaded on miss. |
| **Why Independently Reviewable** | Complete routing feature; default preserves existing behavior |

---

#### PR 7: InstanceScheduler Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add InstanceScheduler with FCFS and PriorityFCFS` |
| **Motivation** | Enable per-instance batch scheduling policies |
| **Depends On** | PR 4 |
| **In Scope** | `InstanceScheduler` interface, `SchedulerContext`, `FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler` |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/policy/scheduler.go` (~180 LOC). Modified: `sim/cluster/instance.go` (~30 LOC), `cmd/root.go` (~15 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --scheduler priority-fcfs` |
| **Tests** | Unit: each template. Integration: batch ordering respects policy |
| **Parallel With** | PR 5, PR 6 |
| **No Dead Code** | All templates exercisable via `--scheduler` flag |
| **LOC Estimate** | ~225 |
| **Architectural Impact** | Extracts batch formation from Simulator; enables preemption policies |
| **Behavioral Guarantees** | `fcfs` (default) matches existing FIFO behavior; priority-fcfs respects Priority field |
| **API Surface Changes** | New CLI flag: `--scheduler` |
| **README Changes** | Add "Instance Schedulers" section |
| **Risks + Mitigations** | Risk: SJF starvation of long requests. Mitigation: Document limitation; recommend FCFS for fairness. |
| **Why Independently Reviewable** | Complete scheduler feature; default preserves existing behavior |

**Note on parallel PR integration:** PRs 5-7 each add flags to `cmd/root.go`. To avoid merge conflicts:
- Each PR adds its flags in a clearly separated block with comment header
- PR 8 (PolicyBundle) consolidates flag handling and resolves any conflicts

---

#### PR 8: RouterState and PolicyBundle

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RouterState and PolicyBundle configuration` |
| **Motivation** | Unified policy configuration via YAML |
| **Depends On** | PR 4, PR 5, PR 6, PR 7 |
| **In Scope** | `RouterState`, `TenantState`, `GlobalMetrics`, `PolicyBundle`, YAML loading |
| **Out of Scope** | AutoScale policy (Phase 4) |
| **Files Changed** | New: `sim/cluster/router_state.go` (~150 LOC), `sim/policy/bundle.go` (~100 LOC). Modified: `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --policy-config policies.yaml` |
| **Tests** | Unit: YAML parsing, validation. Integration: policy config overrides CLI flags |
| **No Dead Code** | `--policy-config` flag exercises PolicyBundle loading |
| **LOC Estimate** | ~280 |
| **Architectural Impact** | Introduces hierarchical config; RouterState aggregates cluster-wide state |
| **Behavioral Guarantees** | CLI flags override YAML defaults; missing config uses defaults |
| **API Surface Changes** | New CLI flag: `--policy-config` |
| **README Changes** | Add "Policy Configuration" section with YAML example |
| **Risks + Mitigations** | Risk: YAML parsing errors. Mitigation: Validate on load; clear error messages. |
| **Why Independently Reviewable** | Integrates PRs 4-7; provides unified config interface |

**⚠️ INTERFACE FREEZE after PR 8** — Policy interfaces are stable. "Freeze" means no breaking changes (no field removals, no type changes, no method signature changes). Adding new fields to structs or new keys to `Extended` maps is permitted in later phases.

---

#### PR 9: RawMetrics, Anomaly Detection, and Pathological Templates

| Aspect | Details |
|--------|---------|
| **Title** | `feat(metrics): Add RawMetrics, anomaly detection, and pathological policy templates` |
| **Motivation** | Enable fitness evaluation and validate anomaly detection |
| **Depends On** | PR 4, PR 5, PR 6, PR 7 |
| **In Scope** | `RawMetrics`, `Distribution`, `FitnessFunction`, `EvaluationResult`, anomaly counters, pathological templates (`RejectAll`, `InvertedSLO`, `AlwaysBusiest`, `ReversePriority`), validation tests |
| **Out of Scope** | Scale oscillation detection (requires AutoScaler in PR 11) |
| **Files Changed** | New: `sim/cluster/metrics.go` (~300 LOC). Modified: `sim/policy/*.go` (~100 LOC total for pathological templates), `sim/cluster/cluster.go` (~50 LOC), `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --fitness-weights "throughput:0.5,p99_ttft:0.3"` |
| **Tests** | Integration tests validating pathological policies trigger expected anomaly counts |
| **No Dead Code** | Pathological templates exercisable via policy flags; anomaly metrics in output |
| **LOC Estimate** | ~470 |
| **Architectural Impact** | Adds metrics collection to ClusterSimulator; enables fitness-based evaluation |
| **Behavioral Guarantees** | Anomaly counters increment when conditions detected; pathological policies trigger expected anomalies |
| **API Surface Changes** | New CLI flag: `--fitness-weights`; new JSON output fields in results |
| **README Changes** | Add "Metrics and Fitness Evaluation" section |
| **Risks + Mitigations** | Risk: False positive anomaly detection. Mitigation: Pathological policy tests validate detection accuracy. |
| **Why Independently Reviewable** | Complete metrics feature; pathological templates immediately testable |

**Pathological templates added:**

| Template | Policy Type | Purpose | Expected Anomaly |
|----------|-------------|---------|------------------|
| `reject-all` | Admission | Baseline for admission metrics | 100% rejection rate |
| `inverted-slo` | Priority | Test priority inversion detection | High `PriorityInversionCount` |
| `always-busiest` | Routing | Test load imbalance detection | High `HOLBlockingEvents` |
| `reverse-priority` | Scheduler | Test scheduler fairness | High `PriorityInversionCount` |

---

### 🎯 RESEARCH-READY CHECKPOINT (After PR 9)

At this point (~5 weeks), BLIS supports:
- ✅ Multi-instance cluster simulation with control plane / data plane separation
- ✅ Online routing (event-driven, not pre-dispatched)
- ✅ InstanceSnapshot with configurable staleness (ObservabilityConfig)
- ✅ All 4 policy interfaces (admission, priority, routing, scheduler)
- ✅ PolicyBundle with YAML configuration
- ✅ RawMetrics with fitness evaluation
- ✅ Anomaly detection validated with pathological templates
- ✅ Existing workload generation (`--workload distribution`)
- ✅ Single-tier KV cache (existing `sim/kvcache.go`)

**You can begin policy research experiments here.**

---

### Phase 3: Enhanced Workloads (1 PR, Optional)

This PR improves workload fidelity but is not required for initial research.

---

#### PR 10: Workload Generator with Edge Case Scenarios

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add multi-tenant workload generator with edge case scenarios` |
| **Motivation** | Enable realistic multi-tenant workloads for policy research |
| **In Scope** | `WorkloadSpec`, `TenantSpec`, `SLOSpec`, `PrefixSpec`, `ArrivalPattern` types, `WorkloadGenerator`, all arrival patterns, prefix reuse, `--workload-spec` flag, edge case scenarios |
| **Out of Scope** | Trace replay (existing `--workload traces` preserved) |
| **Files Changed** | New: `sim/workload/spec.go` (~180 LOC), `sim/workload/generator.go` (~250 LOC), `sim/workload/arrival.go` (~150 LOC), `sim/workload/scenarios.go` (~100 LOC). Modified: `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --workload-spec workload.yaml` |
| **Tests** | Unit: arrival patterns. Integration: scenario generation |
| **No Dead Code** | `--workload-spec` flag exercises all new code |
| **LOC Estimate** | ~700 |
| **Architectural Impact** | Adds workload package; coexists with existing workload generation |
| **Behavioral Guarantees** | Existing `--workload distribution` unchanged; new spec is additive |
| **API Surface Changes** | New CLI flag: `--workload-spec` |
| **README Changes** | Add "Workload Specification" section with YAML examples |
| **Risks + Mitigations** | Risk: Complex YAML schema. Mitigation: Provide built-in scenarios as starting points. |
| **Why Independently Reviewable** | Complete workload feature; existing workload modes preserved |

**Arrival patterns:**
- `Poisson` — constant rate λ
- `Bursty` — Poisson with periodic 10x spikes (configurable spike interval and duration)
- `Diurnal` — sinusoidal rate variation (models day/night traffic patterns)

**Built-in edge case scenarios:**

| Scenario | Description | Tests |
|----------|-------------|-------|
| `bursty-traffic` | Poisson baseline with 10x spikes every 60s | Admission policy resilience |
| `diurnal-cycle` | 24-hour sinusoidal rate (10x peak-to-trough) | Autoscaler response |
| `unfair-tenants` | 90% low-priority, 10% high-priority requests | Multi-tenant fairness, priority inversion |
| `prefix-heavy` | 80% requests share common prefixes | Cache-aware routing effectiveness |
| `mixed-slo` | Equal mix of realtime/interactive/batch | Joint priority scheduling |

**Example scenario config:**
```yaml
scenario: "unfair-tenants"
tenants:
  - id: "low-priority-bulk"
    weight: 0.9
    slo_class: "batch"
    rate_fraction: 0.9
  - id: "high-priority-realtime"
    weight: 0.1
    slo_class: "realtime"
    rate_fraction: 0.1
```

---

### Phase 4: Advanced Features (8 PRs, Parallelizable)

After Research-Ready checkpoint, these three tracks can proceed **in parallel**.

---

**Track A: Auto-Scaling**

#### PR 11: AutoScaler Core

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add AutoScaler with ThresholdScaler and Oscillator` |
| **Motivation** | Enable dynamic scaling based on load |
| **In Scope** | `AutoScaler`, `AutoScalePolicy`, `AutoScaleContext`, `ThresholdScaler`, `Oscillator` (pathological) |
| **Out of Scope** | Provisioning delays (PR 12) |
| **Files Changed** | New: `sim/policy/autoscale.go` (~230 LOC). Modified: `sim/cluster/cluster.go` (~40 LOC), `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --autoscaler-enabled --autoscaler-max 8` |
| **Tests** | Unit: threshold logic. Integration: scale up/down triggers |
| **Parallel With** | PR 13, PR 17 |
| **No Dead Code** | `--autoscaler-enabled` exercises all paths; `Oscillator` validates scale oscillation detection |
| **LOC Estimate** | ~300 |
| **Architectural Impact** | Adds periodic scale check event to ClusterSimulator |
| **Behavioral Guarantees** | Scale decisions respect min/max bounds; oscillation detected and counted |
| **API Surface Changes** | New CLI flags: `--autoscaler-enabled`, `--autoscaler-min`, `--autoscaler-max`, `--autoscaler-threshold` |
| **README Changes** | Add "Auto-Scaling" section |
| **Risks + Mitigations** | Risk: Scale oscillation. Mitigation: Cooldown period; `Oscillator` template validates detection. |
| **Why Independently Reviewable** | Complete autoscaler feature (instant scaling); actuation delays in PR 12 |

---

#### PR 12: Scaling Actuation Model

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add provisioning delays, warmup, and drain` |
| **Motivation** | Model realistic scaling latency |
| **Depends On** | PR 11 |
| **In Scope** | `WarmupProfile`, `DrainPolicy`, `InstanceState` lifecycle |
| **Out of Scope** | Predictive scaling |
| **Files Changed** | Modified: `sim/policy/autoscale.go` (~150 LOC), `sim/cluster/instance.go` (~50 LOC), `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --autoscaler-enabled --provisioning-delay 30s --warmup-duration 60s` |
| **Tests** | Integration: verify delay between scale decision and instance ready |
| **No Dead Code** | Flags exercise warmup/drain logic |
| **LOC Estimate** | ~220 |
| **Architectural Impact** | Adds instance lifecycle states (provisioning, warming, ready, draining) |
| **Behavioral Guarantees** | New instances not routed until warmup complete; draining instances finish existing requests |
| **API Surface Changes** | New CLI flags: `--provisioning-delay`, `--warmup-duration`, `--drain-policy` |
| **README Changes** | Extend "Auto-Scaling" section with actuation model |
| **Risks + Mitigations** | Risk: Complex state machine. Mitigation: Clear state diagram in code comments. |
| **Why Independently Reviewable** | Extends PR 11 with realistic timing; PR 11 works without this |

---

**Track B: Tiered KV + P/D Disaggregation**

#### PR 13: KVTier Types and Configuration

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add KVTier types and multi-tier configuration` |
| **Motivation** | Enable GPU+CPU KV cache modeling |
| **In Scope** | `KVTier` enum, `KVTierConfig`, extend `KVBlock` with `Tier` field |
| **Out of Scope** | Offload/reload mechanics (PR 14) |
| **Files Changed** | New: `sim/kv/tiered.go` (~100 LOC). Modified: `sim/kvcache.go` (~30 LOC), `cmd/root.go` (~15 LOC) |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 10000 --kv-cpu-blocks 50000` |
| **Tests** | Unit: tier assignment. Integration: CPU blocks available |
| **Parallel With** | PR 11, PR 17 |
| **No Dead Code** | `--kv-cpu-blocks` flag exercises tiered KV |
| **LOC Estimate** | ~145 |
| **Architectural Impact** | Extends KVCacheState with tier awareness |
| **Behavioral Guarantees** | `--kv-cpu-blocks 0` (default) preserves existing behavior |
| **API Surface Changes** | New CLI flag: `--kv-cpu-blocks` |
| **README Changes** | Add "Tiered KV Cache" section |
| **Risks + Mitigations** | Risk: Tier confusion in existing code. Mitigation: Default to GPU tier. |
| **Why Independently Reviewable** | Adds tier types; transfer mechanics in PR 14 |

---

#### PR 14: KV Offload/Reload Mechanics

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add offload/reload transfer mechanics` |
| **Motivation** | Model GPU↔CPU KV transfer latency |
| **Depends On** | PR 13 |
| **In Scope** | `KVTransfer` event, offload trigger, reload on CPU hit, transfer latency, `KVThrashingRate` metric |
| **Out of Scope** | P/D architecture (PR 15) |
| **Files Changed** | New: `sim/kv/transfer.go` (~200 LOC). Modified: `sim/cluster/event.go` (~30 LOC), `sim/cluster/metrics.go` (~20 LOC), `cmd/root.go` (~15 LOC) |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 1000 --kv-cpu-blocks 10000 --kv-offload-threshold 0.9` |
| **Tests** | Unit: transfer timing. Integration: thrashing detection |
| **No Dead Code** | `--kv-offload-threshold` exercises transfer logic |
| **LOC Estimate** | ~265 |
| **Architectural Impact** | Adds KVTransfer events to event queue |
| **Behavioral Guarantees** | Offload when GPU > threshold; reload adds latency; thrashing counted |
| **API Surface Changes** | New CLI flags: `--kv-offload-threshold`, `--kv-transfer-bandwidth` |
| **README Changes** | Extend "Tiered KV Cache" with transfer mechanics |
| **Risks + Mitigations** | Risk: Transfer deadlock. Mitigation: Timeout on pending transfers. |
| **Why Independently Reviewable** | Complete tiered KV feature; P/D in PR 15 |

---

#### PR 15: P/D Architecture

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add disaggregated prefill-decode architecture` |
| **Motivation** | Model DistServe/Splitwise style deployments |
| **Depends On** | PR 14 |
| **In Scope** | `DISAGGREGATED_PD` type, `PrefillPool`, `DecodePool`, `PDHandoffEvent`, routing changes |
| **Out of Scope** | KV ownership transfer (PR 16) |
| **Files Changed** | Modified: `sim/cluster/deployment.go` (~50 LOC), `sim/cluster/cluster.go` (~150 LOC), `sim/cluster/event.go` (~100 LOC), `cmd/root.go` (~25 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --prefill-replicas 2 --decode-replicas 4` |
| **Tests** | Integration: request flows P→D; separate pools visible |
| **Parallel With** | PR 11, PR 13 |
| **No Dead Code** | `--architecture pd` exercises P/D paths |
| **LOC Estimate** | ~325 |
| **Architectural Impact** | Major: separate prefill and decode pools with handoff |
| **Behavioral Guarantees** | `--architecture monolithic` (default) preserves existing behavior |
| **API Surface Changes** | New CLI flags: `--architecture`, `--prefill-replicas`, `--decode-replicas` |
| **README Changes** | Add "Prefill-Decode Disaggregation" section |
| **Risks + Mitigations** | Risk: Handoff timing complexity. Mitigation: Model as discrete event. |
| **Why Independently Reviewable** | Complete P/D architecture; KV ownership in PR 16 |

---

#### PR 16: KV Transfer for P/D

| Aspect | Details |
|--------|---------|
| **Title** | `feat(pd): Add KV transfer with ownership tracking` |
| **Motivation** | Model KV cache handoff from prefill to decode |
| **Depends On** | PR 15 |
| **In Scope** | `PDTransferConfig`, `BlockTransferState`, ownership transfer |
| **Out of Scope** | Multi-hop transfers |
| **Files Changed** | Modified: `sim/kv/transfer.go` (~150 LOC), `cmd/root.go` (~15 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --pd-transfer-latency 1ms --pd-transfer-bandwidth 10GB/s` |
| **Tests** | Unit: ownership invariant. Integration: P→D transfer timing |
| **No Dead Code** | P/D transfer flags exercise ownership logic |
| **LOC Estimate** | ~165 |
| **Architectural Impact** | Adds ownership tracking to BlockTransferState |
| **Behavioral Guarantees** | Block has exactly one owner at any time; transfer adds latency |
| **API Surface Changes** | New CLI flags: `--pd-transfer-latency`, `--pd-transfer-bandwidth` |
| **README Changes** | Extend "Prefill-Decode Disaggregation" with KV transfer |
| **Risks + Mitigations** | Risk: Ownership bugs (double-free). Mitigation: Conservation invariant enforced. |
| **Why Independently Reviewable** | Completes P/D feature; ownership clearly tracked |

---

**Track C: Observability**

#### PR 17: Decision Traces

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add DecisionTrace with RoutingRecord` |
| **Motivation** | Enable policy decision debugging and analysis |
| **In Scope** | `SimulationTrace`, `DecisionTrace`, `RoutingRecord`, `TraceConfig`, `--trace-level` flag |
| **Out of Scope** | Counterfactual analysis (PR 18) |
| **Files Changed** | New: `sim/trace/trace.go` (~100 LOC), `sim/trace/record.go` (~150 LOC). Modified: `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --trace-level decisions` |
| **Tests** | Integration: trace JSON includes routing decisions |
| **Parallel With** | PR 11, PR 13 |
| **No Dead Code** | `--trace-level` exercises trace collection |
| **LOC Estimate** | ~270 |
| **Architectural Impact** | Adds trace collection to ClusterSimulator; policies log decisions |
| **Behavioral Guarantees** | `--trace-level none` (default) has no overhead; `decisions` captures all policy calls |
| **API Surface Changes** | New CLI flag: `--trace-level` (none/decisions/detailed) |
| **README Changes** | Add "Decision Tracing" section |
| **Risks + Mitigations** | Risk: Trace memory bloat. Mitigation: Configurable verbosity levels. |
| **Why Independently Reviewable** | Complete trace feature; counterfactual analysis separate |

---

#### PR 18: Counterfactual Analysis and Trace Summary

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add counterfactual analysis and trace summarization` |
| **Motivation** | Enable "what-if" analysis of routing decisions |
| **Depends On** | PR 17 |
| **In Scope** | `TopKCandidates`, `Regret` calculation, `TraceSummary`, `--summarize-trace` flag |
| **Out of Scope** | LLM-based reflection (framework-specific) |
| **Files Changed** | New: `sim/trace/summary.go` (~200 LOC). Modified: `sim/trace/record.go` (~50 LOC), `cmd/root.go` (~15 LOC) |
| **CLI** | `./simulation_worker run --model X --trace-level decisions --counterfactual-k 5 --summarize-trace` |
| **Tests** | Unit: regret calculation. Integration: summary includes counterfactuals |
| **No Dead Code** | `--summarize-trace` exercises summary logic |
| **LOC Estimate** | ~265 |
| **Architectural Impact** | Extends trace with counterfactual analysis |
| **Behavioral Guarantees** | Top-k candidates stored per routing decision; regret computed as best-alternative - actual |
| **API Surface Changes** | New CLI flags: `--counterfactual-k`, `--summarize-trace` |
| **README Changes** | Extend "Decision Tracing" with counterfactual analysis |
| **Risks + Mitigations** | Risk: Expensive for large k. Mitigation: Default k=3; document performance. |
| **Why Independently Reviewable** | Extends PR 17; trace works without counterfactuals |

---

### Phase 5: Framework Adapters (2 PRs, Optional)

BLIS is fully functional without these. They provide convenience for framework integration.

---

#### PR 19: GEPA Adapter

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add GEPA adapter` |
| **Motivation** | Enable GEPA framework integration |
| **Depends On** | PR 18 (traces for `ExtractTracesForReflection()`) |
| **In Scope** | `BLISGEPAAdapter`, `Evaluate()`, `ExtractTracesForReflection()`, `gepa-evaluate` command |
| **Out of Scope** | GEPA framework itself (external dependency) |
| **Files Changed** | New: `sim/adapter/gepa.go` (~150 LOC). Modified: `cmd/root.go` (~40 LOC) |
| **CLI** | `./simulation_worker gepa-evaluate --policy-config p.yaml --workload w.yaml` |
| **Tests** | Integration: adapter returns expected format |
| **Parallel With** | PR 20 |
| **No Dead Code** | `gepa-evaluate` command exercises adapter |
| **LOC Estimate** | ~190 |
| **Architectural Impact** | Adds new CLI subcommand; wraps Run() with GEPA-specific format |
| **Behavioral Guarantees** | Returns fitness + traces in GEPA format |
| **API Surface Changes** | New subcommand: `gepa-evaluate` |
| **README Changes** | Add "Framework Integration" section |
| **Risks + Mitigations** | Risk: GEPA format changes. Mitigation: Version adapter with GEPA release. |
| **Why Independently Reviewable** | Self-contained adapter; BLIS core unchanged |

---

#### PR 20: OpenEvolve Evaluator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add OpenEvolve evaluator` |
| **Motivation** | Enable OpenEvolve framework integration |
| **Depends On** | PR 18 (trace summary for feature extraction) |
| **In Scope** | `BLISEvaluator`, multi-objective fitness, feature extraction, `openevolve-evaluate` command |
| **Out of Scope** | OpenEvolve framework itself (external dependency) |
| **Files Changed** | New: `sim/adapter/openevolve.go` (~150 LOC). Modified: `cmd/root.go` (~40 LOC) |
| **CLI** | `./simulation_worker openevolve-evaluate --config oe.yaml --candidate c.yaml` |
| **Tests** | Integration: adapter returns expected format |
| **Parallel With** | PR 19 |
| **No Dead Code** | `openevolve-evaluate` command exercises evaluator |
| **LOC Estimate** | ~190 |
| **Architectural Impact** | Adds new CLI subcommand; wraps Run() with OpenEvolve-specific format |
| **Behavioral Guarantees** | Returns multi-objective fitness in OpenEvolve format |
| **API Surface Changes** | New subcommand: `openevolve-evaluate` |
| **README Changes** | Extend "Framework Integration" section |
| **Risks + Mitigations** | Risk: OpenEvolve format changes. Mitigation: Version adapter with OpenEvolve release. |
| **Why Independently Reviewable** | Self-contained adapter; BLIS core unchanged |

---

### Phase 6: Validation (1 PR)

#### PR 21: Integration Tests and Examples

| Aspect | Details |
|--------|---------|
| **Title** | `test: Add comprehensive integration test suite and examples` |
| **Motivation** | Validate end-to-end workflows and provide usage examples |
| **In Scope** | Integration tests, sample configs, example policies, CI validation |
| **Out of Scope** | Performance benchmarks (already in PRs) |
| **Files Changed** | New: `test/integration/` (~500 LOC), `examples/` (configs) |
| **CLI** | `go test ./test/integration/...` |
| **Tests** | E2E scenarios: admission+routing+scaling, P/D pipeline, trace analysis |
| **No Dead Code** | Tests exercise all CLI paths |
| **LOC Estimate** | ~500 |
| **Architectural Impact** | None (test-only PR) |
| **Behavioral Guarantees** | All integration tests pass; examples run without errors |
| **API Surface Changes** | None |
| **README Changes** | Add "Examples" section with links to `examples/` |
| **Risks + Mitigations** | Risk: Flaky tests. Mitigation: Deterministic seed for all tests. |
| **Why Independently Reviewable** | Test suite validates cumulative work; examples document usage |

---

## J) Dependency DAG (Restructured, No Scaffolding)

### PR Dependency Graph

```
PHASE 1: FOUNDATION (Sequential, 3 PRs)
════════════════════════════════════════════════════════════════════════════════

  PR 1 ──────► PR 2 ──────► PR 3
  (RNG)      (Instance)   (Cluster+Deploy)
                                │
                                ▼
                    ✅ MOCK STUDY CHECKPOINT (COMPLETED)
                                │
                                ▼
PHASE 2: POLICY INTERFACES + METRICS (6 PRs) ══════════════════════════════════
                                │
                                ▼
                              PR 4
                  (Control Plane + Admission)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
            PR 5              PR 6              PR 7
          (Priority)        (Route)           (Sched)
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                              PR 8
                           (Bundle)
                                │
                                ▼
                    ⚠️ INTERFACE FREEZE
                                │
                                ▼
                              PR 9
                    (Metrics + Pathological)
                                │
                                ▼
                    🎯 RESEARCH-READY CHECKPOINT
                                │
                                ▼
PHASE 3: ENHANCED WORKLOADS (1 PR, Optional) ══════════════════════════════════
                                │
                                ▼
                             PR 10
                    (Workload Generator)
                                │
                                ▼
PHASE 4: ADVANCED FEATURES (8 PRs) ════════════════════════════════════════════
                                │
     ┌──────────────────────────┼──────────────────────────┐
     ▼                          ▼                          ▼
   PR 11                      PR 13                      PR 17
 (AutoScale)                (KV Tier)                  (Traces)
     │                          │                          │
     ▼                          ▼                          ▼
   PR 12                      PR 14                      PR 18
 (Actuation)               (Transfer)                 (Summary)
                                │
                                ▼
                              PR 15
                              (P/D)
                                │
                                ▼
                              PR 16
                           (P/D Xfer)
                                │
                                ▼
PHASE 5: ADAPTERS (2 PRs, Optional) ═══════════════════════════════════════════
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                  PR 19                   PR 20
                 (GEPA)               (OpenEvolve)
                    │                       │
                    └───────────┬───────────┘
                                ▼
PHASE 6: VALIDATION (1 PR) ════════════════════════════════════════════════════
                                │
                                ▼
                              PR 21
                            (Tests)
```

### Parallel Development Matrix

| Gate | Completed PRs | Unlocked for Parallel Development |
|------|---------------|-----------------------------------|
| **G1** | PR 4 (Control Plane + Admission) | PR 5, PR 6, PR 7 (3 parallel) |
| **G2** | PR 9 (Research-Ready) | PR 10, PR 11, PR 13, PR 17 (4 parallel tracks) |
| **G3** | PR 18 | PR 19, PR 20 (2 parallel) |

### Critical Checkpoints

| Checkpoint | Verification | Failure Action |
|------------|--------------|----------------|
| **Mock Study (after PR 3)** | ✅ COMPLETED 2026-02-13. Findings: pre-dispatch routing broken, 4 observable gaps identified | PR 4 restructured with online routing and observation methods |
| **After PR 3** | Determinism: 100 runs identical | Block until fixed |
| **Interface Freeze (after PR 8)** | Policy interfaces frozen (no breaking changes) | Additive changes (new fields, new `Extended` keys) permitted |
| **Research-Ready (after PR 9)** | Pathological policies trigger anomalies | Required for research |
| **After PR 18** | All observability features working | Required for adapters |

### Timeline Estimate (3-4 developers)

```
Week 1-2:   Phase 1 (PR 1-3, sequential, 1 dev)
            + Mock Study (2-3 days) ✅ COMPLETED

Week 3:     Phase 2 PR 4 (Control Plane + Admission, sequential, 1 dev)

Week 4-5:   Phase 2 PRs 5-7 (3 parallel), then PR 8-9
            → 🎯 RESEARCH-READY (~5 weeks)

Week 6-7:   Phase 3 (PR 10, optional, 1 dev)
            Phase 4 Track A (PR 11-12, 1 dev)
            Phase 4 Track B (PR 13-16, 1 dev)
            Phase 4 Track C (PR 17-18, 1 dev)

Week 8-9:   Phase 4 continued

Week 10:    Phase 5 (PR 19-20, 2 devs parallel)

Week 11:    Phase 6 (PR 21)

Total: ~11 weeks with 3-4 developers
Research-ready: ~5 weeks
```

---

## K) Validation Strategy

### K.1 Unit Test Coverage

| Component | Test Focus |
|-----------|------------|
| `PartitionedRNG` | Subsystem isolation, determinism |
| Policy interfaces | Default implementations, edge cases |
| Pathological policies | Known-bad behavior produces expected anomalies |
| `KVCacheState` | Conservation invariant, LRU ordering |
| `EventHeap` | Ordering rules, tie-breaking |
| Workload generator | Distribution correctness, prefix reuse |

### K.2 Integration Tests

| Test | Description | PR |
|------|-------------|-----|
| `TestSingleInstanceCompatibility` | `--num-instances 1` produces identical results to original | PR 3 |
| `TestDeterministicReplay` | 100 runs with same seed produce identical output | PR 3 |
| `TestPolicyPipeline` | Admission → Priority → Routing → Scheduling flow | PR 9 |
| `TestAnomalyDetection_PathologicalPolicies` | Pathological policies trigger expected anomalies | PR 9 |
| `TestKVTierTransfer` | Offload/reload timing and conservation | PR 14 |
| `TestPDHandoff` | Prefill → Transfer → Decode lifecycle | PR 15 |
| `TestAutoScaling` | Scale up/down with warmup and drain | PR 12 |

### K.3 Behavioral Validation

| Invariant | Verification Method |
|-----------|---------------------|
| `request_lifecycle` | Track all requests; assert exactly one terminal state |
| `clock_monotonicity` | Assert `new_clock >= old_clock` after every event |
| `kv_conservation` | Assert `used + free = total` after every KV operation |
| `scale_bounds` | Assert `min <= current <= max` after every scale action |
| `determinism` | Diff outputs of multiple runs |
| `anomaly_detection` | Pathological policies produce non-zero anomaly counts |

### K.4 Sim-to-Real Validation Framework

The research agenda treats sim-to-real transfer as a first-class question. BLIS supports this through structured validation:

**Validation Protocol:**
1. **Policy discovery** — Evolve policies in BLIS simulation
2. **Policy export** — Serialize top-k policies as configuration
3. **Real deployment** — Deploy policies in llm-d with vLLM backends
4. **Metrics collection** — Collect same metrics as BLIS (TTFT, TPOT, SLO attainment, etc.)
5. **Transfer analysis** — Compare sim vs. real rankings and performance gaps

**Transfer Quality Metrics:**
| Metric | Definition |
|--------|------------|
| **Ranking consistency** | Spearman correlation between sim and real policy rankings |
| **Absolute gap** | |sim_metric - real_metric| for each policy |
| **Relative ordering preservation** | % of pairwise comparisons where sim correctly predicts winner |

**BLIS Output for Validation:**
```json
{
  "policy_id": "routing_v42",
  "sim_fitness": {"throughput": 0.85, "p99_ttft": 0.72},
  "sim_metrics": { ... },
  "config_for_deployment": { ... }  // directly loadable by llm-d
}
```

**Coefficient Refinement Loop:**
When sim-to-real gap exceeds threshold:
1. Collect detailed traces from real deployment
2. Identify divergence sources (latency model, cache behavior, etc.)
3. Refine alpha/beta coefficients using real data
4. Re-validate transfer quality

**Sanity Scenario (New in v2):**
Maintain one simple scenario that mirrors a real deployment you can instrument:
- 4 replicas, monolithic architecture
- Poisson arrivals at 80% utilization
- Single SLO class
- Measure: P50/P99 TTFT, throughput, cache hit rate

This scenario is used for:
1. Early coefficient sanity checks
2. Sim-to-real gap baseline
3. Regression detection when models change

**Parallel Trace Collection (New in v2.1):**
Real-world trace collection for coefficient refinement can begin **in parallel with Phase 2** as a separate workstream. This is not a blocking dependency for BLIS development.

Recommended parallel activities:
1. **During Phase 1-2:** Set up instrumented vLLM deployment with trace export
2. **During Phase 3-4:** Collect traces under sanity scenario workload
3. **After Research-Ready:** Use traces to refine alpha/beta coefficients

This parallelization allows sim-to-real validation to begin shortly after the research-ready checkpoint without delaying BLIS core development.

---

## L) Design Bug Prevention Checklist

### L.1 Invariants

| Invariant | Enforcement |
|-----------|-------------|
| `request_lifecycle` | Assert on simulation end: all requests terminated |
| `clock_monotonicity` | Assert in event loop: `new >= old` |
| `kv_conservation` | Assert after every `Allocate`/`Release`: `used + free = total` |
| `scale_bounds` | Assert in `executeScaleAction`: `min <= target <= max` |
| `pd_ownership` | Assert: block has exactly one owner at any time |
| `determinism` | CI: run twice, diff outputs |
| `anomaly_detection` | Pathological policies produce non-zero counts |

### L.2 Regression Surfaces

| Surface | Risk | Mitigation |
|---------|------|------------|
| Event ordering | Non-determinism | Explicit type priorities, event IDs, no map iteration |
| RNG isolation | Cross-contamination | Hash-based derivation, isolation tests |
| KV reference counting | Leaks | Conservation check after every op |
| Policy interfaces | Breaking changes | Interface freeze after PR 8 |
| Floating-point | Non-determinism | Use int64 for time; careful ordering |
| Anomaly detection | False negatives | Pathological policy tests |
| RouterState map iteration | Non-determinism | Use sorted accessor methods; lint for direct iteration |

### L.3 Failure Mode Prevention

| Failure | Prevention |
|---------|------------|
| Non-deterministic ties | Explicit rules documented in code |
| Scale oscillation | Cooldown period, hysteresis in `ThresholdScaler` |
| P/D deadlock | Transfer timeout, backpressure threshold |
| KV thrashing | Offload rate metric, alert if high |
| Missing anomalies | Pathological policy test suite |

---

## M) Performance Expectations

BLIS is a CPU-only discrete-event simulator. Performance should be fast enough for interactive use and batch evolutionary evaluation.

### Target Performance

| Workload | Target | Notes |
|----------|--------|-------|
| 1K requests, 1 instance | < 100ms | Interactive CLI use |
| 10K requests, 4 instances | < 1 second | Batch evaluation |
| 100K requests, 16 instances | < 10 seconds | Large-scale simulation |

### Performance Principles

1. **No allocation in hot path** — Pre-allocate event structs, reuse slices
2. **Efficient event heap** — Standard library `container/heap` is sufficient
3. **Avoid map iteration** — Use slices for ordered iteration
4. **Profile before optimizing** — Measure actual bottlenecks

### Benchmark Requirements

Benchmarks added in PR 3 (ClusterSimulator):

```go
func BenchmarkClusterSimulator_1K_1Instance(b *testing.B) {
    // Setup: 1K requests, 1 instance
    for i := 0; i < b.N; i++ {
        sim.Run()
    }
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
    // Setup: 10K requests, 4 instances
    for i := 0; i < b.N; i++ {
        sim.Run()
    }
}
```

**CI gate:** Performance regression >20% blocks merge.

---

## N) Summary

| Metric | Value |
|--------|-------|
| **Phases** | 6 |
| **Total PRs** | 21 |
| **Total LOC Estimate** | ~5,500 (PR 4 expanded from ~170 to ~450 LOC) |
| **Max Parallel PRs** | 3 (Phase 2 PRs 5-7), 4 (Phase 4) |
| **Research-Ready** | ~5 weeks (PR 4 adds 1 sequential week) |
| **Full Implementation** | ~11 weeks (with 3-4 developers) |

### Key Decisions

1. **BLIS is standalone** — framework adapters are optional conveniences
2. **No arbitrary code execution** — parameterized policy templates instead
3. **Determinism is foundational** — explicit rules for all non-deterministic sources
4. **vLLM-grounded modeling** — explicit citations and simplifications documented
5. **Routing policy freedom** — policies can maintain their own internal state; router provides observables, not mandated tracking
6. **Research-first ordering** — metrics and anomaly detection moved up; tiered KV deferred
7. **Pathological templates consolidated** — added with anomaly detection in PR 9, not scattered across policy PRs
8. **Mock study before freeze** — validate interfaces with real experiments
9. **Interface extension point** — `Extended` map allows Phase 4 observables without breaking freeze (v2.1)
10. **Stateful policies supported** — policy instances persist; internal state is allowed and expected (v2.1)
11. **No scaffolding** — every PR is CLI-exercisable immediately after merge (v2.2)
12. **Online routing architecture** — mock study proved pre-dispatch routing breaks load-aware policies; routing decisions happen during event loop (v2.3)
13. **Control plane / data plane separation** — ClusterSimulator split into cluster event queue + instance event queues (v2.3)
14. **InstanceSnapshot staleness model** — immutable value types with configurable refresh via ObservabilityConfig (v2.3)

### Changes from v1

| Change | Rationale |
|--------|-----------|
| Metrics moved to Phase 2 | Enables research after Phase 2 |
| Tiered KV moved to Phase 4 | Single-tier sufficient for initial research |
| P/D depends on tiered KV | Natural dependency |
| Mock study checkpoint added | De-risk interface freeze |
| Benchmarks added in PR 3 | Early performance visibility |

### Changes from v2.1 (v2.2)

| Change | Rationale |
|--------|-----------|
| PR 3+4 merged | DeploymentConfig introduced with ClusterSimulator (no scaffolding) |
| PR 12+13 merged | WorkloadSpec introduced with generator (no scaffolding) |
| PR 10+11 merged | RawMetrics and anomaly validation combined |
| Pathological templates moved to PR 9 | Consolidated with anomaly detection (no test infra in feature PRs) |
| 24 PRs → 21 PRs | Eliminated all scaffolding PRs |
| Research-ready ~5 weeks → ~4 weeks | Fewer PRs in critical path |

### Changes from v2.2 (v2.3)

| Change | Rationale |
|--------|-----------|
| PR 4 expanded to include cluster event infrastructure | Mock study proved pre-dispatch routing incompatible with load-aware policies |
| Control plane / data plane separation | Online routing requires cluster-level event queue separate from instance event queues |
| InstanceSnapshot staleness model added | Researchers need configurable observation freshness to study policy robustness |
| PRs 5-7 now depend on PR 4 (no longer parallel) | PR 4 provides cluster event infrastructure that policies consume |
| Research-ready ~4 weeks → ~5 weeks | PR 4 expanded scope adds 1 sequential week |
| InstanceSimulator observation methods | Mock study identified 4 observable gaps; PR 4 adds QueueDepth(), BatchSize(), KVUtilization(), FreeKVBlocks() methods. CacheHitRate deferred (requires new KV hit/miss tracking) |

### Research Agenda Alignment

This plan directly supports the four llm-d inference control problems:

| Research Problem | BLIS Support | Key PRs |
|-----------------|--------------|---------|
| **Routing Policy Evolution** | `RoutingPolicy` interface, `InstanceSnapshot` observables, existing prefix caching | PR 6, PR 9 |
| **Admission Control Evolution** | `AdmissionPolicy` interface, multi-tenant `TenantState`, fairness metrics | PR 4, PR 8-9 |
| **Joint Priority Scheduling** | `PriorityPolicy` + `InstanceScheduler` separation, priority inversion detection, HOL blocking detection | PR 5, PR 7, PR 9 |
| **Autoscaling Evolution** | `AutoScalePolicy` interface, provisioning/warmup modeling, cost metrics, scale oscillation detection | PR 11-12 |

**Architectural Locality:** Each policy interface respects control boundaries—routing sees only `RouterState` + `InstanceSnapshot`, instance schedulers see only local state, autoscalers see only aggregate metrics.

**Sim-to-Real Transfer:** BLIS outputs directly deployable policy configurations, with structured validation protocol (Section K.4) to measure transfer quality.

### References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024
3. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
4. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023

### Next Steps

1. Review and approve this plan (v2.3)
2. Create PR 4 micro-level implementation plan (cluster control plane + admission)
3. Begin PR 4 (Control Plane + AdmissionPolicy)
4. (Parallel) Set up instrumented vLLM deployment for trace collection
