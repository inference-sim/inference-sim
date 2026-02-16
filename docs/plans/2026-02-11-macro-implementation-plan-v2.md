# BLIS Evolutionary Policy Optimization: Macro-Level Implementation Plan (v3)

**Date:** 2026-02-11
**Revision:** v3.1 (post-PR7 accuracy update — 16 total PRs, 9 remaining)
**Status:** Draft
**Target:** Multi-replica cluster simulation with pluggable policies
**Based on:** [Design Document](2026-02-06-evolutionary-policy-optimization-design.md)

---

## Revision Notes

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

**v3.0 changes (simplification assessment, 2026-02-13):**
20. **Dropped backward compatibility constraint** — Golden dataset tests (`testdata/goldendataset.json`) remain the verification gate; all other backward compatibility constraints relaxed
21. **Simplification assessment conducted** — Four independent analyses (see `docs/plans/2026-02-13-simplification-assessment.md`) identified constructor collapse, unified CLI path, field privatization, and interface deduplication as high-value simplifications
22. **New PR 5: Architectural Simplification** — `SimConfig` struct replaces 17-param constructors; CLI always uses `ClusterSimulator` (even N=1); ~15 internal `Simulator` fields privatized; `AdmissionPolicy` interface deduplicated to `sim/` base package
23. **6 PR merges** — Priority+Scheduler (→PR 7), AutoScaler+Actuation (→PR 11), TieredKV+Transfer (→PR 12), Traces+Counterfactual (→PR 13), P/D+KVTransfer combined (→PR 14), Adapters combined (→PR 15)
24. **Unified event queue assessed but deferred** — Feasible but low ROI (~50 LOC savings) with risk to golden tests and P/D disaggregation
25. **21 PRs reduced to 16 total** (12 remaining after PR 4): constructor collapse and unified CLI path make every subsequent PR simpler

**v3.1 changes (post-PR7 accuracy update, 2026-02-16):**
26. **PR 6 and PR 7 marked completed** with implementation notes documenting actual file placements (`sim/routing.go`, `sim/priority.go`, `sim/scheduler.go` — all in `sim/`, not `sim/policy/`)
27. **Package structure corrected** — `sim/policy/` subtree replaced with actual flat `sim/` layout; `router_state.go` moved from `sim/cluster/` to `sim/` (bridge type pattern); `ReplicaPool` removed (never implemented)
28. **G.1 interface signatures updated** to show actual frozen signatures (post-PR8), replacing aspirational signatures that diverged during implementation
29. **G.2 state structures annotated** with current vs planned fields per PR
30. **B.4 hardcoded behaviors** updated with completion status (FIFO → PR7, All-admit → PR4)
31. **B.6 CLI flags** reorganized: PRs 1-7 flags moved to "already added"; added missing `--token-bucket-capacity` and `--token-bucket-refill-rate`
32. **Remaining count** updated from 12 to 9 (PRs 1-7 completed)

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
- 6 phases, 16 PRs
- **Research-ready checkpoint after Phase 2** (~5 weeks; PR 4 expanded with cluster event infrastructure)
- Each PR is CLI-exercisable immediately after merge — no scaffolding
- Estimated ~10 weeks with 3-4 developers

---

## B) Repository Recon Summary

### B.1 Package Structure

| Package | SLOC (non-test) | Responsibility |
|---------|------|----------------|
| `sim/` | ~2700 | Core single-instance discrete-event simulator + policy interfaces |
| `sim/cluster/` | ~850 | Multi-replica cluster simulation |
| `cmd/` | ~440 | CLI interface (Cobra), configuration loading |
| `main.go` | 11 | Entry point |

### B.2 Core Data Structures

**Simulator** (`sim/simulator.go` — `SimConfig` at ~line 73, `Simulator` struct at ~line 97):
- Event queue (min-heap), clock (`int64` ticks), wait queue, KV cache, running batch
- `*PartitionedRNG` (partitioned per subsystem), alpha/beta coefficients for latency estimation

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

| Behavior | Location | Target Interface | Status |
|----------|----------|------------------|--------|
| FIFO batch formation | `simulator.go` | `InstanceScheduler` | ✅ PR7 |
| LIFO preemption | `simulator.go` | `InstanceScheduler` | Partial (PR7 added scheduler; preemption policy deferred) |
| Single-tier KV | `kvcache.go` | Tiered `KVCacheState` (Phase 4) | Pending |
| All-admit | implicit | `AdmissionPolicy` | ✅ PR4 |

### B.5 Extension Strategy

- **Composition over modification**: `ClusterSimulator` wraps `InstanceSimulator` instances
- **Interface extraction**: Pull hardcoded behaviors into pluggable interfaces
- **Unified CLI path**: CLI always uses `ClusterSimulator` (even N=1); `--num-instances 1` produces identical results via cluster path

### B.6 CLI Entrypoints and Flag Surface

**Current CLI structure** (`cmd/root.go`):

```
simulation_worker run
  --model STRING           # Required: model identifier
  --seed INT               # RNG seed (default: 42)
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

**Flags already added (PRs 1-7, merged):**
- `--num-instances INT` (PR 3)
- `--admission-policy STRING` (PR 4)
- `--admission-latency INT` (PR 4, microseconds, default 0)
- `--routing-latency INT` (PR 4, microseconds, default 0)
- `--token-bucket-capacity FLOAT` (PR 4, default 10000)
- `--token-bucket-refill-rate FLOAT` (PR 4, default 1000)
- `--routing-policy STRING` (PR 6)
- `--routing-cache-weight FLOAT` (PR 6)
- `--routing-load-weight FLOAT` (PR 6)
- `--priority-policy STRING` (PR 7)
- `--scheduler STRING` (PR 7)

**Flags to be added (PRs 8+):**
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
| `Simulator` constructors accept up to 17 parameters | Adding more params increases constructor complexity | Collapsed to `SimConfig` struct in PR 5 |
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
- Golden dataset tests remain the verification gate for all refactors
- Output JSON schema: additions only, no breaking changes
- **Dropped (v3.0):** Separate single-instance CLI path removed; `ClusterSimulator` handles N=1 transparently

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
| Immediate | QueueDepth, BatchSize, KVUtilization, FreeKVBlocks | Load balancer connection tracking |
| Periodic | CacheHitRate (10ms), RecentTTFT (100ms), RecentTPOT (100ms) | Prometheus scrape, xDS push (planned, PR9+) |
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

**Note:** PR 4 implemented `(timestamp, priority, seqID)` ordering for the `ClusterEventQueue`. The per-instance `EventQueue` retains timestamp-only ordering; cluster-level tie-breaking uses lowest instance index.

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

### E.6 Simplification Risks (New in v3.0)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Golden test breakage from unified CLI path | BLOCKS adoption | Run ALL golden tests through cluster path with N=1 before merging PR 5; existing `TestClusterSimulator_SingleInstance_MatchesGoldenDataset` validates this |
| RNG stream divergence from constructor collapse | BLOCKS adoption | `SimConfig` constructor must call `NewPartitionedRNG(NewSimulationKey(cfg.Seed))` at same point as `newSimulatorBase`; test identical request generation through both paths |
| Big-bang PR risk (PR 5 touches 10+ files) | COMPLICATES review | Atomic commits per simplification: SimConfig → unified CLI → privatization → interface dedup |
| Performance regression from field privatization | COSMETIC | Hot-path fields kept public or use inline accessors; benchmark before/after |

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

**Implemented templates (PR 4-7):**

| Policy Type | Templates Available |
|-------------|---------------------|
| **Admission** | `always-admit`, `token-bucket` |
| **Priority** | `constant`, `slo-based` |
| **Routing** | `round-robin`, `least-loaded`, `weighted` (weighted-scoring), `prefix-affinity` |
| **Scheduler** | `fcfs`, `priority-fcfs`, `sjf` (shortest job first) |

**Planned templates (future PRs):**

| Policy Type | Planned Template | Target PR |
|-------------|-----------------|-----------|
| **Admission** | `rate-limit`, `tenant-quota` | PR9+ (requires TenantID) |
| **Priority** | `tenant-priority`, `deadline-aware` | PR9+ (requires TenantState) |

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

**Example: TokenBucket with internal state (actual implementation in `sim/admission.go`)**

```go
type TokenBucket struct {
    // Configuration (from PolicyBundle/CLI)
    capacity      float64
    refillRate    float64  // tokens per second

    // Internal state (persists across Admit() calls)
    currentTokens float64
    lastRefill    int64
}

func (tb *TokenBucket) Admit(req *Request, state *RouterState) (bool, string) {
    // Refill tokens based on simulated time elapsed (deterministic)
    clock := state.Clock
    elapsed := clock - tb.lastRefill
    if elapsed > 0 {
        refill := float64(elapsed) * tb.refillRate / 1e6
        tb.currentTokens = min(tb.capacity, tb.currentTokens+refill)
        tb.lastRefill = clock
    }
    // Consume tokens if available
    cost := float64(len(req.InputTokens))
    if tb.currentTokens >= cost {
        tb.currentTokens -= cost
        return true, ""
    }
    return false, "insufficient tokens"
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

> **v3.1 note:** The interfaces below show the **actual frozen signatures** (post-PR8 interface freeze).
> Additive changes (new struct fields, new `Extended` map keys) are permitted; breaking changes are not.
> Original aspirational signatures from v2.0 were simplified during PRs 4-7 implementation.

```go
// === Admission === (sim/admission.go)
// Cluster-level: called by AdmissionDecisionEvent.Execute with *RouterState

type AdmissionPolicy interface {
    Admit(req *Request, state *RouterState) (admitted bool, reason string)
}
// Templates: AlwaysAdmit, TokenBucket
// Future: DELAY action may be added via extended return type (PR11 autoscaler)

// === Priority === (sim/priority.go)
// Instance-level: called by Simulator.Step() — no RouterState (instance-local only)

type PriorityPolicy interface {
    Compute(req *Request, clock int64) float64
}
// Templates: ConstantPriority, SLOBasedPriority
// Note: SLO class integration using TenantState planned for PR9+

// === Routing === (sim/routing.go)
// Cluster-level: called by RoutingDecisionEvent.Execute with *RouterState

type RoutingDecision struct {
    TargetInstance string             // Instance ID (must match a snapshot ID)
    Reason         string             // Human-readable explanation
    Scores         map[string]float64 // Instance ID → composite score (nil if N/A)
    Priority       float64            // Cluster-level priority hint; 0 = defer to instance PriorityPolicy
    // Future additive fields (permitted under freeze):
    // PrefillInstance, DecodeInstance (PR14 P/D disaggregation)
    // Candidates []CandidateScore (PR13 counterfactual traces)
}

type RoutingPolicy interface {
    Route(req *Request, state *RouterState) RoutingDecision
}
// Templates: RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity

// === Instance Scheduler === (sim/scheduler.go)
// Instance-level: called by Simulator.Step() — reorders WaitQueue before batch formation

type InstanceScheduler interface {
    OrderQueue(requests []*Request, clock int64)
}
// Templates: FCFSScheduler, PriorityFCFSScheduler, SJFScheduler
// Note: Full MakeBatch/SchedulerContext interface deferred (PR7 implementation notes)

// === Auto-Scale === (planned, sim/autoscale.go, PR11)

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

type AutoScalePolicy interface {
    Evaluate(ctx AutoScaleContext) ScaleDecision
}
// AutoScaleContext details TBD in PR11 micro plan
```

### G.2 State Structures

> **v3.1 note:** Structures below show both **current fields** (implemented) and **planned fields**
> (annotated with target PR). Current fields are frozen after PR 8 — no removals or type changes.

```go
// RouterState — bridge type in sim/ (not sim/cluster/) to avoid import cycles.
// Current implementation (PR8): minimal struct with Snapshots slice + Clock.
// Future PRs add fields additively.
type RouterState struct {
    Snapshots  []RoutingSnapshot           // Current (PR8): one per instance, deterministic order
    Clock      int64                        // Current (PR8): simulation clock in microseconds
    // Planned additions (additive, no breaking change):
    // PerTenant  map[TenantID]*TenantState // PR9+: requires TenantID on Request
    // Global     GlobalMetrics             // PR9+: cluster-wide aggregate metrics
}

type TenantState struct {               // Planned for PR9+
    RequestCount   int
    ActiveRequests int
    RecentRate     float64 // requests/sec over sliding window
    SLOClass       string
}

// InstanceSnapshot — immutable value type in sim/cluster/snapshot.go
type InstanceSnapshot struct {
    // Current fields (implemented, frozen after PR 8):
    ID            InstanceID
    Timestamp     int64           // Clock time when snapshot was captured
    QueueDepth    int
    BatchSize     int
    KVUtilization float64         // GPU tier utilization (0.0-1.0)
    FreeKVBlocks  int64           // TotalBlocks - UsedBlockCnt

    // Planned fields (additive, to be added in future PRs):
    // PoolType          PoolType   // PR14: MONOLITHIC (default), PREFILL, DECODE
    // InFlightRequests  int        // PR9+: requires tracking in InstanceSimulator
    // RecentTTFT        float64    // PR9+: requires sliding window metrics
    // RecentTPOT        float64    // PR9+: requires sliding window metrics
    // CacheHitRate      float64    // PR9+: requires KV hit/miss tracking
    // EstimatedWaitTime float64    // PR9+: derived from queue + batch state
    // Extended map[string]float64  // Phase 4: CPUKVUtilization, TransferQueueDepth, etc.
}

// SnapshotProvider controls when/how snapshots are refreshed.
// Implemented in sim/cluster/snapshot.go (PR4).
type SnapshotProvider interface {
    Snapshot(id InstanceID, clock int64) InstanceSnapshot
    RefreshAll(clock int64)
}

// ObservabilityConfig — current implementation (PR4) has 3 fields.
// Planned additions annotated below.
type ObservabilityConfig struct {
    // Current (implemented):
    QueueDepth    FieldConfig // default: Immediate
    BatchSize     FieldConfig // default: Immediate
    KVUtilization FieldConfig // default: Immediate
    // Planned (future PRs):
    // CacheHitRate  FieldConfig // PR9+: default Periodic(10000 ticks)
    // RecentTTFT    FieldConfig // PR9+: default Periodic(100000 ticks)
    // RecentTPOT    FieldConfig // PR9+: default Periodic(100000 ticks)
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
CURRENT                              TARGET (Phase 2 Research-Ready)
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
├── simulator.go          # Existing: SimConfig, Simulator, event loop, batch formation
├── request.go            # Existing (extended with Priority in PR7; TenantID planned for PR9+)
├── event.go              # Existing: ArrivalEvent, StepEvent, etc.
├── kvcache.go            # Existing (single-tier, sufficient for Phase 2)
├── metrics.go            # Existing (foundation for RawMetrics)
├── rng.go                # PartitionedRNG (PR1)
├── admission.go          # AdmissionPolicy interface, AlwaysAdmit, TokenBucket (PR4/5)
├── routing.go            # RoutingPolicy interface, RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity (PR6)
├── priority.go           # PriorityPolicy interface, ConstantPriority, SLOBasedPriority (PR7)
├── scheduler.go          # InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler (PR7)
├── router_state.go       # RouterState bridge type (PR8) — in sim/ to avoid import cycle
├── bundle.go             # PolicyBundle, YAML loading (PR8)
├── autoscale.go          # AutoScalePolicy interface + templates (planned, PR11)
├── cluster/
│   ├── cluster.go        # ClusterSimulator (restructured Run() with control/data plane)
│   ├── instance.go       # InstanceSimulator wrapper (+ observation methods: QueueDepth, BatchSize, KVUtilization, FreeKVBlocks)
│   ├── cluster_event.go  # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent (New in v2.3)
│   ├── snapshot.go       # InstanceSnapshot, SnapshotProvider, ObservabilityConfig (New in v2.3)
│   ├── workload.go       # Centralized request generation for cluster dispatch
│   ├── deployment.go     # DeploymentConfig
│   └── metrics.go        # ClusterMetrics, RawMetrics, FitnessFunction (planned, PR9)
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

**Golden test guarantee:** `./simulation_worker run --model X` (without new flags) produces identical output to golden dataset tests via unified `ClusterSimulator` path.

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
| **In Scope** | `ClusterSimulator`, `DeploymentConfig`, `EventHeap` with ordering, `--num-instances` flag, basic round-robin dispatch |
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

### Phase 2: Policy Interfaces + Metrics (PR 4 first, then PR 5, then PRs 6-9)

After PR 3 and mock study, PR 4 must land first (cluster event infrastructure). PR 5 (simplification) must land before policy PRs. PRs 6-7 can then be developed **in parallel**.

---

#### PR 4: Cluster Control Plane + AdmissionPolicy

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add cluster event infrastructure, SnapshotProvider, and AdmissionPolicy` |
| **Motivation** | Mock study proved pre-dispatch routing breaks load-aware policies. Cluster needs event-driven control plane with online routing. Admission is the simplest policy to exercise the pipeline. |
| **In Scope** | Cluster-level event queue (`ClusterArrivalEvent`, `AdmissionDecisionEvent`, `RoutingDecisionEvent`), `InstanceSnapshot` with `Timestamp`, `SnapshotProvider` + `CachedSnapshotProvider`, `ObservabilityConfig`, `InstanceSimulator` observation methods (`QueueDepth()`, `BatchSize()`, `KVUtilization()`, `FreeKVBlocks()`), restructured `ClusterSimulator.Run()`, `AdmissionPolicy` interface, `AlwaysAdmit` + `TokenBucket` templates, configurable per-event latency |
| **Out of Scope** | Routing policy templates (PR 6), pathological templates (PR 9) |
| **Files Changed** | New: `sim/cluster/cluster_event.go` (~150 LOC), `sim/cluster/snapshot.go` (~200 LOC), `sim/admission.go` (~65 LOC). Modified: `sim/cluster/cluster.go` (~200 LOC restructure), `sim/cluster/instance.go` (~30 LOC observation methods), `sim/request.go` (~5 LOC TenantID), `cmd/root.go` (~30 LOC) |
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

### Phase 2a: Simplification (1 PR)

#### PR 5: Architectural Simplification

| Aspect | Details |
|--------|---------|
| **Title** | `refactor: SimConfig struct, unified CLI path, field privatization, interface dedup` |
| **Motivation** | Reduce codebase complexity before adding policy features; eliminate duplicated code and multi-param constructors |
| **Depends On** | PR 4 |
| **In Scope** | (1) `SimConfig` options struct replacing 17-param constructors, (2) Unified CLI path (always ClusterSimulator, even N=1), (3) Privatize ~15 internal Simulator fields, (4) Move `AdmissionPolicy` interface to `sim/` base package |
| **Out of Scope** | Unified event queue, workload deduplication, package restructuring |
| **Files Changed** | Modified: `sim/simulator.go` (~80 LOC), `sim/cluster/instance.go` (~40 LOC), `sim/cluster/cluster.go` (~20 LOC), `sim/cluster/deployment.go` (~10 LOC), `cmd/root.go` (~60 LOC), `sim/admission.go` (~10 LOC), test files (~100 LOC). New: none |
| **CLI** | `./simulation_worker run --model X` (same interface, but internally always uses ClusterSimulator) |
| **Tests** | All golden tests must pass. New: `TestUnifiedCLIPath_MatchesGoldenDataset` (verifies N=1 cluster path = single-instance results) |
| **No Dead Code** | SimConfig used by all constructors; unified path eliminates dead single-instance path |
| **LOC Estimate** | ~320 changed, ~200 net reduction |
| **Architectural Impact** | Major internal cleanup; no behavioral change; all golden tests pass |
| **Behavioral Guarantees** | Bit-exact golden test output preserved; CLI interface unchanged |
| **API Surface Changes** | None (internal refactor) |
| **README Changes** | None |
| **Risks + Mitigations** | Risk: RNG stream divergence. Mitigation: Test that request generation produces identical tokens through both paths before removing old path. Risk: Big PR. Mitigation: Atomic commits per simplification (SimConfig first, then unified CLI, then privatization, then interface dedup). |
| **Why Independently Reviewable** | Pure refactor with golden test verification; no new features |

**v2.3 → v3.0 change:** NEW PR. Justified by dropping backward compatibility constraint — this cleanup makes every subsequent PR simpler.

---

### Phase 2b: Policy Interfaces + Metrics (4 PRs, PRs 6-7 parallel after PR 5)

After PR 5 (simplification), PRs 6-7 can be developed **in parallel**.

---

#### PR 6: RoutingPolicy Interface — ✅ COMPLETED 2026-02-16

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RoutingPolicy with RoundRobin and WeightedScoring` |
| **Motivation** | Enable intelligent request routing across instances |
| **Depends On** | PR 5 |
| **In Scope** | `RoutingPolicy` interface, `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity` templates |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/routing.go` (~193 LOC). Modified: `sim/cluster/cluster.go` (~30 LOC), `sim/cluster/cluster_event.go` (~20 LOC), `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --routing-policy weighted --routing-cache-weight 0.6` |
| **Tests** | Unit: each template. Integration: load distribution across instances |
| **Parallel With** | PR 7 |
| **No Dead Code** | All templates exercisable via `--routing-policy` flag |
| **LOC Estimate** | ~270 |
| **Architectural Impact** | Replaces hardcoded round-robin in `RoutingDecisionEvent.Execute` with pluggable policy |
| **Behavioral Guarantees** | `round-robin` (default) matches existing behavior |
| **API Surface Changes** | New CLI flags: `--routing-policy`, `--routing-cache-weight`, `--routing-load-weight` |
| **README Changes** | Add "Routing Policies" section |
| **Risks + Mitigations** | Risk: Prefix affinity cache misses. Mitigation: Fallback to least-loaded on miss. |
| **Why Independently Reviewable** | Complete routing feature; default preserves existing behavior |

**v2.3 → v3.0 change:** Now depends on PR 5 (uses `SimConfig`). Otherwise identical to v2.3 PR 6.

**Implementation notes (COMPLETED 2026-02-16):** Files placed in `sim/routing.go` (not `sim/policy/`) following the same pattern as `sim/admission.go`. `RoutingSnapshot` bridge type avoids `sim/cluster/` import cycle. `PrefixAffinity` reuses `hashTokens` from `sim/kvcache.go` (same package). See `docs/plans/pr6-routing-plan.md` for full plan with behavioral contracts.

---

#### PR 7: PriorityPolicy + InstanceScheduler — ✅ COMPLETED 2026-02-16

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add PriorityPolicy and InstanceScheduler with templates` |
| **Motivation** | Enable request prioritization and per-instance batch scheduling policies |
| **Depends On** | PR 5 |
| **In Scope** | `PriorityPolicy` interface (`ConstantPriority`, `SLOBasedPriority`), `InstanceScheduler` interface (`FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`), `Priority` field on `Request`, `SchedulerContext` |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/priority.go` (~50 LOC), `sim/scheduler.go` (~74 LOC). Modified: `sim/request.go` (~10 LOC), `sim/simulator.go` (~15 LOC), `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --priority-policy slo-based --scheduler priority-fcfs` |
| **Tests** | Unit: each template. Integration: priority ordering in scheduling, batch formation respects scheduler |
| **Parallel With** | PR 6 |
| **No Dead Code** | All templates exercisable via `--priority-policy` and `--scheduler` flags |
| **LOC Estimate** | ~370 |
| **Architectural Impact** | Adds Priority field to Request; extracts batch formation policy |
| **Behavioral Guarantees** | `constant` priority + `fcfs` scheduler (defaults) match existing behavior |
| **API Surface Changes** | New CLI flags: `--priority-policy`, `--scheduler` |
| **README Changes** | Add "Priority Policies" and "Instance Schedulers" sections |
| **Risks + Mitigations** | Risk: SJF starvation. Mitigation: Document limitation. Risk: Priority affecting determinism. Mitigation: Tie-breaking rules documented. |
| **Why Independently Reviewable** | Complete priority + scheduling feature; defaults preserve existing behavior |

**Implementation notes (COMPLETED 2026-02-16):** Files placed in `sim/priority.go` and `sim/scheduler.go` (not `sim/policy/`) to avoid circular dependency — follows PR5 precedent with `sim/admission.go`. `InstanceScheduler` uses simplified `OrderQueue(requests, clock)` interface that sorts the WaitQueue before existing `makeRunningBatch()`, deferring full `MakeBatch`/`SchedulerContext` to when needed. Wiring in `sim/simulator.go` (not `sim/cluster/instance.go`) since the integration point is `Simulator.Step()`. See `docs/plans/pr7-priority-plus-sched-plan.md` for full plan with behavioral contracts.

**v2.3 → v3.0 change:** MERGED from v2.3 PR 5 (Priority) + PR 7 (Scheduler). Both affect request ordering and share the `Priority` field on `Request`. Natural cohesion reduces CLI flag PRs that touch `cmd/root.go`.

**Note on parallel PR integration:** PRs 6-7 each add flags to `cmd/root.go`. To avoid merge conflicts:
- Each PR adds its flags in a clearly separated block with comment header
- PR 8 (PolicyBundle) consolidates flag handling and resolves any conflicts

---

#### PR 8: RouterState and PolicyBundle

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RouterState and PolicyBundle configuration` |
| **Motivation** | Unified policy configuration via YAML |
| **Depends On** | PR 6, PR 7 |
| **In Scope** | `RouterState` (bridge type in `sim/`), `PolicyBundle` YAML loading, `RoutingDecision.Priority` field, interface signature updates for `AdmissionPolicy` and `RoutingPolicy` to accept `*RouterState` |
| **Out of Scope** | AutoScale policy (Phase 4), TenantState/GlobalMetrics (deferred to PR9+ — no TenantID on Request yet) |
| **Files Changed** | New: `sim/router_state.go` (~20 LOC), `sim/bundle.go` (~90 LOC). Modified: `sim/admission.go` (~10 LOC), `sim/routing.go` (~45 LOC), `sim/cluster/cluster_event.go` (~25 LOC), `cmd/root.go` (~40 LOC) |
| **CLI** | `./simulation_worker run --model X --policy-config policies.yaml` |
| **Tests** | Unit: YAML parsing, validation. Integration: policy config overrides CLI flags |
| **No Dead Code** | `--policy-config` flag exercises PolicyBundle loading |
| **LOC Estimate** | ~280 |
| **Architectural Impact** | Introduces hierarchical config; RouterState aggregates cluster-wide state |
| **Behavioral Guarantees** | CLI flags override YAML defaults; missing config uses defaults |
| **API Surface Changes** | New CLI flag: `--policy-config` |
| **README Changes** | Add "Policy Configuration" section with YAML example |
| **Risks + Mitigations** | Risk: YAML parsing errors. Mitigation: Validate on load; clear error messages. |
| **Why Independently Reviewable** | Integrates PRs 6-7; provides unified config interface |

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 8. Dependencies updated.

**INTERFACE FREEZE after PR 8** — Policy interfaces are stable. "Freeze" means no breaking changes (no field removals, no type changes, no method signature changes). Adding new fields to structs or new keys to `Extended` maps is permitted in later phases.

---

#### PR 9: RawMetrics, Anomaly Detection, and Pathological Templates

| Aspect | Details |
|--------|---------|
| **Title** | `feat(metrics): Add RawMetrics, anomaly detection, and pathological policy templates` |
| **Motivation** | Enable fitness evaluation and validate anomaly detection |
| **Depends On** | PR 8 (sequenced after Interface Freeze; PR 8 transitively depends on PR 6, PR 7) |
| **In Scope** | `RawMetrics`, `Distribution`, `FitnessFunction`, `EvaluationResult`, anomaly counters, pathological templates (`RejectAll`, `InvertedSLO`, `AlwaysBusiest`, `ReversePriority`), validation tests |
| **Out of Scope** | Scale oscillation detection (requires AutoScaler in PR 11) |
| **Files Changed** | New: `sim/cluster/metrics.go` (~300 LOC). Modified: `sim/admission.go`, `sim/routing.go`, `sim/priority.go`, `sim/scheduler.go` (~100 LOC total for pathological templates), `sim/cluster/cluster.go` (~50 LOC), `cmd/root.go` (~20 LOC) |
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

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 9. Dependencies updated (now depends on PR 8 instead of PR 4-7 directly).

**Pathological templates added:**

| Template | Policy Type | Purpose | Expected Anomaly |
|----------|-------------|---------|------------------|
| `reject-all` | Admission | Baseline for admission metrics | 100% rejection rate |
| `inverted-slo` | Priority | Test priority inversion detection | High `PriorityInversionCount` |
| `always-busiest` | Routing | Test load imbalance detection | High `HOLBlockingEvents` |
| `reverse-priority` | Scheduler | Test scheduler fairness | High `PriorityInversionCount` |

---

### RESEARCH-READY CHECKPOINT (After PR 9)

At this point (~5 weeks), BLIS supports:
- Multi-instance cluster simulation with control plane / data plane separation
- Online routing (event-driven, not pre-dispatched)
- InstanceSnapshot with configurable staleness (ObservabilityConfig)
- All 4 policy interfaces (admission, priority, routing, scheduler)
- PolicyBundle with YAML configuration
- RawMetrics with fitness evaluation
- Anomaly detection validated with pathological templates
- Existing workload generation (`--workload distribution`)
- Single-tier KV cache (existing `sim/kvcache.go`)

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
| **Depends On** | PR 9 (Research-Ready checkpoint) |
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

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 10.

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

### Phase 4: Advanced Features (4 PRs, Parallelizable)

After Research-Ready checkpoint, three tracks can proceed **in parallel**.

---

#### PR 11: AutoScaler Core + Actuation

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add AutoScaler with ThresholdScaler, provisioning delays, and warmup` |
| **Motivation** | Enable dynamic scaling based on load with realistic timing |
| **Depends On** | PR 9 |
| **In Scope** | `AutoScaler`, `AutoScalePolicy`, `ThresholdScaler`, `Oscillator` (pathological), `WarmupProfile`, `DrainPolicy`, `InstanceState` lifecycle |
| **Out of Scope** | Predictive scaling |
| **Files Changed** | New: `sim/autoscale.go` (~380 LOC). Modified: `sim/cluster/cluster.go` (~70 LOC), `sim/cluster/instance.go` (~50 LOC), `cmd/root.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --autoscaler-enabled --autoscaler-max 8 --provisioning-delay 30s` |
| **Tests** | Unit: threshold logic. Integration: scale up/down with warmup, drain, oscillation detection |
| **Parallel With** | PR 12, PR 13 |
| **No Dead Code** | `--autoscaler-enabled` exercises all paths; `Oscillator` validates scale oscillation detection |
| **LOC Estimate** | ~520 |
| **Architectural Impact** | Adds periodic scale check event to ClusterSimulator; instance lifecycle states (provisioning, warming, ready, draining) |
| **Behavioral Guarantees** | Scale decisions respect min/max bounds; oscillation detected and counted; new instances not routed until warmup complete; draining instances finish existing requests |
| **API Surface Changes** | New CLI flags: `--autoscaler-enabled`, `--autoscaler-min`, `--autoscaler-max`, `--autoscaler-threshold`, `--provisioning-delay`, `--warmup-duration`, `--drain-policy` |
| **README Changes** | Add "Auto-Scaling" section with actuation model |
| **Risks + Mitigations** | Risk: Scale oscillation. Mitigation: Cooldown period; `Oscillator` template validates detection. Risk: Complex state machine. Mitigation: Clear state diagram in code comments. |
| **Why Independently Reviewable** | Complete autoscaler with realistic timing in one PR |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 11 (AutoScaler Core) + PR 12 (Actuation). Actuation without the core is dead code; core without actuation is unrealistic. Combined PR is ~520 LOC, still reviewable.

---

#### PR 12: Tiered KV Cache + Transfer

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add tiered KV cache with GPU/CPU offload/reload mechanics` |
| **Motivation** | Model GPU+CPU KV cache with transfer latency |
| **Depends On** | PR 9 |
| **In Scope** | `KVTier` enum, `KVTierConfig`, offload trigger, reload on CPU hit, transfer latency, `KVThrashingRate` metric |
| **Out of Scope** | P/D architecture (PR 14) |
| **Files Changed** | New: `sim/kv/tiered.go` (~100 LOC), `sim/kv/transfer.go` (~200 LOC). Modified: `sim/kvcache.go` (~30 LOC), `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 1000 --kv-cpu-blocks 10000 --kv-offload-threshold 0.9` |
| **Tests** | Unit: tier assignment, transfer timing. Integration: CPU blocks available, thrashing detection |
| **Parallel With** | PR 11, PR 13 |
| **No Dead Code** | `--kv-cpu-blocks` and `--kv-offload-threshold` flags exercise all tiered KV code |
| **LOC Estimate** | ~360 |
| **Architectural Impact** | Extends KVCacheState with tier awareness; adds KVTransfer events to event queue |
| **Behavioral Guarantees** | `--kv-cpu-blocks 0` (default) preserves existing behavior; offload when GPU > threshold; thrashing counted |
| **API Surface Changes** | New CLI flags: `--kv-gpu-blocks`, `--kv-cpu-blocks`, `--kv-offload-threshold`, `--kv-transfer-bandwidth` |
| **README Changes** | Add "Tiered KV Cache" section with transfer mechanics |
| **Risks + Mitigations** | Risk: Transfer deadlock. Mitigation: Timeout on pending transfers. Risk: Tier confusion in existing code. Mitigation: Default to GPU tier. |
| **Why Independently Reviewable** | Complete tiered KV feature; `--kv-cpu-blocks 0` preserves existing behavior |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 13 (KV Tier Types) + PR 14 (Transfer). Types without transfer is dead code.

---

#### PR 13: Decision Traces + Counterfactual Analysis

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add DecisionTrace with RoutingRecord and counterfactual analysis` |
| **Motivation** | Enable policy decision debugging and "what-if" analysis |
| **Depends On** | PR 9 |
| **In Scope** | `SimulationTrace`, `DecisionTrace`, `RoutingRecord`, `TraceConfig`, `TopKCandidates`, `Regret`, `TraceSummary` |
| **Out of Scope** | LLM-based reflection (framework-specific) |
| **Files Changed** | New: `sim/trace/trace.go` (~100 LOC), `sim/trace/record.go` (~150 LOC), `sim/trace/summary.go` (~200 LOC). Modified: `cmd/root.go` (~35 LOC) |
| **CLI** | `./simulation_worker run --model X --trace-level decisions --counterfactual-k 5 --summarize-trace` |
| **Tests** | Unit: regret calculation. Integration: trace JSON includes routing decisions and counterfactuals |
| **Parallel With** | PR 11, PR 12 |
| **No Dead Code** | `--trace-level` and `--summarize-trace` exercise all trace code |
| **LOC Estimate** | ~485 |
| **Architectural Impact** | Adds trace collection to ClusterSimulator; policies log decisions |
| **Behavioral Guarantees** | `--trace-level none` (default) has no overhead; `decisions` captures all policy calls; top-k candidates stored per routing decision |
| **API Surface Changes** | New CLI flags: `--trace-level` (none/decisions/detailed), `--counterfactual-k`, `--summarize-trace` |
| **README Changes** | Add "Decision Tracing" section with counterfactual analysis |
| **Risks + Mitigations** | Risk: Trace memory bloat. Mitigation: Configurable verbosity levels. Risk: Expensive for large k. Mitigation: Default k=3; document performance. |
| **Why Independently Reviewable** | Complete trace + analysis feature; `--trace-level none` has no overhead |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 17 (Traces) + PR 18 (Counterfactual). Traces without counterfactual has limited value; combined is ~485 LOC, still reviewable.

---

#### PR 14: P/D Disaggregated Architecture + KV Transfer

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add disaggregated prefill-decode architecture with KV transfer` |
| **Motivation** | Model DistServe/Splitwise style deployments |
| **Depends On** | PR 12 (tiered KV) |
| **In Scope** | `DISAGGREGATED_PD` type, `PrefillPool`, `DecodePool`, `PDHandoffEvent`, `PDTransferConfig`, `BlockTransferState`, routing changes, ownership tracking |
| **Out of Scope** | Multi-hop transfers |
| **Files Changed** | Modified: `sim/cluster/deployment.go` (~50 LOC), `sim/cluster/cluster.go` (~200 LOC), `sim/cluster/cluster_event.go` (~100 LOC), `sim/kv/transfer.go` (~150 LOC), `cmd/root.go` (~40 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --prefill-replicas 2 --decode-replicas 4` |
| **Tests** | Integration: request flows P→D; separate pools visible; KV transfer timing; ownership invariant |
| **No Dead Code** | `--architecture pd` exercises all P/D paths |
| **LOC Estimate** | ~540 |
| **Architectural Impact** | Major: separate prefill and decode pools with handoff and KV ownership transfer |
| **Behavioral Guarantees** | `--architecture monolithic` (default) preserves existing behavior; block has exactly one owner at any time; transfer adds latency |
| **API Surface Changes** | New CLI flags: `--architecture`, `--prefill-replicas`, `--decode-replicas`, `--pd-transfer-latency`, `--pd-transfer-bandwidth` |
| **README Changes** | Add "Prefill-Decode Disaggregation" section with KV transfer |
| **Risks + Mitigations** | Risk: Handoff timing complexity. Mitigation: Model as discrete event. Risk: Ownership bugs (double-free). Mitigation: Conservation invariant enforced. |
| **Why Independently Reviewable** | Complete P/D feature; `--architecture monolithic` (default) preserves existing behavior |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 15 (P/D Architecture) + PR 16 (KV Transfer for P/D). P/D without KV transfer is unrealistic.

---

### Phase 5: Framework Adapters (1 PR, Optional)

BLIS is fully functional without this. It provides convenience for framework integration.

---

#### PR 15: Framework Adapters (GEPA + OpenEvolve)

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add GEPA and OpenEvolve framework adapters` |
| **Motivation** | Enable framework integration for evolutionary policy optimization |
| **Depends On** | PR 13 (traces) |
| **In Scope** | `BLISGEPAAdapter`, `BLISEvaluator`, `gepa-evaluate` command, `openevolve-evaluate` command |
| **Out of Scope** | GEPA and OpenEvolve frameworks themselves (external dependencies) |
| **Files Changed** | New: `sim/adapter/gepa.go` (~150 LOC), `sim/adapter/openevolve.go` (~150 LOC). Modified: `cmd/root.go` (~80 LOC) |
| **CLI** | `./simulation_worker gepa-evaluate --policy-config p.yaml` and `./simulation_worker openevolve-evaluate --config oe.yaml` |
| **Tests** | Integration: each adapter returns expected format |
| **No Dead Code** | `gepa-evaluate` and `openevolve-evaluate` commands exercise both adapters |
| **LOC Estimate** | ~380 |
| **Architectural Impact** | Adds two CLI subcommands; wraps Run() with framework-specific formats |
| **Behavioral Guarantees** | Returns fitness + traces in framework-specific formats |
| **API Surface Changes** | New subcommands: `gepa-evaluate`, `openevolve-evaluate` |
| **README Changes** | Add "Framework Integration" section |
| **Risks + Mitigations** | Risk: Framework format changes. Mitigation: Version adapters with framework releases. |
| **Why Independently Reviewable** | Self-contained adapters; BLIS core unchanged |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 19 + PR 20. Both are thin wrappers (~150 LOC each) with no cross-dependencies.

---

### Phase 6: Validation (1 PR)

#### PR 16: Integration Tests and Examples

| Aspect | Details |
|--------|---------|
| **Title** | `test: Add comprehensive integration test suite and examples` |
| **Motivation** | Validate end-to-end workflows and provide usage examples |
| **Depends On** | PR 11, PR 13, PR 14 (tests autoscaling, traces, and P/D features) |
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

**v2.3 → v3.0 change:** Renumbered from PR 21. Content unchanged.

---

## J) Dependency DAG (Restructured, v3.0)

### PR Dependency Graph

```
PHASE 1: FOUNDATION (Sequential, 3 PRs)
════════════════════════════════════════════════════════════════════════════════

  PR 1 ──────► PR 2 ──────► PR 3
  (RNG)      (Instance)   (Cluster+Deploy)
                                │
                                ▼
                    MOCK STUDY CHECKPOINT (COMPLETED)
                                │
                                ▼
PHASE 2: POLICY INTERFACES + METRICS ══════════════════════════════════════════
                                │
                                ▼
                              PR 4
                  (Control Plane + Admission)
                                │
PHASE 2a: SIMPLIFICATION ═════════════════════════════════════════════════════
                                │
                              PR 5
                     (SimConfig + Unified
                      CLI + Privatization)
                                │
PHASE 2b: POLICY INTERFACES ══════════════════════════════════════════════════
                                │
              ┌─────────────────┼───────────────┐
              ▼                                 ▼
            PR 6                              PR 7
          (Routing)                     (Priority+Sched)
              │                                 │
              └─────────────────┬───────────────┘
                                ▼
                              PR 8
                           (Bundle)
                                │
                                ▼
                    INTERFACE FREEZE
                                │
                                ▼
                              PR 9
                    (Metrics + Pathological)
                                │
                                ▼
                    RESEARCH-READY CHECKPOINT
                                │
         ┌──────────────────────┼──────────────────────────────────────┐
         ▼                      │                                      │
       PR 10                    │                                      │
 (Workload, Optional)           │                                      │
                                │                                      │
PHASE 4: ADVANCED FEATURES (4 PRs) ═══════════════════════════════════════════
                                │
     ┌──────────────────────────┼──────────────────────┐
     ▼                        ▼                        ▼
   PR 11                    PR 12                    PR 13
 (AutoScale              (Tiered KV              (Traces +
  + Actuation)            + Transfer)             Counterfactual)
     │                        │                        │
     │                        ▼                        ├──────────────┐
     │                      PR 14                      │              ▼
     │                  (P/D + KV Xfer)                │        PHASE 5: ADAPTERS
     │                        │                        │        (Optional)
     │                        │                        │              │
     │                        │                        │            PR 15
     │                        │                        │      (GEPA + OpenEvolve)
     │                        │                        │
PHASE 6: VALIDATION ══════════════════════════════════════════════════════════
     │                        │                        │
     └────────────────────────┼────────────────────────┘
                              ▼
                            PR 16
                    (Tests: depends on 11+13+14)
```

### Parallel Development Matrix

| Gate | Completed PRs | Unlocked for Parallel Development |
|------|---------------|-----------------------------------|
| **G1** | PR 5 (Simplification) | PR 6, PR 7 (2 parallel) |
| **G2** | PR 9 (Research-Ready) | PR 10, PR 11, PR 12, PR 13 (4 parallel tracks) |
| **G3** | PR 13 | PR 15 |

### Critical Checkpoints

| Checkpoint | Verification | Failure Action |
|------------|--------------|----------------|
| **Mock Study (after PR 3)** | COMPLETED 2026-02-13. Findings: pre-dispatch routing broken, 4 observable gaps identified | PR 4 restructured with online routing and observation methods |
| **After PR 3** | Determinism: 100 runs identical | Block until fixed |
| **After PR 5** | Golden tests pass through unified cluster path | Block until fixed |
| **Interface Freeze (after PR 8)** | Policy interfaces frozen (no breaking changes) | Additive changes (new fields, new `Extended` keys) permitted |
| **Research-Ready (after PR 9)** | Pathological policies trigger anomalies | Required for research |
| **After PR 13** | All observability features working | Required for adapters |

### Timeline Estimate (3-4 developers)

```
Week 1-2:   Phase 1 (PR 1-3, sequential, 1 dev)
            + Mock Study (2-3 days) COMPLETED

Week 3:     Phase 2 PR 4 (Control Plane + Admission, sequential, 1 dev)
            PR 5 (Simplification, 1 dev, ~2 days)

Week 3-4:   PR 6-7 (2 parallel devs), then PR 8  [PR 6, PR 7 COMPLETED 2026-02-16]

Week 4-5:   PR 9
            → RESEARCH-READY (~5 weeks)

Week 6-7:   PR 10 (optional, 1 dev)
            PR 11 (1 dev)
            PR 12 (1 dev)
            PR 13 (1 dev)

Week 7-8:   PR 14 (1 dev, depends on PR 12)

Week 9:     PR 15 (adapters, 1 dev)

Week 10:    PR 16 (tests, 1 dev)

Total: ~10 weeks with 3-4 developers
Research-ready: ~5 weeks
```

### v2.3 → v3.0 PR Mapping

| v3.0 PR | v2.3 PR(s) | Change |
|---------|-----------|--------|
| PR 1 | PR 1 | PartitionedRNG — unchanged |
| PR 2 | PR 2 | InstanceSimulator — unchanged |
| PR 3 | PR 3 | ClusterSimulator — unchanged |
| PR 4 | PR 4 | Control Plane + Admission — unchanged |
| PR 5 | NEW | Simplification (constructor collapse, unified CLI, field privatization, interface dedup) |
| PR 6 | PR 6 | Routing — unchanged except depends on PR 5 |
| PR 7 | PR 5 + PR 7 | MERGED: Priority + Scheduler |
| PR 8 | PR 8 | PolicyBundle — unchanged |
| PR 9 | PR 9 | RawMetrics — unchanged |
| PR 10 | PR 10 | Workload — unchanged |
| PR 11 | PR 11 + PR 12 | MERGED: AutoScaler + Actuation |
| PR 12 | PR 13 + PR 14 | MERGED: Tiered KV + Transfer |
| PR 13 | PR 17 + PR 18 | MERGED: Traces + Counterfactual |
| PR 14 | PR 15 + PR 16 | MERGED: P/D Architecture + KV Transfer |
| PR 15 | PR 19 + PR 20 | MERGED: GEPA + OpenEvolve adapters |
| PR 16 | PR 21 | Integration Tests — renumbered |

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
| `TestUnifiedCLIPath` | N=1 cluster path matches single-instance golden tests | PR 5 |
| `TestKVTierTransfer` | Offload/reload timing and conservation | PR 12 |
| `TestPDHandoff` | Prefill → Transfer → Decode lifecycle | PR 14 |
| `TestAutoScaling` | Scale up/down with warmup and drain | PR 11 |

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
| **Total PRs** | 16 |
| **Total LOC Estimate** | ~5,200 (net reduction from PR 5 simplification) |
| **Max Parallel PRs** | 2 (Phase 2b PRs 6-7), 4 (G2 gate: PRs 10-13 parallel) |
| **Research-Ready** | ~5 weeks |
| **Full Implementation** | ~10 weeks (with 3-4 developers) |

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
15. **Backward compatibility dropped** — golden tests are the verification gate; CLI always uses ClusterSimulator (v3.0)
16. **Constructor collapse** — `SimConfig` struct replaces 17-param constructors (v3.0)

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

### Changes from v2.3 (v3.0)

| Change | Rationale |
|--------|-----------|
| New PR 5: Architectural Simplification | Constructor collapse, unified CLI path, field privatization, interface dedup reduce complexity for all subsequent PRs |
| Priority + Scheduler merged (→ PR 7) | Both affect request ordering and share `Priority` field on `Request`; natural cohesion |
| AutoScaler + Actuation merged (→ PR 11) | Actuation without core is dead code; core without actuation is unrealistic |
| Tiered KV + Transfer merged (→ PR 12) | Types without transfer is dead code |
| Traces + Counterfactual merged (→ PR 13) | Traces without counterfactual has limited value |
| P/D + KV Transfer merged (→ PR 14) | P/D without KV transfer is unrealistic |
| GEPA + OpenEvolve merged (→ PR 15) | Both thin wrappers (~150 LOC each) with no cross-dependencies |
| Integration Tests renumbered (→ PR 16) | Content unchanged |
| 21 PRs → 16 PRs | 6 merges + 1 new simplification PR; ~300 LOC net reduction |
| Backward compatibility dropped | Golden tests remain; separate single-instance path eliminated |
| Unified event queue deferred | Low ROI (~50 LOC), risk to golden tests, complicates P/D |

### Research Agenda Alignment

This plan directly supports the four llm-d inference control problems:

| Research Problem | BLIS Support | Key PRs |
|-----------------|--------------|---------|
| **Routing Policy Evolution** | `RoutingPolicy` interface, `InstanceSnapshot` observables, existing prefix caching | PR 6, PR 9 |
| **Admission Control Evolution** | `AdmissionPolicy` interface, multi-tenant `TenantState`, fairness metrics | PR 4, PR 8-9 |
| **Joint Priority Scheduling** | `PriorityPolicy` + `InstanceScheduler` separation, priority inversion detection, HOL blocking detection | PR 7, PR 9 |
| **Autoscaling Evolution** | `AutoScalePolicy` interface, provisioning/warmup modeling, cost metrics, scale oscillation detection | PR 11 |

**Architectural Locality:** Each policy interface respects control boundaries—routing sees only `RouterState` + `InstanceSnapshot`, instance schedulers see only local state, autoscalers see only aggregate metrics.

**Sim-to-Real Transfer:** BLIS outputs directly deployable policy configurations, with structured validation protocol (Section K.4) to measure transfer quality.

### References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024
3. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
4. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023

### Next Steps

1. ~~Review and approve this plan (v3.0)~~ ✅ Approved
2. ~~PRs 5-7~~ ✅ Completed (SimConfig simplification, RoutingPolicy, PriorityPolicy+Scheduler)
3. Execute PR 8 (RouterState + PolicyBundle) — see `docs/plans/pr8-routing-state-and-policy-bundle-plan.md`
4. Then PR 9 (RawMetrics + Pathological Templates) → Research-Ready Checkpoint
5. (Parallel) Set up instrumented vLLM deployment for trace collection
