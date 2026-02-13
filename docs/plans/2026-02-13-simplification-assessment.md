# BLIS Architectural Simplification Assessment

**Date:** 2026-02-13
**Status:** Assessment Complete
**Based on:** Post-PR4 codebase (~6,300 LOC), macro plan v2.3
**Constraint:** Golden dataset tests (`testdata/goldendataset.json`) must continue to pass. All other backward compatibility dropped.

---

## Executive Summary

Four independent analyses assessed simplification opportunities for the BLIS inference simulator after merging PR4. The key finding: **we can reduce the remaining plan from 17 PRs (5-21) to 13 PRs while simplifying the architecture**, primarily through constructor collapse, unified CLI path, and interface deduplication. A fully unified event queue is feasible but should be deferred—the current hybrid approach (cluster queue + per-instance queues) is adequate and lower risk.

**Net impact:** ~300 LOC eliminated, 4 PRs removed, cleaner foundation for Phase 2-6 development.

---

## 1. Decision Matrix

| # | Simplification | LOC Impact | Risk | Golden-Test-Safe | Recommended | Notes |
|---|---------------|-----------|------|-----------------|-------------|-------|
| S1 | Constructor collapse: 17-param → `SimConfig` struct | -150 LOC | LOW | YES | **YES** | All constructors and call sites updated. No behavioral change. |
| S2 | Unified CLI path: always use `ClusterSimulator` (even N=1) | -80 LOC | **MEDIUM** | **CONDITIONAL** | **YES, with guard** | Requires `admissionLatency=0`, `routingLatency=0` defaults (already the case). RNG parity verified via `SubsystemWorkload`. Must validate single-instance golden tests pass through cluster path. |
| S3 | Eliminate duplicated `AdmissionPolicy` interface | -50 LOC | LOW | YES | **YES** | Move `AdmissionPolicy` interface to `sim/` base package. Both `cluster.go:18` and `policy/admission.go:11` define identical interface. |
| S4 | Field privatization: `Simulator` public fields → private + accessors | -20 LOC (net) | LOW | YES | **YES, incremental** | Start with fields not accessed by tests: `StepCount`, `PreemptionHappened`, `StepEvent`, `ReqNumComputedTokens`, etc. Defer fields heavily used by `InstanceSimulator` (`Clock`, `Metrics`, `WaitQ`, `KVCache`, `RunningBatch`). |
| S5 | Workload deduplication: `cluster/workload.go` calls `sim/workload_config.go` | -60 LOC | LOW | YES | **DEFER** | Both implementations use same RNG and produce same output. Refactoring is clean but low priority—saves minimal LOC vs risk of subtle divergence. |
| S6 | Package restructuring: merge `sim/policy/` into `sim/cluster/` | -30 LOC | MEDIUM | YES | **DEFER** | Import cycle is real but manageable. Better to create `sim/types/` package for shared interfaces when policy interfaces expand (PR 6-7). |
| S7 | Unified event queue: single heap for all events | ~0 LOC net | **HIGH** | **UNCERTAIN** | **NO (defer)** | Feasible per analysis but the hybrid approach (cluster queue + per-instance queues) already works. Unified queue risks subtle ordering changes and complicates P/D disaggregation. Revisit only if interleaving logic becomes a bottleneck. |

### Summary: Implement S1 + S2 + S3 + S4 (incremental) as PR 5. Defer S5, S6, S7.

---

## 2. Detailed Findings

### 2.1 Unified Event Queue Assessment (Teammate 1: unified-queue-analyst)

**Recommendation: QUALIFIED YES, but DEFER implementation.**

The current architecture uses N+1 queues:
- 1 `ClusterEventQueue` (`sim/cluster/cluster_event.go:26`) — min-heap by (timestamp, priority, seqID)
- N per-instance `EventQueue` (`sim/simulator.go:14`) — min-heap by timestamp only
- Interleaving in `ClusterSimulator.Run()` (`sim/cluster/cluster.go:178-220`) compares earliest cluster event vs earliest instance event

A unified queue IS feasible via:
- `UnifiedEvent` interface: `Timestamp() int64`, `Priority() int`, `Execute(*ClusterSimulator)`
- Instance events wrapped in `InstanceEventWrapper` carrying instance ID, delegating to `sim.Event.Execute(*Simulator)`
- Cluster events: priority 0-2 (Arrival, Admission, Routing); instance events: priority 3
- Determinism preserved: cluster events at time T always before instance events at T

**Why defer:**
1. The interleaving loop (`cluster.go:178-220`) is only ~40 lines and already correct
2. Wrapping every instance event adds allocation overhead on the hot path
3. P/D disaggregation (Phase 4) may need separate prefill/decode queues, which is easier with the current N-queue model
4. Risk of subtle ordering changes that break golden tests in the cluster path
5. The `sim.Simulator.Run()` event loop (`simulator.go:196-212`) MUST remain unchanged—unified queue only applies to cluster path anyway

**LOC impact if implemented:** ~200 lines changed, ~50 lines net simpler. Not worth the risk for 50 lines.

### 2.2 Simplification Survey (Teammate 2: simplification-scout)

#### S1: Constructor Collapse — HIGH impact, LOW risk

Current state:
- `newSimulatorBase()`: 15 params (`sim/simulator.go:117-148`)
- `NewSimulator()`: 17 params (`sim/simulator.go:150-171`)
- `NewSimulatorWithoutWorkload()`: 15 params (`sim/simulator.go:175-182`)
- `NewInstanceSimulator()`: 18 params (`sim/cluster/instance.go:30-73`)
- `NewInstanceSimulatorWithoutWorkload()`: 16 params (`sim/cluster/instance.go:114-127`)

Call sites: `cmd/root.go:203-222`, `cmd/root.go:250`, `sim/cluster/cluster.go:110-127`, 6+ test files

**Proposed `SimConfig` struct:**
```go
type SimConfig struct {
    Horizon                   int64
    Seed                      int64
    TotalKVBlocks             int64
    BlockSizeTokens           int64
    MaxRunningReqs            int64
    MaxScheduledTokens        int64
    LongPrefillTokenThreshold int64
    BetaCoeffs                []float64
    AlphaCoeffs               []float64
    ModelConfig               ModelConfig
    HWConfig                  HardwareCalib
    Model                     string
    GPU                       string
    TP                        int
    Roofline                  bool
    // Optional workload (nil = no workload, caller injects requests)
    GuideLLMConfig            *GuideLLMConfig
    TracesWorkloadFilePath    string
}
```

Single constructor: `NewSimulator(cfg SimConfig) *Simulator`. The workload/no-workload distinction becomes: if both `GuideLLMConfig` and `TracesWorkloadFilePath` are empty, no workload is generated. `DeploymentConfig` can embed or convert to `SimConfig`.

#### S2: Unified CLI Path — HIGH impact, MEDIUM risk

Current: two code paths in `cmd/root.go:201-258`:
- Lines 201-224: `numInstances == 1` → `NewInstanceSimulator` → `Run()` → `SaveResults`
- Lines 225-258: `numInstances > 1` → `NewClusterSimulator` → `Run()` → per-instance + aggregated `SaveResults`

Since `ClusterSimulator` with N=1 passes golden equivalence tests (`sim/cluster/cluster_test.go:TestClusterSimulator_SingleInstance_MatchesGoldenDataset`), we can always use the cluster path.

**Critical guard:** The cluster path adds admission and routing latency (`AdmissionLatency`, `RoutingLatency` in `deployment.go:27-28`). Both default to 0, so with default config, the cluster path injects requests at the same time as the single-instance path. The admission policy defaults to `always-admit` (`cluster.go:65`). With these defaults, the event sequence is: `ClusterArrivalEvent(t=X)` → `AdmissionDecisionEvent(t=X+0)` → `RoutingDecisionEvent(t=X+0)` → `InjectRequestOnline(t=X)`. This produces the same `ArrivalEvent` timing as direct `InjectArrival`.

**RNG parity:** Both paths use `SubsystemWorkload` for request generation. The cluster path calls `c.rng.ForSubsystem(sim.SubsystemWorkload)` (`cluster/workload.go:27`), and the single-instance path uses `sim.rng.ForSubsystem(SubsystemWorkload)` (`simulator.go:187`). With the same seed, these produce identical RNG streams. The cluster path generates requests centrally before the event loop (`cluster.go:160`), matching the single-instance behavior of generating all arrivals upfront (`simulator.go:150-171`).

**Edge case: trace replay.** The cluster CSV loader (`cluster/workload.go:68-126`) mirrors the single-instance loader (`sim/workload_config.go`). Both parse the same format and produce equivalent requests.

#### S3: Duplicated AdmissionPolicy — MEDIUM impact, LOW risk

Two identical interfaces:
- `cluster.AdmissionPolicy` (`sim/cluster/cluster.go:18-20`)
- `policy.AdmissionPolicy` (`sim/policy/admission.go:11-13`)

The comment at `cluster.go:15-17` explains: import cycle prevents `cluster` from importing `policy`. Both packages also duplicate `AlwaysAdmit` and `TokenBucket` implementations.

**Solution:** Move `AdmissionPolicy` interface to `sim/` base package (alongside `Request`, `Event`, etc.). Both `cluster` and `policy` already import `sim`. The interface itself only depends on `*sim.Request` and `int64`—no heavy dependencies.

#### S4: Field Privatization — MEDIUM impact, LOW risk

`Simulator` has 25+ public fields (`sim/simulator.go:72-113`). External access:

| Field | Accessed by `InstanceSimulator` | Accessed by tests | Can privatize now? |
|-------|-------------------------------|-------------------|-------------------|
| `Clock` | Yes (`instance.go:98`) | Yes (golden tests) | NO — too many accesses |
| `Horizon` | Yes (`instance.go:109`) | Yes | NO |
| `EventQueue` | No (via `HasPendingEvents`) | Yes (some tests) | YES with accessor |
| `WaitQ` | Yes (`instance.go:158`) | Yes (golden tests) | NO — too many accesses |
| `KVCache` | Yes (`instance.go:171-177`) | Yes | NO |
| `RunningBatch` | Yes (`instance.go:163-166`) | Yes | NO |
| `Metrics` | Yes (`instance.go:104`) | Yes (golden tests) | NO |
| `MaxRunningReqs` | No | Some tests | YES with accessor |
| `MaxScheduledTokens` | No | Some tests | YES with accessor |
| `BetaCoeffs` | No | No | YES |
| `AlphaCoeffs` | No | No | YES |
| `RunningBatchFeatures` | No | No | YES |
| `LongPrefillTokenThreshold` | No | No | YES |
| `StepEvent` | No | No | YES |
| `StepCount` | No | No | YES |
| `ReqNumComputedTokens` | No | No | YES |
| `PreemptionHappened` | No | No | YES |
| `GuideLLMConfig` | No | Some tests | YES with accessor |
| `Model` | No | No | YES |
| `GPU` | No | No | YES |
| `TP` | No | No | YES |
| `Roofline` | No | No | YES |
| `TracesWorkloadFilePath` | No | No | YES |
| `ModelConfig` | No | No | YES |
| `HWConfig` | No | No | YES |

**Recommendation:** Privatize ~15 fields that are only accessed internally. Keep `Clock`, `Horizon`, `WaitQ`, `KVCache`, `RunningBatch`, `Metrics` public for now (or add accessors on `InstanceSimulator`). This is a mechanical change with no behavioral impact.

### 2.3 Revised Plan (Teammate 3: plan-reviser)

**Reduction: 17 remaining PRs → 13 remaining PRs (PR 5 through PR 17)**

Changes from v2.3:
1. **New PR 5 "Simplification"** — constructor collapse + unified CLI path + field privatization + interface dedup
2. **Merged PR 6 = old PR 5 + PR 7** — Priority + Scheduler (both affect request ordering, natural cohesion)
3. **Merged PR 12 = old PR 17 + PR 18** — Traces + Counterfactual (trace infrastructure without consumers is dead code)
4. **Merged PR 16 = old PR 19 + PR 20** — GEPA + OpenEvolve adapters (both are thin wrappers, ~150 LOC each)
5. **Renumbered** all subsequent PRs

### 2.4 Risk Analysis (Teammate 4: devils-advocate)

#### Risk 1: Golden Test Breakage from Unified CLI Path
**Severity:** BLOCKS adoption if not mitigated
**Evidence:** The cluster path processes events through `ClusterArrivalEvent` → `AdmissionDecisionEvent` → `RoutingDecisionEvent` pipeline (`cluster_event.go:65-117`), while single-instance directly schedules `ArrivalEvent` in the constructor (`simulator.go:159-168`). With zero admission/routing latency, timestamps are identical, but the cluster path adds 3 heap operations per request that don't exist in single-instance.

The critical question: does `InjectRequestOnline` (`instance.go:181-183`) produce the same downstream event sequence as the original workload generation in `NewSimulator`? `InjectRequestOnline` calls `InjectArrivalAt` which schedules an `ArrivalEvent` — this IS the same event chain.

**Mitigation:** Before merging PR 5, run ALL golden tests through the cluster path with N=1. The existing `TestClusterSimulator_SingleInstance_MatchesGoldenDataset` (`cluster_test.go`) already validates this. Additionally, ensure the `SaveResults` output format is identical (single-instance currently uses instance ID `"default"`, cluster path uses `"instance_0"`).

#### Risk 2: RNG Stream Divergence
**Severity:** BLOCKS adoption if not mitigated
**Evidence:** `PartitionedRNG.ForSubsystem(SubsystemWorkload)` (`rng.go:55-72`) uses the master seed directly for backward compatibility. The cluster path uses the same subsystem key (`cluster/workload.go:27`). Both produce the same stream from the same seed. However, constructor collapse could change the ORDER of RNG initialization if `NewPartitionedRNG` is called at a different point.

**Mitigation:** The `SimConfig` constructor must call `NewPartitionedRNG(NewSimulationKey(cfg.Seed))` at exactly the same point as `newSimulatorBase` (`simulator.go:146`). Write a test that generates requests through both paths and asserts identical request IDs, arrival times, and token sequences.

#### Risk 3: Big-Bang PR Risk
**Severity:** COMPLICATES adoption
**Evidence:** Constructor collapse touches 10+ files (2 constructors, 2 `InstanceSimulator` constructors, `cmd/root.go`, `DeploymentConfig`, 6+ test files). Field privatization touches `simulator.go` + all files accessing those fields. Combined with unified CLI path, this could be 400+ lines changed.

**Mitigation:** Split PR 5 into two sub-PRs:
- **PR 5a:** Constructor collapse (`SimConfig` struct + new constructors + update all call sites). ~200 LOC changed. Purely mechanical, easy to review.
- **PR 5b:** Unified CLI path + field privatization + interface dedup. ~200 LOC changed. Behavioral change isolated to CLI.

Or keep as one PR but with atomic commits (one per simplification).

#### Risk 4: Performance Regression from Field Privatization
**Severity:** COSMETIC
**Evidence:** The hot path in `Step()` (`simulator.go:226-340`) accesses `EventQueue`, `WaitQ`, `KVCache`, `RunningBatch`, `Metrics` on every iteration. Method call overhead is ~1-2ns per call vs direct field access. With 10K+ events per simulation, this adds <100μs total.

**Mitigation:** Benchmark before/after. If measurable, keep hot-path fields public or use inline accessor methods that the compiler will optimize.

#### Risk 5: Abstraction Loss for P/D Disaggregation
**Severity:** COSMETIC (does not block any proposal)
**Evidence:** None of the recommended simplifications (S1-S4) affect the control plane / data plane abstraction. The unified queue (S7) is deferred. The cluster event pipeline (`ClusterArrivalEvent` → `AdmissionDecisionEvent` → `RoutingDecisionEvent`) remains intact and extensible for P/D routing.

**Mitigation:** None needed. The proposals preserve all extension points.

---

## 3. Revised Architecture (Post-Simplification)

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (cmd/)                           │
│  Always uses ClusterSimulator (even N=1)                    │
│  SimConfig struct → DeploymentConfig → ClusterSimulator     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   ClusterSimulator                          │
│  ┌─────────────────────────────────────────────────┐        │
│  │           ClusterEventQueue (min-heap)           │        │
│  │  (timestamp, priority, seqID)                    │        │
│  │  ClusterArrivalEvent → AdmissionDecision →       │        │
│  │  RoutingDecision → InjectRequestOnline           │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ InstanceSim[0]  │  │ InstanceSim[N-1]│  ...              │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │                   │
│  │ │ EventQueue  │ │  │ │ EventQueue  │ │  Per-instance     │
│  │ │ (timestamp) │ │  │ │ (timestamp) │ │  heaps            │
│  │ └─────────────┘ │  │ └─────────────┘ │                   │
│  │ sim.Simulator   │  │ sim.Simulator   │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
│  Shared-clock loop: process earliest event across all       │
│  queues. Cluster events before instance events at same T.   │
│                                                             │
│  Policies: AdmissionPolicy, RoutingPolicy (pluggable)       │
│  Snapshots: CachedSnapshotProvider (configurable staleness) │
└─────────────────────────────────────────────────────────────┘
```

**What changes from current:**
- `SimConfig` replaces 17-parameter constructors
- CLI always takes the cluster path (single code path)
- `AdmissionPolicy` interface lives in `sim/` package (no duplication)
- ~15 internal `Simulator` fields privatized

**What stays the same:**
- N+1 event queues (cluster + per-instance)
- Control plane / data plane separation
- `InstanceSnapshot` + `CachedSnapshotProvider` + `ObservabilityConfig`
- All event types and their Execute methods
- `sim.Simulator.Run()` internal event loop (golden test path)

---

## 4. Revised PR Series (v3.0)

### Revision Notes (v3.0)

This revision incorporates the simplification assessment findings:

1. **New PR 5** — Architectural simplification (constructor collapse, unified CLI, field privatization, interface dedup)
2. **Merged old PRs 5+7 → new PR 7** — Priority + Scheduler combined (both affect request ordering)
3. **Merged old PRs 17+18 → new PR 13** — Traces + Counterfactual combined (trace without consumers is dead code)
4. **Merged old PRs 19+20 → new PR 15** — GEPA + OpenEvolve adapters combined (both thin wrappers)
5. **Deferred** unified event queue (S7) — not worth the risk for ~50 LOC savings
6. **Total: 13 remaining PRs** (down from 17)

---

### Phase 2a: Simplification (1 PR)

#### PR 5: Architectural Simplification

| Aspect | Details |
|--------|---------|
| **Title** | `refactor: SimConfig struct, unified CLI path, field privatization, interface dedup` |
| **Motivation** | Reduce codebase complexity before adding policy features; eliminate duplicated code and multi-param constructors |
| **Depends On** | PR 4 |
| **In Scope** | (1) `SimConfig` options struct replacing 17-param constructors, (2) Unified CLI path (always ClusterSimulator), (3) Privatize ~15 internal Simulator fields, (4) Move `AdmissionPolicy` interface to `sim/` base package |
| **Out of Scope** | Unified event queue, workload deduplication, package restructuring |
| **Files Changed** | Modified: `sim/simulator.go` (~80 LOC), `sim/cluster/instance.go` (~40 LOC), `sim/cluster/cluster.go` (~20 LOC), `sim/cluster/deployment.go` (~10 LOC), `cmd/root.go` (~60 LOC), `sim/policy/admission.go` (~10 LOC), test files (~100 LOC). New: none |
| **CLI** | `./simulation_worker run --model X` (same interface, but internally always uses ClusterSimulator) |
| **Tests** | All golden tests must pass. New: `TestUnifiedCLIPath_MatchesGoldenDataset` (verifies N=1 cluster path = single-instance results) |
| **LOC Estimate** | ~320 changed, ~200 net reduction |
| **Architectural Impact** | Major internal cleanup; no behavioral change; all golden tests pass |
| **Behavioral Guarantees** | Bit-exact golden test output preserved; CLI interface unchanged |
| **API Surface Changes** | None (internal refactor) |
| **Risks + Mitigations** | Risk: RNG stream divergence. Mitigation: Test that request generation produces identical tokens through both paths before removing old path. Risk: Big PR. Mitigation: Atomic commits per simplification. |
| **Why Independently Reviewable** | Pure refactor with golden test verification; no new features |

**v2.3 → v3.0 change:** NEW PR. Justified by dropping backward compatibility constraint — this cleanup makes every subsequent PR simpler.

---

### Phase 2b: Policy Interfaces (4 PRs)

#### PR 6: RoutingPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RoutingPolicy with RoundRobin and WeightedScoring` |
| **Motivation** | Enable intelligent request routing across instances |
| **Depends On** | PR 5 |
| **In Scope** | `RoutingPolicy` interface, `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity` templates |
| **Out of Scope** | Priority and scheduling policies (PR 7) |
| **Files Changed** | New: `sim/policy/routing.go` (~200 LOC). Modified: `sim/cluster/cluster.go` (~30 LOC), `sim/cluster/cluster_event.go` (~20 LOC), `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --routing-policy weighted` |
| **Tests** | Unit: each template. Integration: load distribution across instances |
| **LOC Estimate** | ~270 |
| **Architectural Impact** | Replaces hardcoded round-robin in `RoutingDecisionEvent.Execute` with pluggable policy |
| **Behavioral Guarantees** | `round-robin` (default) matches existing behavior |
| **API Surface Changes** | New CLI flags: `--routing-policy`, `--routing-cache-weight`, `--routing-load-weight` |
| **Risks + Mitigations** | Risk: Prefix affinity cache misses. Mitigation: Fallback to least-loaded on miss. |
| **Why Independently Reviewable** | Complete routing feature; default preserves existing behavior |

**v2.3 → v3.0 change:** Now depends on PR 5 (uses `SimConfig`). Otherwise identical to v2.3 PR 6.

---

#### PR 7: PriorityPolicy + InstanceScheduler

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add PriorityPolicy and InstanceScheduler with templates` |
| **Motivation** | Enable request prioritization and per-instance batch scheduling policies |
| **Depends On** | PR 5 |
| **In Scope** | `PriorityPolicy` interface (`ConstantPriority`, `SLOBasedPriority`), `InstanceScheduler` interface (`FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`), `Priority` field on `Request`, `SchedulerContext` |
| **Out of Scope** | Pathological templates (deferred to PR 9) |
| **Files Changed** | New: `sim/policy/priority.go` (~120 LOC), `sim/policy/scheduler.go` (~180 LOC). Modified: `sim/request.go` (~10 LOC), `sim/cluster/instance.go` (~30 LOC), `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --priority-policy slo-based --scheduler priority-fcfs` |
| **Tests** | Unit: each template. Integration: priority ordering in scheduling, batch formation respects scheduler |
| **Parallel With** | PR 6 |
| **LOC Estimate** | ~370 |
| **Architectural Impact** | Adds Priority field to Request; extracts batch formation policy |
| **Behavioral Guarantees** | `constant` priority + `fcfs` scheduler (defaults) match existing behavior |
| **API Surface Changes** | New CLI flags: `--priority-policy`, `--scheduler` |
| **Risks + Mitigations** | Risk: SJF starvation. Mitigation: Document limitation. |
| **Why Independently Reviewable** | Complete priority + scheduling feature; defaults preserve existing behavior |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 5 (Priority) + PR 7 (Scheduler). Both affect request ordering and share the `Priority` field on `Request`. Natural cohesion reduces CLI flag PRs that touch `cmd/root.go`.

---

#### PR 8: PolicyBundle and RouterState

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RouterState and PolicyBundle configuration` |
| **Motivation** | Unified policy configuration via YAML |
| **Depends On** | PR 6, PR 7 |
| **In Scope** | `RouterState`, `TenantState`, `GlobalMetrics`, `PolicyBundle`, YAML loading |
| **Out of Scope** | AutoScale policy (Phase 4) |
| **Files Changed** | New: `sim/cluster/router_state.go` (~150 LOC), `sim/policy/bundle.go` (~100 LOC). Modified: `cmd/root.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --policy-config policies.yaml` |
| **Tests** | Unit: YAML parsing, validation. Integration: policy config overrides CLI flags |
| **LOC Estimate** | ~280 |
| **Behavioral Guarantees** | CLI flags override YAML defaults; missing config uses defaults |
| **API Surface Changes** | New CLI flag: `--policy-config` |
| **Why Independently Reviewable** | Integrates PRs 6-7; provides unified config interface |

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 8. Dependencies updated.

**INTERFACE FREEZE after PR 8** — Policy interfaces are stable. No breaking changes; additive extensions permitted.

---

#### PR 9: RawMetrics, Anomaly Detection, and Pathological Templates

| Aspect | Details |
|--------|---------|
| **Title** | `feat(metrics): Add RawMetrics, anomaly detection, and pathological policy templates` |
| **Motivation** | Enable fitness evaluation and validate anomaly detection |
| **Depends On** | PR 6, PR 7 |
| **In Scope** | `RawMetrics`, `Distribution`, `FitnessFunction`, `EvaluationResult`, anomaly counters, pathological templates (`RejectAll`, `InvertedSLO`, `AlwaysBusiest`, `ReversePriority`) |
| **Out of Scope** | Scale oscillation detection (requires AutoScaler) |
| **Files Changed** | New: `sim/cluster/raw_metrics.go` (~300 LOC). Modified: `sim/policy/*.go` (~100 LOC), `sim/cluster/cluster.go` (~50 LOC), `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --fitness-weights "throughput:0.5,p99_ttft:0.3"` |
| **Tests** | Integration tests: pathological policies trigger expected anomaly counts |
| **LOC Estimate** | ~470 |
| **Why Independently Reviewable** | Complete metrics feature; pathological templates immediately testable |

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 9.

---

### RESEARCH-READY CHECKPOINT (After PR 9)

At this point, BLIS supports:
- Multi-instance cluster simulation with online routing pipeline
- All 4 policy interfaces (admission, priority, routing, scheduler)
- PolicyBundle with YAML configuration
- RawMetrics with fitness evaluation
- Anomaly detection validated with pathological templates

**You can begin policy research experiments here.**

---

### Phase 3: Enhanced Workloads (1 PR, Optional)

#### PR 10: Workload Generator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add multi-tenant workload generator with edge case scenarios` |
| **Motivation** | Enable realistic multi-tenant workloads for policy research |
| **Depends On** | PR 9 |
| **In Scope** | `WorkloadSpec`, `TenantSpec`, arrival patterns (Poisson, Bursty, Diurnal), built-in scenarios |
| **Files Changed** | New: `sim/workload/` (~680 LOC). Modified: `cmd/root.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --workload-spec workload.yaml` |
| **LOC Estimate** | ~700 |
| **Why Independently Reviewable** | Complete workload feature; existing `--workload distribution` preserved |

**v2.3 → v3.0 change:** Unchanged from v2.3 PR 10.

---

### Phase 4: Advanced Features (5 PRs)

#### PR 11: AutoScaler Core + Actuation

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add AutoScaler with ThresholdScaler, provisioning delays, and warmup` |
| **Motivation** | Enable dynamic scaling based on load with realistic timing |
| **Depends On** | PR 9 |
| **In Scope** | `AutoScaler`, `AutoScalePolicy`, `ThresholdScaler`, `Oscillator` (pathological), `WarmupProfile`, `DrainPolicy`, `InstanceState` lifecycle |
| **Out of Scope** | Predictive scaling |
| **Files Changed** | New: `sim/policy/autoscale.go` (~380 LOC). Modified: `sim/cluster/cluster.go` (~70 LOC), `sim/cluster/instance.go` (~50 LOC), `cmd/root.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --autoscaler-enabled --autoscaler-max 8 --provisioning-delay 30s` |
| **Tests** | Unit: threshold logic. Integration: scale up/down with warmup, drain, oscillation detection |
| **Parallel With** | PR 12, PR 13 |
| **LOC Estimate** | ~520 |
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
| **Parallel With** | PR 11, PR 13 |
| **LOC Estimate** | ~360 |
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
| **Parallel With** | PR 11, PR 12 |
| **LOC Estimate** | ~485 |
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
| **LOC Estimate** | ~540 |
| **Why Independently Reviewable** | Complete P/D feature; `--architecture monolithic` (default) preserves existing behavior |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 15 (P/D Architecture) + PR 16 (KV Transfer for P/D). P/D without KV transfer is unrealistic.

---

### Phase 5: Framework Adapters (1 PR, Optional)

#### PR 15: Framework Adapters (GEPA + OpenEvolve)

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add GEPA and OpenEvolve framework adapters` |
| **Motivation** | Enable framework integration for evolutionary policy optimization |
| **Depends On** | PR 13 (traces) |
| **In Scope** | `BLISGEPAAdapter`, `BLISEvaluator`, `gepa-evaluate` command, `openevolve-evaluate` command |
| **Files Changed** | New: `sim/adapter/gepa.go` (~150 LOC), `sim/adapter/openevolve.go` (~150 LOC). Modified: `cmd/root.go` (~80 LOC) |
| **CLI** | `./simulation_worker gepa-evaluate --policy-config p.yaml` and `./simulation_worker openevolve-evaluate --config oe.yaml` |
| **LOC Estimate** | ~380 |
| **Why Independently Reviewable** | Self-contained adapters; BLIS core unchanged |

**v2.3 → v3.0 change:** MERGED from v2.3 PR 19 + PR 20. Both are thin wrappers (~150 LOC each) with no cross-dependencies.

---

### Phase 6: Validation (1 PR)

#### PR 16: Integration Tests and Examples

| Aspect | Details |
|--------|---------|
| **Title** | `test: Add comprehensive integration test suite and examples` |
| **Motivation** | Validate end-to-end workflows and provide usage examples |
| **Depends On** | PR 14 |
| **In Scope** | Integration tests, sample configs, example policies, CI validation |
| **Files Changed** | New: `test/integration/` (~500 LOC), `examples/` (configs) |
| **LOC Estimate** | ~500 |
| **Why Independently Reviewable** | Test-only PR validating cumulative work |

**v2.3 → v3.0 change:** Renumbered from PR 21. Content unchanged.

---

## 5. Dependency DAG

```
PHASE 2a: SIMPLIFICATION (1 PR) ═══════════════════════════════════════════════
                              │
                            PR 5
                     (SimConfig + Unified
                      CLI + Privatization)
                              │
PHASE 2b: POLICY INTERFACES (4 PRs) ═══════════════════════════════════════════
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
            PR 6                            PR 7
          (Routing)                   (Priority+Sched)
              │                               │
              └───────────────┬───────────────┘
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
PHASE 3: ENHANCED WORKLOADS (Optional) ════════════════════════════════════════
                              │
                            PR 10
                    (Workload Generator)
                              │
PHASE 4: ADVANCED FEATURES (5 PRs) ════════════════════════════════════════════
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
   PR 11                    PR 12                    PR 13
 (AutoScale              (Tiered KV              (Traces +
  + Actuation)            + Transfer)             Counterfactual)
                              │
                              ▼
                            PR 14
                        (P/D + KV Xfer)
                              │
PHASE 5: ADAPTERS (Optional) ══════════════════════════════════════════════════
                              │
                            PR 15
                      (GEPA + OpenEvolve)
                              │
PHASE 6: VALIDATION ═══════════════════════════════════════════════════════════
                              │
                            PR 16
                           (Tests)
```

### Parallel Development Matrix

| Gate | Completed PRs | Unlocked for Parallel Development |
|------|---------------|-----------------------------------|
| **G1** | PR 5 (Simplification) | PR 6, PR 7 (2 parallel) |
| **G2** | PR 9 (Research-Ready) | PR 10, PR 11, PR 12, PR 13 (4 parallel tracks) |
| **G3** | PR 13 | PR 15 |

### Timeline Estimate (3-4 developers)

```
Week 3:     PR 5 (Simplification, 1 dev, ~2 days)

Week 3-4:   PR 6-7 (2 parallel devs), then PR 8

Week 4-5:   PR 9
            → 🎯 RESEARCH-READY (~5 weeks, same as v2.3)

Week 6-7:   PR 10 (optional, 1 dev)
            PR 11 (1 dev)
            PR 12 (1 dev)
            PR 13 (1 dev)

Week 7-8:   PR 14 (1 dev, depends on PR 12)

Week 9:     PR 15 (adapters, 1 dev)

Week 10:    PR 16 (tests, 1 dev)

Total: ~10 weeks with 3-4 developers (was ~11)
Research-ready: ~5 weeks (unchanged)
```

---

## 6. Migration Strategy

### Step 1: PR 5 — Architectural Simplification (FIRST)

This PR is the foundation. Do it first because every subsequent PR benefits:

1. **Constructor collapse** — All future PRs use `SimConfig` instead of 17 params
2. **Unified CLI path** — All future PRs only need to add flags to one code path
3. **Interface dedup** — Future policy PRs import `AdmissionPolicy` from one location

**Safe migration order within PR 5:**
1. Introduce `SimConfig` struct alongside existing constructors (additive)
2. Add `NewSimulatorFromConfig(cfg SimConfig)` that calls old constructors internally
3. Update all call sites to use `SimConfig`
4. Remove old constructors
5. Privatize internal fields
6. Unify CLI path (last, since it's the behavioral change)

**Verification gate:** ALL golden tests pass after each step.

### Step 2: Policy PRs (PR 6-9) — Standard Development

These proceed as in v2.3 but with simplified constructor signatures and single CLI path. No special migration concerns.

### Step 3: Phase 4 PRs (PR 11-14) — Parallel Tracks

Three independent tracks after research-ready checkpoint. The unified event queue decision can be revisited here if the interleaving loop becomes a bottleneck for P/D disaggregation.

### Step 4: Phase 5-6 — Adapters and Validation

Standard development, no migration concerns.

---

## 7. Deferred Items

| Item | Reason for Deferral | Revisit Trigger |
|------|--------------------|-----------------|
| Unified event queue (S7) | Low ROI (~50 LOC savings), risk to golden tests, complicates P/D | If interleaving loop becomes bottleneck or P/D needs it |
| Workload deduplication (S5) | Low impact (~60 LOC), risk of subtle divergence | When adding new workload patterns (PR 10) |
| Package restructuring (S6) | Better context after policy interfaces stabilize (post PR 8) | After interface freeze when we know final package shape |

---

## Appendix: v2.3 → v3.0 PR Mapping

| v3.0 PR | v2.3 PR(s) | Change |
|---------|-----------|--------|
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
