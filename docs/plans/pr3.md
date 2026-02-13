# PR 3: ClusterSimulator with DeploymentConfig — Micro-Design Plan

**Date:** 2026-02-12
**PR Title:** `feat(cluster): Add ClusterSimulator with shared-clock multi-instance event loop`
**Macro Plan Reference:** Phase 1, PR 3 in `2026-02-11-macro-implementation-plan-v2.md`
**Status:** Draft (v3 — review fixes)
**Depends On:** PR 1 (PartitionedRNG), PR 2 (InstanceSimulator) — both merged

---

## A) Executive Summary

PR3 introduces `ClusterSimulator`, a multi-instance orchestrator that runs N `InstanceSimulator` replicas behind a **shared clock** with round-robin request dispatch. Events from all instances are processed in global timestamp order. This is the final foundation PR before Phase 2 policy work begins.

**Key design decision: shared-clock event loop.** The cluster drives execution by repeatedly selecting the instance whose next event has the earliest timestamp, processing that single event, and repeating. Each event still calls `Execute(*Simulator)` on its own instance's `Simulator` — no cross-instance state access. The cluster merely controls *which instance advances next*. This is achieved by decomposing `sim.Simulator.Run()` into three primitives (`HasPendingEvents`, `PeekNextEventTime`, `ProcessNextEvent`) without altering any existing behavior.

**Why shared clock matters for PR3:**
1. The macro plan explicitly specifies it (PR3 Motivation: "Run N instances with shared clock", Architectural Impact: "shared clock across instances").
2. The mock study checkpoint after PR3 requires writing hand-coded policies that observe instance state *during* execution — impossible if instances run independently to completion.
3. Phase 2 routing (PR6) needs to observe instance state at the current simulation time. Establishing the shared clock now avoids a disruptive refactor later.

**Scope:**
- Decompose `sim.Simulator.Run()` into step-based primitives (no behavior change)
- Export workload generation helpers from `sim` package (eliminate duplication)
- `ClusterSimulator` struct with shared-clock `Run()` method
- `DeploymentConfig` struct for multi-instance configuration
- Centralized workload generation calling shared helpers
- `--num-instances` CLI flag
- Round-robin request dispatch
- Aggregated metrics via merged `*sim.Metrics`

**Not in scope:** Policy interfaces, P/D disaggregation, cluster-level event types (PriorityAdmission, PriorityRouting — those arrive with PRs 4-6).

**Concurrency assumptions:** Single-goroutine execution throughout. No goroutines in the event loop. Determinism is maintained by construction.

**Intentional simplifications vs macro plan:** The macro plan lists `ReplicaPool` and `EventHeap` as in-scope for PR3. This micro plan intentionally simplifies both: `ReplicaPool` becomes `[]*InstanceSimulator` directly (no separate container type needed at this stage), and `EventHeap` becomes a linear scan across N instances' `PeekNextEventTime()` (functionally equivalent global ordering, O(N) per event, negligible for target range N=2-16). The macro plan's `sim/cluster/event.go` (~150 LOC) is also omitted — cluster-level event types arrive with PRs 4-6 (policy events). These simplifications reduce ~150 LOC of abstractions that would be dead code until Phase 2.

---

## B) Targeted Recon Summary

### B.1 Confirmed Facts — Files This PR Touches

| File | Current State | PR3 Impact |
|------|--------------|------------|
| `sim/simulator.go` | `NewSimulator()` calls workload generation at L147-153. `Run()` (L170-186) is a tight loop: pop event, advance clock, execute, check horizon. `EventQueue` is a min-heap ordered by timestamp. `EventQueue[0]` is guaranteed minimum (Go `container/heap` contract). `Schedule()` pushes events via `heap.Push`. | Decompose `Run()` into `HasPendingEvents`/`PeekNextEventTime`/`ProcessNextEvent`/`Finalize`. Add `newSimulatorBase` + `NewSimulatorWithoutWorkload`. Add `InjectArrival`. |
| `sim/workload_config.go` | `generateLengthGauss` (L98-107) and `generateRandomTokenIDs` (L111-119) are unexported methods on `*Simulator` that call `sim.WorkloadRNG()`. `generateWorkloadDistribution` (L122-186) calls these in a specific RNG order per request. | Export `GenerateLengthGauss` and `GenerateRandomTokenIDs` as standalone functions taking `*rand.Rand`. Refactor existing methods to delegate to them. |
| `sim/event.go` | `ArrivalEvent` has **unexported** fields `time int64` and `Request *Request`. All event types use unexported `time` fields. `Event` interface: `Timestamp() int64`, `Execute(*Simulator)`. | No changes. `InjectArrival` on `*Simulator` (same package) can access unexported fields. |
| `sim/request.go` | `Request` struct — **all fields exported**: `ID`, `InputTokens`, `OutputTokens`, `State`, `ArrivalTime`, etc. | No changes. Cluster code can construct `*sim.Request` directly. |
| `sim/metrics.go` | `Metrics` struct — all fields exported. `SaveResults(instanceID string, horizon int64, totalBlocks int64, startTime time.Time, outputFilePath string)`. Maps keyed by request ID (`RequestTTFTs`, `RequestE2Es`, `RequestITLs`, `RequestSchedulingDelays`, `Requests`). | No changes to struct or methods. Cluster aggregation merges maps into a new `*Metrics`, then calls existing `SaveResults`. |
| `sim/metrics_utils.go` | `RequestMetrics` struct (all exported): `ID`, `ArrivedAt`, `NumPrefillTokens`, `NumDecodeTokens`, `TTFT`, `ITL`, `E2E`, `SchedulingDelay`. `CalculatePercentile` and `CalculateMean` — both divide by 1000 (ticks → ms). | No changes. |
| `sim/rng.go` | `SubsystemWorkload = "workload"` uses master seed directly. `SubsystemRouter = "router"` (unused). `SubsystemInstance(id int)` returns `"instance_N"`. `ForSubsystem` caches results — same name returns same `*rand.Rand` pointer. | No changes. Cluster uses `ForSubsystem("workload")` for workload gen. |
| `sim/cluster/instance.go` | `InstanceSimulator` wraps `*sim.Simulator` (unexported field `sim`). `Run()` delegates to `i.sim.Run()` with run-once panic guard. Accessors: `ID()`, `Clock()`, `Metrics()`, `Horizon()`. | Add `NewInstanceSimulatorWithoutWorkload`, `InjectRequest`, `SetRequestRate`, and delegation methods for step-based execution. |
| `cmd/root.go` | Creates `cluster.NewInstanceSimulator(...)` at L183-202, calls `instance.Run()` at L203, calls `instance.Metrics().SaveResults(...)` at L206. Flags defined in `init()` at L220-264. | Add `--num-instances` flag. Branch: N==1 uses existing path; N>1 uses `ClusterSimulator`. |

### B.2 Current `Run()` Semantics (Critical for Decomposition)

```go
// sim/simulator.go L170-186
func (sim *Simulator) Run() {
    for len(sim.EventQueue) > 0 {
        ev := heap.Pop(&sim.EventQueue).(Event)  // pop min
        sim.Clock = ev.Timestamp()                // advance clock
        ev.Execute(sim)                           // execute (may push new events)
        if sim.Clock > sim.Horizon { break }      // horizon check AFTER execution
    }
    sim.Metrics.SimEndedTime = min(sim.Clock, sim.Horizon)
}
```

**Horizon boundary:** The event that crosses the horizon IS executed. Then the loop breaks. `SimEndedTime` is capped at `Horizon`.

**Empty queue:** If `EventQueue` is empty initially, the loop body never executes. `Clock` stays 0. `SimEndedTime = min(0, Horizon) = 0`.

### B.3 Workload Generation RNG Call Sequence

Per `generateWorkloadDistribution` (L122-186), using `sim.WorkloadRNG()`:

```
1. generateRandomTokenIDs(PrefixTokens)  →  PrefixTokens × Intn(128000)
FOR each request:
  2. generateLengthGauss(PromptTokens...)  →  1 × NormFloat64()
  3. generateRandomTokenIDs(promptLen)     →  promptLen × Intn(128000)
  4. append(prefix, prompt...)              →  no RNG call
  5. generateLengthGauss(OutputTokens...)   →  1 × NormFloat64()
  6. generateRandomTokenIDs(outputLen)      →  outputLen × Intn(128000)
  7. arrival time: currentTime += int64(1 / sim.Metrics.RequestRate)
```

**`sim.Metrics.RequestRate`** is set from `GuideLLMConfig.Rate` at L151 *before* generation starts.

**`append(prefix, prompt...)`** at L143: reuses `prefix` slice backing array. This is the existing behavior and must be replicated exactly (not "fixed").

### B.4 Relevant Invariants

1. **Determinism:** Same seed → identical output. `PartitionedRNG` isolates subsystems.
2. **Request lifecycle:** Every request reaches exactly one terminal state.
3. **Clock monotonicity:** `sim.Clock` never decreases within an instance (events are processed in heap order).
4. **KV conservation:** `allocated + free = total` per instance.
5. **BC-1:** `--num-instances 1` (or omitted) → bit-for-bit identical to current.

---

## C) Expanded Contracts

### C.1 Behavioral Contracts

**BC-1: Single-Instance Backward Compatibility**
- GIVEN `--num-instances 1` or flag omitted (default 1)
- WHEN simulation runs
- THEN output is bit-for-bit identical to current behavior
- MECHANISM: When `numInstances == 1`, bypass `ClusterSimulator` entirely — use existing `InstanceSimulator` code path unchanged.

**BC-2: Multi-Instance Determinism**
- GIVEN `--num-instances N` (N > 1), fixed `--seed S`
- WHEN simulation runs M times
- THEN all M runs produce identical per-instance and aggregated metrics
- MECHANISM: Centralized workload gen via `PartitionedRNG.ForSubsystem("workload")`. Round-robin dispatch (no randomness). Shared-clock processes events in deterministic global order. Ties broken by instance index (lowest first, by iteration order with strict `<`).

**BC-3: Round-Robin Dispatch**
- GIVEN N instances and K requests (indexed 0..K-1)
- WHEN requests are dispatched
- THEN request i goes to instance `i % N`

**BC-4: Globally Unique Request IDs**
- GIVEN N instances
- THEN all request IDs are unique (centrally generated: `"request_0"`, `"request_1"`, ...).

**BC-5: Per-Instance Isolation**
- GIVEN N instances
- THEN each instance's KV cache, WaitQueue, RunningBatch, and Metrics are independent. No cross-instance state sharing during event execution.

**BC-6: Shared-Clock Global Ordering**
- GIVEN N instances with pending events
- WHEN the cluster event loop runs
- THEN events are processed in non-decreasing global timestamp order across all instances
- Ties broken by lowest instance index first (deterministic, by iteration order)

**BC-7: Aggregated Metrics Correctness**
- GIVEN N instances that have completed
- WHEN aggregated metrics are computed
- THEN: `CompletedRequests = Σ per-instance`, `TotalInputTokens = Σ`, `TotalOutputTokens = Σ`, `SimEndedTime = max per-instance`
- Latency percentiles are computed by merging all per-request values across instances into one sorted list, then computing percentiles on the merged data (not averaging per-instance percentiles).

**BC-8: Horizon Enforcement**
- GIVEN horizon H
- THEN no request is generated with `ArrivalTime > H`
- The shared-clock loop breaks after processing the first event with timestamp > H (matching `Simulator.Run()` behavior).

**BC-9: Decomposed Run Equivalence**
- GIVEN any `sim.Simulator` S with workload
- WHEN S is run via the refactored `Run()` (which uses `HasPendingEvents`/`ProcessNextEvent`/`Finalize`)
- THEN output is bit-for-bit identical to the original `Run()` implementation
- This is verified by existing golden dataset tests passing unchanged.

### C.2 "MUST NOT" Contracts

- `ClusterSimulator` MUST NOT modify any instance's `Simulator` fields directly — only via `InstanceSimulator` methods.
- `ProcessNextEvent()` MUST NOT check horizon — that is the caller's responsibility (`Run()` or cluster loop).
- The exported `GenerateLengthGauss`/`GenerateRandomTokenIDs` MUST NOT change the RNG call order relative to the existing unexported methods.
- Workload generation MUST NOT happen between `NewClusterSimulator` and `Run()` — it happens inside `Run()` to keep RNG state pristine.

### C.3 API Contracts

#### `sim.Simulator` — New Methods (sim/simulator.go)

```go
// HasPendingEvents returns true if the EventQueue is non-empty.
func (s *Simulator) HasPendingEvents() bool

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
func (s *Simulator) PeekNextEventTime() int64

// ProcessNextEvent pops the earliest event, advances Clock, and executes it.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
// Does NOT check horizon — caller is responsible.
func (s *Simulator) ProcessNextEvent()

// Finalize sets SimEndedTime and logs completion. Call once after event loop ends.
func (s *Simulator) Finalize()

// InjectArrival schedules an ArrivalEvent for req and registers it in Metrics.Requests.
func (s *Simulator) InjectArrival(req *Request)

// NewSimulatorWithoutWorkload creates a Simulator with no workload generation.
// EventQueue is empty. Caller injects requests via InjectArrival before running.
func NewSimulatorWithoutWorkload(horizon, seed, totalKVBlocks, blockSizeTokens,
    maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64,
    betaCoeffs, alphaCoeffs []float64, modelConfig ModelConfig,
    hwConfig HardwareCalib, model, GPU string, tp int, roofline bool) *Simulator
```

#### `sim` — Exported Workload Helpers (sim/workload_config.go)

```go
// GenerateLengthGauss samples from a clamped Gaussian distribution.
// RNG calls: 1 × NormFloat64() (or 0 if min == max).
func GenerateLengthGauss(rng *rand.Rand, mean, std, min, max int) int

// GenerateRandomTokenIDs creates a slice of random token IDs in [0, MaxTokenID).
// RNG calls: length × Intn(MaxTokenID).
func GenerateRandomTokenIDs(rng *rand.Rand, length int) []int
```

The existing unexported methods (`generateLengthGauss`, `generateRandomTokenIDs`) are refactored to delegate to these, preserving their signatures.

#### `cluster.InstanceSimulator` — New Methods (sim/cluster/instance.go)

```go
func NewInstanceSimulatorWithoutWorkload(id InstanceID, /* same params as
    NewSimulatorWithoutWorkload */) *InstanceSimulator

// InjectRequest delegates to sim.InjectArrival. Panics if called after Run().
func (i *InstanceSimulator) InjectRequest(req *sim.Request)

func (i *InstanceSimulator) SetRequestRate(rate float64)

// Step-based execution delegation (no hasRun guard — cluster manages lifecycle):
func (i *InstanceSimulator) HasPendingEvents() bool
func (i *InstanceSimulator) PeekNextEventTime() int64
func (i *InstanceSimulator) ProcessNextEvent()
func (i *InstanceSimulator) Finalize()
```

#### `cluster.DeploymentConfig` (sim/cluster/deployment.go)

```go
// DeploymentConfig describes a homogeneous cluster deployment.
type DeploymentConfig struct {
    NumInstances              int
    Horizon                   int64
    Seed                      int64
    TotalKVBlocks             int64
    BlockSizeTokens           int64
    MaxRunningReqs            int64
    MaxScheduledTokens        int64
    LongPrefillTokenThreshold int64
    BetaCoeffs                []float64
    AlphaCoeffs               []float64
    ModelConfig               sim.ModelConfig
    HWConfig                  sim.HardwareCalib
    Model                     string
    GPU                       string
    TP                        int
    Roofline                  bool
}
```

#### `cluster.ClusterSimulator` (sim/cluster/cluster.go)

```go
type ClusterSimulator struct {
    config            DeploymentConfig
    instances         []*InstanceSimulator
    rng               *sim.PartitionedRNG
    clock             int64
    workload          *sim.GuideLLMConfig
    tracesPath        string
    hasRun            bool
    aggregatedMetrics *sim.Metrics // populated by Run(), returned by AggregatedMetrics()
}

func NewClusterSimulator(config DeploymentConfig, workload *sim.GuideLLMConfig,
    tracesPath string) *ClusterSimulator
    // Panics if config.NumInstances < 1.

func (c *ClusterSimulator) Run()
func (c *ClusterSimulator) Clock() int64
func (c *ClusterSimulator) Instances() []*InstanceSimulator
func (c *ClusterSimulator) AggregatedMetrics() *sim.Metrics
    // Returns a merged Metrics (reuses existing sim.Metrics type — no new metrics type needed).
```

### C.4 Edge Case Behavior

| Edge Case | Behavior |
|-----------|----------|
| `--num-instances 0` | CLI: `logrus.Fatalf("num-instances must be >= 1")` |
| `--num-instances 1` | Bypass ClusterSimulator, use existing InstanceSimulator path |
| `MaxPrompts < NumInstances` | Some instances get 0 requests; `Finalize` sets `SimEndedTime=0`; `CompletedRequests=0`; no panic |
| `MaxPrompts == 0` | All instances empty; aggregated `CompletedRequests=0` |
| `--workload traces` with N > 1 | CSV requests dispatched round-robin |
| Empty CSV file | 0 requests generated; same as `MaxPrompts == 0` |
| All events on one instance | Other instances finalize immediately; shared clock still correct |

---

## D) Detailed Implementation Plan

### D.1 Modified: `sim/simulator.go`

**D.1a: Decompose `Run()` into primitives.**

```go
func (s *Simulator) HasPendingEvents() bool {
    return len(s.EventQueue) > 0
}

func (s *Simulator) PeekNextEventTime() int64 {
    return s.EventQueue[0].Timestamp() // heap[0] is min; caller must check HasPendingEvents
}

func (s *Simulator) ProcessNextEvent() {
    ev := heap.Pop(&s.EventQueue).(Event)
    s.Clock = ev.Timestamp()
    logrus.Infof("[tick %07d] Executing %T", s.Clock, ev)
    ev.Execute(s)
}

func (s *Simulator) Finalize() {
    s.Metrics.SimEndedTime = min(s.Clock, s.Horizon)
    logrus.Infof("[tick %07d] Simulation ended", s.Clock)
}
```

**Refactor `Run()` to use them (exact behavioral equivalence):**

```go
func (s *Simulator) Run() {
    for s.HasPendingEvents() {
        s.ProcessNextEvent()
        if s.Clock > s.Horizon {
            break
        }
    }
    s.Finalize()
}
```

This is semantically identical to the original L170-186. The horizon-boundary behavior is preserved: the event at the crossing timestamp IS executed, then the loop breaks.

**D.1b: Add `InjectArrival`.**

```go
func (s *Simulator) InjectArrival(req *Request) {
    s.Schedule(&ArrivalEvent{time: req.ArrivalTime, Request: req})
    s.Metrics.Requests[req.ID] = RequestMetrics{
        ID:               req.ID,
        ArrivedAt:        float64(req.ArrivalTime) / 1e6,
        NumPrefillTokens: len(req.InputTokens),
        NumDecodeTokens:  len(req.OutputTokens),
    }
}
```

This consolidates the two operations that `generateWorkloadDistribution` (L163-173) performs: scheduling the ArrivalEvent and registering in Metrics.Requests. Only `generateWorkloadDistribution` is refactored to call `InjectArrival`.

**`generateWorkloadFromCSV` is NOT refactored** to use `InjectArrival`. The CSV generator sets `ArrivedAt` from the raw parsed float (`arrivalFloat`), while `InjectArrival` computes it as `float64(req.ArrivalTime) / 1e6`. Since `req.ArrivalTime = int64(arrivalFloat * 1e6)`, the round-trip loses sub-microsecond precision. Leaving the CSV generator as-is preserves its existing output behavior exactly.

**D.1c: Add `newSimulatorBase` + `NewSimulatorWithoutWorkload`.**

Extract the struct initialization (L118-144) + RNG creation (L146) into unexported `newSimulatorBase`. `NewSimulator` calls it then does workload generation. `NewSimulatorWithoutWorkload` calls it and returns immediately. `NewSimulator` signature is unchanged.

### D.2 Modified: `sim/workload_config.go`

**D.2a: Export workload helpers.**

```go
// GenerateLengthGauss samples a length from a clamped Gaussian distribution.
func GenerateLengthGauss(rng *rand.Rand, mean, std, min, max int) int {
    if min == max {
        return min
    }
    val := rng.NormFloat64()*float64(std) + float64(mean)
    clampedVal := math.Min(float64(max), val)
    clampedVal = math.Max(float64(min), clampedVal)
    return int(math.Round(clampedVal))
}

// GenerateRandomTokenIDs creates a slice of random token IDs in [0, MaxTokenID).
func GenerateRandomTokenIDs(rng *rand.Rand, length int) []int {
    tokens := make([]int, length)
    for i := range tokens {
        tokens[i] = rng.Intn(MaxTokenID)
    }
    return tokens
}
```

**D.2b: Refactor existing methods to delegate.**

```go
func (s *Simulator) generateLengthGauss(mean, std, min, max int) int {
    return GenerateLengthGauss(s.WorkloadRNG(), mean, std, min, max)
}

func (s *Simulator) generateRandomTokenIDs(length int) []int {
    return GenerateRandomTokenIDs(s.WorkloadRNG(), length)
}
```

Same signatures, same behavior, same RNG call order. Existing golden tests verify no regression.

**D.2c: Refactor `generateWorkloadDistribution` to use `InjectArrival`.**

In `generateWorkloadDistribution` — replace L163-173:
```go
// Before:
sim.Schedule(&ArrivalEvent{time: currentTime, Request: req})
sim.Metrics.Requests[reqID] = detail

// After:
sim.InjectArrival(req)
```

**`generateWorkloadFromCSV` is left unchanged.** It sets `ArrivedAt` from the raw CSV float (`arrivalFloat`), which would lose sub-microsecond precision if routed through `InjectArrival` (which computes `float64(int64(arrivalFloat * 1e6)) / 1e6`). Keeping it as-is preserves existing CSV output behavior exactly.

### D.3 Modified: `sim/cluster/instance.go`

**Add methods:**

```go
func NewInstanceSimulatorWithoutWorkload(id InstanceID, horizon, seed, totalKVBlocks,
    blockSizeTokens, maxRunningReqs, maxScheduledTokens,
    longPrefillTokenThreshold int64, betaCoeffs, alphaCoeffs []float64,
    modelConfig sim.ModelConfig, hwConfig sim.HardwareCalib,
    model, GPU string, tp int, roofline bool) *InstanceSimulator {
    s := sim.NewSimulatorWithoutWorkload(horizon, seed, totalKVBlocks,
        blockSizeTokens, maxRunningReqs, maxScheduledTokens,
        longPrefillTokenThreshold, betaCoeffs, alphaCoeffs,
        modelConfig, hwConfig, model, GPU, tp, roofline)
    return &InstanceSimulator{id: id, sim: s}
}

func (i *InstanceSimulator) InjectRequest(req *sim.Request) {
    if i.hasRun {
        panic("InstanceSimulator.InjectRequest() called after Run()")
    }
    i.sim.InjectArrival(req)
}

func (i *InstanceSimulator) SetRequestRate(rate float64) {
    i.sim.Metrics.RequestRate = rate
}

func (i *InstanceSimulator) HasPendingEvents() bool  { return i.sim.HasPendingEvents() }
func (i *InstanceSimulator) PeekNextEventTime() int64 { return i.sim.PeekNextEventTime() }
func (i *InstanceSimulator) ProcessNextEvent()         { i.sim.ProcessNextEvent() }
func (i *InstanceSimulator) Finalize()                 { i.sim.Finalize() }
```

Note: The step-based methods do NOT set `hasRun`. The `hasRun` guard is only for `Run()` and `InjectRequest()`. The cluster manages lifecycle via the step-based API and never calls `Run()` on instances.

### D.4 New: `sim/cluster/deployment.go` (~30 LOC)

`DeploymentConfig` struct as defined in C.3. Pure data, no methods.

### D.5 New: `sim/cluster/cluster.go` (~230 LOC)

**Constructor:**

```go
func NewClusterSimulator(config DeploymentConfig, workload *sim.GuideLLMConfig,
    tracesPath string) *ClusterSimulator {
    if config.NumInstances < 1 {
        panic("ClusterSimulator: NumInstances must be >= 1")
    }
    instances := make([]*InstanceSimulator, config.NumInstances)
    for idx := range instances {
        instances[idx] = NewInstanceSimulatorWithoutWorkload(
            InstanceID(fmt.Sprintf("instance_%d", idx)),
            config.Horizon,
            config.Seed, // all instances share seed; instance-local RNG unused in PR3
            config.TotalKVBlocks,
            config.BlockSizeTokens,
            config.MaxRunningReqs,
            config.MaxScheduledTokens,
            config.LongPrefillTokenThreshold,
            config.BetaCoeffs,
            config.AlphaCoeffs,
            config.ModelConfig,
            config.HWConfig,
            config.Model,
            config.GPU,
            config.TP,
            config.Roofline,
        )
    }
    return &ClusterSimulator{
        config:    config,
        instances: instances,
        rng:       sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed)),
        workload:  workload,
        tracesPath: tracesPath,
    }
}
```

All instances receive the same `config.Seed`. Each instance creates its own `PartitionedRNG` internally, but instance-local RNG is unused in PR3 since workload is generated centrally by the cluster. The cluster's own `PartitionedRNG` (also seeded with `config.Seed`) is used for centralized workload generation via `ForSubsystem("workload")`, which returns master seed directly — producing the same RNG stream as the single-instance path.

**`Run()` — shared-clock event loop:**

```go
func (c *ClusterSimulator) Run() {
    if c.hasRun {
        panic("ClusterSimulator.Run() called more than once")
    }
    c.hasRun = true

    // 1. Generate requests centrally
    requests := c.generateRequests()

    // 2. Dispatch round-robin and set request rate
    for i, req := range requests {
        c.instances[i%c.config.NumInstances].InjectRequest(req)
    }
    if c.workload != nil {
        for _, inst := range c.instances {
            inst.SetRequestRate(c.workload.Rate)
        }
    }

    // 3. Shared-clock event loop
    for {
        // Find instance with earliest pending event (ties: lowest index wins)
        earliestTime := int64(math.MaxInt64)
        earliestIdx := -1
        for idx, inst := range c.instances {
            if inst.HasPendingEvents() {
                t := inst.PeekNextEventTime()
                if t < earliestTime {
                    earliestTime = t
                    earliestIdx = idx
                }
            }
        }
        if earliestIdx == -1 {
            break // all instances drained
        }
        c.clock = earliestTime
        c.instances[earliestIdx].ProcessNextEvent()
        if c.clock > c.config.Horizon {
            break
        }
    }

    // 4. Finalize all instances + aggregate
    for _, inst := range c.instances {
        inst.Finalize()
    }
    c.aggregatedMetrics = c.aggregateMetrics()
}
```

**Tie-breaking:** We iterate `idx` from 0 upward using strict `<`. The first instance at the minimum timestamp wins. This gives deterministic lowest-index-first ordering, matching the macro plan's tie-breaking rule ("lowest InstanceID, lexicographic" — our instance IDs are `"instance_0"`, `"instance_1"`, ..., which sort by index).

**`aggregateMetrics()`:** Merges per-instance `*sim.Metrics` into a single `*sim.Metrics`:

```go
func (c *ClusterSimulator) aggregateMetrics() *sim.Metrics {
    merged := sim.NewMetrics()
    for _, inst := range c.instances {
        m := inst.Metrics()
        merged.CompletedRequests += m.CompletedRequests
        merged.TotalInputTokens += m.TotalInputTokens
        merged.TotalOutputTokens += m.TotalOutputTokens
        merged.TTFTSum += m.TTFTSum
        if m.SimEndedTime > merged.SimEndedTime {
            merged.SimEndedTime = m.SimEndedTime
        }
        // Merge per-request maps (IDs are globally unique — no collisions)
        for k, v := range m.RequestTTFTs { merged.RequestTTFTs[k] = v }
        for k, v := range m.RequestE2Es { merged.RequestE2Es[k] = v }
        for k, v := range m.RequestITLs { merged.RequestITLs[k] = v }
        for k, v := range m.RequestSchedulingDelays { merged.RequestSchedulingDelays[k] = v }
        for k, v := range m.RequestCompletionTimes { merged.RequestCompletionTimes[k] = v }
        for k, v := range m.Requests { merged.Requests[k] = v }
        merged.AllITLs = append(merged.AllITLs, m.AllITLs...)
        merged.RequestStepCounters = append(merged.RequestStepCounters, m.RequestStepCounters...)
    }
    if c.workload != nil {
        merged.RequestRate = c.workload.Rate
    }
    return merged
}
```

By reusing `*sim.Metrics`, we get `SaveResults` for free — no new output type or method needed. Percentile calculations in `SaveResults` sort the merged maps, so the merge-then-compute approach gives correct cross-instance percentiles.

### D.6 New: `sim/cluster/workload.go` (~120 LOC)

Workload generation using the **exported** `sim.GenerateLengthGauss` and `sim.GenerateRandomTokenIDs`:

```go
func (c *ClusterSimulator) generateRequests() []*sim.Request {
    if c.tracesPath != "" && c.workload == nil {
        return c.generateRequestsFromCSV()
    }
    return c.generateRequestsFromDistribution()
}

func (c *ClusterSimulator) generateRequestsFromDistribution() []*sim.Request {
    rng := c.rng.ForSubsystem(sim.SubsystemWorkload)
    cfg := c.workload
    horizon := c.config.Horizon

    var requests []*sim.Request
    currentTime := int64(0)
    reqIdx := 0

    prefix := sim.GenerateRandomTokenIDs(rng, cfg.PrefixTokens) // exported helper

    for currentTime < horizon && reqIdx < cfg.MaxPrompts {
        promptLen := sim.GenerateLengthGauss(rng, cfg.PromptTokens,
            cfg.PromptTokensStdDev, cfg.PromptTokensMin, cfg.PromptTokensMax)
        prompt := sim.GenerateRandomTokenIDs(rng, promptLen)
        input := append(prefix, prompt...) // intentionally matches original behavior

        outputLen := sim.GenerateLengthGauss(rng, cfg.OutputTokens,
            cfg.OutputTokensStdDev, cfg.OutputTokensMin, cfg.OutputTokensMax)
        output := sim.GenerateRandomTokenIDs(rng, outputLen)

        req := &sim.Request{
            ID:          fmt.Sprintf("request_%v", reqIdx), // matches L151 format
            ArrivalTime: currentTime,
            InputTokens: input,
            OutputTokens: output,
            State:       "queued",
        }
        requests = append(requests, req)

        currentTime += int64(1 / cfg.Rate) // matches L176: cfg.Rate == sim.Metrics.RequestRate
        reqIdx++
        if currentTime > horizon {
            break
        }
    }
    return requests
}
```

**No code duplication.** The RNG-consuming logic calls the same `sim.GenerateLengthGauss` and `sim.GenerateRandomTokenIDs` that the original `Simulator` methods now delegate to. The RNG call sequence is identical by construction.

**`cfg.Rate` note:** In `generateWorkloadDistribution`, the arrival interval is `int64(1 / sim.Metrics.RequestRate)`, where `Metrics.RequestRate` was set from `GuideLLMConfig.Rate` at L151. Here we use `cfg.Rate` which IS the same `GuideLLMConfig.Rate` value — set at `cmd/root.go:167` as `rate / 1e6`.

**`generateRequestsFromCSV`:** Mirrors `sim.Simulator.generateWorkloadFromCSV` (workload_config.go:15-93), but returns `[]*sim.Request` instead of pushing events directly:

```go
func (c *ClusterSimulator) generateRequestsFromCSV() []*sim.Request {
    file, err := os.Open(c.tracesPath)
    if err != nil {
        logrus.Fatalf("failed to open csv file: %v", err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    if _, err := reader.Read(); err != nil { // skip header
        logrus.Fatalf("failed to read csv header: %v", err)
    }

    var requests []*sim.Request
    reqIdx := 0
    for {
        record, err := reader.Read()
        if err == io.EOF { break }
        if err != nil {
            logrus.Fatalf("error reading csv at row %d: %v", reqIdx, err)
        }
        arrivalFloat, err := strconv.ParseFloat(record[0], 64)
        if err != nil {
            logrus.Fatalf("invalid arrival time at row %d: %v", reqIdx, err)
        }
        arrivalTime := int64(arrivalFloat * 1e6)
        if arrivalTime > c.config.Horizon { break }

        var inputTokens, outputTokens []int
        if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
            logrus.Fatalf("failed to parse prefill_tokens at row %d: %v", reqIdx, err)
        }
        if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
            logrus.Fatalf("failed to parse decode_tokens at row %d: %v", reqIdx, err)
        }

        requests = append(requests, &sim.Request{
            ID:          fmt.Sprintf("request_%d", reqIdx),
            ArrivalTime: arrivalTime,
            InputTokens: inputTokens,
            OutputTokens: outputTokens,
            State:       "queued",
        })
        reqIdx++
    }
    return requests
}
```

Note: Unlike `generateRequestsFromDistribution`, this does NOT call any RNG functions — CSV provides all token data directly. The `ArrivedAt` field for per-request metrics is computed inside `InjectArrival` (called via `InjectRequest` during dispatch) as `float64(arrivalTime) / 1e6`. This differs from the existing `Simulator.generateWorkloadFromCSV` which uses the raw `arrivalFloat`. The precision difference is sub-microsecond and only affects the JSON reporting field `arrived_at`, not simulation behavior. (The single-instance CSV path is unchanged — only the multi-instance path uses this code.)

### D.7 Modified: `cmd/root.go`

```go
// New package-level var:
var numInstances int

// In init():
runCmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

// In runCmd.Run, after workload config setup:
if numInstances < 1 {
    logrus.Fatalf("num-instances must be >= 1")
}

if numInstances == 1 {
    // Existing single-instance code path (L183-206, UNCHANGED)
    instance := cluster.NewInstanceSimulator(/* existing args */)
    instance.Run()
    instance.Metrics().SaveResults(string(instance.ID()), simulationHorizon,
        totalKVBlocks, startTime, resultsPath)
} else {
    config := cluster.DeploymentConfig{
        NumInstances: numInstances,
        Horizon: simulationHorizon, Seed: seed,
        TotalKVBlocks: totalKVBlocks, BlockSizeTokens: blockSizeTokens,
        MaxRunningReqs: maxRunningReqs, MaxScheduledTokens: maxScheduledTokens,
        LongPrefillTokenThreshold: longPrefillTokenThreshold,
        BetaCoeffs: betaCoeffs, AlphaCoeffs: alphaCoeffs,
        ModelConfig: modelConfig, HWConfig: hwConfig,
        Model: model, GPU: gpu, TP: tensorParallelism, Roofline: roofline,
    }
    cs := cluster.NewClusterSimulator(config, guideLLMConfig, tracesWorkloadFilePath)
    cs.Run()
    // Print per-instance metrics to stdout
    for _, inst := range cs.Instances() {
        inst.Metrics().SaveResults(string(inst.ID()), config.Horizon,
            totalKVBlocks, startTime, "")
    }
    // Save aggregated metrics to file
    cs.AggregatedMetrics().SaveResults("cluster", config.Horizon,
        totalKVBlocks, startTime, resultsPath)
}
```

### D.8 Summary of All File Changes

| File | Action | LOC Delta |
|------|--------|-----------|
| `sim/simulator.go` | Add `newSimulatorBase`, `NewSimulatorWithoutWorkload`, `InjectArrival`, `HasPendingEvents`, `PeekNextEventTime`, `ProcessNextEvent`, `Finalize`. Refactor `Run()`. | +50 |
| `sim/workload_config.go` | Export `GenerateLengthGauss`, `GenerateRandomTokenIDs`. Refactor existing methods to delegate. Refactor `generateWorkloadDistribution` to use `InjectArrival` (CSV generator unchanged — see D.2c). | +15 (net, replacing inlined code with calls) |
| `sim/cluster/instance.go` | Add constructor, `InjectRequest`, `SetRequestRate`, 4 delegation methods | +40 |
| `sim/cluster/deployment.go` | **New:** `DeploymentConfig` struct | ~30 |
| `sim/cluster/cluster.go` | **New:** `ClusterSimulator`, `NewClusterSimulator`, shared-clock `Run()`, `aggregateMetrics()` | ~230 |
| `sim/cluster/workload.go` | **New:** `generateRequestsFromDistribution`, `generateRequestsFromCSV` | ~120 |
| `cmd/root.go` | Add `--num-instances` flag, branch on N==1 vs N>1 | +40 |
| **Total** | | ~525 |

### D.9 Dead Code Check

Every new symbol is on a call path from the CLI:
- `--num-instances > 1` → `NewClusterSimulator` → `NewInstanceSimulatorWithoutWorkload` → `NewSimulatorWithoutWorkload` → `newSimulatorBase`
- `ClusterSimulator.Run()` → `generateRequests` (calls `GenerateLengthGauss`, `GenerateRandomTokenIDs`) → `InjectRequest` → `InjectArrival` → shared-clock loop (calls `HasPendingEvents`, `PeekNextEventTime`, `ProcessNextEvent`, `Finalize`) → `aggregateMetrics` → `AggregatedMetrics`
- `--num-instances 1` → existing path. `Run()` now calls `HasPendingEvents`/`ProcessNextEvent`/`Finalize` internally.
- `GenerateLengthGauss`/`GenerateRandomTokenIDs` called by BOTH existing `Simulator` methods (via delegation) AND cluster `workload.go`.

No dead code. No unused types. No orphaned methods.

### D.10 No Speculative Scaffolding

- No policy interfaces.
- No `InstanceSnapshot` or `RouterState`.
- No cluster-level event types (PriorityAdmission, etc. — those come with PRs 4-6).
- No `PolicyBundle`.
- No autoscaling hooks.
- No `ClusterMetrics` type — aggregation reuses `*sim.Metrics`.

---

## E) CLI Exercise Proof

### E.1 BC-1: Single-Instance Backward Compatibility

```bash
# Without flag (existing behavior):
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 10 --max-prompts 50 --results-path /tmp/a.json

# With --num-instances 1 (same path):
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 10 --max-prompts 50 --num-instances 1 --results-path /tmp/b.json

diff /tmp/a.json /tmp/b.json  # identical (excluding wall-clock timestamps)
```

### E.2 Multi-Instance

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 20 --max-prompts 100 --num-instances 4 \
  --results-path /tmp/cluster.json
```

### E.3 Determinism

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 20 --max-prompts 100 --num-instances 4 --results-path /tmp/r1.json
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 20 --max-prompts 100 --num-instances 4 --results-path /tmp/r2.json
diff /tmp/r1.json /tmp/r2.json  # identical
```

### E.4 More Instances Than Requests

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 42 --rate 10 --max-prompts 2 --num-instances 4
# instances 2,3 get 0 requests, complete with 0 metrics, no panic
```

### E.5 Traces + Multi-Instance

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --workload traces --workload-traces-filepath traces.csv \
  --num-instances 2 --results-path /tmp/traces_cluster.json
```

### E.6 Invalid Input

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 0
# Fatal: "num-instances must be >= 1"
```

---

## F) Test Matrix

All tests are **behavioral** — they verify observable outcomes (metrics, completion counts, determinism) via actual function calls. No tests inspect internal queue state or compare struct fields.

### F.1 Regression Tests — `sim/simulator_test.go` / `sim/cluster/instance_test.go`

These are EXISTING tests that MUST CONTINUE TO PASS, unchanged. They validate BC-9 (decomposed Run equivalence):

| Existing Test | What It Proves |
|---------------|----------------|
| `TestSimulator_GoldenDataset` | Refactored `Run()` using `ProcessNextEvent`/`Finalize` produces identical output |
| `TestInstanceSimulator_GoldenDataset_Equivalence` | `InstanceSimulator.Run()` still matches golden values |
| `TestInstanceSimulator_Determinism` | Same seed still produces identical output |
| All other existing tests in `sim/` and `sim/cluster/` | No regressions from exported helpers or `InjectArrival` refactor |

### F.2 Unit Tests — `sim/simulator_test.go` (new)

| Test | Contract | Description |
|------|----------|-------------|
| `TestNewSimulatorWithoutWorkload_RunsEmpty` | C.4 | GIVEN `NewSimulatorWithoutWorkload()` WHEN `Run()` called without injecting requests THEN `CompletedRequests == 0`, `SimEndedTime == 0`, no panic |
| `TestInjectArrival_RequestCompletes` | C.3 | GIVEN `NewSimulatorWithoutWorkload()` WHEN one request is injected via `InjectArrival` and `Run()` called THEN `CompletedRequests == 1` and request appears in `Metrics.Requests` |
| `TestInjectArrival_MultipleRequests` | C.3 | GIVEN `NewSimulatorWithoutWorkload()` WHEN 10 requests injected at staggered times and `Run()` called THEN `CompletedRequests == 10` |

### F.3 Unit Tests — `sim/cluster/cluster_test.go` (new)

| Test | Contract | Description |
|------|----------|-------------|
| `TestClusterSimulator_SingleInstance_GoldenEquivalence` | BC-7, BC-9 | GIVEN each golden dataset test case configured as `NumInstances=1` via `ClusterSimulator` WHEN `Run()` called THEN `CompletedRequests`, `TotalInputTokens`, `TotalOutputTokens` match golden values exactly. This proves: workload gen parity (shared exported functions), shared-clock correctness (single instance = same as `Run()`), and aggregation correctness (N=1 = identity). |
| `TestClusterSimulator_MultiInstance_Determinism` | BC-2 | GIVEN N=4, seed=42, 100 requests WHEN run twice THEN per-instance `CompletedRequests` and aggregated `CompletedRequests` are identical across both runs |
| `TestClusterSimulator_MultiInstance_AllComplete` | BC-3, BC-5 | GIVEN N=4, 100 requests WHEN run THEN aggregated `CompletedRequests == 100` AND each instance's `CompletedRequests > 0` |
| `TestClusterSimulator_RoundRobin_EvenDistribution` | BC-3 | GIVEN N=3, 9 requests WHEN run THEN each instance has `CompletedRequests == 3` |
| `TestClusterSimulator_RoundRobin_UnevenDistribution` | BC-3 | GIVEN N=3, 10 requests WHEN run THEN instance 0 has `CompletedRequests == 4`, instances 1,2 have `CompletedRequests == 3` |
| `TestClusterSimulator_ZeroRequestInstances` | C.4 | GIVEN N=4, 2 requests WHEN run THEN instances 0,1 have `CompletedRequests == 1`, instances 2,3 have `CompletedRequests == 0`, no panic |
| `TestClusterSimulator_AggregatedMetrics_Correctness` | BC-7 | GIVEN N=2 WHEN run THEN `aggregated.CompletedRequests == sum(per-instance)` AND `aggregated.TotalInputTokens == sum(per-instance)` AND `aggregated.SimEndedTime == max(per-instance)` |
| `TestClusterSimulator_SharedClock_MonotonicGlobal` | BC-6 | GIVEN N=2 WHEN run THEN `cluster.Clock()` is >= every instance's `Clock()`. (This verifies the cluster clock tracked the latest event processed.) |
| `TestClusterSimulator_RunOnce_Panics` | C.3 | GIVEN cluster has `Run()` WHEN `Run()` called again THEN panic |
| `TestNewClusterSimulator_ZeroInstances_Panics` | C.4 | GIVEN `NumInstances=0` WHEN `NewClusterSimulator()` THEN panic |

### F.4 Unit Tests — Workload Parity (new, in `sim/cluster/workload_test.go`)

| Test | Contract | Description |
|------|----------|-------------|
| `TestClusterWorkloadGen_MatchesSimulator` | D.6 | GIVEN same seed and workload config WHEN `sim.NewSimulator` generates workload (inspect its `Metrics.Requests` map) AND `ClusterSimulator.generateRequestsFromDistribution` generates requests THEN count matches, and for each request index: `ArrivalTime`, `len(InputTokens)`, `len(OutputTokens)` match. Uses actual `sim.NewSimulator` output as reference — no duplicated logic in the test. |
| `TestClusterWorkloadGen_Determinism` | BC-2 | GIVEN same seed WHEN called twice THEN request lists are identical (count, IDs, token lengths) |

### F.5 Failure Mode Tests

| Test | Contract | Description |
|------|----------|-------------|
| `TestInstanceSimulator_InjectAfterRun_Panics` | C.3 | GIVEN instance has `Run()` WHEN `InjectRequest()` called THEN panic with expected message |
| `TestClusterSimulator_RunOnce_Panics` | (above) | |
| `TestNewClusterSimulator_ZeroInstances_Panics` | (above) | |

### F.6 Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `BenchmarkClusterSimulator_1K_1Instance` | 1 instance, 1000 requests (baseline, matches macro plan naming) |
| `BenchmarkClusterSimulator_10K_4Instances` | 4 instances, 10000 requests (matches macro plan naming) |
| `BenchmarkClusterSimulator_1K_10Instances` | 10 instances, 1000 requests (scaling test) |

### F.7 Lint

```bash
golangci-lint run ./...
```

All new code passes with zero issues.

---

## G) Risk Analysis

### G.1 Run() Decomposition Regression (HIGH)

**Risk:** Refactoring `Run()` to use `ProcessNextEvent`/`Finalize` subtly changes behavior.

**Prevention:** The refactored `Run()` is a mechanical extraction — same operations in same order. The existing golden dataset tests (`TestSimulator_GoldenDataset`, `TestInstanceSimulator_GoldenDataset_Equivalence`) verify bit-for-bit identical output. Any deviation fails CI immediately.

### G.2 Workload Generation Parity (HIGH)

**Risk:** Cluster workload generation diverges from `Simulator.generateWorkloadDistribution` despite calling the same exported helpers.

**Prevention:** Both code paths call `sim.GenerateLengthGauss` and `sim.GenerateRandomTokenIDs` with the same `*rand.Rand` (from `ForSubsystem("workload")` with the same seed). `TestClusterWorkloadGen_MatchesSimulator` creates an actual `sim.NewSimulator`, extracts its generated requests via `Metrics.Requests`, and compares against the cluster's output field-by-field. Any divergence in RNG call order fails this test.

### G.3 Shared-Clock Overhead (MEDIUM)

**Risk:** The per-event scan across N instances (`O(N)` per event) slows simulation.

**Impact:** For small N (2-16 instances, the target range), the linear scan is negligible compared to event execution. `BenchmarkClusterSimulator` measures this. If N grows large, the scan can be replaced with a heap-of-heaps, but that's a future optimization.

### G.4 Backward Compatibility (HIGH)

**Risk:** Changes to `sim/simulator.go` or `sim/workload_config.go` break existing behavior.

**Prevention:** BC-1 path (N==1) uses the existing `InstanceSimulator` code path completely unchanged. All changes to `sim/` are additive (new exported functions) or mechanical refactors (delegation, `InjectArrival` consolidation). Existing golden tests catch any regression.

### G.5 Metrics Aggregation: Map Iteration Order (MEDIUM)

**Risk:** `aggregateMetrics` iterates over per-instance `Metrics` maps (Go maps have non-deterministic iteration order). Could this affect percentile computation?

**Prevention:** Not a problem. The map *values* are copied into the merged `Metrics` maps. `SaveResults` then extracts values, sorts them, and computes percentiles. The sort step eliminates any iteration-order dependency. Global uniqueness of request IDs (BC-4) ensures no key collisions during merge.

### G.6 Instance Seed Identity (LOW, known limitation)

**Risk:** All instances are created with the same seed. Their internal `PartitionedRNG` instances are identical. If future PRs add instance-local randomness (e.g., scheduler jitter), all instances would draw identical streams.

**Impact:** Acceptable for PR3 (no instance-local RNG usage). Future PRs should use `SubsystemInstance(idx)` for per-instance randomness. Noted as a known limitation.

### G.7 Observability Gap (LOW)

**Risk:** Cluster-mode JSON output uses existing `MetricsOutput` schema. No `num_instances` field in output.

**Impact:** The `instance_id` field distinguishes per-instance (`"instance_0"`) from aggregated (`"cluster"`) output. Adding a dedicated `num_instances` field can be done when the output format is revisited (PR 9, RawMetrics). Not a PR3 blocker.

---

## H) Sanity Checklist

- [x] **No unnecessary abstractions.** No interfaces, no `ClusterMetrics` type (reuses `*sim.Metrics`), no cluster event types.
- [x] **No feature creep.** No policies, no autoscaling, no P/D, no `InstanceSnapshot`, no `RouterState`.
- [x] **No unexercised flags.** `--num-instances` exercised by N==1 (passthrough) and N>1 (cluster).
- [x] **No partial implementations.** ClusterSimulator delivers complete shared-clock multi-instance simulation with aggregated metrics.
- [x] **No breaking changes.** `NewSimulator` signature unchanged. All existing CLI flags work. BC-1 path is existing code.
- [x] **No hidden global state.** All state is struct-local. No new package-level vars beyond `numInstances` in `cmd/root.go`.
- [x] **Passes golangci-lint.** All exported types/functions have doc comments. No unused vars. No shadowed imports.
- [x] **No dead code.** Every symbol is on a CLI-exercisable call path (verified in D.9).
- [x] **No speculative scaffolding.** No TODOs, no placeholder methods, no empty interfaces.
- [x] **No code duplication.** Workload helpers are exported once in `sim/`, called from both `sim/` methods and `sim/cluster/` generator. `InjectArrival` consolidates event+metrics registration for distribution workloads. CSV generator left unchanged to preserve `ArrivedAt` precision.
- [x] **Shared clock implemented.** Macro plan requirement satisfied: events processed in global timestamp order across instances.
- [x] **Macro plan deviations documented.** `ReplicaPool` simplified to `[]*InstanceSimulator`, `EventHeap` replaced with linear scan, `sim/cluster/event.go` deferred to PRs 4-6. All noted in Executive Summary.
