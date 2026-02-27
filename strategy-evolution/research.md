# Research Background: Optimal Joint Scheduling + KV Cache Strategy for LLM Inference

## 1. Problem Statement

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation.

The current default configuration (llm-d parity) uses:
- **Routing**: Weighted scoring with `prefix-affinity:3, queue-depth:2, kv-utilization:2`
- **Scheduling**: FCFS (first-come-first-served) at each instance
- **KV Cache**: Single-tier GPU (tiered GPU+CPU available but not default)
- **Priority**: Constant (all requests equal)
- **Admission**: Always-admit
- **Batch Formation**: vLLM-style FCFS with chunked prefill

**Core problem**: Current scheduling and KV caching policies are designed independently -- the router does not know what the scheduler will do with a request, the scheduler does not know why the router sent a request to this instance, and KV eviction is oblivious to both routing intent and scheduling priorities. We need to discover **jointly optimized strategies** where routing, scheduling, KV cache management, and priority assignment work together as a coherent system.

**Target workload**: Mixed production with prefix-heavy multi-turn sessions, bursty arrivals (gamma CV=2-3), mixed SLO classes (critical/standard/sheddable), heavy-tailed token distributions (ParetoLogNormal), near-saturation load (~80% capacity).

**Constraints**: (1) KV offloading must always be enabled, (2) must map to BLIS extension points, (3) defensible to vLLM/llm-d/distributed inference experts, (4) must beat baselines on throughput, TTFT P99, E2E P99, and SLO-class fairness.

**Success criteria**: >15% TTFT P99 improvement, >5% throughput improvement, no SLO class starvation, robust across 3+ seeds, mechanistically explained, parameter sensitivity characterized.

---

## 2. BLIS Architecture

### 2.1 Overall Data Flow

```
Request Arrival -> Admission -> Routing -> WaitQueue -> Batch Formation -> Step Execution -> Completion
                                               |              |
                                         KV Allocation   Latency Estimation (alpha/beta or roofline)
```

The simulator uses a discrete-event architecture with a min-heap event queue. The cluster simulator maintains a single global clock shared across all instances. Cluster-level events at time T are processed before instance events at time T.

### 2.2 Shared-Clock Event Loop

The cluster event pipeline processes events in this order:
1. **ClusterArrivalEvent** -- request enters system
2. **AdmissionDecisionEvent** -- admission policy evaluates (admit/reject)
3. **RoutingDecisionEvent** -- routing policy selects target instance, increments PendingRequests
4. **QueuedEvent** -- request enters instance wait queue, decrements PendingRequests
5. **StepEvent** -- 4-phase step cycle: scheduleBatch -> executeBatchStep -> processCompletions -> scheduleNextStep
6. **RequestLeftEvent** -- request completes, metrics recorded

### 2.3 Instance Isolation

Each instance is fully independent: own event queue, wait queue, running batch, KV cache, latency model, scheduling and priority policies. Instances share the global clock but have no direct communication. All inter-instance coordination happens through the routing layer via routing snapshots.

### 2.4 Step Phases (the per-instance engine)

**Phase 1 (scheduleBatch)**: Assign priority scores to all queued requests via priority policy, reorder wait queue via scheduling policy, invoke batch formation to select which requests enter the running batch.

**Phase 2 (executeBatchStep)**: Compute step time via latency model. For each request: advance progress through input tokens (respecting chunked prefill) or decode one output token. Record TTFT at prefill-to-decode boundary.

**Phase 3 (processCompletions)**: Identify completed requests, release KV blocks, record E2E metrics.

**Phase 4 (scheduleNextStep)**: If running batch or wait queue non-empty, schedule next StepEvent at `now + stepTime` (work-conserving INV-8).

---

## 3. All Existing Policies and Interfaces

### 3.1 Admission Policies (`sim/admission.go`)

Interface: `AdmissionPolicy.Admit(req *Request, state *RouterState) (admitted bool, reason string)`

| Policy | Behavior |
|--------|----------|
| `always-admit` | Accept all requests (default) |
| `token-bucket` | Rate-limiting: consumes tokens equal to input token count, refills at constant rate |
| `reject-all` | Reject all (pathological) |

**Key insight**: The `TokenBucket` has access to `*RouterState` (cluster-wide snapshots + clock) but currently only uses `state.Clock` for refill timing and `req.InputTokens` for cost. It does NOT use any instance-level information from snapshots. This is an unexploited opportunity -- admission could consider cluster-wide load or KV pressure.

### 3.2 Routing Policies (`sim/routing.go`)

Interface: `RoutingPolicy.Route(req *Request, state *RouterState) RoutingDecision`

The `RoutingDecision` includes `TargetInstance`, `Reason`, `Scores` (per-instance), and critically a `Priority` hint field (currently zero-valued by all policies) that sets `req.Priority` for first-step scheduling. This is an **unexploited cross-layer information path** from router to scheduler.

| Policy | Selection Rule | Notes |
|--------|---------------|-------|
| `round-robin` | Cyclic assignment | Perfect distribution, no load awareness |
| `least-loaded` | Min `EffectiveLoad()` | Positional tie-breaking bias (H4) |
| `weighted` | Composable scorer pipeline | Default: `pa:3,qd:2,kv:2` |
| `prefix-affinity` | Prefix hash mapping + LL fallback | Unbounded prefix map growth |
| `always-busiest` | Max load (pathological) | Creates single-instance bottleneck |

The `WeightedScoring` pipeline:
1. Each scorer produces per-instance scores in [0,1]
2. Scores are clamped to [0,1]
3. Multiplied by normalized weights (sum to 1.0)
4. Summed across scorers per instance
5. Argmax selects instance (ties broken by first occurrence in snapshot order)
6. Observers notified after selection (stateful scorers update state)

### 3.3 Scorers (`sim/routing_scorers.go`, `sim/routing_prefix_scorer.go`)

Interface: `scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64`

| Scorer | Formula | Signal Freshness | Stateful? |
|--------|---------|-----------------|-----------|
| `prefix-affinity` | proportion of request block hashes found in instance's router-side cache | Tier 1 (always fresh, observer updates synchronously) | Yes (LRU cache per instance, 10K blocks) |
| `queue-depth` | min-max normalization of EffectiveLoad (lower load = higher score) | Tier 1+2 (QD+BatchSize immediate, PendingRequests synchronous) | No |
| `kv-utilization` | `1 - KVUtilization` | Tier 3 (Periodic, stale across batch steps) | No |
| `load-balance` | `1/(1+EffectiveLoad)` | Tier 1+2 (same as queue-depth) | No |

**RoutingSnapshot fields available** (from `sim/routing.go`):
- `ID string` -- instance identifier
- `QueueDepth int` -- number of requests in wait queue
- `BatchSize int` -- number of requests in running batch
- `KVUtilization float64` -- fraction of KV blocks used
- `FreeKVBlocks int64` -- absolute free block count
- `CacheHitRate float64` -- cache hit rate
- `PendingRequests int` -- requests routed but not yet enqueued

### 3.4 Scheduling Policies (`sim/scheduler.go`)

Interface: `InstanceScheduler.OrderQueue(requests []*Request, clock int64)`

| Policy | Ordering | Notes |
|--------|----------|-------|
| `fcfs` | No reordering (arrival order) | Default, fair |
| `priority-fcfs` | Priority descending, then arrival ascending | Requires non-constant PriorityPolicy |
| `sjf` | Input token count ascending | Shortest-job-first; can starve long requests |
| `reverse-priority` | Priority ascending (pathological) | Starves high-priority requests |

**Critical gap**: All schedulers operate on `[]*Request` but do not have access to KV cache state. A scheduler cannot know which request would benefit most from currently-cached blocks. Similarly, schedulers do not know the routing reason -- whether the request was sent here for cache affinity or load balancing.

### 3.5 Priority Policies (`sim/priority.go`)

Interface: `PriorityPolicy.Compute(req *Request, clock int64) float64`

| Policy | Formula | Notes |
|--------|---------|-------|
| `constant` | Fixed score (default 0.0) | No differentiation |
| `slo-based` | `BaseScore + AgeWeight * age` | Older requests get higher priority. AgeWeight default = 1e-6. Does NOT currently use `req.SLOClass` despite the name. |
| `inverted-slo` | `BaseScore - AgeWeight * age` (pathological) | Starves older requests |

**Critical gap**: `SLOBasedPriority` does NOT use SLO class metadata. The `req.SLOClass` field (`"critical"`, `"standard"`, `"sheddable"`, `"batch"`, `"background"`) is carried through the pipeline but never read by any priority policy. This is a major unexploited opportunity for SLO-aware differentiation.

### 3.6 Batch Formation (`sim/batch_formation.go`)

Interface: `BatchFormation.FormBatch(ctx BatchContext) BatchResult`

Only one implementation exists: `VLLMBatchFormation`.

**Phase 1 (continuing requests)**: Process requests already in the running batch. Apply chunked prefill limits. Allocate token budget for decode tokens. If KV allocation fails, preempt from batch tail.

**Phase 2 (new requests)**: Dequeue from wait queue. Compute cached prefix blocks. Allocate KV blocks. Stop when max batch size, allocation failure, token budget exhausted, or preemption occurred.

**Key constraint**: New request dequeuing stops entirely if any preemption occurred in Phase 1. This is a conservative safety measure but means a single large request's KV allocation failure halts all new admissions for that step.

### 3.7 KV Cache (`sim/kv_store.go`, `sim/kv/cache.go`, `sim/kv/tiered.go`)

Interface: `KVStore` (11 methods)

**Single-tier** (`KVCacheState`): GPU-only with block allocation, prefix caching (hierarchical block hashing), LRU eviction, reference counting, transactional allocation with rollback.

**Tiered** (`TieredKVCache`): Composes GPU `KVCacheState` with a simple CPU tier.
- **Offload trigger**: When GPU utilization exceeds `KVOffloadThreshold` (default 0.9), free blocks with cached content are copied to CPU.
- **Reload**: On GPU allocation failure, blocks are reloaded from CPU to GPU hash table with transfer latency: `baseLatency + ceil(blockSize/bandwidth)` per block.
- **Thrashing detection**: Offloaded and reloaded within 1000 ticks increments thrashing counter.

**Key observation for strategy design**: The tiered cache's offload decision is currently a fixed threshold -- it does not consider which blocks to offload based on likely future demand. A smarter eviction policy that considers routing intent (which sessions are active on this instance) could significantly reduce thrashing.

---

## 4. Extension Points

From `docs/extension-recipes.md`, there are six categories of extensions:

### 4.1 Adding a New Scorer (2 touch points)
1. Implement `scorerFunc` in `sim/routing_scorers.go` (or separate file for stateful)
2. Register in `validScorerNames` map + `newScorerWithObserver` factory switch

Stateful scorers return an `observerFunc` that updates state after each routing decision. This is the mechanism by which the router can maintain cross-request state.

### 4.2 Adding a New Scheduling Policy (2 touch points)
1. Implement `InstanceScheduler` interface in `sim/scheduler.go`
2. Register in `NewScheduler` factory + `validSchedulers` map in `sim/bundle.go`

### 4.3 Adding a New Priority Policy (2 touch points)
1. Implement `PriorityPolicy` interface in `sim/priority.go`
2. Register in `NewPriorityPolicy` factory + `validPriorityPolicies` map in `sim/bundle.go`

### 4.4 Adding a New Admission Policy (2 touch points)
1. Implement `AdmissionPolicy` interface in `sim/admission.go`
2. Register in `NewAdmissionPolicy` factory + `validAdmissionPolicies` map in `sim/bundle.go`

### 4.5 Adding a New Batch Formation Strategy (2+ touch points)
Currently only `VLLMBatchFormation` exists. Adding a second requires: implementation, `BatchFormation` field in config, CLI flag, validation, and selection logic.

### 4.6 Adding a New KV Tier (4+ touch points)
Implement `KVStore` interface, update factory, add CLI flags, wire into `KVCacheConfig`.

---

## 5. Prior Hypothesis Findings

### 5.1 H4: Round-Robin vs Least-Loaded
- **Mean metrics equivalent** at low load (<5% diff)
- **LL TTFT P99 is 12-21% worse** at low load due to positional tie-breaking bias (`routing.go:113-114` initializes with `snapshots[0]`, strict `<` at line 118)
- **RR and LL produce identical results** under constant-token overload due to PendingRequests tracking creating virtual round-robin
- **Routing differentiation requires workload heterogeneity** -- with constant tokens, no policy can outperform cyclic assignment

**Implication for strategy design**: Any novel routing policy must be tested with variable-token workloads. Tie-breaking in argmax matters at the tail.

### 5.2 H7: Horizontal Scaling
- **7.4x super-linear TTFT P99 improvement** (4->8 instances at rate=500) due to non-linear queue growth rate reduction: `excess = lambda/k - mu` drops 92.5% going 4->8 instances
- **E2E insensitive** (1.06x) because decode time dominates
- **Effect vanishes at sub-saturation** (1.064x at rate=100)

**Implication for strategy design**: Near the saturation boundary, even small improvements in effective capacity yield outsized TTFT benefits. A strategy that shifts the effective saturation point even slightly lower should produce super-linear TTFT P99 improvement.

### 5.3 H20: Heavy-Tailed Input Distributions
- **ParetoLogNormal produces FEWER preemptions** than Gaussian at same mean (0.71x), **refuting** the hypothesis that heavy tails cause more KV pressure
- **Distribution MEDIAN drives KV pressure**, not mean or tail. Pareto component (70%) median ~79 tokens (5 blocks) vs Gaussian median ~256 (16 blocks) -> 3.2x occupancy ratio
- Short requests cycle fast, creating "breathing room"
- TTFT P99 at sub-saturation is 2.9x higher for ParetoLN = intrinsic prefill cost, not HOL blocking

**Implication for strategy design**: For KV capacity planning, focus on the median token count of the request distribution, not the mean or tail. Strategies that manage KV pressure should consider the occupancy profile (block-seconds), not just block count.

### 5.4 H21: Extreme Scorer Weights
- **Single-scorer prefix-affinity is degenerate**: cold-start ties pile ALL requests onto instance_0 (Jain FI = 0.250)
- **Even 1/101 (0.99%) queue-depth weight prevents this** -- the tiebreaker is binary (present/absent), not proportional
- **Weight magnitude is irrelevant**: 100:1 and 100000:1 are byte-identical (normalization makes only ratios matter)
- **Observer-seeded positive feedback loop**: first routing decision seeds cache on instance_0, subsequent requests see cache hit on instance_0, reinforcing the cycle

**Implication for strategy design**: Any stateful scorer MUST be paired with a load-balancing scorer. The cold-start cascade is structural, not a parameter issue.

### 5.5 H24: Combined Pathological Policies
- **Combined pathological produces 4.9x TTFT degradation** (routing + scheduling together)
- **Routing dominates** (~95% of TTFT degradation). Scheduling alone contributes only ~18%.
- **Super-additive interaction**: combined inversions (9,963) >> routing-only (2,859) + scheduling-only (2,158) = 5,017
- **Realtime class most vulnerable** to priority inversion (2.8x E2E degradation ratio)

**Implication for strategy design**: The converse is also likely true -- jointly GOOD routing + scheduling policies could produce super-additive IMPROVEMENT. If bad policies compound, good policies should compound too. This is the core thesis of the joint strategy approach.

### 5.6 H27: Chunked Prefill TTFT
- **52% short-request TTFT P99 improvement** with `--long-prefill-token-threshold=256` in bimodal workloads
- **Mechanism**: Splits 2048-token prefill into 8 chunks of ~11ms vs one ~43ms step, reducing HOL blocking
- **Tradeoff**: Long-request TTFT degrades 60-69%, P50 degrades 8-21%
- **E2E insensitive** (<4%) because decode time dominates

**Implication for strategy design**: Chunked prefill is a powerful TTFT P99 lever for bimodal workloads. The threshold parameter is strategy-tunable. Combined with SLO-aware scheduling, critical requests could get scheduling priority between chunks.

### 5.7 H28: Chunked Prefill ITL
- **REFUTED**: Chunked prefill provides ZERO ITL improvement (-0.5%)
- **Root cause**: ITL is dominated by decode-only steps (~255 of ~256 per request). Even a 43ms co-batched prefill step inflates only 1/255 of ITL samples.
- **DES insight**: Chunked prefill benefits TTFT (scheduling of new requests), NOT ITL (decode-phase token generation)

**Implication for strategy design**: Do not target ITL improvement through chunked prefill. ITL improvements require reducing decode step time, which means reducing batch size or improving decode efficiency.

### 5.8 H29: Snapshot Staleness
- **KV staleness degrades TTFT P99 by +242% to +548%** for kv-utilization:1 scorer at 100ms refresh
- **Queue-depth is immune** (0.0% change) -- always uses Immediate mode
- **Composite scorer mitigates ~99%** of the staleness effect (queue-depth:2,kv-utilization:2 -> only +3.8% degradation)
- **Safe zone <5ms** (~1 step time), threshold 10ms (14%), monotonic super-linear to 1717% at 500ms
- **Deterministic herding**: Stale KV signals create 120/120/120/140 distribution pattern (not stochastic)

**Implication for strategy design**: Any novel scorer that reads KV-related signals must be Tier 1 or paired with a Tier 1 signal. Alternatively, designs should prefer signals that are inherently fresh (queue depth, pending requests) over periodically-refreshed signals (KV utilization).

### 5.9 Prefix-Affinity Experiments
- **Prefix-affinity is 2.45x better** than queue-depth for multi-turn workloads (28.2 vs 69.0ms TTFT)
- **Queue-depth DESTROYS cache locality** for sessions -- actively avoids returning to cached instance
- **Round-robin is surprisingly strong** at low load due to cyclic cache reuse accident (62.9% cache hit vs 55.7% for prefix-affinity)
- **At high load, prefix-affinity wins** -- cache reuse reduces prefill time, outweighing concentration overhead
- **Session stickiness is inherently load-balanced** for multi-turn workloads by law of large numbers
- **Cache-heavy scoring (`pa:5,qd:1`) dominates ALL metrics even at 3x overload** for multi-turn
- **Queue-depth and kv-utilization are redundant** when KV blocks are abundant -- both track correlated signals

**Implication for strategy design**: For multi-turn workloads, session stickiness (via prefix-affinity) is the single most impactful routing signal. The load-balancing concern is largely self-solving for session-based workloads.

### 5.10 H-Reasoning-KV: Reasoning Context Accumulation
- **63.8% prefix cache hit rate** for multi-turn reasoning (context accumulation creates inter-round prefix reuse)
- **TTFT monotonicity** (rho=+1.000) with only 32% growth despite 13x input growth -- caching reduces beta
- **Cliff shift refuted**: Mean demand drives cliff, not per-request peak. Both workloads had 72 blocks mean -> same cliff.
- **Cascading preemptions** at low blocks: preemption rates >100x (thrashing). Known phenomenon (#349).

**Implication for strategy design**: Context accumulation workloads have excellent cache reuse potential (63.8%). KV management strategies should prioritize keeping active session KV blocks in GPU memory rather than evicting them to CPU.

### 5.11 H15: Fitness Evaluation
- **Fitness correctly ranks prefix-affinity higher** (3/3 seeds) for prefix workloads
- **1/(1+x/1000) normalization compresses** raw 14-38% TTFT P99 improvement to 2.6-8.2% fitness score difference
- **Weight sensitivity**: TTFT-heavy weights amplify ranking signal (+4.4%), throughput-heavy weights compress it (+0.7%)

**Implication for strategy design**: When comparing strategies via fitness, always examine raw metrics alongside. The normalization is aggressive.

### 5.12 H23: Low-Load Equivalence
- **All 4 policies within 4.40%** at rate=1 (< 5% threshold)
- **Uniform workloads eliminate differentiation** even at 8.7x overload (0.18% max deviation)
- **Prefix-affinity = least-loaded exactly** for random-token workloads (100% cache-miss -> falls back to LL)

**Implication for strategy design**: Novel strategies must be tested on heterogeneous workloads. Uniform workloads cannot distinguish policies.

---

## 6. The llm-d Blog: Precise vs Approximate KV-Aware Routing

The problem statement references a key finding: **precise KV-aware routing achieves 57x TTFT improvement** over approximate prefix-aware routing by maintaining real-time global cache state.

In BLIS's current architecture:
- The **router-side prefix cache index** (`PrefixCacheIndex`) is an approximation of actual per-instance KV state. It tracks block hashes via LRU with 10,000 blocks per instance, updated synchronously by the observer after routing decisions.
- The **actual instance KV cache** may diverge from the router-side view due to: (a) LRU evictions in the actual cache not reflected in the router index, (b) preemptions freeing blocks the router thinks are still cached, (c) the router index using a simplified LRU vs the actual cache's reference-counting and eviction policy.

The implication is that the router's prefix-affinity score can be inaccurate -- it may believe blocks are cached at an instance when they have been evicted. This is exactly the "approximate" approach the llm-d blog shows is 57x worse than precise.

**Opportunity**: A scorer that reads `FreeKVBlocks` or `CacheHitRate` from the actual routing snapshot (Tier 3 signals, periodically refreshed) could approximate "precise" KV state awareness. Even better, if the snapshot refresh interval is very low (<5ms per H29), these signals are fresh enough to be useful.

---

## 7. Key Performance Characteristics

### 7.1 Latency Model Coefficients (llama-3.1-8b, H100, TP=2)

```
Beta coefficients (step time):
  beta0 = 6910.42 us  (fixed overhead per step)
  beta1 = 17.67 us    (per cache-miss token during prefill)
  beta2 = 2.84 us     (per decode token)

Alpha coefficients (non-GPU overhead):
  alpha0 = 1601.35 us (fixed queueing overhead)
  alpha1 = 3.51 us    (per input token queueing overhead)
  alpha2 = 1805.54 us (per output token processing overhead)

Total KV blocks: 132,139
Block size: 16 tokens
```

### 7.2 Capacity Estimates

For constant 512/512 workload:
- Step time = 6910 + 17.67*512 + 2.84*512 = **17,412 us = 17.4ms**
- Per-instance capacity = **~57.4 req/s**
- 4 instances = **~229.7 req/s**
- Alpha overhead per request: 1601 + 3.51*512 = **3,398 us** (non-blocking)

### 7.3 Request Lifecycle Timing

For a single request (512 input, 512 output):
- Queueing delay (alpha): ~3.4ms (non-blocking)
- Prefill step: ~16ms (1 step if cached, more if not)
- Decode steps: ~512 steps at ~6.9ms each = ~3.5s
- Alpha2 overhead: ~512 * 1.8ms = ~925ms
- **Total E2E: ~4.5-5.5s** (dominated by decode)

### 7.4 KV Block Economics

- 132,139 blocks at 16 tokens/block = 2,114,224 tokens capacity
- A 512-token request uses ceil(512/16) = 32 blocks
- Max concurrent requests per instance (KV-limited): 132,139 / 32 = ~4,129 (not a bottleneck for this model)
- KV becomes a bottleneck for longer inputs (e.g., 4096 tokens = 256 blocks -> max ~516 concurrent)

---

## 8. The Weighted Scoring Framework

### 8.1 Architecture

The `WeightedScoring` router (`sim/routing.go`) implements a composable pipeline:

```go
type WeightedScoring struct {
    scorers   []scorerFunc        // each produces map[instanceID]float64 in [0,1]
    weights   []float64           // normalized to sum to 1.0
    observers []observerFunc      // called after routing decision (stateful scorers)
}
```

Weight normalization: absolute weights are irrelevant, only ratios matter. `pa:3,qd:2,kv:2` is identical to `pa:30,qd:20,kv:20` after normalization.

### 8.2 Score Computation Details

**queue-depth**: Min-max normalization. If all loads equal, all score 1.0. Otherwise `(maxLoad - load) / (maxLoad - minLoad)`. Range: [0, 1].

**kv-utilization**: Simple complement: `1 - KVUtilization`. Range: [0, 1] when utilization is in [0, 1].

**load-balance**: Inverse transform: `1/(1 + EffectiveLoad)`. Range: (0, 1]. Preserves absolute load differences (unlike min-max which is relative).

**prefix-affinity**: Proportion of request's block hashes found in instance's router-side cache index. Range: [0, 1]. Requires observer to maintain per-instance LRU cache.

### 8.3 Adding a New Scorer

A new scorer needs:
1. A function: `func(req *Request, snapshots []RoutingSnapshot) map[string]float64`
2. Registration in `validScorerNames` map + `newScorerWithObserver` switch
3. Optional observer if stateful

The scorer has access to:
- **Full request metadata**: `req.InputTokens`, `req.OutputTokens`, `req.SLOClass`, `req.TenantID`, `req.SessionID`, `req.RoundIndex`, `req.Priority`, `req.ArrivalTime`
- **Per-instance snapshots**: `QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`, `CacheHitRate`, `PendingRequests`

This is a rich input set that current scorers barely exploit. No existing scorer uses `req.SLOClass`, `req.ArrivalTime`, `req.SessionID`, `req.OutputTokens`, or `CacheHitRate`.

---

## 9. Tiered KV Cache Mechanics

### 9.1 Offload Trigger

In `TieredKVCache.maybeOffload()` (`sim/kv/tiered.go:207-238`), after any block release:
- Check if GPU utilization > `offloadThreshold` (default 0.9)
- Find free blocks with cached content (hash != "")
- Copy to CPU tier: `offloadedBlock{OriginalID, Tokens, Hash, OffloadTime}`
- Remove from GPU free list and hash table
- Re-add to GPU free list as empty block (at tail)

### 9.2 Reload Mechanism

In `TieredKVCache.AllocateKVBlocks()`: if GPU allocation fails, try `tryReloadFromCPU()`:
- Iterate CPU blocks in sorted order (deterministic)
- For each block with a hash not already on GPU: pop a GPU free block, fill with CPU content
- Accumulate transfer latency: `baseLatency + ceil(blockSize/bandwidth)` per block
- Check thrashing (offload+reload within 1000 ticks)

### 9.3 Transfer Latency

Transfer latency is consumed via `ConsumePendingTransferLatency()`, called by `Simulator.Step()`. This latency is added to the step time, slowing all requests in the current batch. This is non-blocking in the sense that the clock is not advanced, but the step takes longer.

### 9.4 Thrashing Detection

A thrashing counter increments when a block is reloaded within 1000 ticks of its offload. `KVThrashingRate()` returns `thrashingCount / offloadCount`. Currently, this counter is only for monitoring -- no policy reacts to it.

**Opportunity**: A proactive offload policy that considers session affinity and routing intent could dramatically reduce thrashing. If the router is sending session S to instance I, the offload policy on instance I should keep session S's blocks in GPU memory.

---

## 10. Batch Formation with Chunked Prefill

### 10.1 Chunked Prefill Mechanics

When `longPrefillTokenThreshold > 0`, a long request's prefill is split across multiple steps:
- Each step processes at most `threshold` tokens
- Between chunks, the wait queue is re-evaluated (new requests can enter)
- Total prefill time increases due to per-chunk beta0 overhead: ceil(inputTokens/threshold) * beta0

### 10.2 Token Budget

`maxScheduledTokens` limits total new tokens across all requests per step. This creates a natural batching limit even beyond `maxRunningReqs`.

### 10.3 Key Tunable Parameters

| Parameter | CLI Flag | Effect |
|-----------|----------|--------|
| `maxRunningReqs` | `--max-num-running-reqs` | Max batch size |
| `maxScheduledTokens` | `--max-num-scheduled-tokens` | Token budget per step |
| `longPrefillTokenThreshold` | `--long-prefill-token-threshold` | Chunked prefill threshold |

---

## 11. SLO Classes and Priority Policies

### 11.1 Available SLO Classes

Defined in `sim/workload/spec.go`: `"critical"`, `"standard"`, `"sheddable"`, `"batch"`, `"background"`.

### 11.2 How SLO Classes Flow Through the System

1. **Workload generation**: Set on `ClientSpec.SLOClass`
2. **Request creation**: Propagated to `Request.SLOClass`
3. **Routing**: Available on `req.SLOClass` but **NOT used by any current routing policy or scorer**
4. **Priority**: Available on `req.SLOClass` but **NOT used by `SLOBasedPriority`** (only uses age)
5. **Scheduling**: Uses `req.Priority` (set by PriorityPolicy), which does NOT incorporate SLO class
6. **Metrics**: Per-SLO-class TTFT and E2E distributions are collected and reported

**This is the single largest gap in the current system**: SLO class metadata exists end-to-end but is never used for any routing, scheduling, or priority decision. The infrastructure is in place; the policies just do not read it.

### 11.3 Priority Policy Parameters

The `SLOBasedPriority` uses:
- `BaseScore` (default 0.0)
- `AgeWeight` (default 1e-6: a request waiting 1 second gets +1.0 priority)

These are hardcoded in the factory. A parameterized version could:
- Set different `BaseScore` per SLO class (critical=10.0, standard=5.0, sheddable=0.0)
- Set different `AgeWeight` per SLO class (critical gets faster aging for urgency escalation)

---

## 12. Gaps, Redundancies, and Unexploited Opportunities

### 12.1 Unexploited Cross-Layer Information Paths

1. **RoutingDecision.Priority**: The router can set a priority hint that the instance scheduler sees on first-step scheduling. Currently ALWAYS zero. A SLO-aware router could boost critical requests' initial scheduling priority.

2. **Request.SLOClass in Scorers/Policies**: Available but never read. A scorer could differentiate routing based on SLO class (e.g., route critical requests to less-loaded instances, route sheddable requests to cache-hot instances).

3. **Request.OutputTokens in Routing**: Available but never read. Knowing expected output length enables predicted-latency scoring: a request with 1000 output tokens will occupy instance resources for ~1000 decode steps. The router could avoid sending long-output requests to already-busy instances.

4. **KV CacheHitRate in Routing**: Available in `RoutingSnapshot.CacheHitRate` but never read by any scorer. Combined with `FreeKVBlocks`, this could provide a "KV efficiency" signal -- route to instances with high cache hit rates (good prefix locality) and sufficient free blocks.

5. **Batch Formation has no access to routing intent**: The batch formation strategy does not know why requests were sent to this instance. If the router sent a request for cache affinity, the batch formation should prioritize scheduling it while its prefix blocks are still cached.

### 12.2 Redundancies

1. **queue-depth and kv-utilization are redundant** when KV blocks are abundant (confirmed by H-prefix-affinity: byte-identical results). Both track correlated signals (more load -> more KV usage). Removing one and using the freed weight for a novel signal is free.

2. **queue-depth and load-balance** serve the same purpose (load balancing) with different normalization. No need for both.

### 12.3 Missing Policy Dimensions

1. **SLO-aware admission**: The `TokenBucket` admission policy could shed `sheddable` requests first when under pressure, instead of applying a uniform rate limit.

2. **Deadline-driven priority**: No policy computes a "time remaining to SLO deadline" urgency score. With SLO class metadata, this is straightforward: `urgency = max_allowed_TTFT - (clock - arrivalTime)`.

3. **KV-pressure-aware routing**: No scorer considers `FreeKVBlocks` directly. The `kv-utilization` scorer uses the complementary signal, but a scorer that combines `FreeKVBlocks` with the request's expected block demand (`ceil(inputTokens/blockSize)`) could predict allocation success probability.

4. **Expected service time scorer**: No scorer estimates how long a request will take on each instance. With `req.InputTokens` and `req.OutputTokens`, plus beta coefficients, one could predict total service time and route to minimize makespan.

5. **Proactive KV offload policy**: The current offload is reactive (triggered by utilization threshold after release). A proactive policy could pre-offload cold blocks when a burst arrival is detected (from admission pipeline signals).

6. **Cache-aware scheduling**: No scheduler considers which requests have blocks currently cached. A scheduler that prioritizes requests with high cache overlap would maximize cache reuse within each batch step.

### 12.4 Structural Opportunities

1. **Super-additivity of joint policies (H24 converse)**: If pathological routing + scheduling produce super-additive degradation, well-chosen joint routing + scheduling should produce super-additive improvement. This is the fundamental hypothesis behind joint strategy optimization.

2. **Near-saturation regime amplification (H7)**: A strategy that effectively increases per-instance capacity by even 10% (through better cache reuse, shorter prefills, or smarter scheduling) would yield super-linear TTFT P99 improvement near the saturation boundary.

3. **Multi-turn session stickiness is self-balancing (prefix-affinity)**: For the target mixed-production workload with prefix-heavy multi-turn sessions, aggressive session stickiness (high prefix-affinity weight) may be optimal with minimal load-balancing penalty, because sessions distribute themselves uniformly across instances by the law of large numbers.

4. **Chunked prefill threshold as an SLO knob**: Setting a lower threshold for instances handling critical requests, and a higher threshold for instances handling batch requests, could optimize TTFT P99 for critical workloads while maximizing throughput for batch workloads.

---

## 13. Summary: Key Constraints and Invariants for Strategy Design

### Must-Honor Invariants
- INV-1: Request conservation (injected = completed + queued + running + dropped)
- INV-3: Clock monotonicity
- INV-4: KV block conservation (allocated + free = total)
- INV-5: Causality (arrival <= enqueue <= schedule <= completion)
- INV-6: Determinism (same seed = byte-identical output)
- INV-8: Work-conserving (if WaitQ non-empty, StepEvent must exist)

### Must-Avoid Antipatterns
- R1: No silent data loss (every error path must be handled)
- R4: Construction site audit (grep ALL construction sites when adding fields)
- R6: No Fatalf in library code
- R8: No exported mutable maps
- R11: Guard division (zero denominators)
- R13: Multi-impl interfaces (new interfaces need 2+ implementations)
- R14: Single-module methods (no method spans scheduling + latency + metrics)
- R19: Livelock protection (unbounded retry loops need circuit breakers)

### Key Parameters for Bayesian Optimization

| Parameter | Range | Affected Module |
|-----------|-------|-----------------|
| Scorer weights (w_prefix, w_queue, w_kv, w_custom) | [0, 10] | Routing |
| Offload threshold | [0.5, 1.0] | KV Cache |
| Prefill token threshold | [64, 2048] | Batch Formation |
| Priority base scores per SLO class | [0, 10] | Priority |
| Priority age weight | [1e-7, 1e-4] | Priority |
| Token bucket capacity/refill rate | various | Admission |
| Snapshot refresh interval | [0, 100000] us | Snapshot Provider |

### Extension Friction Summary

| Extension Type | Touch Points | Complexity |
|---------------|-------------|------------|
| New scorer | 2 | Low |
| New scheduler | 2 | Low |
| New priority policy | 2 | Low |
| New admission policy | 2 | Low |
| New batch formation | 4+ | Medium |
| New KV tier | 4+ | High |

The lowest-friction, highest-impact extensions for joint strategy design are: **new scorers** (access to full request metadata + instance snapshots), **new priority policies** (access to SLO class + age + clock), and **new schedulers** (access to request queue + clock). These three together can implement cross-layer information sharing without touching the batch formation or KV cache internals.

---

## 14. Empirical Baseline (llm-d Parity)

### Workload Design: Orthogonal SLO × Workload

**Critical design principle:** SLO tiers and workload types are ORTHOGONAL. All 3 tiers (critical/standard/sheddable) share the identical multi-turn prefix-heavy pattern (prefix_group="system-prompt", prefix_length=512, input~Gaussian(256,100), output~Exponential(128), gamma arrival CV=2.0). The SLO class is the ONLY differentiator. This forces strategies to use `SLOClass` metadata — exploiting token-length differences as a proxy for SLO is not possible.

### Baseline Results (3 seeds, 2000 req/s, 1000 requests, 8 instances)

| Seed | TTFT P99 | E2E P99 | TPS | Sched Delay P99 |
|------|----------|---------|-----|-----------------|
| 42 | 237.9 ms | 5,423.5 ms | 19,062 | 174.7 ms |
| 43 | 267.2 ms | 5,365.1 ms | 16,673 | 198.9 ms |
| 44 | 302.3 ms | 5,486.1 ms | 15,859 | 233.5 ms |
| **Mean** | **269.1 ms** | **5,424.9 ms** | **17,198** | **202.4 ms** |

### Per-SLO TTFT P99 (seed 42)

| SLO Class | TTFT P99 (ms) | Gap from Critical |
|-----------|---------------|-------------------|
| Critical | 233.1 | — |
| Standard | 238.2 | +2.2% |
| Sheddable | 242.2 | +3.9% |

**All tiers within ~4% of each other — zero meaningful SLO differentiation.** This confirms the baseline treats all SLO classes identically (FCFS + constant priority + no SLO-aware scoring).

### Targets

| Metric | Baseline (mean) | Target |
|--------|-----------------|--------|
| TTFT P99 (cluster) | 269.1 ms | <229 ms (>15% improvement) |
| Throughput | 17,198 tps | >18,058 tps (>5% improvement) |
| SLO differentiation | ~4% gap | >30% gap between critical and sheddable |
| Critical TTFT P99 | ~269 ms (same as others) | Significantly lower than sheddable |

### What the Baseline Teaches Us

1. **No SLO differentiation at all** — FCFS + constant priority treats premium and free-tier users identically
2. **Zero preemptions** — KV pressure is manageable at this load with 132K GPU + 44K CPU blocks
3. **Scheduling delay P99 = 202ms** — significant queuing exists, so scheduling order MATTERS
4. **Throughput varies 15,859-19,062 across seeds** — gamma burstiness creates seed-dependent load patterns

**Each strategy is a parameterized template** with tunable parameters optimized via Bayesian optimization (scikit-optimize). The optimizer sweeps parameter space across 3+ seeds, minimizing a multi-objective fitness function (TTFT P99 primary, throughput constraint, SLO differentiation constraint).

---

# Strategy Ideas

Three strategies were generated, iteratively refined based on cross-model reviews (GPT-4o, Claude Opus 4.6, Gemini 2.5 Flash). All strategies are updated for the v2 orthogonal workload design where SLO tiers share IDENTICAL token distributions -- the SLO class is the ONLY differentiator.

---

## Idea 1: SLO-Tiered Priority Cascade

### Core Mechanism
A three-layer cross-cutting strategy that exploits the SLO class metadata flowing through the entire BLIS pipeline but never read by any policy. The strategy introduces:

1. **SLO-Aware Priority Policy**: Replaces `ConstantPriority` with a policy that assigns per-SLO-class base scores (critical=10.0, standard=5.0, sheddable=1.0, batch=0.5, background=0.1) combined with per-class age escalation weights. Critical requests age faster (higher urgency escalation), while batch/background requests age slowly. Formula: `BaseSLO[class] + AgeWeight[class] * (clock - arrivalTime)`.

2. **SLO-Weighted Routing Scorer**: A new scorer that biases routing based on SLO class. Critical requests get routed to less-loaded instances (scorer score = queue-depth * slo_boost), while sheddable requests get routed to cache-hot instances (scorer score = prefix-affinity * slo_boost). The SLO class determines the weighting between load-awareness and cache-awareness PER REQUEST.

3. **Priority-FCFS Scheduling**: Enabled at instance level to honor the priority scores from (1).

The synergy: critical requests go to less-loaded instances (low TTFT) AND get scheduled first (low scheduling delay). Sheddable requests go to cache-hot instances (good throughput via cache reuse) but yield scheduling priority to critical requests if co-located.

### Why It Should Beat the Baseline
The baseline treats all requests identically. With the v2 orthogonal workload, the ~4% SLO gap is pure noise. This strategy reads `SLOClass` at every decision point:
- H24 shows joint policies are super-additive: if bad routing + scheduling compound, good routing + scheduling should compound too
- The RoutingDecision.Priority field is implemented but always zero -- we finally use it as the cross-layer information path from router to scheduler
- Scheduling delay P99 is 202ms -- significant enough for reordering to produce measurable differentiation

### Parameterized Template (10 parameters)
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `base_score_critical` | float64 | [5.0, 20.0] | Priority base score for critical class |
| `base_score_standard` | float64 | [2.0, 10.0] | Priority base score for standard class |
| `base_score_sheddable` | float64 | [0.0, 5.0] | Priority base score for sheddable class |
| `age_weight_critical` | float64 | [1e-5, 1e-3] | Age escalation for critical (fast aging) |
| `age_weight_standard` | float64 | [1e-6, 1e-4] | Age escalation for standard |
| `age_weight_sheddable` | float64 | [1e-7, 1e-5] | Age escalation for sheddable (slow aging) |
| `slo_load_bias` | float64 | [0.0, 1.0] | How much SLO class influences load vs cache routing |
| `w_slo_scorer` | float64 | [0.5, 5.0] | Weight of SLO scorer in weighted routing |
| `w_prefix` | float64 | [1.0, 5.0] | Weight of prefix-affinity scorer |
| `w_queue` | float64 | [1.0, 5.0] | Weight of queue-depth scorer |

### Hypotheses (4)
- **H1**: Critical TTFT P99 < 210ms (baseline ~269ms), sheddable TTFT P99 > 300ms. Target >30% gap.
- **H2**: Joint strategy produces >1.5x the TTFT P99 improvement of either component alone (super-additivity).
- **H3**: Total throughput changes <3% from baseline (reordering, not discarding).
- **H4**: SLO differentiation ratio >2x at 80% load, <1.3x at 40% load.

### Self-Critique
1. Sheddable starvation risk from aggressive critical priority
2. 10 parameters is a large search space for Bayesian optimization
3. SLO scorer blending load/cache signals may anti-correlate under load
4. H24 converse (super-additivity of good policies) is unproven

### Reviews

#### GPT-4o Review
- **Rating**: Solid plan with clear mechanisms
- **Strengths**: Correctly identifies the unused SLOClass metadata gap; three-layer synergy is well-reasoned
- **Concerns**: 10 parameters may have too many local minima for optimization; SLO scorer complexity may not add enough signal beyond just priority-FCFS; load/cache anti-correlation is a real risk
- **Suggestions**: Start with priority-only ablation to measure the 80% value before investing in the scorer; add starvation circuit breaker

#### Claude Opus Review
- **Rating**: Strong architectural insight
- **Concerns**: Priority data flow between router and scheduler is ambiguous -- which formula wins? Router hint is one-shot but priority policy recomputes each step. The SLO scorer needs clearer specification of how it interacts with existing scorers (additive? multiplicative?). Base scores are workload-dependent constants with no principled derivation.
- **Suggestions**: Use same formula in both router and instance to eliminate ambiguity; reduce parameters by sharing age weight across classes

#### Gemini Review
- **Rating**: Well-structured, implementable
- **Concerns**: Empty SLOClass handling not specified (crash or default?); performance overhead of per-request SLO lookup in scorer needs characterization; interaction with chunked prefill is unanalyzed
- **Suggestions**: Map empty SLOClass to "standard" by default; add overhead benchmark; document interaction with long-prefill-token-threshold

---

## Idea 2: Deadline-Driven Urgency Scheduling with Predictive Service Time Routing

### Core Mechanism
Builds on Idea 1's SLO exploitation but addresses three key criticisms: (a) the priority data flow ambiguity, (b) the load/cache anti-correlation, and (c) the overly large parameter space.

Introduces a **deadline-driven urgency** model where each SLO class has a TTFT budget. Priority = `1.0 / max(epsilon, (budget[class] - elapsed) / budget[class])`. As a request approaches its TTFT deadline, urgency grows hyperbolically, naturally handling aging without separate weights.

The routing layer introduces a **predictive service time scorer** that estimates per-instance completion time using beta coefficients and current batch sizes. The cross-layer bridge uses the same urgency formula in both router and instance policy.

### Why It Should Beat the Baseline
- Deadline awareness replaces arbitrary base scores with SLO-grounded urgency
- Predictive routing avoids the load/cache anti-correlation
- Reduced parameter space: 6 parameters vs 10
- With orthogonal workloads, the budget ratios are the ONLY mechanism for differentiation

### Parameterized Template (6 parameters)
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `ttft_budget_critical` | int64 (us) | [50000, 200000] | TTFT deadline for critical |
| `ttft_budget_standard` | int64 (us) | [200000, 500000] | TTFT deadline for standard |
| `ttft_budget_sheddable` | int64 (us) | [500000, 2000000] | TTFT deadline for sheddable |
| `w_predicted_service` | float64 | [1.0, 5.0] | Weight of predicted service time scorer |
| `w_prefix` | float64 | [1.0, 5.0] | Weight of prefix-affinity scorer |
| `urgency_epsilon` | float64 | [0.01, 0.1] | Floor to prevent division by zero |

### Hypotheses (4)
- **H1**: Critical TTFT P99 decreases >20% vs baseline (from ~269ms to <215ms).
- **H2**: Predicted-service-time scorer improves overall TTFT P99 by >10% vs queue-depth.
- **H3**: Sheddable TTFT P99 stays within 2x of its budget (starvation ceiling).
- **H4**: Budget value sensitivity <10% variance across 4x parameter range.

### Self-Critique
1. Predicted service time scorer needs beta coefficients -- scorers cannot access latency model in current architecture
2. Hyperbolic urgency has numerical instability near epsilon floor
3. Scorer requires interface changes (context injection for beta coefficients)
4. With orthogonal workloads, the predictive service time scorer adds no differentiation (identical token distributions mean identical predicted times)

### Reviews

#### GPT-4o Review
- **Rating**: Improvement over Idea 1 in parameter reduction and urgency model
- **Strengths**: Hyperbolic urgency is more principled than linear aging; predictive service time is novel
- **Concerns**: Predictive scorer needs beta coefficients that scorers cannot access; epsilon tuning is a hidden parameter; at 80% load, hyperbolic urgency may create priority oscillations
- **Suggestions**: Use closure pattern to inject beta coefficients; test stability with various epsilon values; consider piecewise-linear ramp as alternative to hyperbolic

#### Claude Opus Review
- **Rating**: Strong conceptual improvement, implementation concerns
- **Concerns**: The predicted-service-time scorer requires passing latency model parameters through a path that does not exist in the scorer interface. This is not a 2-touch-point extension -- it requires modifying the scorer factory to accept additional context. With orthogonal workloads (identical input/output distributions), the predictive scorer produces IDENTICAL scores for all SLO classes, making it useless for differentiation. The differentiation comes entirely from the priority policy, not the scorer.
- **Key insight**: "The predictive scorer is solving a problem that only exists with heterogeneous token distributions. With orthogonal workloads, it degenerates to queue-depth."
- **Suggestions**: Drop the predictive scorer entirely for orthogonal workloads; focus investment on the priority policy which IS the sole differentiator; consider a simpler scorer that reads SLOClass directly

#### Gemini Review
- **Rating**: Well-reasoned evolution from Idea 1
- **Concerns**: Empty SLOClass defaults to "standard" budget -- correct. Performance overhead is minimal (one division). But the TTFT budget values need calibration against actual system timing -- a 50ms budget for critical is below the compute floor (prefill of 256 tokens takes ~11ms + queueing), leaving only ~39ms for scheduling delay. This may be infeasible.
- **Suggestions**: Calibrate budgets against measured baseline scheduling delays per SLO class; the minimum feasible budget = compute time + minimal queueing

---

## Idea 3: SLO-Gated Priority Cascade with Cache-Protected Scheduling

### Core Mechanism
A four-layer strategy synthesizing the best elements of Ideas 1 and 2 while addressing their key weaknesses. The critical insight: with orthogonal workloads (identical token distributions across all SLO tiers), the ONLY lever for differentiation is how the system treats the `SLOClass` field at every decision point.

1. **SLO-Gated Priority Policy** (`slo-tiered`): Two-term formula combining static tier separation with piecewise-linear urgency escalation: `priority = base[class] + max(0, ageWeight * (age - threshold[class]))`. The base score provides immediate separation at enqueue time. The threshold gives sheddable requests a "grace period" where they stay at base priority, letting critical requests dominate scheduling without competition. Avoids Idea 2's hyperbolic instability.

2. **SLO-Aware Routing Scorer** (`slo-priority`): First scorer in BLIS to read `req.SLOClass`. Interpolates between load-awareness and cache-awareness per request: `score = bias[class] * queueScore + (1-bias[class]) * cacheScore`. Critical requests get `bias=0.8` (latency-focused); sheddable gets `bias=0.2` (cache-focused).

3. **Router-to-Scheduler Priority Bridge**: `WeightedScoring.Route()` sets `RoutingDecision.Priority` based on SLO class base score, activating the dormant cross-layer information path. Instance-level priority policy takes over with age escalation.

4. **Priority-FCFS Scheduling**: Uses existing scheduler. No new implementation.

### Why It Should Beat Baseline AND Ideas 1-2
- **vs Baseline**: Reads `SLOClass` at routing (layer 2), priority (layer 1), and scheduling (layer 4) -- zero SLO-aware components in baseline
- **vs Idea 1**: 7 parameters (not 10); piecewise-linear urgency (not raw linear aging); explicit priority bridge (not assumed)
- **vs Idea 2**: No beta-coefficient dependency in scorer; no hyperbolic instability; implementable within existing scorer interface; works with orthogonal workloads (SLOClass-based scoring, not token-based prediction)

### Parameterized Template (7 parameters)
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `base_critical` | float64 | [5.0, 15.0] | Priority base score for critical |
| `base_standard` | float64 | [2.0, 8.0] | Priority base for standard |
| `base_sheddable` | float64 | [0.0, 3.0] | Priority base for sheddable |
| `age_weight` | float64 | [1e-6, 1e-4] | Shared age escalation rate (threshold provides differentiation) |
| `threshold_sheddable` | int64 (us) | [50000, 500000] | Grace period before sheddable urgency activates |
| `slo_load_bias_critical` | float64 | [0.5, 1.0] | Load vs cache blend for critical routing |
| `slo_load_bias_sheddable` | float64 | [0.0, 0.5] | Load vs cache blend for sheddable routing |

### Hypotheses (4)
- **H1**: >30% gap between critical and sheddable TTFT P99 (critical < 210ms, sheddable > 300ms). Baseline gap is ~4%.
- **H2**: Joint strategy produces >1.3x the TTFT P99 gap of best single-component ablation (super-additivity test).
- **H3**: Total throughput changes <5% from baseline.
- **H4**: Sheddable TTFT P99 < 2x baseline (starvation ceiling via threshold mechanism).

### Self-Critique
1. SLO-gated routing may have weak signal if KV utilization is uniform across instances (132K blocks = abundant)
2. Seven parameters still large for Bayesian optimization (3-parameter subset may capture 90%)
3. Priority bridge assumes WeightedScoring routing policy
4. Cache protection is indirect (routing sheddable to cache-hot instances, not modifying KV eviction)
5. H24 converse (super-additivity of good policies) is unproven

### Reviews

#### GPT-4o Review
- **Rating**: Strong -- comprehensive, actionable, iterative improvement over Ideas 1-2
- **Strengths**: Well-structured four-layer approach; reduced parameter space; piecewise-linear urgency is cleaner than hyperbolic; self-critique is honest and anticipates real issues
- **Concerns**: Seven parameters still complex for Bayesian optimization; KVUtilization signal may be weak/redundant when blocks are abundant; priority bridge depends on WeightedScoring; indirect cache protection may not be strong enough
- **Suggestions**: Simplify parameter space further (collapse base_standard and base_sheddable); validate KVUtilization signal strength empirically; explore direct KV eviction policy modification; add dynamic threshold adaptation; handle edge cases for mixed SLO workloads with overlapping prefixes

#### Claude Opus Review
- **Rating**: Strong -- best of the three ideas
- **Critical issue**: The cache score (`1 - KVUtilization`) is NOT a cache affinity signal -- it is a free-capacity signal. It says nothing about whether an instance holds relevant prefix cache entries. At 132K blocks, this signal will be nearly uniform across instances, making the scorer collapse to noise-weighted queue-depth for all tiers. Should replace with actual prefix-affinity scorer signal.
- **Math concern**: Starvation ceiling is higher than claimed. With base_critical=10, base_sheddable=1, age_weight=1e-5, threshold=100ms: sheddable needs ~1 second to overtake fresh critical. The H4 prediction needs recalibration.
- **Missing parameter**: `threshold_standard` should be explicit -- with 50% of traffic at zero threshold, standard requests compete with critical after ~500ms aging.
- **Priority bridge risk**: Verify that `priority-fcfs` scheduler re-evaluates priority each step (calls PriorityPolicy.Compute) rather than caching the router-set initial priority.
- **Suggestions**: Replace cache score with prefix-affinity signal; add threshold_standard; verify scheduler re-evaluation behavior; add SLOClass distribution metric for debugging

#### Gemini Review
- **Rating**: Well-structured, detailed, strong understanding of problem space
- **Strengths**: Incorporates feedback from prior iterations; clear implementation path; honest self-critique
- **Concerns**: (review was truncated but key themes align with GPT-4o and Claude Opus concerns about signal strength and parameter complexity)

---

# Executive Summary

## Rankings

| Rank | Strategy | Expected Impact | Confidence | Key Risk |
|------|----------|----------------|------------|----------|
| 1 | **Idea 3: SLO-Gated Priority Cascade** | High | Medium-High | Cache score signal may be degenerate |
| 2 | Idea 2: Deadline-Driven Urgency | Medium | Medium | Predictive scorer unusable with orthogonal workloads |
| 3 | Idea 1: SLO-Tiered Priority Cascade | Medium | Medium-Low | Too many parameters; starvation risk |

## Top Strategy: Idea 3

**Idea 3 is the recommended first implementation** for these reasons:

1. **Orthogonal workload compatibility**: Unlike Idea 2's predictive service time scorer (which degenerates to queue-depth with identical token distributions), Idea 3's SLO-priority scorer explicitly reads `SLOClass` to differentiate routing behavior. This is the only approach that works by design with the v2 orthogonal workload.

2. **Lowest implementation risk**: 5-6 touch points, all within the policy template extension recipe. No interface changes required (unlike Idea 2's beta-coefficient injection). The `priority-fcfs` scheduler and `WeightedScoring` router are both existing, tested components.

3. **Balanced parameter space**: 7 parameters is a manageable search space for Bayesian optimization (Idea 1 has 10, which is more likely to get stuck in local minima). The 3-parameter subset (base_critical, base_sheddable, slo_load_bias_critical) captures the core mechanism and can be optimized first, with remaining parameters as refinements.

4. **Addresses all three reviewer concerns from Ideas 1-2**: eliminates priority data flow ambiguity (same base score table), reduces parameters (shared age_weight), avoids hyperbolic instability (piecewise-linear ramp).

## Recommended Implementation Changes (from reviews)

Before implementing Idea 3, incorporate these critical reviewer findings:

1. **Replace cache score with prefix-affinity signal** (Claude Opus): The `1 - KVUtilization` signal is a free-capacity proxy, not a cache-affinity signal. Replace with actual prefix-affinity scorer output for the cache component of the SLO-priority scorer. This makes the `slo_load_bias` parameter genuinely meaningful.

2. **Add `threshold_standard` parameter** (Claude Opus): Without it, 50% of traffic (standard tier) competes with critical after ~500ms of aging, eroding the differentiation. Either add an explicit threshold or set `threshold_standard = threshold_sheddable / 2`.

3. **Verify scheduler re-evaluation** (Claude Opus): Confirm that `PriorityFCFSScheduler.OrderQueue()` is called every step with recomputed priorities (not cached from initial routing hint). Code inspection shows it IS called every step in `scheduleBatch()` -- this is safe.

4. **Calibrate starvation ceiling** (Claude Opus): Recalculate the crossover time with actual parameter ranges. At midpoint values, sheddable needs ~1 second to overtake fresh critical. This may be acceptable (well within the 2x baseline target of 538ms TTFT P99), but H4 predictions should reflect this realistic timescale.

## Cross-Cutting Themes from Reviews

All three reviewers across all three ideas converged on these themes:

1. **Signal quality matters more than mechanism complexity**: The most sophisticated routing formula is useless if the underlying signal (KVUtilization, predicted service time) is degenerate or noisy. Prefix-affinity is the only proven, high-signal scorer for multi-turn workloads.

2. **The priority policy is the primary differentiator**: With orthogonal workloads, routing can only influence WHICH instance processes a request, but ALL instances see the same token distributions. The priority policy is the only mechanism that directly controls scheduling ORDER within an instance. Every reviewer rated the priority mechanism as more impactful than the routing scorer.

3. **Parameter count drives optimization difficulty**: Every reviewer flagged parameter space size as a risk. The consensus is that 5-7 parameters is workable; 10+ is risky for Bayesian optimization over 3 seeds.

4. **Super-additivity is a hypothesis, not a guarantee**: All three reviewers noted that H24's pathological super-additivity does not guarantee the converse. The ablation experiments (priority-only, scorer-only, joint) are essential to test this claim empirically.

5. **Starvation prevention must be quantified**: Every strategy needs an explicit starvation ceiling analysis with concrete crossover times, not just "the age weight will prevent it."

## Implementation Order

1. **Phase 1 (Priority-only baseline)**: Implement `SLOTieredPriority` policy + `priority-fcfs` scheduler. No new scorer. This tests the pure scheduling differentiation signal and establishes the 80% value. ~3 touch points. Expected SLO gap: 15-25%.

2. **Phase 2 (Add SLO-priority scorer)**: Add the SLO-priority scorer with prefix-affinity signal (per Claude Opus recommendation) + router priority bridge. This tests the routing-scheduling compounding hypothesis. ~3 additional touch points. Expected SLO gap: 25-40%.

3. **Phase 3 (Bayesian optimization)**: Sweep the 7-parameter space across 3+ seeds. Start with 3-parameter subset (base_critical, base_sheddable, slo_load_bias_critical), then expand. Target: identify parameter regime that maximizes SLO gap while keeping throughput within 5% of baseline.

4. **Phase 4 (Ablation experiments)**: Run the four hypotheses with full ablation controls to validate super-additivity, starvation ceiling, throughput preservation, and load-regime sensitivity.
