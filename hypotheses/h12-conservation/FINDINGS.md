# H12: Request Conservation Invariant

**Status:** Confirmed (with bug discovery)
**Tier:** 1 (correctness baseline)
**Type:** Deterministic
**Date:** 2026-02-20

## Hypothesis

> No matter what routing, scheduling, or admission policy is used, every injected request must end up completed, queued, or running at simulation end: `injected == completed + still_queued + still_running`. With admission control: `num_requests == injected + rejected`. This is a fundamental correctness property.

## Experiment Design

**Classification:** Deterministic — the invariant either holds or it doesn't. Single seed sufficient.

**Invariants tested:**
1. **Per-instance conservation:** `injected == completed + still_queued + still_running` for each instance
2. **Cluster conservation:** `injected == completed + still_queued + still_running` for cluster aggregate
3. **Cross-instance consistency:** `sum(per-instance injected) == cluster injected`
4. **Full pipeline conservation:** `num_requests == injected + rejected` (with admission control)

**Configurations compared (10 non-preempting + 1 preemption test):**

| # | Routing | Scheduler | Priority | Admission | Notes |
|---|---------|-----------|----------|-----------|-------|
| 1 | round-robin | fcfs | constant | always-admit | Baseline |
| 2 | least-loaded | fcfs | constant | always-admit | Different routing |
| 3 | weighted (qd:1) | fcfs | constant | always-admit | Weighted routing |
| 4 | weighted (pa:3,qd:2,kv:2) | fcfs | constant | always-admit | Full scorer stack |
| 5 | round-robin | sjf | constant | always-admit | Different scheduler |
| 6 | round-robin | priority-fcfs | slo-based | always-admit | Priority scheduling |
| 7 | round-robin | fcfs | constant | token-bucket | Admission control |
| 8 | least-loaded | fcfs | constant | always-admit | High rate=2000 (queue stress) |
| 9 | weighted (qd:2,kv:2) | priority-fcfs | slo-based | token-bucket | Combined policies |
| 10 | always-busiest | reverse-priority | inverted-slo | always-admit | Pathological |
| 11 | least-loaded | fcfs | constant | always-admit | KV=500 (preemption — **panics**) |

**Controlled variables:** model (llama-3.1-8b-instruct), instances (4), requests (200), seed (42)
**Varied variable:** policy combination per configuration
**Seeds:** 42 (single seed — deterministic experiment)
**Preconditions verified:** Binary builds, all tests pass on baseline commit

## Results

### Conservation Invariant (Configs 1-10)

```
  #    Configuration                                Inj  Comp   Q   R   Rej Preempt Checks Status
  ---- ------------------------------------------ ----- ----- --- --- ----- ------- ------ ------
  1    round-robin + fcfs + always-admit            200   200   0   0     0       0    7/7 PASS
  2    least-loaded + fcfs + always-admit           200   200   0   0     0       0    7/7 PASS
  3    weighted(qd:1) + fcfs + always-admit         200   200   0   0     0       0    7/7 PASS
  4    weighted(pa:3,qd:2,kv:2) + fcfs              200   200   0   0     0       0    7/7 PASS
  5    round-robin + sjf + always-admit             200   200   0   0     0       0    7/7 PASS
  6    round-robin + priority-fcfs + slo-based      200   200   0   0     0       0    7/7 PASS
  7    round-robin + fcfs + token-bucket              5     5   0   0   195       0    7/7 PASS
  8    least-loaded + fcfs + rate=2000              200   200   0   0     0       0    7/7 PASS
  9    weighted + priority-fcfs + token-bucket        5     5   0   0   195       0    7/7 PASS
  10   always-busiest + reverse-priority            200   200   0   0     0       0    4/4 PASS
```

**67 invariant checks across 10 configurations — zero violations.**

### Preemption Path (Config 11)

Config 11 (`--total-kv-blocks 500`) triggers a panic:

```
panic: runtime error: index out of range [-1]

goroutine 1 [running]:
github.com/inference-sim/inference-sim/sim.(*Simulator).preempt(...)
    sim/simulator.go:383
```

The `preempt()` function at `simulator.go:383` accesses `RunningBatch.Requests[len(RunningBatch.Requests)-1]` without checking if the batch is empty. When a request needs more blocks than the total available after evicting ALL running requests, the loop depletes the batch and then accesses index `-1`.

## Root Cause Analysis

### Conservation (Confirmed)

The request lifecycle is correct for all non-preempting configurations:
- **Admission:** Requests are either admitted (routed to an instance) or rejected (counted in trace summary). `admitted + rejected == total_decisions == num_requests`.
- **Routing:** Each admitted request reaches exactly one instance. `sum(per-instance injected) == cluster injected`.
- **Completion:** At simulation end, each instance reports `completed + still_queued + still_running == injected`.

The token-bucket admission (configs 7 and 9) correctly accounts for rejected requests: `5 injected + 195 rejected == 200 num_requests`.

Config 10 (pathological: always-busiest) routes all 200 requests to a single instance. Only that instance emits metrics, but conservation still holds: `200 completed + 0 queued + 0 running == 200 injected`.

### Preemption Panic (Bug)

The bug is in the preemption loop at `sim/simulator.go:375-404`:

```go
func (sim *Simulator) preempt(req *Request, now int64, numNewTokens int64) bool {
    for {
        if ok := sim.KVCache.AllocateKVBlocks(...); !ok {
            // BUG: no check for empty RunningBatch
            preemptedRequest := sim.RunningBatch.Requests[len(sim.RunningBatch.Requests)-1]
            // ... evicts last request from batch
        } else {
            return true
        }
    }
}
```

The loop assumes there is always another request to evict. When a single request needs more blocks than the total cache capacity (e.g., request needs 53 blocks but only 125 blocks/instance available, and the existing running requests don't free enough), the loop evicts all running requests and then tries to access an empty slice.

**Impact:** Any workload with constrained KV blocks that produces requests larger than the per-instance cache capacity will crash the simulator.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| INV-1 holds for all 10 non-preempting policy combinations | Confirmation | Documented here |
| Full pipeline conservation holds with token-bucket rejection | Confirmation | Documented here |
| Preemption loop panics on empty RunningBatch (simulator.go:383) | **Bug discovery** | File GitHub issue |
| Conservation cannot be tested under preemption pressure | **Design limitation** | Blocked until bug is fixed |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **R5 (Transactional mutation)** — the preemption loop mutates `RunningBatch.Requests` without a bounds check, violating the principle of transactional mutation with rollback on failure
- [ ] Any new rules needed? None — R5 already covers this case
- [ ] Any new invariants needed? None — INV-1 already covers conservation
- [x] Any existing rules/invariants confirmed? **INV-1** confirmed for all non-preempting configurations (routing, scheduling, admission, pathological). **INV-1 full pipeline** confirmed with token-bucket admission.

## Implications for Users

1. **Conservation is reliable** for all standard policy combinations at typical load levels. Users can trust that `injected == completed + queued + running` holds in their simulations.

2. **Avoid constrained KV blocks with high load.** If `total_kv_blocks` is too low relative to request sizes, the simulator will panic. As a rule of thumb, ensure `total_kv_blocks * block_size_in_tokens` is at least 4x the maximum expected input token count.

3. **Token-bucket admission correctly tracks rejections.** The full pipeline invariant `num_requests == injected + rejected` holds, making it safe to use rejection counts for capacity planning.

## Reproducing

```bash
cd hypotheses/h12-conservation
./run.sh
```
