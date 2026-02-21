# H13: Determinism Invariant

**Status:** Confirmed
**Tier:** 1 (correctness baseline)
**Type:** Deterministic
**Date:** 2026-02-20

## Hypothesis

> Same seed must produce byte-identical stdout across runs. BLIS uses PartitionedRNG for deterministic simulation — running the same configuration with the same seed twice should produce identical output. This is critical for reproducible research.

## Experiment Design

**Classification:** Deterministic (Type 1) — single seed, exact pass/fail. One failure = non-determinism bug.

**Configurations compared (5 pairs, each run twice with seed=42):**
1. `round-robin + fcfs + constant + always-admit` — simplest code path
2. `least-loaded + fcfs + constant + always-admit` — PendingRequests bookkeeping
3. `weighted (qd:2,kv:2) + fcfs + constant + always-admit` — scorer pipeline
4. `weighted (pa:3,qd:2,kv:2) + fcfs + constant + always-admit` — **stateful scorer (highest risk)**
5. `least-loaded + priority-fcfs + slo-based + always-admit` — priority ordering

**Controlled variables:** model (llama-3.1-8b), instances (4), requests (200), rate (1000), seed (42)
**Varied variable:** None — each configuration runs twice identically
**Seeds:** 42 (single seed — determinism is the point)

**Preconditions verified:**
- Configuration 4 uses prefix-affinity scorer with enough requests (200) to trigger LRU eviction in PrefixCacheIndex — this exercises the `evictOldest` map iteration path that is the most likely non-determinism source (R2: sort map keys)

## Results

| # | Configuration                              | Run1 Size | Run2 Size | Status |
|---|-------------------------------------------|----------:|----------:|--------|
| 1 | round-robin + fcfs (simplest)             |     3,967 |     3,967 | **PASS** |
| 2 | least-loaded + fcfs (PendingRequests)     |     4,076 |     4,076 | **PASS** |
| 3 | weighted (qd:2,kv:2) (scorer pipeline)   |     4,073 |     4,073 | **PASS** |
| 4 | weighted (pa:3,qd:2,kv:2) (stateful)     |     4,073 |     4,073 | **PASS** |
| 5 | least-loaded + priority-fcfs + slo-based  |     4,170 |     4,170 | **PASS** |

**Result: ALL PASS — 5 configuration pairs, byte-identical output.**

## Root Cause Analysis

Determinism is maintained because:

1. **PartitionedRNG**: Each subsystem (workload generation, routing, scheduling) draws from its own partition of a seeded PRNG. Partitioning prevents ordering dependencies between subsystems.

2. **Sorted map iteration**: R2 compliance ensures that map iteration feeding output ordering or float accumulation always sorts keys first. The PrefixCacheIndex `evictOldest` function maintains the monotonic clock invariant instead of iterating the underlying map directly.

3. **Event ordering**: The DES event queue uses `(timestamp, priority, seqID)` ordering with `seqID` as a deterministic tiebreaker. No wall-clock timing enters the event ordering.

4. **Output separation**: Deterministic results go to stdout; diagnostic timing goes to stderr. The `2>/dev/null` redirect ensures no wall-clock information reaches stdout.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| INV-6 holds across all 5 routing/scheduling/priority configurations | Confirmation | Documented here |
| Stateful prefix-affinity scorer with LRU eviction is deterministic | Confirmation | Documented here |
| Priority ordering (slo-based + priority-fcfs) is deterministic | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found**
- [x] Any new rules needed? **None** — existing R2 (sort map keys) is sufficient
- [x] Any new invariants needed? **None** — INV-6 is confirmed
- [x] Any existing rules/invariants confirmed? **INV-6 (determinism) confirmed across 5 policy configurations.** R2 compliance verified implicitly — map iteration determinism holds.

## Implications for Users

- Users can rely on `--seed` for fully reproducible experiments: same seed = identical output, regardless of routing policy, scheduler, or priority configuration.
- This includes the stateful prefix-affinity scorer, which maintains an internal LRU cache that could theoretically break determinism — it does not.
- Multi-seed experiments (comparing seeds 42, 123, 456) produce genuinely independent samples, not artifacts of non-determinism.

## Reproducing

```bash
cd hypotheses/h13-determinism
./run.sh
```
