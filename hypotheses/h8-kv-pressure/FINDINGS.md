# H8: KV Cache Pressure

**Status:** Confirmed
**Tier:** 3 (System Understanding)
**Type:** Statistical / Monotonicity
**Date:** 2026-02-20

## Hypothesis

> Reducing total KV blocks should increase preemption frequency and worsen tail latency. KV blocks are the memory currency — each running request holds blocks proportional to its token count. With fewer blocks, the cache fills up faster, forcing preemptions (evictions of running requests to make room). Preempted requests restart from scratch, increasing tail latency.

**Mechanism under test:**
- `sim/simulator.go:375-408` — `preempt()` evicts last running request when KV allocation fails
- `sim/simulator.go:436,455` — `makeRunningBatch` calls `preempt()` during batch formation
- `sim/simulator.go:391-394` — preempted request requeued at front of WaitQ with `ProgressIndex = 0`

## Result Summary

| KV Blocks | Preempt Rate | TTFT p99 (ms) | E2E p99 (ms) | vs Baseline |
|:---------:|:------------:|:-------------:|:-------------:|:-----------:|
| 5000 | 0.0000 | 460.6 | 3610.9 | baseline |
| 3000 | 0.0000 | 460.6 | 3610.9 | 1.00x |
| 2200 | 0.0000 | 460.6 | 3610.9 | 1.00x |
| 2100 | 0.1117 | 2174.6 | 6193.3 | 4.72x |
| 2000 | 0.5100 | 3048.0 | 7194.8 | 6.62x |

*4 instances, 200 requests, rate=2000, block_size=16, averaged across seeds 42/123/456.*

**Verdict: CONFIRMED — Both preemption rate and TTFT p99 monotonically increase as KV blocks decrease, across all 3 seeds.**

## Experiment Design

**Classification:** Statistical / Monotonicity

**Configurations compared:**
- A: `--total-kv-blocks 5000` (abundant — baseline)
- B: `--total-kv-blocks 3000` (ample)
- C: `--total-kv-blocks 2200` (threshold)
- D: `--total-kv-blocks 2100` (constrained)
- E: `--total-kv-blocks 2000` (severely constrained)

**Controlled variables:** model (llama-3.1-8b), 4 instances, rate=2000, 200 requests, block_size=16, Poisson arrivals, Gaussian input (mean=512, std=50), Gaussian output (mean=256, std=50), always-admit, round-robin routing, FCFS scheduling

**Varied variable:** `--total-kv-blocks` (per-instance)

**Seeds:** 42, 123, 456 (both CLI `--seed` and YAML `seed:` field)

**Preconditions verified:**
- `--total-kv-blocks` CLI flag works correctly (bug #285 fixed in commit cbb0de7)
- Rate=2000 creates enough concurrent KV pressure (verified via feasibility testing)
- Block counts ≥ 2000 complete within 120s (block counts < 1000 cause livelock)

## Results

### Per-Seed Detail

**Seed 42:**

| Blocks | Preempt Rate | Preempt # | TTFT p99 | E2E p99 | Throughput |
|--------|:------------:|:---------:|:--------:|:-------:|:----------:|
| 5000 | 0.0000 | 0 | 473.8 | 3608.9 | 64.3 |
| 3000 | 0.0000 | 0 | 473.8 | 3608.9 | 64.3 |
| 2200 | 0.0000 | 0 | 473.8 | 3608.9 | 64.3 |
| 2100 | 0.1750 | 35 | 2305.0 | 6193.9 | 41.5 |
| 2000 | 0.5950 | 119 | 2859.7 | 7071.7 | 38.0 |

**Seed 123:**

| Blocks | Preempt Rate | Preempt # | TTFT p99 | E2E p99 | Throughput |
|--------|:------------:|:---------:|:--------:|:-------:|:----------:|
| 5000 | 0.0000 | 0 | 453.4 | 3583.1 | 61.5 |
| 3000 | 0.0000 | 0 | 453.4 | 3583.1 | 61.5 |
| 2200 | 0.0000 | 0 | 453.4 | 3583.1 | 61.5 |
| 2100 | 0.0550 | 11 | 2053.3 | 6110.4 | 46.3 |
| 2000 | 0.2750 | 55 | 2666.3 | 6835.2 | 40.4 |

**Seed 456:**

| Blocks | Preempt Rate | Preempt # | TTFT p99 | E2E p99 | Throughput |
|--------|:------------:|:---------:|:--------:|:-------:|:----------:|
| 5000 | 0.0000 | 0 | 454.8 | 3640.8 | 63.0 |
| 3000 | 0.0000 | 0 | 454.8 | 3640.8 | 63.0 |
| 2200 | 0.0000 | 0 | 454.8 | 3640.8 | 63.0 |
| 2100 | 0.1050 | 21 | 2165.3 | 6275.5 | 44.0 |
| 2000 | 0.6600 | 132 | 3618.1 | 7677.6 | 32.6 |

### Conservation Invariant (INV-1)

All 15 configurations (5 block counts × 3 seeds) pass: `injected == completed + still_queued + still_running`. Zero violations.

### Monotonicity Check

All 3 seeds pass monotonicity for both preemption rate and TTFT p99. No inversions.

## Root Cause Analysis

### Why the cliff is so sharp

The transition from "no preemptions" to "frequent preemptions" occurs between 2200 and 2100 blocks because of how batch formation interacts with KV allocation:

1. **Concurrent block demand:** At rate=2000 with 4 instances, ~500 requests/s per instance. With 512 mean input tokens / 16 block_size = 32 blocks per request for prefill alone. Plus 256 mean output / 16 = 16 blocks for decode. Total: ~48 blocks per request at completion.

2. **Peak concurrent occupancy:** With `max_running_reqs=256` (default) and batch formation greedily dequeuing, the peak concurrent KV usage depends on how many requests are simultaneously running. At rate=500/instance, with ~4-8s per request E2E, roughly 2000-4000 blocks are needed concurrently.

3. **Binary threshold:** If peak concurrent demand exceeds total blocks, the `preempt()` function at `simulator.go:375` enters a loop that evicts running requests one by one until the allocation succeeds. Each preempted request is requeued with `ProgressIndex = 0` (line 392), meaning it must re-prefill entirely — wasting the blocks it previously consumed.

4. **Cascade effect:** Preempted requests re-enter the queue and need blocks again. This creates a positive feedback loop: preemptions free blocks temporarily, but the re-prefill of the preempted request consumes them again, triggering more preemptions.

### Why 5000/3000/2200 produce identical output

These block counts are all above the peak concurrent demand (~2200 blocks). Since no preemptions occur, the simulation follows the exact same execution path — same batch formation, same step times, same completions. The KV cache never becomes the bottleneck, so extra blocks are irrelevant. This confirms that BLIS correctly models KV capacity as a threshold effect, not a gradual degradation.

### Preemption count variance across seeds

| Blocks | Seed 42 | Seed 123 | Seed 456 | Explanation |
|--------|:-------:|:--------:|:--------:|-------------|
| 2100 | 35 | 11 | 21 | Seed-dependent arrival clustering determines which requests overlap at the threshold |
| 2000 | 119 | 55 | 132 | More preemptions → more requeuing → more re-preemption (cascade amplification) |

The high variance at 2000 blocks (55 to 132) reflects the cascade effect: seeds that produce more simultaneous arrivals (due to Poisson clustering) trigger earlier preemptions, which compound.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Preemption monotonicity confirmed | Confirmation | Documented here |
| TTFT p99 monotonicity confirmed | Confirmation | Documented here |
| Conservation (INV-1) holds under pressure | Confirmation (INV-1) | Documented here |
| Sharp cliff between 2200-2100 blocks | Design limitation | Documented here — users should plan for threshold, not gradual degradation |
| `preemption_count` was missing from JSON output | Bug (minor) | Fixed in this PR — added to `MetricsOutput` |
| Blocks < 1000 cause simulation livelock | Design limitation | See "Livelock Finding" below |

### Livelock Finding

During feasibility testing, block counts below ~1000 (with this workload) caused the simulation to run indefinitely. The `preempt()` function at `simulator.go:375-408` evicts requests one at a time and retries allocation. When no single request can complete because every request needs more blocks than are available after eviction, the simulation enters a cycle: evict → requeue → re-prefill → evict.

This is not strictly a bug — the simulation accurately models what would happen in a real system with catastrophically undersized KV cache. However, the lack of a circuit breaker (e.g., maximum preemption attempts per step) means the simulation never terminates. Users who accidentally set very low block counts will see a hung process.

**Recommendation:** File an issue to add a configurable preemption limit or a warning when preemption count exceeds a threshold (e.g., 10× num_requests).

## Standards Audit

Findings checked against `docs/standards/`:

- [x] Any violations of existing rules? **None found.**
  - R4 (construction sites): `MetricsOutput` has one construction site in `SaveResults` — updated.
  - R11 (guard division): No new division operations.
- [x] Any new rules needed? **None**, but the livelock finding suggests a future safeguard.
- [x] Any new invariants needed? **None** — INV-1 (conservation) was confirmed to hold.
- [x] Any existing rules/invariants confirmed?
  - **INV-1 confirmed** — request conservation holds under KV pressure across all 15 configurations.
  - **INV-6 confirmed** — all abundant-block configs (5000/3000/2200) produce byte-identical output across runs with the same seed.

## Implications for Users

1. **KV blocks have a cliff, not a slope.** Performance is identical above a threshold and collapses sharply below it. For capacity planning, determine the peak concurrent block demand and provision above it. There is no "graceful degradation" zone.

2. **The threshold depends on workload concurrency.** For this workload (rate=2000, 512-token input, 256-token output, 4 instances): the threshold is ~2200 blocks per instance. Your threshold will differ based on `rate / num_instances × (input_tokens + output_tokens) / block_size`.

3. **Preemptions cascade.** A single preemption event doesn't just add one restart — it triggers a chain because the restarted request competes for the same scarce blocks. The preemption count at 2000 blocks (55-132 for 200 requests) shows 28-66% of requests are preempted at least once.

4. **Conservation holds under pressure.** Even with extreme preemption rates (66%), no requests are lost. All 200 requests complete correctly. The preemption → requeue → re-prefill cycle is correctness-preserving.

## Reproducing

```bash
cd hypotheses/h8-kv-pressure
./run.sh           # ~3 minutes, all experiments
./run.sh --rebuild # rebuild binary first
```

Requires: Go 1.24+, Python 3
