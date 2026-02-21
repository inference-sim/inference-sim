# H10: Tiered KV Cache (GPU+CPU Offload)

**Status:** Confirmed
**Resolution:** Confirmation with bug discovery. Tiered cache reduces preemptions (17.5% → 8.5%) and improves TTFT (417ms → 299ms, 28% better) via prefix hash preservation. An analyzer bug (`analyze.py` parsed the wrong output field) masked these preemptions for 2 rounds.
**Family:** Structural model
**VV&UQ:** Validation
**Tier:** 3 (system understanding)
**Type:** Statistical / Dominance
**Date:** 2026-02-20
**Rounds:** 4

## Hypothesis

> When GPU KV blocks are exhausted, the single-tier cache preempts requests. With a CPU tier, blocks can be offloaded to CPU instead of being evicted entirely. The tradeoff: reload incurs transfer latency, but avoids full recomputation.

## Experiment Design

**Classification:** Statistical / Dominance — tiered preemption count < single-tier preemption count.

**Configurations compared:**
- A: Single-tier (`--kv-cpu-blocks 0`) — GPU blocks=2100, no CPU tier
- B: Tiered (`--kv-cpu-blocks 500 --kv-offload-threshold 0.8 --kv-transfer-bandwidth 100 --kv-transfer-base-latency 10`)

**Controlled variables:** model (llama-3.1-8b), instances (4), workload (Gaussian input mean=512, Poisson rate=2000, 200 requests), GPU blocks (2100), block size (16 tokens)
**Varied variable:** CPU tier presence and size
**Seeds:** 42, 123, 456 (Experiments 1-3); seed=42 only (Experiment 4 confound matrix)
**Preconditions verified:** H8 found the preemption cliff at 2100-2200 blocks with similar workload parameters.
**Reference experiment:** `hypotheses/h8-kv-pressure/run.sh` — H10 reuses H8's workload distributions and block counts.

## Results

### Experiments 1-3: Primary 3-seed runs (via `run_sim` wrapper)

These experiments used `--routing-policy least-loaded` and were analyzed by `analyze.py`, which had a regex bug (see Experiment 4 note). The preemption column below shows what the analyzer reported (0), which was incorrect.

| Seed | Tier | TTFT Mean | TTFT P99 | E2E P99 | Reported Preempt | Completed | INV-1 |
|------|------|----------:|---------:|---------:|--------:|----------:|-------|
| 42 | single-tier | 417.2 | 2,305.0 | 6,193.9 | 0* | 200 | OK |
| 42 | tiered(CPU=500) | 299.4 | 1,616.8 | 5,532.9 | 0* | 200 | OK |
| 123 | single-tier | 316.9 | 2,053.3 | 6,110.4 | 0* | 200 | OK |
| 123 | tiered(CPU=500) | 274.6 | 1,683.0 | 4,462.6 | 0* | 200 | OK |
| 456 | single-tier | 338.8 | 2,165.3 | 6,275.5 | 0* | 200 | OK |
| 456 | tiered(CPU=500) | 326.4 | 1,987.6 | 6,237.5 | 0* | 200 | OK |

*\* Reported as 0 due to analyzer bug. Actual preemption rates are in Experiment 4.*

TTFT improvement across 3 seeds: 3.7-28.2% (mean), 8.2-29.8% (P99). The tiered configuration consistently improves TTFT.

### Experiment 2: CPU Tier Size Scaling (seed=42)

| CPU Blocks | TTFT Mean | TTFT P99 | E2E P99 | Completed |
|-----------:|----------:|---------:|---------:|----------:|
| 0 (single) | 417.2 | 2,305.0 | 6,193.9 | 200 |
| 100 | 299.4 | 1,615.1 | 5,531.6 | 200 |
| 250 | 299.4 | 1,616.8 | 5,532.9 | 200 |
| 500 | 299.4 | 1,616.8 | 5,532.9 | 200 |
| 1,000 | 299.4 | 1,616.8 | 5,532.9 | 200 |

**Threshold effect:** 100 CPU blocks = full benefit. Outputs for 250, 500, 1,000 are byte-identical.

### Experiment 4: Confound Matrix with Corrected Metrics (seed=42)

This experiment was added in Round 3 to isolate the mechanism, then re-analyzed in Round 4 using the KV cache summary section (not the per-instance JSON that `analyze.py` was incorrectly parsing).

**Analyzer bug:** The original `analyze.py` used regex `r"Preemptions?: (\d+)"` which does not match the actual output `"Preemption Rate: 0.1750"` (wrong field name + integer pattern for a float). This caused 0 preemptions to be reported for all configurations in Experiments 1-3 and the initial Experiment 4 run. The corrected extraction reads `"Preemption Rate"` and `"Cache Hit Rate"` from the KV cache summary section printed by `cmd/root.go:544-546`.

**Corrected confound matrix (seed=42, standalone runs bypassing `run_sim` wrapper):**

| Config | TTFT Mean | TTFT P99 | Preemption Rate | Cache Hit Rate |
|---|---:|---:|---:|---:|
| single-tier (CPU=0) | 417.2 | 2,305.0 | **17.5%** | **0.51%** |
| tiered (CPU=500, offload=0.8) | 299.4 | 1,616.8 | **8.5%** | **4.52%** |
| tiered (CPU=500, offload=1.0) | 417.2 | 2,305.0 | **17.5%** | **0.50%** |

**Key results:**
- **Preemptions DO occur** (17.5% in single-tier) and **tiered DOES halve them** (→ 8.5%)
- **Cache hit rate INCREASES 9x** (0.51% → 4.52%) with tiered cache
- **Control (offload=1.0)** produces output byte-identical to single-tier — confirming `maybeOffload` as the sole mechanism
- **Routing policy is irrelevant** at this operating point — round-robin and least-loaded produce identical TTFT (both 417.2ms single-tier, both 299.4ms tiered)

**Note on primary vs corrected data:** The TTFT values are identical between Experiments 1-3 and the corrected Experiment 4 (e.g., 417.2ms for single-tier seed=42 in both). The discrepancy is only in the preemption and cache hit metrics, which were incorrectly parsed by the original analyzer.

## Root Cause Analysis

**The complete mechanism (RCV-1 compliant, file:line cited):**

1. **Preemptions occur in single-tier** (17.5%): At 2100 GPU blocks with rate=2000 and gaussian input mean=512, the GPU cache fills up. When `makeRunningBatch` (`simulator.go:449-570`) calls `AllocateKVBlocks` and GPU is full, `preempt()` (`simulator.go:408-446`) evicts running requests via LRU. Evicted blocks lose their prefix hashes permanently.

2. **`maybeOffload` preserves prefix hashes on CPU**: When `ReleaseKVBlocks` is called (`kvcache_tiered.go:149-151`), `maybeOffload()` (lines 199-230) checks if GPU utilization > 0.8. If so, it copies prefix hash entries from GPU free blocks to the CPU tier (`kvcache_tiered.go:214-219` creates the `offloadedBlock`) and removes the hash from GPU's `HashToBlock` map (line 224).

3. **`tryReloadFromCPU` restores hashes when GPU allocation fails**: When `AllocateKVBlocks` fails on GPU, the tiered cache calls `tryReloadFromCPU` (`kvcache_tiered.go:95-143`, called at line 70) which brings prefix hashes back from CPU to GPU, making blocks available as cached prefix blocks instead of requiring full recomputation.

4. **Cache hit rate increases 9x** (0.51% → 4.52%): Prefix hashes preserved on CPU are reloaded when the same prefix recurs, producing more cached blocks via `GetCachedBlocks` (`simulator.go:521`). More cached blocks → lower `numNewTokens` → shorter prefill → lower TTFT.

5. **Preemption rate halves** (17.5% → 8.5%): Better cache reuse means fewer blocks need fresh allocation, reducing GPU pressure.

6. **Control confirms** (offload=1.0 = byte-identical to single-tier): `maybeOffload` is the sole differentiating mechanism.

**Threshold effect:** Once ANY CPU tier exists, `maybeOffload` activates. The CPU tier size only matters if it fills up (`t.cpu.used >= t.cpu.capacity` guard at `kvcache_tiered.go:205`). With 100+ CPU blocks, the offloaded hashes fit easily.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might not be:**
The confirmation rests on a single-seed confound matrix (seed=42), not the 3-seed primary experiment. The primary 3-seed experiment's preemption data was corrupt (analyzer bug), so the directional claim (tiered reduces preemptions) is only demonstrated for one seed. Additionally, the TTFT improvement varies substantially across seeds (3.7% to 28.2%), suggesting the effect magnitude is operating-point-sensitive. At seed=456, the improvement is barely above the "significant" threshold.

**If this were "Inconclusive," argue why it IS Confirmed:**
The control experiment (offload=1.0 → byte-identical to single-tier) definitively proves that `maybeOffload` is the mechanism. The TTFT values match between the primary experiment and the confound matrix (417.2ms in both), confirming the data is consistent. The mechanism is verified against code with file:line citations.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Tiered cache reduces preemptions 17.5% → 8.5% and TTFT by 28% | Confirmation | Hypothesis confirmed via corrected confound matrix |
| `maybeOffload` confirmed as sole mechanism (offload=1.0 = byte-identical to single-tier) | Confirmation | Control experiment (RCV-4) |
| Cache hit rate increases 9x (0.51% → 4.52%) via prefix hash preservation on CPU | Confirmation | Resolves mechanism direction |
| Threshold effect: 100 CPU blocks = full benefit | Confirmation | `maybeOffload` saturates with minimal CPU capacity |
| Analyzer bug: `analyze.py` regex `Preemptions?:` didn't match actual `Preemption Rate:` output | Bug (experiment) | Fixed; added to code review checklist as canonical example |
| INV-1 (conservation) holds across all configurations | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found.** All configurations complete 200/200 requests with INV-1 satisfied.
- [x] Any new rules needed? **Candidate: Silent analyzer defaults should emit warnings.** The `analyze.py` regex failure was silent, defaulting to 0 without warning. Added to code review checklist.
- [x] Any new invariants needed? **None** — existing INV-1 and INV-4 cover the relevant properties.
- [x] Any existing rules/invariants confirmed? **INV-1 confirmed** across all configurations. **INV-4 implied** (no allocation failures in completed runs).

## Scope and Limitations (RCV-6)

- **Operating point tested:** GPU=2100 blocks, block_size=16 tokens, rate=2000, 200 requests, 4 instances
- **Routing:** Primary experiments used least-loaded; confound matrix tested both round-robin and least-loaded (identical results at this operating point)
- **Seeds:** 3 seeds for TTFT comparison (Exp 1); single seed (42) for preemption/cache-hit metrics (Exp 4 confound matrix)
- **Parameters findings depend on:** Operating near the preemption threshold; sufficient prefix reuse in workload; `maybeOffload` threshold < 1.0
- **What was NOT tested:** Different GPU block counts; bandwidth sensitivity under actual preemption conditions; seeds 123/456 for preemption metrics; workloads with low prefix reuse
- **Generalizability:** TTFT improvement demonstrated across 3 seeds (3.7-28.2%). Preemption reduction and cache hit increase demonstrated for seed=42 only. Effect magnitude is operating-point-sensitive.
- **Uncertainty quantification:** Not performed. The 28% TTFT improvement is a point estimate at seed=42; the range across seeds is 3.7-28.2%.

## Evidence Quality

| Metric | Value | Confidence |
|---|---|---|
| TTFT improvement | 3.7-28.2% across 3 seeds | High — consistent direction, variable magnitude |
| Preemption reduction | 51% (17.5%→8.5%) at seed=42 | Medium — single seed, corrected data |
| Cache hit rate increase | 9x (0.51%→4.52%) at seed=42 | Medium — single seed, corrected data |
| Mechanism | `maybeOffload` prefix hash preservation | High — byte-identical control confirms |
| Analyzer bug | Regex mismatch masked preemptions for 2 rounds | Confirmed — root cause identified and fixed |

## Implications for Users

1. **Tiered KV cache reduces preemptions and improves TTFT near the preemption cliff.** At GPU=2100 blocks, adding 500 CPU blocks with offload threshold 0.8 halves preemptions (17.5%→8.5%) and improves TTFT by 3.7-28.2% depending on seed.

2. **The mechanism is prefix hash preservation.** `maybeOffload` copies prefix hashes to CPU before LRU eviction destroys them. When the same prefix recurs, `tryReloadFromCPU` restores the hash, enabling cache reuse (9x higher cache hit rate).

3. **100 CPU blocks = full benefit.** The offloaded hash entries are small. Over-provisioning has no additional effect.

4. **Bandwidth tuning was not validated under preemption conditions.** Experiments 1-3 showed zero bandwidth effect, but the corrected data shows preemptions ARE occurring. A future experiment should sweep bandwidth under preemption conditions.

## Reproducing

```bash
cd hypotheses/h10-tiered-kv
./run.sh
```
