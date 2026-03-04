# Problem Statement: BLIS Simulator Wall-Clock Performance

**Issue:** #484
**Family:** Structural Model (implementation performance)
**Date:** 2026-03-04

---

## Baseline

**Configuration:**
- Model: `meta-llama/llama-3.1-8b-instruct`
- Instances: 4
- Routing: weighted (prefix-affinity:1.0, load-balance:1.0)
- Scheduler: FCFS
- Admission: always-admit
- Priority: constant

**Measured baseline (3 workloads):**

| Workload | Requests | Rate | Wall-clock |
|----------|----------|------|------------|
| cache_warmup | 5000 | 1000 | 9.3s |
| load_spikes | 5000 | 1000 | 7.7s |
| multiturn | ~150 | 150 | 0.25s |
| **Total** | | | **17.2s** |

**Platform:** Apple M1 Max, Go 1.23+, macOS

## Target Workload

The three `workload-mert/` YAML files as-is. They stress prefix-affinity routing
(cache_warmup has 3 prefix groups, load_spikes has a 50% heavy-hitter prefix),
which exercises the prefix cache index and hash computation hot paths.

## Quantitative Success Criteria

- **Primary:** >50% reduction in total wall-clock time (17s → <8.5s)
- **Secondary:** >70% reduction in the two dominant workloads (cache_warmup + load_spikes)
- **Hard constraint:** INV-6 determinism preserved — byte-identical stdout for same seed
- **Hard constraint:** `go test ./...` passes, golden dataset unchanged

## Prior Knowledge (from CPU profiling)

| Rank | Hotspot | CPU % | Location | Root Cause |
|------|---------|-------|----------|------------|
| 1 | `lruBlockCache.evictOldest` | 46% | `sim/prefix_cache_index.go:117-129` | O(n) map scan per eviction |
| 2 | `hash.HashBlock` (strconv.Itoa) | 22% | `sim/internal/hash/hash.go:37-45` | String conversion per token |
| 3 | `KVCacheState.AllocateKVBlocks` | 21% | `sim/kv/cache.go` | Unknown — needs investigation |
| 4 | `VLLMBatchFormation.preemptForTokens` | 9% | `sim/batch_formation.go` | Unknown — needs investigation |

**Key insight:** Hotspots 1+2 account for **68%** of CPU. Both are in the
prefix-affinity routing path, which is only active when prefix-affinity scorer
is enabled. Workloads without prefix-affinity should already be fast.
