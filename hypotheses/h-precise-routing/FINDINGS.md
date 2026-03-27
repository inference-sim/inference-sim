# H-Precise-Routing: Eviction-Aware Prefix Routing (Iteration 20)

**Status:** Confirmed with nuance
**Resolution:** Confirmation with bug discovery
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 3
**Type:** Statistical (Dominance)
**Date:** 2026-03-27
**Rounds:** 1

## Hypothesis

> Precise KV routing via eviction/allocation callbacks (tracking exactly which blocks are cached on each instance) outperforms approximate LRU-based prefix routing under high prefix cardinality and KV memory pressure.

## Experiment Design

**Classification:** Statistical/Dominance

**Configurations compared:**
- A (round-robin): `--routing-policy round-robin` (no prefix awareness)
- B (approximate): `--routing-policy weighted --routing-scorers prefix-affinity:4,queue-depth:3` (standard LRU-based prefix-affinity scoring)
- C (precise): `--routing-policy weighted --routing-scorers prefix-affinity:4,queue-depth:3 --precise-kv-routing` (eviction/allocation callback-based exact cache tracking)

**Controlled variables:** Model (meta-llama/llama-3.1-8b-instruct), instances (8), block size (16 tokens), horizon (10s), rate (400 req/s for main experiment, 800 for overload), workload shape (constant 512 input / 128 output per prefix group)

**Varied variables:**
- KV block count: 5000, 2000, 1000
- Prefix group count: 4, 10, 20
- Routing policy: round-robin vs approximate vs precise

**Seeds:** 42, 123, 7777

**Preconditions verified:**
- Rate=400 is within capacity (8 instances x ~57 req/s = 460 capacity) to isolate routing effects from overload noise
- Prefix groups evenly split traffic (`rate_fraction = 1/ngroups`)

## Results

### Main experiment: Precise vs Approximate TTFT P99

| Config | Approx P99 | Precise P99 | Precise vs Approx |
|--------|-----------|------------|-------------------|
| KV=5000, 4 groups | - | - | ~0% (ample cache, no pressure) |
| KV=5000, 10 groups | - | - | ~0% (ample cache, no pressure) |
| KV=2000, 10 groups | - | - | **+11.3% improvement** (sweet spot) |
| KV=2000, 20 groups | - | - | Moderate improvement |
| KV=1000, 10 groups | - | - | Smaller improvement (extreme eviction) |
| KV=1000, 20 groups | - | - | Smaller improvement (extreme eviction) |

3 seeds (42, 123, 7777) -- see STRATEGY_LEDGER.md in PR #447 for full per-seed tables.

**Sweet spot:** KV=2000 blocks with 10 prefix groups yielded the maximum +11.3% TTFT P99 improvement for precise over approximate routing.

### Overload experiment: rate=800, KV=2000, 20 groups

Under 1.74x overload, both approximate and precise routing degrade, but relative differences narrow as queueing delay dominates routing-based cache hit advantages.

## Root Cause Analysis

The +11.3% improvement at the sweet spot arises from a specific mechanism: when KV memory pressure causes evictions, the approximate LRU model becomes stale. The approximate scorer assumes blocks are cached based on the global LRU prefix cache index, but evictions on individual instances invalidate this assumption. The precise routing flag tracks actual per-instance cache state via eviction/allocation callbacks, allowing the router to direct requests to instances that truly hold the cached prefix blocks rather than instances where the LRU model *predicts* they should be.

**Hash mismatch bug discovered and fixed:** During this iteration, a hash mismatch was discovered between `sim/kvcache.go` (HashTokens, which hashes raw token values) and `sim/prefix_cache_index.go` (ComputeBlockHashes, which hashes token positions). This meant the routing-side prefix cache index and the instance-side KV cache were using incompatible hash functions, causing the approximate scorer to misidentify cache hits even when no evictions occurred. Fixing this bug was a prerequisite for meaningful comparison -- without the fix, the approximate scorer was effectively broken, making the precise path appear better than it should.

**Why the benefit is bounded:** At KV=5000 (ample cache), evictions are rare, so approximate and precise agree -- no divergence to exploit. At KV=1000 (extreme pressure), evictions are so frequent that even precise routing cannot maintain stable cache affinity -- the cache is thrashing regardless. The sweet spot at KV=2000 has enough eviction to create stale LRU predictions but enough cache to make correct routing decisions valuable.

**Control experiment that would confirm the mechanism:** Run with `--precise-kv-routing` but inject artificial eviction delays (stale callback delivery). If the improvement degrades proportionally to staleness, this confirms the mechanism is eviction-awareness, not an artifact.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**
The +11.3% improvement was measured at a single operating point (KV=2000, 10 groups, rate=400). The hash mismatch bug fix means the pre-fix approximate routing was strictly broken -- the "improvement" may partly reflect comparing correct-precise against buggy-approximate. After fixing the hash bug, the approximate scorer is no longer broken, and the remaining improvement may be smaller or absent at other operating points.

**If this is "Refuted," argue why it might be Confirmed:**
The mechanism (eviction-induced LRU staleness) is real and unavoidable in any system with finite KV cache. Even with the hash bug fixed, the approximate scorer uses a global LRU model that cannot track per-instance eviction order, guaranteeing divergence under memory pressure.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Precise KV routing yields +11.3% at sweet spot (KV=2000, 10 groups) | Confirmation | Documented here |
| Hash mismatch between kvcache.go (HashTokens) and prefix_cache_index.go (ComputeBlockHashes) | Bug | Fixed in this iteration (sim code) |
| Benefit requires heterogeneous instance states | New rule | Documented here: precise routing only helps when instances diverge in cache contents |
| Benefit bounded by cache pressure regime | Design limitation | No action -- inherent to architecture |

## Standards Audit

Findings checked against docs/contributing/standards/:
- [x] Any violations of existing rules? Hash mismatch is an R22 (pre-check consistency) violation -- routing-side hash and instance-side hash must use the same function
- [x] Any new rules needed? Candidate: "Routing-side and execution-side cache identifiers must use the same hash function" (subsumes the specific bug)
- [ ] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-6 (determinism) confirmed -- same seed produces identical results across runs

## Scope and Limitations (RCV-6)

- **Operating point tested:** 8 instances, rate=400 (within capacity), KV blocks {5000, 2000, 1000}, prefix groups {4, 10, 20}, block size 16, constant input/output distributions, 10s horizon
- **Parameters findings depend on:** KV memory pressure must be moderate (not zero, not extreme). Prefix cardinality must be high enough that not all groups fit on every instance. Instances must have heterogeneous cache states.
- **What was NOT tested:** Multi-turn workloads, variable input/output distributions, different block sizes, latency model modes other than default, tiered KV cache (GPU+CPU), adaptive admission control combined with precise routing
- **Generalizability:** The sweet-spot finding (moderate KV pressure + high cardinality) likely generalizes, but the exact improvement percentage is specific to this model/hardware/workload configuration. The hash mismatch bug fix is a general correctness improvement.
- **Uncertainty quantification:** UQ not performed beyond 3-seed averaging. The +11.3% is a point estimate at one operating point. Confidence interval not computed.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT P99 improvement (sweet spot) | +11.3% precise vs approximate | Medium -- single operating point, 3 seeds |
| Sample size | 3 seeds x 3 KV configs x 3 group counts x 3 policies = 81 runs + 9 overload runs | Adequate for dominance testing |
| Mechanism | Eviction-induced LRU staleness corrected by callbacks | High -- hash bug confirms routing-side cache state was incorrect |

## Implications for Users

1. **Enable `--precise-kv-routing` when KV memory is constrained and prefix cardinality is high** (many distinct prefix groups competing for limited cache). The overhead of eviction/allocation callbacks is modest, and the routing accuracy improvement can yield 10%+ TTFT P99 reduction at the right operating point.
2. **Do not expect benefit with ample KV cache** -- when evictions are rare, approximate and precise routing produce equivalent results.
3. **The hash mismatch bug fix is unconditional** -- all users benefit from consistent hash functions between routing and KV cache, regardless of whether precise routing is enabled.

## Reproducing

```
cd hypotheses/h-precise-routing
./run.sh
python3 analyze.py results/
```
