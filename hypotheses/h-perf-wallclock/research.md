# Research: BLIS Wall-Clock Performance Optimization

## Problem Statement

**Issue:** #484
**Family:** Structural Model (implementation performance)

**Baseline:** 17.2s total wall-clock for 3 workloads (cache_warmup 9.3s, load_spikes 7.7s, multiturn 0.25s) on Apple M1 Max with prefix-affinity routing enabled.

**Target:** >50% reduction (17s to <8.5s), INV-6 determinism preserved.

**Prior profiling:** Four hotspots -- LRU eviction (46%), token hashing (22%), KV allocation (21%), batch formation/preemption (9%). Hotspots 1+2 are both on the prefix-affinity routing path.

---

# Background

**Status: completed**

## Architecture of the Hot Path

The performance bottleneck chain flows through the prefix-affinity routing scorer. For every request routed through the cluster:

1. **`RoutingDecisionEvent.Execute`** (`sim/cluster/cluster_event.go:148`) calls `cs.routingPolicy.Route(req, state)`
2. **`WeightedScoring.Route`** (`sim/routing.go:155`) iterates over all scorers, including the prefix-affinity scorer
3. **Prefix-affinity scorer** (`sim/routing_prefix_scorer.go:23-39`) does TWO expensive operations per `Route()` call:
   - **Scoring phase** (line 28): `idx.ComputeBlockHashes(req.InputTokens)` -- hashes ALL input tokens into blocks
   - **Observer phase** (line 45): `idx.ComputeBlockHashes(req.InputTokens)` AGAIN -- identical computation repeated for the observer callback
4. **`RecordBlocks`** (`sim/prefix_cache_index.go:76-88`) calls `touch(h)` for each block hash, which may trigger `evictOldest()` per insertion

### Hotspot 1: O(n) LRU Eviction (46% CPU)

`lruBlockCache.evictOldest()` (`sim/prefix_cache_index.go:117-129`) scans the ENTIRE `hashes` map to find the minimum timestamp. With `defaultLRUCapacity = 10000` blocks per instance, every eviction requires iterating 10,000 map entries. For a request with B blocks being recorded, if the cache is at capacity, eviction happens B times per `RecordBlocks` call, giving O(B * capacity) per request.

The data structure is a `map[string]int64` (hash to timestamp). There is no secondary index for finding the minimum -- it's a full linear scan every time.

### Hotspot 2: Token Hashing (22% CPU)

`hash.HashBlock` (`sim/internal/hash/hash.go:37-45`) converts each token integer to a string via `strconv.Itoa(t)`, writes it to a SHA256 hasher, and hex-encodes the result. For each block of `blockSize` tokens (default 16), this means 16 `strconv.Itoa` calls plus SHA256 computation plus hex encoding.

Additionally, `hash.ComputeBlockHashes` is called TWICE per routing decision -- once during scoring and once during the observer callback (`sim/routing_prefix_scorer.go:28` and `:45`). This is pure redundant computation.

In the KV cache layer (`sim/kv/cache.go`), `hash.HashTokens` uses a different (non-hierarchical) hashing scheme that hashes the ENTIRE token prefix for each block check. `GetCachedBlocks` (line 128-139) calls `HashTokens(tokens[:(i+1)*blockSizeTokens])` for block i, giving O(n^2) total token processing for n blocks. `AllocateKVBlocks` has similar patterns at lines 212 and 257.

### Hotspot 3: KV Allocation (21% CPU)

`KVCacheState.AllocateKVBlocks` (`sim/kv/cache.go:146-270`) is called from batch formation (`sim/batch_formation.go`) for every request scheduled into the running batch. The cost comes from: (a) the HashTokens calls within it (shared with hotspot 2), (b) linked-list manipulation for the free list, and (c) the per-block allocation loop with rollback tracking.

### Hotspot 4: Batch Formation (9% CPU)

`VLLMBatchFormation.preemptForTokens` (`sim/batch_formation.go:149-183`) runs a retry loop: if KV allocation fails, it evicts from the batch tail and retries. Each retry involves `ReleaseKVBlocks` + `AllocateKVBlocks`. Under memory pressure, this loop may execute multiple times per step.

### Call Frequency Analysis

For 5000 requests with prefix-affinity routing and 4 instances:
- `Route()` is called 5000 times (once per request)
- `ComputeBlockHashes` is called 10,000 times (2x per route -- scorer + observer)
- For a request with 256 input tokens at block_size=16 = 16 blocks: 160,000 `HashBlock` calls, each with 16 `strconv.Itoa` conversions = 2,560,000 string conversions
- `RecordBlocks` with 16 blocks at near-capacity cache triggers up to 16 evictions, each scanning 10,000 entries = 160,000 map iterations per request, 800 million iterations total

---

# Idea 1: Replace O(n) LRU Map Scan with Doubly-Linked List + Map

**Status: completed**

## Description

Replace the `lruBlockCache` implementation from a flat `map[string]int64` with the classic O(1) LRU data structure: a `map[string]*lruNode` for O(1) lookup combined with a doubly-linked list for O(1) eviction and O(1) touch (move-to-front).

Current implementation (`sim/prefix_cache_index.go:100-129`):
```go
type lruBlockCache struct {
    hashes   map[string]int64 // block hash -> access timestamp
    capacity int
    clock    int64
}

func (c *lruBlockCache) evictOldest() {
    var oldestHash string
    oldestTime := int64(math.MaxInt64)
    for h, t := range c.hashes {      // O(n) scan!
        if t < oldestTime {
            oldestTime = t
            oldestHash = h
        }
    }
    if oldestHash != "" {
        delete(c.hashes, oldestHash)
    }
}
```

Proposed replacement:
```go
type lruNode struct {
    hash       string
    prev, next *lruNode
}

type lruBlockCache struct {
    lookup   map[string]*lruNode  // O(1) lookup
    head     *lruNode             // most recently used
    tail     *lruNode             // least recently used (eviction target)
    capacity int
    size     int
}
```

- `touch(hash)`: If exists, remove node from list and move to head (O(1)). If new, create node at head, evict tail if at capacity (O(1)).
- `evictOldest()`: Remove tail node, delete from map (O(1)).

## Expected Impact

This is the #1 hotspot at 46% of CPU. The improvement is from O(capacity) per eviction to O(1). With capacity=10,000 and frequent evictions (cache stays near capacity for prefix-heavy workloads), the speedup on this path alone is approximately 10,000x per eviction operation.

**Estimated wall-clock improvement:** The 46% CPU share would drop to near-zero for this specific function. However, the remaining work in `touch()` (map lookups, pointer manipulation) and the other hotspots remain. Conservative estimate: **30-40% total wall-clock reduction** from this change alone.

## Implementation Complexity

**Low.** This is a textbook data structure replacement. The LRU cache is entirely internal to `prefix_cache_index.go` (unexported `lruBlockCache` type). No API changes needed -- `RecordBlocks`, `MatchLength`, and `InstanceBlockCount` all keep the same signatures.

Files to modify:
- `sim/prefix_cache_index.go` -- replace `lruBlockCache` internals

Tests to update:
- `sim/prefix_cache_index_test.go` -- existing tests should pass unchanged (they test behavior, not internals)

Estimated effort: ~50 lines of Go, 1-2 hours.

## Self-Critique

- **Determinism (INV-6):** The doubly-linked list approach is fully deterministic. Eviction always removes the tail, `touch()` always moves to head. No map iteration order dependency. Determinism is PRESERVED.
- **Memory overhead:** Each node adds two pointers (16 bytes on 64-bit). For 10,000 entries x 4 instances = 40,000 nodes = 640KB additional. Negligible.
- **Correctness risk:** Low. The LRU linked-list is a well-understood data structure. The monotonic clock field becomes unnecessary and can be removed.
- **Potential issue:** If `MatchLength` needs to check membership, the map lookup is unchanged (O(1)). No regression there.

## Risks

- **Minimal.** This is a contained refactor of an internal data structure with comprehensive test coverage. The behavioral contract (LRU eviction, bounded capacity) is unchanged.
- **One subtlety:** The current implementation has no tie-breaking issue because timestamps are unique (monotonic clock). The linked-list approach inherently orders by recency, so this is a non-issue.

---

# Idea 2: Eliminate Redundant Hashing and Replace SHA256 with a Faster Hash

**Status: completed**

## Description

This idea has three sub-optimizations targeting the #2 hotspot (22% CPU):

### 2A: Eliminate Double ComputeBlockHashes Call

The prefix-affinity scorer computes block hashes TWICE per routing decision:

```go
// sim/routing_prefix_scorer.go
scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
    hashes := idx.ComputeBlockHashes(req.InputTokens)  // FIRST call (line 28)
    // ... use hashes for scoring ...
}

observer := func(req *Request, targetInstance string) {
    hashes := idx.ComputeBlockHashes(req.InputTokens)  // SECOND call (line 45) -- identical!
    idx.RecordBlocks(hashes, targetInstance)
}
```

Fix: Cache the hashes from the scorer call and reuse in the observer. Since both the scorer and observer are closures over the same `PrefixCacheIndex`, add a `lastHashes` field or use a shared variable:

```go
var cachedHashes []string
var cachedReqID string

scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
    cachedHashes = idx.ComputeBlockHashes(req.InputTokens)
    cachedReqID = req.ID
    // ... use cachedHashes ...
}

observer := func(req *Request, targetInstance string) {
    hashes := cachedHashes
    if req.ID != cachedReqID {
        hashes = idx.ComputeBlockHashes(req.InputTokens)  // fallback
    }
    idx.RecordBlocks(hashes, targetInstance)
}
```

This halves the number of `ComputeBlockHashes` calls.

### 2B: Replace strconv.Itoa with Binary Encoding

`HashBlock` (`sim/internal/hash/hash.go:37-45`) converts each token to a string via `strconv.Itoa`. This allocates a string per token. Instead, write the token's raw 8-byte binary representation directly:

```go
func HashBlock(prevHash string, tokens []int) string {
    h := sha256.New()
    h.Write([]byte(prevHash))
    var buf [8]byte
    for _, t := range tokens {
        binary.LittleEndian.PutUint64(buf[:], uint64(t))
        h.Write(buf[:])
    }
    return hex.EncodeToString(h.Sum(nil))
}
```

This eliminates all `strconv.Itoa` allocations and produces a fixed 8-byte write per token instead of variable-length string conversion.

### 2C: Replace SHA256 with a Faster Non-Cryptographic Hash

SHA256 is cryptographically secure but overkill for prefix matching. Alternatives:
- **FNV-128** (stdlib `hash/fnv`): ~5-10x faster than SHA256, 128-bit output
- **xxhash** (external): ~20-50x faster, 64-bit or 128-bit
- **maphash** (stdlib `hash/maphash`): ~10-20x faster, 64-bit, process-local seed

For prefix cache matching, we need: (a) determinism across runs (INV-6), (b) low collision probability for ~10K distinct block hashes. FNV-128 from the standard library satisfies both requirements without external dependencies.

**Important:** `hash.HashTokens` (used by `sim/kv/cache.go`) must produce the same hashes as before OR both systems must be updated together. Since `HashTokens` and `HashBlock/ComputeBlockHashes` are separate functions used by different subsystems, they can be changed independently as long as each is internally consistent.

## Expected Impact

- **2A (eliminate double hash):** 50% reduction in `ComputeBlockHashes` calls. Estimated: **~11% total wall-clock reduction** (half of the 22% hotspot).
- **2B (binary encoding):** Eliminates `strconv.Itoa` allocation overhead. Estimated: **~3-5% additional reduction** (strconv.Itoa is a significant fraction of the hash hotspot based on profiling).
- **2C (faster hash function):** SHA256 -> FNV-128 is roughly 10x faster for small inputs. Estimated: **~5-8% additional reduction** (the remaining hash computation after 2A+2B).

**Combined estimate:** ~15-20% total wall-clock reduction.

## Implementation Complexity

- **2A: Low.** Closure-local variable caching. ~10 lines changed in `sim/routing_prefix_scorer.go`.
- **2B: Low.** Replace `strconv.Itoa` with `binary.LittleEndian.PutUint64`. ~5 lines in `sim/internal/hash/hash.go`. BUT: changes hash output format, requiring golden dataset regeneration (R12).
- **2C: Medium.** Requires changing the hash function in `sim/internal/hash/hash.go` and ensuring no downstream code depends on the specific SHA256 output format (e.g., hash string length). Also requires golden dataset regeneration.

## Self-Critique

- **INV-6 (determinism):** 2A preserves determinism (same hashes computed, just cached). 2B changes hash output but is deterministic (binary encoding is platform-consistent with LittleEndian). 2C with FNV-128 is deterministic (pure function, no randomness). All three preserve INV-6.
- **Hash collisions (2C):** FNV-128 has 128 bits of output. With ~10K block hashes per instance x 4 instances = 40K hashes, collision probability is approximately 40000^2 / 2^128 ~= 4.7 x 10^-30. Negligible.
- **Golden dataset (2B, 2C):** Changing hash output changes KV cache hit patterns, which changes simulation output. The golden dataset MUST be regenerated (R12). This is a known cost but not a blocker.
- **Cross-system consistency:** `HashTokens` (KV cache) and `HashBlock/ComputeBlockHashes` (prefix cache index) use different hashing approaches -- `HashTokens` hashes the full cumulative prefix while `HashBlock` chains previous block hashes. They can be changed independently since they're not compared against each other.
- **Risk of 2A:** If the scorer is ever called without a subsequent observer call (or vice versa), the cached hashes could be stale. This would be a correctness bug. However, in `WeightedScoring.Route()`, the observer is always called immediately after scoring (`sim/routing.go:190-192`), so this is safe in practice. A defensive check (comparing `req.ID`) mitigates this.

## Risks

- **Golden dataset regeneration** for 2B/2C adds review overhead
- **External dependency** if using xxhash (mitigated by using stdlib FNV-128)
- **2B changes the hash format** from human-readable to binary -- debugging prefix cache issues becomes slightly harder (mitigated by hex encoding of the final hash)

---

# Idea 3: Algorithmic Optimizations Beyond Data Structure Swaps

**Status: completed**

## Description

Three sub-ideas that address performance at the algorithmic level:

### 3A: Lazy/Incremental Block Hash Computation

Currently, `ComputeBlockHashes` recomputes ALL block hashes from scratch every time. For requests in the same prefix group (which is the entire point of prefix-affinity routing), most blocks are identical across requests.

**Approach:** Add a per-request hash cache on the `Request` struct:

```go
type Request struct {
    // ... existing fields ...
    cachedBlockHashes []string  // computed once, reused
}
```

In `ComputeBlockHashes`, check if `req.cachedBlockHashes` is already populated:
- If populated and `len(req.InputTokens)` hasn't changed: return cached hashes
- Otherwise: compute and cache

Since requests are routed exactly once and their `InputTokens` never change after creation, this cache is always valid. Combined with Idea 2A (eliminate double call), this means block hashes are computed ONCE per request lifetime instead of twice per routing decision.

**Alternative (more aggressive):** Maintain a global hash memo `map[*int][]string` keyed by the backing array pointer of `InputTokens`. Requests sharing the same prefix tokens (from the same prefix group) would share the same slice backing array and thus the same hashes. However, this depends on workload generation internals and is fragile.

### 3B: Short-Circuit Prefix-Affinity Scoring When It Cannot Affect the Decision

When the prefix-affinity scorer's weight is low relative to other scorers, or when all instances have the same match length (0 or equal), the prefix-affinity score doesn't change the routing decision. We can detect this cheaply:

1. **Early exit for zero-block requests:** If `len(InputTokens) < blockSize`, there are no blocks to hash. Return 0.0 for all instances immediately. (This is already handled by the `totalBlocks == 0` check at line 31, but the `ComputeBlockHashes` call on line 28 still runs.)

2. **Early exit for cold cache:** If the prefix cache index is empty for all instances (first few requests), skip the scoring entirely and return 0.0 for all.

3. **Weight-based skip:** If the maximum possible prefix-affinity contribution (weight * 1.0) is less than the gap between the top two non-prefix scores, prefix-affinity cannot change the winner. This requires a two-pass approach (score other dimensions first, then conditionally score prefix-affinity) which changes the scorer pipeline contract.

### 3C: Reduce Hash Operations in KV Cache Layer

`GetCachedBlocks` in `sim/kv/cache.go:128-139` uses `HashTokens` which hashes the FULL prefix up to each block boundary. For block i, it hashes `tokens[:i*blockSize]`. This is O(B^2) in total tokens processed (where B = number of blocks), because block 1 hashes `blockSize` tokens, block 2 hashes `2*blockSize`, etc.

**Fix:** Switch `GetCachedBlocks` to use hierarchical block hashing (`ComputeBlockHashes` from `sim/internal/hash/hash.go:50-64`) which is O(B * blockSize) -- each block hashes only its own `blockSize` tokens plus a constant-size previous hash. This requires changing the hash format stored in `KVBlock.Hash` from flat prefix hashes to chained block hashes.

Similarly, `AllocateKVBlocks` at lines 212 and 257 calls `HashTokens(fullTokens)` and `HashTokens(fullPrefix)` which hash growing prefixes. Switching these to `HashBlock(prevHash, blockTokens)` would reduce from O(B * blockSize * B) to O(B * blockSize) total.

**Caveat:** This changes the KV cache's hash scheme from flat prefix hashing to hierarchical block hashing, unifying it with the router-side prefix cache index. This is a larger refactor but eliminates a fundamental O(n^2) complexity.

### 3D: Batch Block Hash Computation in ComputeBlockHashes

Rather than calling `sha256.New()` per block in `ComputeBlockHashes`, reuse a single hasher with `Reset()`:

```go
func ComputeBlockHashes(blockSize int, tokens []int) []string {
    numBlocks := len(tokens) / blockSize
    if numBlocks == 0 {
        return nil
    }
    hashes := make([]string, numBlocks)
    h := sha256.New()  // allocate once
    prevHash := ""
    for i := 0; i < numBlocks; i++ {
        h.Reset()      // reuse
        h.Write([]byte(prevHash))
        // ... write tokens ...
        hashes[i] = hex.EncodeToString(h.Sum(nil))
        prevHash = hashes[i]
    }
    return hashes
}
```

This avoids allocating a new SHA256 hasher per block (each allocation involves internal state initialization).

## Expected Impact

- **3A (request-level hash caching):** Eliminates redundant computation for repeated routing of similar requests. With Idea 2A, this reduces `ComputeBlockHashes` from 2x to 1x per request. Estimated: **~5-8% wall-clock reduction** (primarily helps when combined with 2A -- together they go from 2 calls to 0 recomputation).
- **3B (short-circuit scoring):** Depends on workload. For the first N requests (where N = number of instances x LRU capacity / blocks per request), the cache is cold and short-circuit applies. For the benchmark workloads with 5000 requests and 4 instances, the cache warms up quickly, so the benefit is modest. Estimated: **~1-3% wall-clock reduction**.
- **3C (unify KV hash scheme):** Eliminates O(B^2) complexity in `GetCachedBlocks` and `AllocateKVBlocks`. For requests with 16 blocks, this reduces from ~136 block-hash computations (sum 1..16) to 16. For the 21% KV allocation hotspot, estimated: **~10-15% wall-clock reduction** from this path.
- **3D (hasher reuse):** Minor optimization. SHA256 internal state is 112 bytes. Estimated: **~1-2% wall-clock reduction**.

## Self-Critique

- **3A correctness:** Request `InputTokens` must be immutable after creation. Inspecting `sim/request.go` and the workload generation pipeline, tokens are set at creation and never modified during simulation. Safe.
- **3A adds a field to Request:** Per R4 (construction site audit), must grep all `Request{}` literal constructions and ensure the new field is zero-valued by default (which it is -- nil slice). No construction site updates needed since `cachedBlockHashes` is set lazily.
- **3B changes the scorer pipeline:** The weight-based skip (3B.3) requires knowing other scorers' results first, which breaks the current parallel-evaluation model. Sub-ideas 3B.1 and 3B.2 are safe and simple.
- **3C is a significant refactor:** Changing the KV cache hash scheme from flat prefix to hierarchical block hashing affects `GetCachedBlocks`, `AllocateKVBlocks`, and all tests that verify hash values. This is the highest-risk change but also addresses a genuine algorithmic complexity issue (O(B^2) to O(B)).
- **3C changes simulation output:** Different hash scheme means different cache hit patterns, requiring golden dataset regeneration.
- **3D is trivial but low-impact:** Hasher allocation is cheap relative to the actual SHA256 computation. Only worth doing if we're already touching the hash code.

## Risks

- **3A:** Minimal risk. Lazy caching with immutable inputs.
- **3B:** Low risk for sub-ideas 1 and 2. High complexity for sub-idea 3 (cross-scorer dependency).
- **3C:** Highest risk -- changes fundamental KV cache behavior. Requires careful testing of cache hit rates and prefix matching correctness. Must verify that the KV cache and router-side prefix cache index agree on block identity.
- **3D:** Negligible risk.

---

# Executive Summary

**Status: completed**

## Comparison Table

| Idea | Target Hotspot | Expected Impact | Complexity | Risk | INV-6 Safe | Golden Regen |
|------|---------------|----------------|------------|------|------------|--------------|
| **1: O(1) LRU** | #1 (46% CPU) | 30-40% | Low | Low | Yes | No |
| **2A: Eliminate double hash** | #2 (22% CPU) | ~11% | Low | Low | Yes | No |
| **2B: Binary token encoding** | #2 (22% CPU) | ~3-5% | Low | Low | Yes | Yes |
| **2C: FNV-128 hash** | #2 (22% CPU) | ~5-8% | Medium | Low | Yes | Yes |
| **3A: Request hash caching** | #2 (22% CPU) | ~5-8% | Low | Low | Yes | No |
| **3B: Short-circuit scoring** | #2 (22% CPU) | ~1-3% | Low | Low | Yes | No |
| **3C: Unify KV hash scheme** | #3 (21% CPU) | ~10-15% | High | Medium | Yes | Yes |
| **3D: Hasher reuse** | #2 (22% CPU) | ~1-2% | Low | Low | Yes | No |

## Ranking (by impact-to-risk ratio)

1. **Idea 1: O(1) LRU** -- Highest impact (30-40%), lowest risk, minimal code change. Clear winner.
2. **Idea 2A: Eliminate double hash** -- Second highest impact-to-effort ratio. ~10 lines changed, ~11% improvement.
3. **Idea 3A: Request hash caching** -- Synergizes with 2A. Together they eliminate nearly all redundant hash computation.
4. **Idea 2B: Binary encoding** -- Small win, small risk, but requires golden regen.
5. **Idea 3C: Unify KV hash scheme** -- High impact (10-15%) but high complexity and risk. Should be a separate follow-up PR.
6. **Idea 2C: FNV-128 hash** -- Good speedup but changes hash format across the board. Bundle with 2B if doing golden regen anyway.
7. **Idea 3B: Short-circuit scoring** -- Low impact, only worth it if other changes are insufficient.
8. **Idea 3D: Hasher reuse** -- Trivial to implement but negligible impact. Include as polish.

## Recommended Hypothesis Bundle

**Phase 1 (PR-A): No golden regen required, high confidence**
- Idea 1: O(1) LRU linked list
- Idea 2A: Eliminate double ComputeBlockHashes call
- Idea 3A: Request-level hash caching
- Idea 3D: Hasher reuse in ComputeBlockHashes

Expected combined impact: **40-50% wall-clock reduction** (17s to ~8.5-10s). These four changes are independent, low-risk, and preserve the existing golden dataset. They can be implemented and validated in a single PR.

**Phase 2 (PR-B): Golden regen required, higher complexity**
- Idea 2B: Binary token encoding
- Idea 2C: FNV-128 hash function
- Idea 3C: Unify KV cache hash scheme (O(B^2) to O(B))

Expected additional impact: **15-25% further reduction** (bringing total from ~9s to ~6-7s). These require golden dataset regeneration and more careful testing.

## Proposed Hypotheses

**H-perf-1 (Structural Model):** Replacing the O(n) map-scan LRU eviction in `lruBlockCache.evictOldest()` with an O(1) doubly-linked-list LRU will reduce the `evictOldest` CPU share from 46% to <1% and total wall-clock time by at least 30%, while preserving INV-6 determinism (byte-identical stdout for same seed).

**H-perf-2 (Structural Model):** Eliminating the redundant `ComputeBlockHashes` call in the prefix-affinity observer (by caching hashes from the scorer phase) and adding per-request hash memoization will reduce `ComputeBlockHashes` invocations from 2x per routing decision to effectively 1x per request lifetime, reducing the hash-related CPU share from 22% to <5% and total wall-clock by at least 10%.

**H-perf-3 (Structural Model):** Combining Phase 1 optimizations (O(1) LRU + hash deduplication + hasher reuse) will achieve the >50% wall-clock reduction target (17s to <8.5s) on the benchmark workloads without requiring golden dataset regeneration or changing simulation output.
