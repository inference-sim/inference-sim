## BLIS PR Review — #1725

**Archon verdict: `ARCHITECTURAL_CHANGE`** (public surface rename on `sim/kv`: `ReloadedPrefixEnd` → `ReloadablePrefixEnd`; no new package edges, no dependency cycles). This is the expected shape for a followup that cleans up the interface semantics from #1699.

**CI signals:** build/test/lint: ✅ `success`. No archon plan (standalone fix). Dismissal state: `none`.

---

### Sub-issue contracts (issue #1706)

**C1 — pure pre-allocation query (`ReloadablePrefixEnd`), analogue of vLLM `get_num_new_matched_tokens`.**
**HOLDS.** `sim/kv_store.go` replaces the one-shot post-hoc `ReloadedPrefixEnd(reqID string)` with the pure pre-query `ReloadablePrefixEnd(req *Request, startIndex int64) (reloadableEnd int64, ok bool)`. The implementation in `offload_chain.go:222-256` and `tiered.go:392-429` walk the uncached tail counting CPU-resident blocks without calling `popFreeBlock`, `appendToFreeList`, or `cpu.touchKey`. The purity test is explicit: `TestOffload_ReloadablePrefixEnd_ReportsBoundary` ("pure CPU-resident prefix reports InputLen and is a pure repeatable query") asserts `oc.reloadCount == 0` after two calls, proving no GPU mutation occurred.

**C2 — fold-in before the clamp (vLLM ordering).**
**HOLDS.** `batch_formation.go:355-360` computes `computedStart = max(startIndex, reloadableEnd)` before any of the three clamps (`PrefillTokenThreshold`, `tokenBudget`, `MaxModelLen`), and `numNewTokens = InputLen - computedStart` is then subject to all three. The `endIndex = computedStart + numNewTokens`. Verified: `TestFormBatch_ReloadThenChunkedRemainder` (4-of-6 blocks reloadable, cap 16 → `NumNewTokens=16`, `ComputedTokens=80 = 64+16`) and `TestFormBatch_ReloadBeyondCappedChunk` (6-of-6 blocks reloadable, cap 16 → `NumNewTokens=0`, `ComputedTokens=64`).

**C3 — commit `[startIndex, computedStart)` + allocate tail.**
**HOLDS.** `batch_formation.go:379` calls `AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks)` with the original `startIndex` (not `computedStart`), so `allocateThroughChain` still runs the actual reload internally. Because `endIndex ≥ computedStart` (the reloaded boundary), `allocateThroughChain`'s path at line 421-429 fires the `newStart >= endIndex` branch which commits the full reloaded prefix `[startIndex, computedStart)` and then returns true. No structural test is needed here since the correctness flows through C2 (if `ComputedTokens = endIndex` and `NumNewTokens = numNewTokens`, the commit must have covered `[startIndex, computedStart)` — otherwise the request would not be admitted).

**C4 — progress and billing.**
**HOLDS.** `batch_formation.go:403-404`: `next.NumNewTokens = int(numNewTokens)` and `ctx.ComputedTokens[next.ID] = endIndex`. The old `billedTokens = endIndex - min(newStart, endIndex)` special-case is gone. `TestFormBatch_ReloadBeyondCappedChunk` now asserts `ComputedTokens["A"] == 64` (was `16` — the cap — under the old code), which is the discriminating assertion the issue requested.

**C5 — both offload paths fixed.**
**HOLDS.** `OffloadCache.ReloadablePrefixEnd` in `offload_chain.go:222` and `TieredKVCache.ReloadablePrefixEnd` in `tiered.go:392` both implement the pure query. Compile-time assertions (`var _ sim.ReloadReportingKVStore = (*OffloadCache)(nil)` and `(*TieredKVCache)(nil)`) are present in both files.

**INV-6 (byte-identity, offload off).**
**HOLDS.** `KVCacheState` does not implement `ReloadReportingKVStore` → `reloadReporter` is nil → `computedStart == startIndex` → `numNewTokens = InputLen - startIndex` (same as pre-#1699) → identical output. `TestFormBatch_NonReloadingStoreUnchanged` pins this.

**INV-13 (run/replay parity).**
**HOLDS.** The fold-in is a pure function of `(req, KVStore state)` — both deterministic, no RNG, no wall-clock. Replay shares the same `FormBatch` path.

---

### Ten code-review perspectives

**1. Substance & design — logic correctness.**
The key invariant is: `endIndex = computedStart + numNewTokens ≥ computedStart = reloadableEnd`, so `allocateThroughChain` always sees `newStart (= reloadableEnd after reload) ≥ endIndex` when the reload covers the full requested range, committing all reloaded blocks. This is sound.

One acknowledged edge: `ReloadablePrefixEnd` is unbounded (no `maxReloads`/`countFreeBlocks` cap), while `consultAndReload`/`reloadPrefixFromCPU` are bounded. If GPU free blocks are exhausted, the pure query may report `reloadableEnd > allocateThroughChain`'s actual new boundary, slightly over-crediting `computedStart`. The PR and CLAUDE.md document this explicitly as matching vLLM's model: "credit external, then allocation failure may preempt." This is an accepted design choice with documented rationale, not a hidden bug.

**2. Code quality & error handling.**
The cleanup is net-positive: `reloadedPrefixEnd map[string]int64` fields and the `recordReloadedPrefix` / `ClearDeferred`'s `delete(o.reloadedPrefixEnd, id)` machinery are removed from both stores, eliminating the one-shot stateful read-then-consume pattern (the source of leak risk in #1699). The `ReleaseKVBlocks` cleanup in both `offload_chain.go` and `tiered.go` similarly simplifies. No exported mutable maps introduced.

**3. Test behavioral quality.**
Tests are behavioral and vLLM-grounded:
- `TestFormBatch_ReloadBeyondCappedChunk`: the key discriminating test — old code asserted `ComputedTokens==16`; new code asserts `ComputedTokens==64`. This test fails on base (the old value), passes on the PR.
- `TestFormBatch_ReloadThenChunkedRemainder`: new test for the partial-reload+chunked case (4 of 6 blocks reloadable, cap 16).
- `TestOffload_ReloadablePrefixEnd_ReportsBoundary`: purity subtests cover full-prefix, partial, miss, running-continuation, non-zero-startIndex.
- `TestDeferral_ResolvedSecondaryBecomesReloadable`: discriminating secondary-path guard — proves a secondary-only prefix is NOT reported before deferral resolves (H3 fence), and IS reported after (the #1699 signal).
- `TestTieredKVCache_ReloadablePrefixEnd_ReportsBoundary`: legacy twin with purity + miss tests.

The INV-4 GPU conservation subtest from the old `TestOffload_ReloadedPrefixEnd_ReportsBoundary` ("request must own all 4 blocks after reload+tail alloc") was removed in the non-zero-startIndex subtest. This is a minor coverage gap: the new pure query test doesn't commit anything, so it can't verify block ownership. The block conservation check was useful as a defense-in-depth on `allocateThroughChain`. However, `allocateThroughChain` itself is unchanged (the removed `recordReloadedPrefix` call was the only diff), so INV-4 is still guarded by the existing chain tests. **Not a blocker.**

**4. Getting-started experience.**
CLAUDE.md is updated with the full #1706 narrative in the `#1699` block. The documentation is accurate and matches the implementation.

**5. Automated reviewer simulation.**
- No new exported mutable maps (R8).
- No new YAML fields.
- No NaN/Inf paths.
- `reloadCount == 0` assertion in the purity test catches a future implementer who accidentally mutates.
- The `prevHash` seeding in both `ReloadablePrefixEnd` implementations mirrors the `consultAndReload`/`reloadPrefixFromCPU` hash chain; any divergence would be caught by the existing block-hash consistency invariant.

**6. DES expert.**
Phase 2 of `FormBatch` is a synchronous read-then-admit loop (no events). The pure query runs in Phase 2's hot path, before `AllocateKVBlocks`. Since it is O(blocks) and adds one `GetCachedBlocks` call (GPU map lookup), this adds work proportional to the GPU-cached prefix length. For the typical case (short uncached tail), this is negligible. For a long fully-cached prefix (reloadableEnd = InputLen), both `GetCachedBlocks` and the walk over `n` blocks are called before the loop guard exits. This is the same complexity as `consultAndReload` itself, so the hot-path penalty is bounded by what already existed.

**7. vLLM/SGLang expert.**
The ordering mirrors vLLM `scheduler.py:1006-1044` as the issue specifies: `num_computed_tokens = local + external` first, then the chunk cap on `num_new_tokens = num_tokens - num_computed_tokens`. The secondary-tier-only exclusion (H3 deferral path) correctly maps to vLLM's `WAITING_FOR_REMOTE_KVS` path (`:1020-1023`). The `!running && !IsDecodeSubRequest` gate is correct: running continuations bill `ProgressIndex` (Phase 1), and PD decode sub-requests bypass Phase 2's new-prefill path entirely.

**8. Distributed inference platform expert.**
The feature is instance-local (single KVStore per instance, no cross-instance state). Cross-path parity (run/replay) holds via INV-13. `observe` dispatches to a live server with no KVStore → no change needed. No routing scorer is affected (`GetCachedBlocks`/`SnapshotCachedBlocksFn` stay GPU-only, unchanged).

**9. Performance & scalability.**
The pure query adds one O(uncached-prefix-length) walk per new admission per step when offload is enabled. The `DeriveChunkKeys` call is the dominant cost (SHA-256 per block); this matches what `consultAndReload` already does. For offload-off runs: `reloadReporter` is nil → zero extra work. No unbounded allocations.

**10. Security & robustness.**
- `startBlock >= n` early-return guards the case where `startIndex` is at or past the input end.
- The running-request gate (`o.gpu.RequestMap[req.ID]`) guards against double-billing a running continuation.
- `reloadableEnd > startIndex` is checked before returning `ok=true` (both implementations).
- No user-controlled panic paths introduced.

---

### Cross-path parity

This feature is `blis run`/`blis replay` symmetric (shared `FormBatch` + shared KVStore interface). `blis observe` dispatches to a live server with no simulator KVStore — parity is N/A and correctly excluded. No new CLI flags.

---

### Archon `ARCHITECTURAL_CHANGE` verdict

Archon flags the `sim/kv` surface change (method rename). No new package edges were added; the dependency graph is unchanged (`sim/kv → sim` via implements, as before). The rename is a clean correctness improvement: the old name (`ReloadedPrefixEnd`, past tense, post-hoc) described the one-shot consumed-record semantics; the new name (`ReloadablePrefixEnd`, present tense, pre-allocation) describes the pure query semantics. The `ARCHITECTURAL_CHANGE` verdict is expected for any interface-surface rename and does not indicate a structural concern here.

---

### Summary

All five issue contracts hold with corresponding tests. The discriminating test `TestFormBatch_ReloadBeyondCappedChunk` correctly fails on base and passes on the PR (the pre-#1706 assertion `ComputedTokens==16` becomes `ComputedTokens==64`). INV-4, INV-6, INV-13 hold. The over-credit edge under free-block exhaustion is documented and accepted. No blocking findings.

DELIVER-VERDICT: GREEN
