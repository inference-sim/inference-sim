# Design: Check-Then-Act KV Cache Allocation (vLLM Alignment)

**Type:** Decision Record
**Status:** Draft
**Closes:** #1061
**Replaces:** PR #1065

## Problem

`rollbackAllocation` in `sim/kv/cache.go` unconditionally deletes `RequestMap[reqID]` when a decode allocation fails. This orphans all previously allocated blocks for continuing requests, causing `UsedBlockCnt` to drift to `TotalBlocks` with no request holding those blocks — a permanent deadlock in FormBatch Phase 2.

Two confirmed leak paths:
1. **`preemptForTokens` retry:** First `AllocateKVBlocks` fails → rollback deletes RequestMap → eviction frees blocks → retry succeeds but only tracks 1 new block, orphaning all prior blocks.
2. **`processCompletions` final-token failure:** `ReleaseKVBlocks` after the failed allocation finds an empty RequestMap and releases nothing.

## Decision

Adopt vLLM's check-then-act allocation pattern. Eliminate rollback entirely.

### What vLLM Does

vLLM's `allocate_slots` (`kv_cache_manager.py:325-336`) separates allocation into two phases:

1. **Pure query:** `get_num_blocks_to_allocate()` computes how many new blocks are needed by inspecting `req_to_blocks` and `block.ref_cnt` — zero state mutations.
2. **Gate:** `if num_blocks_to_allocate > block_pool.get_num_free_blocks(): return None` — rejects before any mutation.
3. **Commit:** `allocate_new_blocks()` runs only after the gate passes. Post-gate failure raises `ValueError` (defensive panic), not a recoverable condition.

vLLM maintains a **direct counter** (`FreeKVCacheBlockQueue.num_free_blocks`) on the free list, incremented/decremented in lockstep with `append`/`remove`/`popleft`. It never derives free count from arithmetic.

### What BLIS Will Do

Mirror the same pattern:

1. **Universal pre-check:** Before any state mutation in `AllocateKVBlocks`, compute the number of new blocks needed for both prefill and decode paths. Compare against free capacity. Return `false` immediately if insufficient.

2. **Remove rollback:** Delete `rollbackAllocation`, `cachedBlockMutation`, `newBlockMutation`, and `prependToFreeList`. Convert `popFreeBlock() == nil` after a passing pre-check into a `panic` (INV-4 invariant violation — structurally unreachable in single-threaded DES).

3. **Direct free block counter:** Replace `UsedBlockCnt` with `FreeBlockCnt`, maintained by `appendToFreeList` (increment) and `removeFromFreeList` (decrement). `countFreeBlocks()` returns `FreeBlockCnt` directly. `UsedBlocks()` derives as `TotalBlocks - FreeBlockCnt` for callers.

4. **Conservation assertion:** Add `verifyBlockConservation()` for debug-mode step-boundary assertions (from PR #1065).

## Code Proofs: Behavioral Equivalence with vLLM

### Proof 1: Universal Pre-Check

**vLLM** (`single_type_kv_cache_manager.py:69-128`):
```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, ...):
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
    if request_id in self.num_cached_block:
        # Running request (decode): simple delta
        return max(num_required_blocks - num_req_blocks, 0)
    # New request (prefill): full calculation
    ...
```

**BLIS equivalent:**

| Case | vLLM calculation | BLIS calculation |
|------|-----------------|------------------|
| Decode (running) | `max(cdiv(existing_tokens+1, block_size) - len(req_blocks), 0)` → 0 or 1 | Last block full? → 1. Has spare? → 0. |
| Prefill (new) | `cdiv(num_tokens, block_size) - max(skipped, cached)` | `ceil(effectiveTokens / BlockSizeTokens)` after spare-capacity adjustment (existing `cache.go:207-217`) |

Both return a pure integer count without touching state. Both gate against a free capacity check before any mutation.

### Proof 2: No Rollback Mechanism

**vLLM** (`block_pool.py:299-311`):
```python
def get_new_blocks(self, num_blocks):
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)  # Defensive panic, structurally unreachable
    ret = self.free_block_queue.popleft_n(num_blocks)
```

**BLIS equivalent:**
```go
blk := kvc.popFreeBlock()
if blk == nil {
    panic("popFreeBlock nil after pre-check: INV-4 violation")
}
```

Both treat post-pre-check allocation failure as an invariant violation, not a recoverable condition. Neither has a rollback path.

### Proof 3: Preemption Loop Preserves Request State

**vLLM** (`scheduler.py:812-863`): `allocate_slots` returns `None` → `req_to_blocks` is untouched → evict a request → retry → request's existing blocks are still tracked.

**BLIS:** `AllocateKVBlocks` returns `false` via pre-check → `RequestMap[reqID]` is untouched → `preemptForTokens` evicts tail request → retry → request's existing blocks are still tracked.

The #1061 bug: rollback deleted `RequestMap[reqID]`, so the retry lost all prior blocks. With pre-check, no mutations happen on failure, so `RequestMap` is preserved.

### Proof 4: processCompletions Safety

**vLLM:** `free()` pops `req_to_blocks[request_id]` and frees all blocks. No preceding operation can empty the map.

**BLIS after refactor:** `AllocateKVBlocks` fails via pre-check (no state mutation) → `ReleaseKVBlocks` finds full `RequestMap[reqID]` → frees all blocks correctly. The incorrect comment at `simulator.go:688-690` ("AllocateKVBlocks only modifies RequestMap on success") becomes provably true.

### Proof 5: Direct Free Counter Alignment

**vLLM** (`kv_cache_utils.py:156-344`):
```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)     # init
    def popleft(self):
        self.num_free_blocks -= 1              # remove
    def append(self, block):
        self.num_free_blocks += 1              # add
    def remove(self, block):
        self.num_free_blocks -= 1              # mid-list remove
```

**BLIS equivalent:**
```go
type KVCacheState struct {
    FreeBlockCnt int64  // Direct counter, replaces UsedBlockCnt
}
func (kvc *KVCacheState) appendToFreeList(block *KVBlock) {
    kvc.FreeBlockCnt++  // add
}
func (kvc *KVCacheState) removeFromFreeList(block *KVBlock) {
    kvc.FreeBlockCnt--  // remove (covers popFreeBlock via removeFromFreeList)
}
func (kvc *KVCacheState) countFreeBlocks() int64 {
    return kvc.FreeBlockCnt  // direct read, not arithmetic
}
```

Both maintain the counter in lockstep with list mutations. No arithmetic derivation.

## Scope

### In Scope
- Universal pre-check in `AllocateKVBlocks` (prefill already has one; add decode)
- Remove `rollbackAllocation`, `cachedBlockMutation`, `newBlockMutation`, `prependToFreeList`
- Replace `UsedBlockCnt` with `FreeBlockCnt` (direct counter on free list)
- `verifyBlockConservation()` for debug-mode assertions
- Fix incorrect comment at `simulator.go:688-690`
- Replace rollback-testing tests with behavioral check-then-act tests

### Out of Scope
- Splitting `AllocateKVBlocks` into separate query/commit methods (Approach C — YAGNI)
- Changing the `KVStore` interface contract
- Tiered cache structural changes (it delegates to `gpu.AllocateKVBlocks` which gets the fix automatically)

## Invariants Affected

- **INV-4 (KV cache conservation):** Strengthened. `FreeBlockCnt + sum(len(RequestMap[r]))` must equal `TotalBlocks`. `verifyBlockConservation` asserts this.
- **INV-8 (Work-conserving):** Preserved. Pre-check returns `false` exactly when rollback would have, so `FormBatch` flow is unchanged.

## Risks

- **Tiered cache:** `commitCachedBlocks` (`cache.go:437`) operates outside rollback tracking. This is safe today per the comment at lines 428-436 (single-threaded DES guarantee). With rollback removed, this code path becomes simpler — it's just a direct commit, no longer an exception to a rollback regime.
- **Test churn:** Several tests verify rollback behavior specifically. These become obsolete and are replaced with check-then-act behavioral tests.
