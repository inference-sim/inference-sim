package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
)

// cpuLookupResult is the readability state of a key in the CPU offload tier. It
// mirrors the readable subset of vLLM's LookupResult (base.py): a block is either
// absent, present-and-readable, or present-but-write-in-flight. The RETRY state
// (async secondary existence check) is H3 (#1591) and is not modeled here.
type cpuLookupResult int

const (
	cpuMiss        cpuLookupResult = iota // absent
	cpuHit                                // present, ref_cnt >= 0 (readable)
	cpuHitPending                         // present, ref_cnt == -1 (promotion write in flight)
)

// offloadCPUBlock is one content-addressed block in the CPU staging tier. ref_cnt
// is a STATE, not merely a count (vLLM BlockStatus, cpu/policies/base.py):
//
//	-1  allocated-not-ready: a promotion write is in flight. Unreadable
//	    (HIT_PENDING) AND unevictable. Off the LRU list.
//	 0  ready-idle: readable (HIT) AND evictable. On the LRU list.
//	 n  ready with n in-flight readers / cascade-writers: readable (HIT) but NOT
//	    evictable. Off the LRU list.
//
// The block stores no token payload: content is addressed by key, and every
// CPU->GPU reload is driven by a request that already holds the matching prefix
// tokens (content-addressing guarantees they equal the key), so the tier only
// needs to track presence + ref_cnt.
type offloadCPUBlock struct {
	key    kvkey.BlockKey
	refCnt int
	prev   *offloadCPUBlock // LRU: older (evict first). Non-nil only when ref_cnt==0.
	next   *offloadCPUBlock // LRU: newer.
}

// offloadCPUTier is the CPU staging tier: a capacity-bounded (in blocks),
// content-addressed store with vLLM's ref_cnt state machine and an O(1) evictable
// counter (BC-C8). The LRU list holds exactly the ready-idle (ref_cnt==0) blocks,
// so eviction is O(1) and never selects a locked block (BC-C4); the evictable
// counter is maintained only at ref_cnt transitions that cross zero.
type offloadCPUTier struct {
	blocks    map[kvkey.BlockKey]*offloadCPUBlock
	lruHead   *offloadCPUBlock // oldest evictable (evict first)
	lruTail   *offloadCPUBlock // newest evictable
	capacity  int64
	used      int64 // == len(blocks): every allocated block (ref_cnt -1, 0, or n)
	evictable int64 // == count of ref_cnt==0 blocks (== LRU list length); maintained O(1)
	evictions int64 // total blocks evicted (diagnostic)
}

func newOffloadCPUTier(capacity int64) *offloadCPUTier {
	if capacity <= 0 {
		panic(fmt.Sprintf("newOffloadCPUTier: capacity must be > 0, got %d", capacity))
	}
	return &offloadCPUTier{blocks: make(map[kvkey.BlockKey]*offloadCPUBlock), capacity: capacity}
}

// lookup reports the readability state of key (BC-C3). ref_cnt>=0 is a readable
// HIT (reads are allowed while a cascade write is in flight); only ref_cnt==-1 is
// HIT_PENDING.
func (t *offloadCPUTier) lookup(key kvkey.BlockKey) cpuLookupResult {
	blk, ok := t.blocks[key]
	if !ok {
		return cpuMiss
	}
	if blk.refCnt == -1 {
		return cpuHitPending
	}
	return cpuHit
}

// store lands a READY (ref_cnt==0) block copied from GPU (the MirrorToCPU path;
// the content is already computed on GPU). If key is already present it is a
// recency touch. If the tier is full it evicts one evictable victim; when none
// exists (all blocks locked) it returns false so the caller SKIPS the mirror+
// cascade — never force-evicting a locked block (BC-C4) or dropping data
// silently (R1).
func (t *offloadCPUTier) store(key kvkey.BlockKey) bool {
	if blk, ok := t.blocks[key]; ok {
		if blk.refCnt == 0 {
			t.touch(blk)
		}
		return true
	}
	if t.used >= t.capacity {
		if t.evict(1) == 0 {
			return false // full and no evictable victim
		}
	}
	blk := &offloadCPUBlock{key: key, refCnt: 0}
	t.blocks[key] = blk
	t.appendToTail(blk)
	t.used++
	t.evictable++
	return true
}

// prepareStore allocates NOT-READY (ref_cnt==-1) slots for a promotion of the
// given keys (the secondary->CPU read). It is all-or-nothing over the fresh
// (not-already-present) keys: it succeeds iff they fit under the EVICTABLE gate
// (BC-C5) — k <= free + evictable — evicting exactly k-free evictable victims.
// It returns the number of slots granted (0 = the whole promotion is refused ->
// the caller recomputes). Because the LRU holds only evictable blocks, the count
// gate is the sole decision and the subsequent evict always succeeds within it;
// a mutant that gates on the FREE count instead grants promotions this refuses
// (the mandatory mutation is caught by TestOffloadCPUTier_PrepareStoreEvictableGate).
func (t *offloadCPUTier) prepareStore(reqKeys []kvkey.BlockKey) int {
	fresh := make([]kvkey.BlockKey, 0, len(reqKeys))
	for _, k := range reqKeys {
		if _, present := t.blocks[k]; !present {
			fresh = append(fresh, k)
		}
	}
	k := int64(len(fresh))
	if k == 0 {
		return 0
	}
	free := t.capacity - t.used
	if k > free+t.evictable { // BC-C5: evictable gate, NOT free gate
		return 0
	}
	if toEvict := k - free; toEvict > 0 {
		t.evict(toEvict)
	}
	for _, key := range fresh {
		t.blocks[key] = &offloadCPUBlock{key: key, refCnt: -1} // not ready, off LRU
		t.used++
	}
	return len(fresh)
}

// completeStore resolves an in-flight promotion write (ref_cnt==-1). On success
// the block transitions -1 -> 0 (readable + evictable). On failure it is removed
// and freed (vLLM cpu/manager.py complete_store) — it never becomes readable.
func (t *offloadCPUTier) completeStore(key kvkey.BlockKey, ok bool) {
	blk := t.blocks[key]
	if blk == nil || blk.refCnt != -1 {
		return // defensive: only a -1 block completes a store
	}
	if ok {
		blk.refCnt = 0
		t.appendToTail(blk)
		t.evictable++
	} else {
		delete(t.blocks, key)
		t.used-- // a -1 block is off the LRU, so no unlink / evictable change
	}
}

// pin marks a ready block as having one more in-flight reader / cascade-writer
// (ref_cnt++). The evictable counter changes ONLY on the 0->1 crossing (BC-C8);
// pinning an already-pinned block does not touch it. Not-ready (-1) blocks are
// never pinned.
func (t *offloadCPUTier) pin(key kvkey.BlockKey) {
	blk := t.blocks[key]
	if blk == nil || blk.refCnt < 0 {
		return
	}
	if blk.refCnt == 0 {
		t.unlink(blk)
		t.evictable--
	}
	blk.refCnt++
}

// unpin drops one in-flight reader / cascade-writer (ref_cnt--). It re-adds the
// block to the LRU and bumps evictable ONLY on the 1->0 crossing. Unpinning a
// block that is absent or not pinned is a safe no-op.
func (t *offloadCPUTier) unpin(key kvkey.BlockKey) {
	blk := t.blocks[key]
	if blk == nil || blk.refCnt <= 0 {
		return
	}
	blk.refCnt--
	if blk.refCnt == 0 {
		t.appendToTail(blk)
		t.evictable++
	}
}

// evict removes up to n oldest ready-idle blocks from the LRU head and returns
// how many it removed. It never touches a locked block (they are off the list).
func (t *offloadCPUTier) evict(n int64) int64 {
	var cnt int64
	for cnt < n && t.lruHead != nil {
		victim := t.lruHead
		t.unlink(victim)
		delete(t.blocks, victim.key)
		t.used--
		t.evictable--
		t.evictions++
		cnt++
	}
	return cnt
}

func (t *offloadCPUTier) touch(blk *offloadCPUBlock) {
	t.unlink(blk)
	t.appendToTail(blk)
}

func (t *offloadCPUTier) appendToTail(blk *offloadCPUBlock) {
	blk.next = nil
	blk.prev = t.lruTail
	if t.lruTail != nil {
		t.lruTail.next = blk
	} else {
		t.lruHead = blk
	}
	t.lruTail = blk
}

func (t *offloadCPUTier) unlink(blk *offloadCPUBlock) {
	if blk.prev != nil {
		blk.prev.next = blk.next
	} else if t.lruHead == blk {
		t.lruHead = blk.next
	}
	if blk.next != nil {
		blk.next.prev = blk.prev
	} else if t.lruTail == blk {
		t.lruTail = blk.prev
	}
	blk.prev = nil
	blk.next = nil
}

func (t *offloadCPUTier) evictableCount() int64 { return t.evictable }
func (t *offloadCPUTier) freeCount() int64       { return t.capacity - t.used }
func (t *offloadCPUTier) usedCount() int64       { return t.used }

// scanEvictable is an O(n) reference count of ready-idle blocks, used by tests to
// validate the maintained O(1) evictable counter (BC-C8). Not used at runtime.
func (t *offloadCPUTier) scanEvictable() int64 {
	var n int64
	for _, blk := range t.blocks {
		if blk.refCnt == 0 {
			n++
		}
	}
	return n
}
