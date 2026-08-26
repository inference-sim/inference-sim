package kv

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
)

// cpuTestKey turns an int into a distinct BlockKey for tests.
func cpuTestKey(i int) kvkey.BlockKey { return kvkey.BlockKey(fmt.Sprintf("k%08d", i)) }

func keys(base, n int) []kvkey.BlockKey {
	out := make([]kvkey.BlockKey, n)
	for i := 0; i < n; i++ {
		out[i] = cpuTestKey(base + i)
	}
	return out
}

// BC-C3: the ref_cnt state machine. -1 = allocated-not-ready (HIT_PENDING,
// unreadable, unevictable); 0 = ready+evictable (HIT); n>0 = ready+pinned (HIT,
// not evictable). Only completeStore(true) performs -1 -> 0; completeStore(false)
// removes+frees.
func TestOffloadCPUTier_RefCntTransitions(t *testing.T) {
	c := newOffloadCPUTier(8)
	k := cpuTestKey(1)

	// A GPU->CPU store lands a READY block (content already computed on GPU).
	if ok := c.store(k); !ok {
		t.Fatalf("store into empty tier must succeed")
	}
	if c.lookup(k) != cpuHit {
		t.Fatalf("stored block must be a readable HIT, got %v", c.lookup(k))
	}
	if c.evictableCount() != 1 {
		t.Fatalf("ready-idle block must be evictable, evictable=%d", c.evictableCount())
	}

	// prepareStore for a promotion allocates a NOT-READY block (-1).
	p := cpuTestKey(2)
	if granted := c.prepareStore([]kvkey.BlockKey{p}); granted != 1 {
		t.Fatalf("prepareStore should grant 1, got %d", granted)
	}
	if c.lookup(p) != cpuHitPending {
		t.Fatalf("promotion-allocated block must be HIT_PENDING (-1), got %v", c.lookup(p))
	}
	if c.evictableCount() != 1 {
		t.Fatalf("a -1 block is NOT evictable; evictable must stay 1, got %d", c.evictableCount())
	}

	// completeStore(true): -1 -> 0, becomes readable+evictable.
	c.completeStore(p, true)
	if c.lookup(p) != cpuHit {
		t.Fatalf("after completeStore(true) block must be a HIT")
	}
	if c.evictableCount() != 2 {
		t.Fatalf("completeStore(true) must make the block evictable, got %d", c.evictableCount())
	}

	// pin (0 -> 1): readable but NOT evictable.
	c.pin(p)
	if c.lookup(p) != cpuHit {
		t.Fatalf("pinned block (ref_cnt>0) must STILL be a readable HIT")
	}
	if c.evictableCount() != 1 {
		t.Fatalf("pin must drop evictable to 1, got %d", c.evictableCount())
	}
	// pin again (1 -> 2): no counter change (BC-C8: only zero-crossings move it).
	c.pin(p)
	if c.evictableCount() != 1 {
		t.Fatalf("second pin must not change evictable, got %d", c.evictableCount())
	}
	// unpin (2 -> 1): still no change.
	c.unpin(p)
	if c.evictableCount() != 1 {
		t.Fatalf("unpin to 1 must not change evictable, got %d", c.evictableCount())
	}
	// unpin (1 -> 0): re-evictable.
	c.unpin(p)
	if c.evictableCount() != 2 {
		t.Fatalf("unpin to 0 must restore evictable to 2, got %d", c.evictableCount())
	}

	// completeStore(false): a failed promotion removes+frees the block.
	q := cpuTestKey(3)
	c.prepareStore([]kvkey.BlockKey{q})
	c.completeStore(q, false)
	if c.lookup(q) != cpuMiss {
		t.Fatalf("failed promotion must remove the block (MISS), got %v", c.lookup(q))
	}
}

// BC-C5 (the trap) as a DIRECT correct-vs-mutant boundary assertion. Gate:
// promotion of k blocks succeeds iff k <= free + evictable. A mutant that gates on
// the FREE count (fail iff k-free > free, i.e. k > 2*free) would grant k=4 here
// where the correct gate fails.
func TestOffloadCPUTier_PrepareStoreEvictableGate(t *testing.T) {
	build := func() *offloadCPUTier {
		c := newOffloadCPUTier(10)
		for i := 0; i < 8; i++ { // 8 ready blocks, 2 free
			if !c.store(cpuTestKey(i)) {
				t.Fatalf("store %d failed", i)
			}
		}
		for i := 0; i < 7; i++ { // pin 7 -> evictable=1
			c.pin(cpuTestKey(i))
		}
		if c.freeCount() != 2 || c.evictableCount() != 1 {
			t.Fatalf("precondition free=2 evictable=1, got free=%d evictable=%d", c.freeCount(), c.evictableCount())
		}
		return c
	}

	// k=3: 3 <= free(2)+evictable(1)=3 -> SUCCEED (evict 1).
	if g := build().prepareStore(keys(100, 3)); g != 3 {
		t.Fatalf("k=3 must be granted (k<=free+evictable), got %d", g)
	}
	// k=4: 4 > free(2)+evictable(1)=3 -> FAIL (grant 0). A free-count mutant
	// (fail iff k>2*free=4) grants 4 here. THIS is the mutation guard.
	if g := build().prepareStore(keys(200, 4)); g != 0 {
		t.Fatalf("k=4 must FAIL under the evictable gate (BC-C5); a free-count mutant grants it. got %d", g)
	}
}

// BC-C4 + I3: eviction never selects a block with ref_cnt != 0, and store returns
// false when no evictable victim exists (so MirrorToCPU must skip, not force-evict
// or drop silently).
func TestOffloadCPUTier_NeverEvictsPinned(t *testing.T) {
	c := newOffloadCPUTier(2)
	c.store(cpuTestKey(0))
	c.store(cpuTestKey(1))
	c.pin(cpuTestKey(0))
	c.pin(cpuTestKey(1)) // both pinned, tier full, zero evictable
	if ok := c.store(cpuTestKey(2)); ok {
		t.Fatalf("store must FAIL when tier is full and all blocks pinned (BC-C4)")
	}
	if c.lookup(cpuTestKey(0)) != cpuHit || c.lookup(cpuTestKey(1)) != cpuHit {
		t.Fatalf("pinned blocks must survive a failed store")
	}
}

// BC-C2 + BC-C8: over a random op sequence, used+free==capacity always, and the
// maintained O(1) evictable counter always equals a reference scan of ref_cnt==0.
func TestOffloadCPUTier_ConservationAndEvictableCounter(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const capacity = 16
	c := newOffloadCPUTier(capacity)
	for step := 0; step < 5000; step++ {
		k := cpuTestKey(rng.Intn(24)) // small key space -> collisions exercise dedup
		switch rng.Intn(5) {
		case 0: // store (ready)
			if c.lookup(k) == cpuMiss {
				c.store(k)
			}
		case 1: // prepareStore (promotion, -1)
			if c.lookup(k) == cpuMiss {
				c.prepareStore([]kvkey.BlockKey{k})
			}
		case 2: // completeStore(true)
			if c.lookup(k) == cpuHitPending {
				c.completeStore(k, true)
			}
		case 3: // pin a ready block
			if c.lookup(k) == cpuHit {
				c.pin(k)
			}
		case 4: // unpin
			c.unpin(k)
		}
		if c.usedCount()+c.freeCount() != capacity {
			t.Fatalf("step %d: conservation broken used=%d free=%d cap=%d", step, c.usedCount(), c.freeCount(), capacity)
		}
		if got, want := c.evictableCount(), c.scanEvictable(); got != want {
			t.Fatalf("step %d: evictable counter=%d != scan=%d", step, got, want)
		}
		if c.usedCount() < 0 || c.freeCount() < 0 {
			t.Fatalf("step %d: negative counts used=%d free=%d", step, c.usedCount(), c.freeCount())
		}
	}
}
