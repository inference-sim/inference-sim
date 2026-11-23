// Original to blis-vllm.

package kv

import (
	"runtime"
	"strings"
	"testing"
)

// Freeing a block that holds no reference is a caller's double free: vLLM's free_blocks
// assumes it never happens (vllm/v1/core/block_pool.py:774-805@5df63a72). The pool refuses it
// loudly rather than leaving a negative count and a block queued twice, and a normal free
// still returns every block to the free queue.
func TestPoolFree_RefusesADoubleFree(t *testing.T) {
	p := NewPool(4, true)
	before := p.NumFree()
	bs := p.New(2)
	p.Free(bs)
	if p.NumFree() != before {
		t.Fatalf("after freeing what it took the pool has %d free, want %d", p.NumFree(), before)
	}
	defer func() {
		r := recover()
		if r == nil || !strings.Contains(r.(string), "more times than its") {
			t.Fatalf("a double free: recovered %v, want a panic naming it", r)
		}
		if p.NumFree() != before {
			t.Errorf("a refused double free changed the free count to %d, want %d", p.NumFree(), before)
		}
	}()
	p.Free(bs[:1])
}

// A block listed twice in one call that holds one reference is refused before anything
// changes: the pool's free count and the block's reference count are as they were, so a
// refused call cannot leave a block unreferenced but missing from the free queue.
func TestPoolFree_RefusesADuplicateWithinOneCallAndChangesNothing(t *testing.T) {
	p := NewPool(4, true)
	bs := p.New(2)
	freeBefore, refBefore := p.NumFree(), bs[0].RefCnt
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(r.(string), "more times than its") {
				t.Fatalf("a duplicate in one call: recovered %v, want a panic naming it", r)
			}
		}()
		p.Free([]*Block{bs[0], bs[1], bs[0]})
	}()
	if p.NumFree() != freeBefore || bs[0].RefCnt != refBefore || bs[1].RefCnt != 1 {
		t.Fatalf("a refused call changed the pool: free %d (want %d), ref counts %d, %d (want %d, 1)",
			p.NumFree(), freeBefore, bs[0].RefCnt, bs[1].RefCnt, refBefore)
	}
	// The pool is still usable: freeing each once returns both blocks.
	p.Free(bs)
	if p.NumFree() != freeBefore+2 {
		t.Errorf("after a proper free the pool has %d free, want %d", p.NumFree(), freeBefore+2)
	}
}

// A block the queue does not hold, or one whose neighbours do not link back to it, is refused
// by remove rather than spliced: touching a block that is not free twice must not corrupt the
// queue's count.
func TestFreeQueueRemove_RefusesABlockItDoesNotHold(t *testing.T) {
	p := NewPool(4, true)
	b := p.New(1)[0]
	stray := &Block{ID: 99, prev: b, next: b}
	defer func() {
		if r := recover(); r == nil || !strings.Contains(r.(string), "not in the free queue") {
			t.Fatalf("removing a block the queue does not hold: recovered %v, want a panic", r)
		}
	}()
	p.free.remove(stray)
}

// A block consistently linked into another pool's free queue is not this queue's: remove
// refuses it rather than splicing the other queue and miscounting this one.
func TestFreeQueueRemove_RefusesABlockOfAnotherQueue(t *testing.T) {
	p, other := NewPool(4, true), NewPool(4, true)
	theirs := other.free.popLeft() // free and correctly linked, but in the other pool's queue
	other.free.pushBack(theirs)
	before, otherBefore := p.NumFree(), other.NumFree()
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(r.(string), "not in the free queue") {
				t.Fatalf("removing another queue's block: recovered %v, want a panic", r)
			}
		}()
		p.free.remove(theirs)
	}()
	if p.NumFree() != before || other.NumFree() != otherBefore {
		t.Errorf("a refused remove changed the counts: %d and %d, want %d and %d",
			p.NumFree(), other.NumFree(), before, otherBefore)
	}
}

// The free queue starts ordered by block ID and gives blocks back at its head or tail as vLLM's
// does; building a block when it is first handed out must not change the order any sequence of
// pops, prepends and appends yields, against a queue that holds every block from the start.
func TestFreeQueue_OrderIsThatOfAnEagerQueue(t *testing.T) {
	const n = 40
	var q freeQueue
	q.init(n)
	eager := make([]int, n)
	for i := range eager {
		eager[i] = i
	}
	var held []*Block
	seed := uint32(7)
	next := func(m int) int { seed = seed*1664525 + 1013904223; return int(seed>>16) % m }
	for step := 0; step < 400; step++ {
		switch op := next(3); {
		case op == 0 && len(eager) > 0:
			b := q.popLeft()
			if b.ID != eager[0] {
				t.Fatalf("step %d: popped block %d, want %d", step, b.ID, eager[0])
			}
			eager = eager[1:]
			held = append(held, b)
		case op == 1 && len(held) > 0:
			k := 1 + next(len(held))
			back := held[len(held)-k:]
			held = held[:len(held)-k]
			q.prependN(back)
			ids := make([]int, 0, k)
			for _, b := range back {
				ids = append(ids, b.ID)
			}
			eager = append(ids, eager...)
		case op == 2 && len(held) > 0:
			k := 1 + next(len(held))
			back := held[len(held)-k:]
			held = held[:len(held)-k]
			q.appendN(back)
			for _, b := range back {
				eager = append(eager, b.ID)
			}
		}
		if q.n != len(eager) {
			t.Fatalf("step %d: the queue counts %d free, want %d", step, q.n, len(eager))
		}
	}
	for len(eager) > 0 {
		if b := q.popLeft(); b.ID != eager[0] {
			t.Fatalf("draining: popped block %d, want %d", b.ID, eager[0])
		}
		eager = eager[1:]
	}
}

// A pool's cost is in the blocks used, not the blocks it has: a pool of millions of blocks (a
// large-memory GPU with a small KV footprint per token) costs far less than a block each, and a
// cluster of dozens of them is run.
func TestNewPool_BuildsBlocksWhenHandedOut(t *testing.T) {
	const n = 8_000_000
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	p := NewPool(n, true)
	runtime.ReadMemStats(&after)
	if p.NumFree() != n-1 || p.NumBlocks() != n {
		t.Fatalf("pool has %d free of %d, want %d of %d", p.NumFree(), p.NumBlocks(), n-1, n)
	}
	if grown := after.TotalAlloc - before.TotalAlloc; grown > 1<<20 {
		t.Fatalf("making a pool of %d blocks allocated %d bytes, want under 1 MiB", n, grown)
	}
	if got := p.New(1)[0].ID; got != 1 {
		t.Fatalf("first block handed out has ID %d, want 1", got)
	}
}
