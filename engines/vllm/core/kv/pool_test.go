// Original to blis-vllm.

package kv

import (
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
