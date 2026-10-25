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
		if r == nil || !strings.Contains(r.(string), "holds no reference") {
			t.Fatalf("a double free: recovered %v, want a panic naming it", r)
		}
		if p.NumFree() != before {
			t.Errorf("a refused double free changed the free count to %d, want %d", p.NumFree(), before)
		}
	}()
	p.Free(bs[:1])
}
