// Original to blis-sglang.

package cache

import (
	"slices"
	"testing"
)

// TotalSize counts every token the tree holds, on every branch, and an empty tree holds none.
func TestTotalSizeCountsEveryBranch(t *testing.T) {
	r := newRig(t, 32, RadixConfig{})
	if n, _ := r.cache.TotalSize(); n != 0 {
		t.Errorf("an empty tree holds %d tokens", n)
	}
	for _, toks := range [][]int64{{1, 2, 3}, {1, 2, 9}, {7, 8}} {
		req := r.finished(toks, len(toks))
		r.cache.Unlock(req.Lock)
	}
	if n, _ := r.cache.TotalSize(); n != 3+1+2 {
		t.Errorf("the tree holds %d tokens, want 6 (the shared prefix once)", n)
	}
}

// Requests are ordered by a depth-first walk that visits the heavier subtree last-first as
// SGLang's DFS weight order does: handles on the same node keep their order, and an empty list is empty.
func TestDFSWeightOrderGroupsRequestsByTheirSubtree(t *testing.T) {
	r := newRig(t, 64, RadixConfig{})
	if got := r.cache.DFSWeightOrder(nil); len(got) != 0 {
		t.Errorf("no handles gave %v", got)
	}
	var handles []NodeID
	for _, toks := range [][]int64{{1, 2, 3}, {5, 6}, {1, 2, 4}, {1, 2, 3}} {
		req := r.finished(toks, len(toks))
		handles = append(handles, *req.LastNode)
		r.cache.Unlock(req.Lock)
	}
	got := r.cache.DFSWeightOrder(handles)
	if len(got) != 4 {
		t.Fatalf("order %v", got)
	}
	pos := func(i int) int { return slices.Index(got, i) }
	// Requests 0 and 3 end on one node and 0 and 2 share a subtree: they sit together, 1 apart.
	if abs(pos(0)-pos(3)) != 1 || max(pos(0), pos(2), pos(3))-min(pos(0), pos(2), pos(3)) != 2 || pos(0) > pos(3) {
		t.Errorf("order %v does not group the shared subtree and keep equal handles in order", got)
	}
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// LFU evicts the least-hit leaf first, then the least recently used among equals, then the
// lower id; SLRU shields nodes hit twice or more; LRU and FIFO order by access and creation.
func TestEvictionPoliciesOrderLeavesAsSGLangDoes(t *testing.T) {
	build := func(policy string) *radixRig {
		r := newRig(t, 64, RadixConfig{EvictionPolicy: policy})
		for _, toks := range [][]int64{{1}, {2}, {3}} {
			req := r.finished(toks, len(toks))
			r.cache.Unlock(req.Lock)
		}
		return r
	}
	hit := func(r *radixRig, tok int64, times int) {
		for range times {
			req := r.finished([]int64{tok}, 1) // an overlap: the node counts a hit
			r.cache.Unlock(req.Lock)
		}
	}
	remaining := func(r *radixRig) []int64 {
		var out []int64
		for _, tok := range []int64{1, 2, 3} {
			if r.cache.MatchPrefix(MatchPrefixParams{Key: Key{Tokens: []int64{tok}}}).DevicePrefixLen == 1 {
				out = append(out, tok)
			}
		}
		return out
	}
	// LRU: token 1 was touched first, so it goes first.
	r := build("lru")
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{2, 3}) {
		t.Errorf("lru left %v", got)
	}
	// LFU: token 1 is hit most, token 2 least among the rest by recency, so token 2 goes.
	r = build("lfu")
	hit(r, 1, 2)
	hit(r, 3, 1)
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{1, 3}) {
		t.Errorf("lfu left %v, want the never-hit token 2 gone", got)
	}
	// LFU with equal hits falls back to recency: 1 and 2 hit once, 1 earlier.
	r = build("lfu")
	hit(r, 1, 1)
	hit(r, 2, 1)
	hit(r, 3, 1)
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{2, 3}) {
		t.Errorf("lfu on equal hits left %v, want the least recently used gone", got)
	}
	// SLRU: tokens hit twice are protected; the oldest unprotected one goes.
	r = build("slru")
	hit(r, 1, 2)
	hit(r, 2, 2)
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{1, 2}) {
		t.Errorf("slru left %v, want the unprotected token 3 gone", got)
	}
	// FIFO and FILO order by creation, not by touches.
	r = build("fifo")
	hit(r, 1, 3)
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{2, 3}) {
		t.Errorf("fifo left %v", got)
	}
	r = build("filo")
	r.cache.Evict(EvictParams{NumTokens: 1})
	if got := remaining(r); !slices.Equal(got, []int64{1, 2}) {
		t.Errorf("filo left %v", got)
	}
}

// Matching reads page-aligned prefixes: a bigram key matches one pair fewer than its tokens, a
// page size above one rounds down to whole pages, and keys of other namespaces are refused.
func TestMatchAtPageAndBigramBoundaries(t *testing.T) {
	a := Key{Tokens: []int64{1, 2, 3, 4, 5}}
	b := Key{Tokens: []int64{1, 2, 3, 9, 9}}
	for _, c := range []struct{ page, want int }{{1, 3}, {2, 2}, {3, 3}, {4, 0}} {
		if got := a.matchAt(b, 0, c.page); got != c.want {
			t.Errorf("page %d matched %d, want %d", c.page, got, c.want)
		}
	}
	ba := Key{Tokens: []int64{1, 2, 3, 4, 5}, Bigram: true}
	bb := Key{Tokens: []int64{1, 2, 3, 9, 9}, Bigram: true}
	for _, c := range []struct{ page, want int }{{1, 2}, {2, 2}, {3, 0}} {
		if got := ba.matchAt(bb, 0, c.page); got != c.want {
			t.Errorf("bigram page %d matched %d, want %d", c.page, got, c.want)
		}
	}
	x := "x"
	mustPanic(t, "another namespace", func() { a.matchAt(Key{Tokens: a.Tokens, ExtraKey: &x}, 0, 1) })
	mustPanic(t, "another bigram mode", func() { a.matchAt(Key{Tokens: a.Tokens, Bigram: true}, 0, 1) })
}

func (r *radixRig) insert(tok int64, insertedLen int) {
	r.t.Helper()
	r.cache.Insert(InsertParams{Key: Key{Tokens: []int64{tok}}, Value: []int64{tok + 100}, InsertedLen: insertedLen})
}

func (r *radixRig) has(tok int64) bool {
	return r.cache.MatchPrefix(MatchPrefixParams{Key: Key{Tokens: []int64{tok}}}).DevicePrefixLen == 1
}

// A touch makes a node the most recently used, whatever its id: LRU evicts the node untouched
// the longest, not the one with the lowest id.
func TestLRUEvictsTheLeastRecentlyTouchedNotTheLowestID(t *testing.T) {
	r := newRig(t, 64, RadixConfig{})
	r.insert(1, 0)
	r.insert(2, 0)
	r.insert(1, 1) // touches node 1 again
	r.cache.Evict(EvictParams{NumTokens: 1})
	if !r.has(1) || r.has(2) {
		t.Errorf("after a touch of 1, LRU kept 1: %v, kept 2: %v", r.has(1), r.has(2))
	}
}

// A request counts a hit on a node only for the part it has not already inserted: under SLRU a
// node is protected after two counted hits, so overlapping inserts at or below what the request
// already inserted count none, and a first insert of a new leaf counts one only past that length.
func TestHitsCountOnlyWhatTheRequestHasNotInserted(t *testing.T) {
	protectedAfter := func(setup func(r *radixRig)) bool {
		r := newRig(t, 64, RadixConfig{EvictionPolicy: "slru"})
		setup(r)
		r.cache.Evict(EvictParams{NumTokens: 1})
		return r.has(1) // node 1 is the oldest: it survives only if protected
	}
	touch2 := func(r *radixRig) { r.has(2) }
	// Overlaps at exactly the inserted length count nothing.
	if protectedAfter(func(r *radixRig) {
		r.insert(1, 1)
		r.insert(2, 1)
		r.insert(1, 1)
		r.insert(1, 1)
		touch2(r)
	}) {
		t.Error("two overlaps at the inserted length protected the node")
	}
	// Overlaps below the inserted length count two.
	if !protectedAfter(func(r *radixRig) {
		r.insert(1, 1)
		r.insert(2, 1)
		r.insert(1, 0)
		r.insert(1, 0)
		touch2(r)
	}) {
		t.Error("two counted hits did not protect the node")
	}
	// A new leaf counts a hit only past the inserted length: one new leaf (1) and one overlap (1).
	if protectedAfter(func(r *radixRig) {
		r.insert(1, 1)
		r.insert(2, 1)
		r.insert(1, 1)
		touch2(r)
	}) {
		t.Error("an uncounted new leaf and an uncounted overlap protected the node")
	}
	// One overlap counts one hit, not two (the walk's, and again as if the leaf were new).
	if protectedAfter(func(r *radixRig) {
		r.insert(1, 1)
		r.insert(2, 1)
		r.insert(1, 0)
		touch2(r)
	}) {
		t.Error("one overlap counted twice")
	}
	// A new leaf counts one hit past the inserted length: it plus one overlap protects.
	if !protectedAfter(func(r *radixRig) {
		r.insert(1, 0)
		r.insert(2, 1)
		r.insert(1, 0)
		touch2(r)
	}) {
		t.Error("a new leaf's hit and an overlap's did not protect the node")
	}
	// A new leaf at exactly the inserted length counts nothing: LFU evicts it first.
	r := newRig(t, 64, RadixConfig{EvictionPolicy: "lfu"})
	r.insert(1, 1) // hits 0
	r.insert(2, 0) // hits 1
	r.has(1)       // node 1 is the most recent
	r.cache.Evict(EvictParams{NumTokens: 1})
	if r.has(1) || !r.has(2) {
		t.Error("LFU kept the node whose new leaf counted no hit")
	}
}

// Evicting from an empty tree, or past everything it holds, evicts nothing and never takes the root.
func TestEvictingPastEverythingTakesNoRoot(t *testing.T) {
	r := newRig(t, 64, RadixConfig{})
	if got := r.cache.Evict(EvictParams{NumTokens: 5}).NumTokensEvicted; got != 0 {
		t.Errorf("evicted %d from an empty tree", got)
	}
	r.insert(1, 0)
	if got := r.cache.Evict(EvictParams{NumTokens: 5}).NumTokensEvicted; got != 1 {
		t.Errorf("evicted %d of 1", got)
	}
	if got := r.cache.Evict(EvictParams{NumTokens: 5}).NumTokensEvicted; got != 0 {
		t.Errorf("evicted %d from the emptied tree", got)
	}
}
