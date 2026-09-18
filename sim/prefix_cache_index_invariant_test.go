// prefix_cache_index_invariant_test.go — companion invariant test for INV-18
// (prefix-cache LRU structural consistency), promoted by #1772.
//
// The production enforcement is a single panic in evictOldest: a nil tail when the
// lookup map is non-empty. That fires only at the moment an eviction is finally
// attempted, which is long after a list/map divergence begins to degrade routing.
// This test asserts the structural equality directly and at every step, so a mutator
// that updates one structure and not the other fails on the property.
package sim

import (
	"fmt"
	"testing"
)

// walkLRUForward returns the hashes reachable from head via next, and verifies the
// forward links are well-formed (no cycle within a bounded budget, correct prev
// back-pointers, and the last node is exactly c.tail).
func walkLRUForward(t *testing.T, c *lruBlockCache) []string {
	t.Helper()
	var got []string
	var prev *lruNode
	budget := len(c.lookup) + 1 // a well-formed list cannot exceed len(lookup)
	for node := c.head; node != nil; node = node.next {
		if len(got) > budget {
			t.Fatalf("INV-18 violated: forward walk exceeded %d nodes for len(lookup)=%d — the list has a cycle",
				budget, len(c.lookup))
		}
		if node.prev != prev {
			t.Errorf("INV-18 violated: node %q has prev=%v, want %v (back-pointer inconsistent)",
				node.hash, node.prev, prev)
		}
		got = append(got, node.hash)
		prev = node
	}
	if prev != c.tail {
		t.Errorf("INV-18 violated: forward walk ended at %v but c.tail=%v", prev, c.tail)
	}
	return got
}

// assertINV18 checks the whole statement at one point in time: the list and the
// lookup map describe the same block set (in both walk directions), and the
// biconditional `tail == nil ⟺ len(lookup) == 0` holds.
func assertINV18(t *testing.T, c *lruBlockCache, after string) {
	t.Helper()

	// The biconditional the production panic depends on, checked in BOTH directions:
	// a nil tail with a non-empty map is the panic case, and a non-nil tail with an
	// empty map is the mirror leak (an orphaned node the map can no longer reach).
	if (c.tail == nil) != (len(c.lookup) == 0) {
		t.Errorf("INV-18 violated after %s: tail==nil is %v but len(lookup)==0 is %v",
			after, c.tail == nil, len(c.lookup) == 0)
	}
	// head and tail are nil together — a list with one terminal but not the other is
	// unwalkable from at least one end.
	if (c.head == nil) != (c.tail == nil) {
		t.Errorf("INV-18 violated after %s: head==nil is %v but tail==nil is %v",
			after, c.head == nil, c.tail == nil)
	}
	if c.head != nil && c.head.prev != nil {
		t.Errorf("INV-18 violated after %s: head %q has a non-nil prev", after, c.head.hash)
	}
	if c.tail != nil && c.tail.next != nil {
		t.Errorf("INV-18 violated after %s: tail %q has a non-nil next", after, c.tail.hash)
	}

	// The list's node set equals the map's key set, and each map entry points at the
	// node that actually carries that hash.
	forward := walkLRUForward(t, c)
	if len(forward) != len(c.lookup) {
		t.Fatalf("INV-18 violated after %s: list holds %d nodes but lookup holds %d keys",
			after, len(forward), len(c.lookup))
	}
	seen := make(map[string]struct{}, len(forward))
	for _, h := range forward {
		if _, dup := seen[h]; dup {
			t.Errorf("INV-18 violated after %s: hash %q appears twice in the list", after, h)
		}
		seen[h] = struct{}{}
		node, ok := c.lookup[h]
		if !ok {
			t.Errorf("INV-18 violated after %s: list holds %q but lookup does not", after, h)
			continue
		}
		if node.hash != h {
			t.Errorf("INV-18 violated after %s: lookup[%q] points at a node carrying %q", after, h, node.hash)
		}
	}
	for h := range c.lookup {
		if _, ok := seen[h]; !ok {
			t.Errorf("INV-18 violated after %s: lookup holds %q but the list does not", after, h)
		}
	}

	// Capacity is the bound eviction defends; exceeding it means an eviction was missed.
	if len(c.lookup) > c.capacity {
		t.Errorf("INV-18 violated after %s: len(lookup)=%d exceeds capacity=%d", after, len(c.lookup), c.capacity)
	}
}

// TestINV18_ListAndLookupStayConsistent records blocks in a sequence that crosses
// capacity repeatedly — filling below capacity, filling exactly to capacity, driving
// evictions, re-touching resident blocks (the move-to-head path, which is a remove
// plus a push and so touches every link) and re-touching the tail specifically (the
// one-node-list edge). The structural equality is asserted after every step.
func TestINV18_ListAndLookupStayConsistent(t *testing.T) {
	const capacity = 4
	idx := NewPrefixCacheIndex(2, capacity)
	const instanceID = "inst-0"

	// Empty cache: the biconditional's base case. RecordBlocks with no hashes creates
	// the cache without touching anything.
	idx.RecordBlocks(nil, instanceID)
	cache := idx.instances[instanceID]
	if cache == nil {
		t.Fatal("RecordBlocks did not create the per-instance cache")
	}
	assertINV18(t, cache, "empty cache")

	hashes := make([]string, 12)
	for i := range hashes {
		hashes[i] = fmt.Sprintf("h%02d", i)
	}

	// One at a time up to and past capacity, so the at-capacity and first-eviction
	// steps are each observed on their own.
	for i, h := range hashes {
		idx.RecordBlocks([]string{h}, instanceID)
		assertINV18(t, cache, fmt.Sprintf("insert %s (step %d)", h, i))
	}
	if len(cache.lookup) != capacity {
		t.Fatalf("after %d inserts, len(lookup)=%d, want capacity=%d", len(hashes), len(cache.lookup), capacity)
	}

	// Re-touch the current tail: this exercises removeNode on the tail (which must
	// reassign c.tail) followed by pushHead, the mutation pair most likely to leave
	// the terminals inconsistent.
	tailHash := cache.tail.hash
	idx.RecordBlocks([]string{tailHash}, instanceID)
	assertINV18(t, cache, "re-touch tail "+tailHash)
	if cache.head.hash != tailHash {
		t.Errorf("re-touching the tail did not promote it to head: head=%q, want %q", cache.head.hash, tailHash)
	}

	// Re-touch the head: removeNode on the head (which must reassign c.head).
	headHash := cache.head.hash
	idx.RecordBlocks([]string{headHash}, instanceID)
	assertINV18(t, cache, "re-touch head "+headHash)

	// Batch insert larger than capacity in a single call — every block after the
	// capacity-th drives an eviction inside one RecordBlocks.
	idx.RecordBlocks(hashes, instanceID)
	assertINV18(t, cache, "batch insert of 12 into capacity 4")
	if len(cache.lookup) != capacity {
		t.Errorf("after an oversized batch, len(lookup)=%d, want %d", len(cache.lookup), capacity)
	}

	// A capacity-1 cache: every insert evicts, so the list is repeatedly reduced to a
	// single node and the head==tail case is exercised on every step.
	single := NewPrefixCacheIndex(2, 1)
	idx1 := "inst-1"
	for i, h := range hashes {
		single.RecordBlocks([]string{h}, idx1)
		assertINV18(t, single.instances[idx1], fmt.Sprintf("capacity-1 insert %s (step %d)", h, i))
	}
	if got := single.InstanceBlockCount(idx1); got != 1 {
		t.Errorf("capacity-1 cache holds %d blocks, want 1", got)
	}
}

// TestINV18_EvictOldestPanicsOnInconsistentState asserts the production guard is live
// rather than dead code: a cache whose lookup map is non-empty while the list is empty
// — precisely the divergence INV-18 forbids — panics on eviction instead of silently
// returning. Without this, the entry's "eviction at capacity can never fail" clause
// would rest on an unasserted assumption.
func TestINV18_EvictOldestPanicsOnInconsistentState(t *testing.T) {
	c := &lruBlockCache{
		lookup:   map[string]*lruNode{"orphan": {hash: "orphan"}},
		capacity: 1,
		// head and tail deliberately left nil: the map says one block is resident,
		// the list says none. This is the state the panic exists to report.
	}
	defer func() {
		if r := recover(); r == nil {
			t.Error("INV-18 guard did not fire: evictOldest with a nil tail and a non-empty lookup must panic")
		}
	}()
	c.evictOldest()
}
