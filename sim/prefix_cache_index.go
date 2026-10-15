package sim

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// PrefixCacheIndex maintains a router-side approximate prefix cache,
// tracking which block hashes each instance has seen. Uses hierarchical
// block hashing (each block's hash chains the previous) and LRU eviction
// per instance to bound memory.
//
// This is an approximation of the actual per-instance KV cache state —
// the router doesn't query instances directly, matching production
// systems like llm-d's Endpoint Picker.
type PrefixCacheIndex struct {
	blockSize   int
	lruCapacity int
	instances   map[string]*lruBlockCache
	// capacityFor, when non-nil, sizes each instance's LRU when that instance's
	// cache is first created (llm-d approx-prefix-cache-producer sizes per server
	// at first Add: indexer.go:59-70). Nil ⇒ lruCapacity for every instance.
	capacityFor func(instanceID string) int
}

// lruNode is an element of the doubly-linked list used for O(1) LRU eviction.
type lruNode struct {
	hash       string
	prev, next *lruNode
}

// lruBlockCache is a per-instance LRU cache of block hashes.
// Uses a doubly-linked list + map for O(1) touch, lookup, and eviction.
// Head = most recently used, tail = least recently used (eviction target).
type lruBlockCache struct {
	lookup   map[string]*lruNode // O(1) lookup by hash
	head     *lruNode            // most recently used
	tail     *lruNode            // least recently used (eviction target)
	capacity int
}

// NewPrefixCacheIndex creates a prefix cache index with the given block size
// and per-instance LRU capacity (maximum blocks tracked per instance).
func NewPrefixCacheIndex(blockSize int, lruCapacity int) *PrefixCacheIndex {
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewPrefixCacheIndex: blockSize must be > 0, got %d", blockSize))
	}
	if lruCapacity <= 0 {
		panic(fmt.Sprintf("NewPrefixCacheIndex: lruCapacity must be > 0, got %d", lruCapacity))
	}
	return &PrefixCacheIndex{
		blockSize:   blockSize,
		lruCapacity: lruCapacity,
		instances:   make(map[string]*lruBlockCache),
	}
}

// NewPrefixCacheIndexWithCapacity creates a prefix cache index whose per-instance
// LRU capacity is resolved by capacityFor when the instance is first recorded.
// Panics on blockSize <= 0 or nil capacityFor; a non-positive capacity returned
// by capacityFor panics at record time (R3).
func NewPrefixCacheIndexWithCapacity(blockSize int, capacityFor func(instanceID string) int) *PrefixCacheIndex {
	if capacityFor == nil {
		panic("NewPrefixCacheIndexWithCapacity: capacityFor must not be nil")
	}
	idx := NewPrefixCacheIndex(blockSize, 1)
	idx.capacityFor = capacityFor
	return idx
}

// BlockSize returns the index's block size in tokens.
func (idx *PrefixCacheIndex) BlockSize() int { return idx.blockSize }

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each block hash incorporates the previous block's hash, creating prefix-semantic
// hashes: two requests sharing the first K blocks produce identical hashes for those K blocks.
// Tokens shorter than one block produce an empty slice.
// Delegates to hash.ComputeBlockHashes for the shared implementation (BC-3).
func (idx *PrefixCacheIndex) ComputeBlockHashes(tokens []TokenID) []string {
	return hash.ComputeBlockHashes(idx.blockSize, tokens)
}

// MatchLength returns the number of consecutive blocks (from the start) that
// the given instance has cached. Returns 0 if the instance has no history.
// Does not refresh LRU ordering — only RecordBlocks updates recency.
func (idx *PrefixCacheIndex) MatchLength(hashes []string, instanceID string) int {
	cache, exists := idx.instances[instanceID]
	if !exists {
		return 0
	}
	matched := 0
	for _, h := range hashes {
		if _, ok := cache.lookup[h]; ok {
			matched++
		} else {
			break // consecutive from start only
		}
	}
	return matched
}

// RecordBlocks records that the given instance now has the given block hashes.
// Refreshes LRU ordering for existing blocks and evicts the oldest if at capacity.
func (idx *PrefixCacheIndex) RecordBlocks(hashes []string, instanceID string) {
	cache, exists := idx.instances[instanceID]
	if !exists {
		capacity := idx.lruCapacity
		if idx.capacityFor != nil {
			capacity = idx.capacityFor(instanceID)
			if capacity <= 0 {
				panic(fmt.Sprintf("PrefixCacheIndex.RecordBlocks: capacity for %q must be > 0, got %d", instanceID, capacity))
			}
		}
		cache = &lruBlockCache{
			lookup:   make(map[string]*lruNode),
			capacity: capacity,
		}
		idx.instances[instanceID] = cache
	}
	for _, h := range hashes {
		cache.touch(h)
	}
}

// RecordBlocksTailFirst records a prompt's hashes last block first, so the prompt's
// head is the most recently used and the LRU evicts its tail first (llm-d main
// approx indexer Add: requestcontrol/dataproducer/approximateprefix/indexer.go:94-103
// at 981acb8a).
func (idx *PrefixCacheIndex) RecordBlocksTailFirst(hashes []string, instanceID string) {
	idx.RecordBlocks(nil, instanceID) // create the instance cache with its capacity
	cache := idx.instances[instanceID]
	for i := len(hashes) - 1; i >= 0; i-- {
		cache.touch(hashes[i])
	}
}

// MatchLengthGalloping returns the leading-run length found by llm-d main's
// galloping search (approximateprefix/indexer.go:133-160 at 981acb8a): an
// exponential probe for the first missing block followed by a binary search. It
// equals MatchLength whenever the instance's held blocks of the prompt form a
// prefix; otherwise it reports the run llm-d reports. Read-only (no recency).
func (idx *PrefixCacheIndex) MatchLengthGalloping(hashes []string, instanceID string) int {
	cache, exists := idx.instances[instanceID]
	if !exists || len(hashes) == 0 {
		return 0
	}
	has := func(i int) bool { _, ok := cache.lookup[hashes[i]]; return ok }
	if !has(0) {
		return 0
	}
	held, missing := 1, len(hashes)+1
	for step := 1; held+step < missing; step *= 2 {
		if !has(held + step - 1) {
			missing = held + step
			break
		}
		held += step
	}
	for missing-held > 1 {
		mid := held + (missing-held)/2
		if has(mid - 1) {
			held = mid
		} else {
			missing = mid
		}
	}
	return held
}

// InstanceBlockCount returns the number of cached blocks for an instance.
// Used for testing LRU capacity bounds.
func (idx *PrefixCacheIndex) InstanceBlockCount(instanceID string) int {
	cache, exists := idx.instances[instanceID]
	if !exists {
		return 0
	}
	return len(cache.lookup)
}

// touch adds or refreshes a block hash in the LRU cache, evicting the tail if at capacity.
func (c *lruBlockCache) touch(h string) {
	if node, exists := c.lookup[h]; exists {
		// Move existing node to head (most recently used)
		c.removeNode(node)
		c.pushHead(node)
		return
	}
	// New entry — evict tail if at capacity
	if len(c.lookup) >= c.capacity {
		c.evictOldest()
	}
	node := &lruNode{hash: h}
	c.lookup[h] = node
	c.pushHead(node)
}

// evictOldest removes the least recently used block hash (the tail). O(1).
// Panics if tail is nil — callers only invoke this when len(lookup) >= capacity,
// so a nil tail indicates linked-list corruption.
//
// This is INV-18's assertion point (docs/contributing/standards/invariants.md#inv-18):
// the list and the lookup map describe the same block set, so `tail == nil` iff
// `len(lookup) == 0` and eviction at capacity can never fail. The two structures are
// maintained by separate statements, so a mutator that updates one and not the other
// degrades routing silently (fewer prefix hits) until it finally reaches this panic.
func (c *lruBlockCache) evictOldest() {
	if c.tail == nil {
		panic(fmt.Sprintf("lruBlockCache.evictOldest: tail is nil but len(lookup)=%d — INV-18 violation (list/lookup inconsistent)", len(c.lookup)))
	}
	delete(c.lookup, c.tail.hash)
	c.removeNode(c.tail)
}

// pushHead inserts a node at the head of the doubly-linked list.
func (c *lruBlockCache) pushHead(node *lruNode) {
	if node == nil {
		panic("lruBlockCache.pushHead: node must not be nil")
	}
	node.prev = nil
	node.next = c.head
	if c.head != nil {
		c.head.prev = node
	}
	c.head = node
	if c.tail == nil {
		c.tail = node
	}
}

// removeNode removes a node from the doubly-linked list.
func (c *lruBlockCache) removeNode(node *lruNode) {
	if node == nil {
		panic("lruBlockCache.removeNode: node must not be nil")
	}
	if node.prev != nil {
		node.prev.next = node.next
	} else {
		c.head = node.next
	}
	if node.next != nil {
		node.next.prev = node.prev
	} else {
		c.tail = node.prev
	}
	node.prev = nil
	node.next = nil
}
