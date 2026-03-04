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

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each block hash incorporates the previous block's hash, creating prefix-semantic
// hashes: two requests sharing the first K blocks produce identical hashes for those K blocks.
// Tokens shorter than one block produce an empty slice.
// Delegates to hash.ComputeBlockHashes for the shared implementation (BC-3).
func (idx *PrefixCacheIndex) ComputeBlockHashes(tokens []int) []string {
	return hash.ComputeBlockHashes(idx.blockSize, tokens)
}

// MatchLength returns the number of consecutive blocks (from the start) that
// the given instance has cached. Returns 0 if the instance has no history.
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
		cache = &lruBlockCache{
			lookup:   make(map[string]*lruNode),
			capacity: idx.lruCapacity,
		}
		idx.instances[instanceID] = cache
	}
	for _, h := range hashes {
		cache.touch(h)
	}
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
func (c *lruBlockCache) evictOldest() {
	if c.tail == nil {
		panic(fmt.Sprintf("lruBlockCache.evictOldest: tail is nil but len(lookup)=%d (invariant violation)", len(c.lookup)))
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
