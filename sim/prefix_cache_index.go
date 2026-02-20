package sim

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"strconv"
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

// lruBlockCache is a per-instance LRU cache of block hashes.
type lruBlockCache struct {
	hashes   map[string]int64 // block hash → access timestamp
	capacity int
	clock    int64 // monotonic counter for LRU ordering
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
func (idx *PrefixCacheIndex) ComputeBlockHashes(tokens []int) []string {
	numBlocks := len(tokens) / idx.blockSize
	if numBlocks == 0 {
		return nil
	}
	hashes := make([]string, numBlocks)
	prevHash := ""
	for i := 0; i < numBlocks; i++ {
		start := i * idx.blockSize
		end := start + idx.blockSize
		hashes[i] = hashBlock(prevHash, tokens[start:end])
		prevHash = hashes[i]
	}
	return hashes
}

// hashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
func hashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	for _, t := range tokens {
		h.Write([]byte(strconv.Itoa(t)))
		h.Write([]byte("|"))
	}
	return hex.EncodeToString(h.Sum(nil))
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
		if _, ok := cache.hashes[h]; ok {
			matched++
		} else {
			break // consecutive from start only
		}
	}
	return matched
}

// RecordBlocks records that the given instance now has the given block hashes.
// Updates LRU timestamps for existing blocks and evicts oldest if at capacity.
func (idx *PrefixCacheIndex) RecordBlocks(hashes []string, instanceID string) {
	cache, exists := idx.instances[instanceID]
	if !exists {
		cache = &lruBlockCache{
			hashes:   make(map[string]int64),
			capacity: idx.lruCapacity,
		}
		idx.instances[instanceID] = cache
	}
	for _, h := range hashes {
		cache.touch(h)
	}
}

// InstanceBlockCount returns the number of cached blocks for an instance.
// Used for testing INV-7 (bounded growth).
func (idx *PrefixCacheIndex) InstanceBlockCount(instanceID string) int {
	cache, exists := idx.instances[instanceID]
	if !exists {
		return 0
	}
	return len(cache.hashes)
}

// touch adds or refreshes a block hash in the LRU cache, evicting the oldest if at capacity.
func (c *lruBlockCache) touch(hash string) {
	c.clock++
	if _, exists := c.hashes[hash]; exists {
		// Refresh existing entry
		c.hashes[hash] = c.clock
		return
	}
	// New entry — evict if at capacity
	if len(c.hashes) >= c.capacity {
		c.evictOldest()
	}
	c.hashes[hash] = c.clock
}

// evictOldest removes the least recently used block hash.
// Uses monotonic timestamps so there is always a unique minimum (no tie-breaking needed).
func (c *lruBlockCache) evictOldest() {
	var oldestHash string
	oldestTime := int64(math.MaxInt64)
	for h, t := range c.hashes {
		if t < oldestTime {
			oldestTime = t
			oldestHash = h
		}
	}
	if oldestHash != "" {
		delete(c.hashes, oldestHash)
	}
}
