package hash

import (
	"slices"

	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// chainCacheMaxEntries bounds a ChainCache; it is cleared when full.
const chainCacheMaxEntries = 128

// ChainCache memoises the hierarchical block-hash chain of token sequences, so a prompt
// looked up again (a waiting request re-examined every step) is not re-hashed. An entry is
// found by the sequence's backing array and kept with a copy of the tokens it covers; a
// lookup reuses the longest cached prefix whose tokens still match, so a view that grew (a
// later turn of a session buffer) or an array written in place is hashed past the match only.
// Chain(tokens) always equals ComputeBlockHashes(blockSize, tokens). Not safe for concurrent use.
type ChainCache struct {
	blockSize int
	entries   map[*tokenid.TokenID]*chainEntry
}

type chainEntry struct {
	tokens []tokenid.TokenID // the complete blocks hashed, copied
	hashes []string
}

// NewChainCache returns an empty cache for blockSize-token blocks.
func NewChainCache(blockSize int) *ChainCache {
	return &ChainCache{blockSize: blockSize, entries: make(map[*tokenid.TokenID]*chainEntry)}
}

// Chain returns the block hashes of tokens' complete blocks. The caller must not modify it.
func (c *ChainCache) Chain(tokens []tokenid.TokenID) []string {
	bs := c.blockSize
	n := len(tokens) / bs
	if n == 0 {
		return nil
	}
	key := &tokens[0]
	e := c.entries[key]
	if e == nil {
		if len(c.entries) >= chainCacheMaxEntries {
			clear(c.entries)
		}
		e = &chainEntry{}
		c.entries[key] = e
	}
	valid := 0
	for valid < len(e.hashes) && valid < n &&
		slices.Equal(e.tokens[valid*bs:(valid+1)*bs], tokens[valid*bs:(valid+1)*bs]) {
		valid++
	}
	if valid >= n {
		return e.hashes[:n:n]
	}
	e.hashes, e.tokens = e.hashes[:valid], e.tokens[:valid*bs]
	prev := ""
	if valid > 0 {
		prev = e.hashes[valid-1]
	}
	for i := valid; i < n; i++ {
		blk := tokens[i*bs : (i+1)*bs]
		prev = HashBlock(prev, blk)
		e.hashes = append(e.hashes, prev)
		e.tokens = append(e.tokens, blk...)
	}
	return e.hashes[:n:n]
}
