package hash

import (
	"container/list"
	"encoding/binary"
	"hash/maphash"
	"slices"
	"sync/atomic"

	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// blockHashes counts HashBlock calls: the counting hook INV-K11's tests read.
var blockHashes atomic.Int64

// BlockHashes returns the number of block hashes HashBlock has computed in this process.
func BlockHashes() int64 { return blockHashes.Load() }

// promptHashes counts the prompt blocks ChainCaches hashed: INV-K11's counting hook.
var promptHashes atomic.Int64

// PromptBlockHashes returns how many prompt blocks every ChainCache of the process has hashed.
func PromptBlockHashes() int64 { return promptHashes.Load() }

// ChainCacheCapacity is how many token sequences a ChainCache holds; the least recently used
// is evicted beyond it. INV-K11 holds while an engine's live prompts number no more.
const ChainCacheCapacity = 1024

// ChainCache memoises the hierarchical block-hash chain of token sequences (INV-K11), so a
// prompt looked up again -- a waiting request re-examined every step -- is not re-hashed. An
// entry is found by the sequence's backing array; a longer view of it (a later turn of a
// session buffer) is hashed past the cached blocks only, and a block seen before in any array
// is served from the content memo. Chain(tokens) always equals ComputeBlockHashes(blockSize, tokens).
// Not safe for concurrent use.
type ChainCache struct {
	blockSize int
	entries   map[*tokenid.TokenID]*list.Element
	lru       list.List // of *chainEntry, most recent first

	memo       map[uint64][]memoEntry
	memoBlocks int
}

type chainEntry struct {
	key    *tokenid.TokenID
	tokens []tokenid.TokenID // the array hashed, held so its address is not reused
	hashes []string
}

// NewChainCache returns an empty cache for blockSize-token blocks.
func NewChainCache(blockSize int) *ChainCache {
	return &ChainCache{blockSize: blockSize, entries: make(map[*tokenid.TokenID]*list.Element),
		memo: make(map[uint64][]memoEntry)}
}

// Chain returns the block hashes of tokens' complete blocks. The caller must not modify it.
func (c *ChainCache) Chain(tokens []tokenid.TokenID) []string {
	if chainCacheOff {
		return computeChain(c.blockSize, tokens)
	}
	bs := c.blockSize
	n := len(tokens) / bs
	if n == 0 {
		return nil
	}
	key := &tokens[0]
	var e *chainEntry
	if el, ok := c.entries[key]; ok {
		c.lru.MoveToFront(el)
		e = el.Value.(*chainEntry)
	} else {
		if c.lru.Len() >= ChainCacheCapacity {
			old := c.lru.Remove(c.lru.Back()).(*chainEntry)
			delete(c.entries, old.key)
		}
		e = &chainEntry{key: key}
		c.entries[key] = c.lru.PushFront(e)
	}
	// The entry holds the array it was built from, so the address names this array for the
	// entry's life; prompt arrays are never written in place (sim.Request.FullInputTokens:
	// every writer allocates a fresh array), so the cached blocks are still these tokens.
	valid := min(len(e.hashes), n)
	if valid >= n {
		return e.hashes[:n:n]
	}
	e.hashes = e.hashes[:valid]
	prev := ""
	if valid > 0 {
		prev = e.hashes[valid-1]
	}
	for i := valid; i < n; i++ {
		blk := tokens[i*bs : (i+1)*bs]
		prev = c.block(prev, blk)
		e.hashes = append(e.hashes, prev)
	}
	e.tokens = tokens
	return e.hashes[:n:n]
}

// block is HashBlock(prev, blk), served from the content memo when this (prev, tokens) pair
// was hashed before -- by any sequence, so a later turn rendered into a fresh array reuses
// its predecessor's chain. The memo is checked against the stored tokens, so it is exact.
func (c *ChainCache) block(prev string, blk []tokenid.TokenID) string {
	var h maphash.Hash
	h.SetSeed(memoSeed)
	h.WriteString(prev)
	var b [4]byte
	for _, t := range blk {
		binary.LittleEndian.PutUint32(b[:], uint32(t))
		h.Write(b[:])
	}
	k := h.Sum64()
	for _, m := range c.memo[k] {
		if m.prev == prev && slices.Equal(m.tokens, blk) {
			return m.hash
		}
	}
	if c.memoBlocks >= MemoCapacity {
		clear(c.memo)
		c.memoBlocks = 0
	}
	promptHashes.Add(1)
	out := HashBlock(prev, blk)
	c.memo[k] = append(c.memo[k], memoEntry{prev: prev, tokens: slices.Clone(blk), hash: out})
	c.memoBlocks++
	return out
}

// MemoCapacity is how many distinct blocks a ChainCache's content memo holds before it is
// cleared; INV-K11 holds while an engine's distinct prompt blocks number no more.
const MemoCapacity = 1 << 21

var memoSeed = maphash.MakeSeed()

type memoEntry struct {
	prev   string
	tokens []tokenid.TokenID
	hash   string
}

// chainCacheOff makes every ChainCache recompute (SetChainCacheOff): the uncached reference
// path INV-K11's tests compare against.
var chainCacheOff bool

// SetChainCacheOff turns memoisation off (true) or on, process-wide. For tests.
func SetChainCacheOff(off bool) { chainCacheOff = off }

func computeChain(bs int, tokens []tokenid.TokenID) []string {
	promptHashes.Add(int64(len(tokens) / bs))
	var out []string
	prev := ""
	for i := 0; i+bs <= len(tokens); i += bs {
		prev = HashBlock(prev, tokens[i:i+bs])
		out = append(out, prev)
	}
	return out
}
