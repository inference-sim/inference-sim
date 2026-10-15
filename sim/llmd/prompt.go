package llmd

import (
	"container/list"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// promptCanon gives each simulated prompt the token values the router sees.
//
// BLIS token values are placeholders: what a workload models is the prompts' lengths and which
// prompts share a prefix. A replayed trace reproduces exactly that (INV-13), but draws new
// placeholder values, while the router reads the prompt's content (its prefix-cache hashes, its
// size). So the router is shown canonical values instead: the prompt is cut into KV blocks (the
// last one possibly partial), each block is identified by its prefix chain (the block's tokens and
// every block before it, hashed as the engines' prefix caches hash it), and every chain gets fixed
// canonical tokens the first time it is seen.
// Two prompts therefore present equal tokens exactly where they share a chain of whole blocks,
// as the engines' prefix caches see them, and a run and its replay present identical prompts.
// A router block size that is a multiple of the engine's block size hashes the canonical prompt
// as it would the original.
//
// Chains are remembered in a least-recently-used set bounded by capacity: a chain no instance
// could still hold in its prefix cache can be forgotten. Eviction order depends only on the
// sequence of prompts, so it is the same in a run and its replay.
type promptCanon struct {
	seed      uint64
	blockSize int
	capacity  int
	next      uint64 // canonical id of the next new chain
	ids       map[string]*list.Element
	lru       *list.List
}

type chainEntry struct {
	key string
	id  uint64
}

func newPromptCanon(seed int64, blockSize, capacity int) *promptCanon {
	return &promptCanon{
		seed: uint64(seed), blockSize: max(blockSize, 1), capacity: max(capacity, 1),
		ids: map[string]*list.Element{}, lru: list.New(),
	}
}

// canonical returns the router's view of prompt. It records the prompt's chains, so it must be
// called once per request, in the order requests reach the router.
func (c *promptCanon) canonical(prompt []sim.TokenID) []uint32 {
	out := make([]uint32, len(prompt))
	chain := ""
	for start := 0; start < len(prompt); start += c.blockSize {
		end := min(start+c.blockSize, len(prompt))
		// The engines' prefix caches key a block the same way (sim/internal/hash), so a KV event's
		// block hash names the chain the router was shown (canonicalBlock).
		chain = hash.HashBlock(chain, prompt[start:end])
		id := c.chainID(chain)
		for i := start; i < end; i++ {
			out[i] = c.token(id, i-start)
		}
	}
	return out
}

// canonicalBlock returns the router's tokens for the n-token block an engine's prefix cache keys
// as blockHash: the tokens the router was shown for that block, or new canonical tokens for a block
// no prompt contained (one an engine filled while decoding).
func (c *promptCanon) canonicalBlock(blockHash string, n int) []uint32 {
	id := c.chainID(blockHash)
	out := make([]uint32, n)
	for i := range out {
		out[i] = c.token(id, i)
	}
	return out
}

func (c *promptCanon) chainID(key string) uint64 {
	if e, ok := c.ids[key]; ok {
		c.lru.MoveToFront(e)
		return e.Value.(chainEntry).id
	}
	id := c.next
	c.next++
	c.ids[key] = c.lru.PushFront(chainEntry{key: key, id: id})
	if c.lru.Len() > c.capacity {
		oldest := c.lru.Back()
		c.lru.Remove(oldest)
		delete(c.ids, oldest.Value.(chainEntry).key)
	}
	return id
}

// token is the canonical value at offset i of the block with canonical id, drawn uniformly from
// the same range as BLIS's placeholder tokens.
func (c *promptCanon) token(id uint64, i int) uint32 {
	x := c.seed ^ (id*0x9e3779b97f4a7c15 + uint64(i)*0xbf58476d1ce4e5b9)
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return uint32(x % sim.MaxTokenID)
}
