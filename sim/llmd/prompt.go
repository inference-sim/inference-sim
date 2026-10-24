package llmd

import (
	"container/list"

	hash "github.com/inference-sim/blis-vllm/compat/prefix"
	"github.com/inference-sim/inference-sim/sim"
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
// Chains an engine reports cached (KV events) are kept as long as it caches them; the others are
// remembered in a least-recently-used set bounded by capacity. Eviction depends only on the
// sequence of prompts and events, so it is the same in a run and its replay.
type promptCanon struct {
	seed      uint64
	blockSize int
	capacity  int
	next      uint64 // canonical id of the next new chain
	chains    map[string]*chain
	lru       *list.List // the unpinned chains, most recent first
}

// chain is one prefix chain's canonical id. A chain an engine reports cached (pins > 0, from its
// KV events) is out of the LRU: forgetting it would show the router new tokens for a block the
// router has indexed under the old ones.
type chain struct {
	id   uint64
	pins int
	elem *list.Element // nil while pinned
}

func newPromptCanon(seed int64, blockSize, capacity int) *promptCanon {
	return &promptCanon{
		seed: uint64(seed), blockSize: max(blockSize, 1), capacity: max(capacity, 1),
		chains: map[string]*chain{}, lru: list.New(),
	}
}

// canonical returns the router's view of prompt. It records the prompt's chains, so it must be
// called once per request, in the order requests reach the router.
func (c *promptCanon) canonical(prompt []sim.TokenID) []uint32 {
	out := make([]uint32, len(prompt))
	chain := ""
	for start := 0; start < len(prompt); start += c.blockSize {
		end := min(start+c.blockSize, len(prompt))
		// The engines' prefix caches key a block the same way (blis-vllm compat/prefix), so a KV event's
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
	if ch, ok := c.chains[key]; ok {
		if ch.elem != nil {
			c.lru.MoveToFront(ch.elem)
		}
		return ch.id
	}
	ch := &chain{id: c.next}
	c.next++
	c.chains[key] = ch
	ch.elem = c.lru.PushFront(key)
	c.trim()
	return ch.id
}

// pin keeps chain key while an engine caches the block it names (a stored KV event).
func (c *promptCanon) pin(key string) {
	c.chainID(key)
	ch := c.chains[key]
	ch.pins++
	if ch.elem != nil {
		c.lru.Remove(ch.elem)
		ch.elem = nil
	}
}

// unpin releases one pin of chain key (a removed KV event, or a cleared engine).
func (c *promptCanon) unpin(key string) {
	ch, ok := c.chains[key]
	if !ok || ch.pins == 0 {
		return
	}
	ch.pins--
	if ch.pins == 0 {
		ch.elem = c.lru.PushFront(key)
		c.trim()
	}
}

// trim forgets the least recently used unpinned chains beyond capacity.
func (c *promptCanon) trim() {
	for c.lru.Len() > c.capacity {
		oldest := c.lru.Back()
		c.lru.Remove(oldest)
		delete(c.chains, oldest.Value.(string))
	}
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
