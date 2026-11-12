// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/block_pool.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/kv_cache_utils.py (KVCacheBlock, FreeKVCacheBlockQueue).

package kv

import (
	"bytes"
	"fmt"
)

// Block is one KV block: vLLM's KVCacheBlock (kv_cache_utils.py:178-235). A block's ID
// is an allocator detail; what a block holds is its hash and reference count.
type Block struct {
	ID     int
	RefCnt int
	// Hash is set while the block is a full, cached prefix block, nil otherwise. It is the
	// block hash with its KV cache group ID appended as 4 big-endian bytes, as vLLM keys
	// its cache (make_block_hash_with_group_id, kv_cache_utils.py:74-84).
	Hash []byte
	// HashTokens is the prefix length Hash covers: a full block's end, or inside the block for a
	// partial entry (block_hash_num_tokens, kv_cache_utils.py:192-224@5df63a72). Valid while
	// Hash is set.
	HashTokens int
	// extra holds the further hash keys that also reach this block, which a partial entry adds
	// to a block that already has a primary hash (cached_block_hashes_by_block,
	// block_pool.py:176-178@5df63a72).
	extra [][]byte
	// Null marks the pool's reserved null block, which never holds KV.
	Null bool

	prev, next *Block
	// in is the free queue that holds the block, nil when it is in none.
	in *freeQueue
}

// Pool is vLLM's BlockPool (block_pool.py:136-901) for one KV cache group with prefix
// caching: a fixed set of blocks, a free queue ordered for eviction, and the map from a
// cached block's hash to the blocks that hold it.
type Pool struct {
	blocks  []*Block
	free    freeQueue
	cached  map[string][]*Block // hash -> blocks holding it, in insertion order
	caching bool
	null    *Block
	// events, when non-nil, collects the KV cache events vLLM's BlockPool queues
	// (kv_event_queue, block_pool.py:179-185) until TakeEvents.
	events *[]Event
}

// Event is one KV cache event (vllm/distributed/kv_events.py:56-138@5df63a72), by content.
type Event struct {
	// Removed distinguishes BlockRemoved (true) from BlockStored.
	Removed bool
	// Hashes are the block hashes, without the group suffix.
	Hashes []BlockHash
	// Parent is BlockStored's parent_block_hash: the hash before the first stored block, nil
	// for a run starting at the prompt's first block.
	Parent BlockHash
	// Tokens are BlockStored's token_ids: the stored blocks' tokens.
	Tokens []int64
	// BlockSize is BlockStored's block_size.
	BlockSize int
	// LoRA is BlockStored's lora_name, "" for the base model.
	LoRA string
	// Group is the event's group_idx: the KV cache group whose blocks it reports.
	Group uint32
}

// EnableEvents makes the pool record KV cache events (enable_kv_cache_events). Turning them on
// again keeps the events not yet taken.
func (p *Pool) EnableEvents() {
	if p.events == nil {
		p.events = &[]Event{}
	}
}

// TakeEvents returns the events since the last call (take_events, block_pool.py:890-901).
func (p *Pool) TakeEvents() []Event {
	if p.events == nil {
		return nil
	}
	out := *p.events
	*p.events = nil
	return out
}

// NewPool builds a pool of numBlocks blocks. Block 0 becomes the null block, as vLLM pops
// the first free block for it (block_pool.py:187-188), so numBlocks-1 can hold KV.
func NewPool(numBlocks int, enableCaching bool) *Pool {
	if numBlocks < 1 {
		panic(fmt.Sprintf("kv: a pool needs at least one block, got %d", numBlocks))
	}
	p := &Pool{blocks: make([]*Block, numBlocks), cached: map[string][]*Block{}, caching: enableCaching}
	for i := range p.blocks {
		p.blocks[i] = &Block{ID: i}
	}
	p.free.init(p.blocks)
	p.null = p.free.popLeft()
	p.null.Null = true
	return p
}

// NumBlocks is the pool's total block count, including the null block.
func (p *Pool) NumBlocks() int { return len(p.blocks) }

// NumFree is get_num_free_blocks (block_pool.py:868-875).
func (p *Pool) NumFree() int { return p.free.n }

// NumCached is how many blocks hold a cached prefix hash: what the published stores less the
// published removals hold (KV events).
func (p *Pool) NumCached() int {
	n := 0
	for _, bs := range p.cached {
		n += len(bs)
	}
	return n
}

// Usage is get_usage (block_pool.py:877-888): one minus the free share of the blocks that
// can hold KV, the null block excluded.
func (p *Pool) Usage() float64 {
	total := len(p.blocks) - 1
	if total == 0 {
		return 0
	}
	return 1 - float64(p.free.n)/float64(total)
}

// Cached returns the first block caching hash in group 0, or nil (get_cached_block,
// block_pool.py:198-224, for one group).
func (p *Pool) Cached(h BlockHash) *Block { return p.CachedIn(h, 0) }

// CachedIn returns the first block caching hash in KV cache group g, or nil.
func (p *Pool) CachedIn(h BlockHash, g uint32) *Block {
	if bs := p.cached[string(WithGroup(h, g))]; len(bs) > 0 {
		return bs[0]
	}
	return nil
}

// New takes n blocks from the free queue's head, evicting any cached hash they still hold
// (get_new_blocks, block_pool.py:666-700). It panics if fewer than n are free: a caller
// checks first, as vLLM's does.
func (p *Pool) New(n int) []*Block {
	if n > p.free.n {
		panic(fmt.Sprintf("kv: cannot get %d free blocks from a pool with %d free", n, p.free.n))
	}
	out := make([]*Block, 0, n)
	for i := 0; i < n; i++ {
		b := p.free.popLeft()
		if p.caching {
			p.evict(b)
		}
		if b.RefCnt != 0 {
			panic(fmt.Sprintf("kv: free block %d has ref count %d", b.ID, b.RefCnt))
		}
		b.RefCnt++
		out = append(out, b)
	}
	return out
}

// Touch takes a reference on prefix-hit blocks, removing a free one from the free queue
// (touch, block_pool.py:752-768).
func (p *Pool) Touch(bs []*Block) {
	for _, b := range bs {
		if b.RefCnt == 0 && !b.Null {
			p.free.remove(b)
		}
		b.RefCnt++
	}
}

// Free drops one reference from each block, in the order given, which is eviction
// priority: a block that becomes free without a cached hash goes to the free queue's head
// (evicted first), one with a hash to its tail (free_blocks, block_pool.py:774-805). Each
// block must hold a reference: vLLM's free_blocks assumes it and checks nothing, and freeing
// a free block (a double free in a caller) would leave a negative count and a block both free
// and queued, so it panics here, before any block changes, instead of corrupting the pool.
func (p *Pool) Free(ordered []*Block) {
	// Check every block before changing any, so a refused call leaves the pool as it was: a
	// block listed more times than it holds references is refused as a whole.
	held := make(map[*Block]int, len(ordered))
	for _, b := range ordered {
		if b.Null {
			continue
		}
		held[b]++
		if held[b] > b.RefCnt {
			panic(fmt.Sprintf("kv: freeing block %d more times than its %d reference(s)", b.ID, b.RefCnt))
		}
	}
	var first, last []*Block
	for _, b := range ordered {
		b.RefCnt--
		if b.RefCnt == 0 && !b.Null {
			if b.Hash == nil || !p.caching {
				first = append(first, b)
			} else {
				last = append(last, b)
			}
		}
	}
	p.free.prependN(first)
	p.free.appendN(last)
}

// StoredRun is what a BlockStored event reports about the request whose blocks are cached.
type StoredRun struct {
	// Tokens are the request's tokens (prompt then output), at least through numFull blocks.
	Tokens []int64
	// LoRA is the request's adapter name, "" for the base model.
	LoRA string
	// NumPromptTokens is request.num_prompt_tokens, which a Mamba align group reads to place its
	// prompt checkpoint (single_type_kv_cache_manager.py:2033-2040@5df63a72); no event reports it.
	NumPromptTokens int
	// SharedPrefixBoundary is request.shared_prefix_boundary, read by a Mamba align group.
	SharedPrefixBoundary int
}

// CacheFull records blocks[numCached:numFull] as cached under their hashes in KV cache
// group 0 (cache_full_blocks, block_pool.py:226-345), and with events on, queues one
// BlockStored event for the run (:301-343).
func (p *Pool) CacheFull(blocks []*Block, hashes []BlockHash, numCached, numFull int, run StoredRun, blockSize int) {
	p.CacheFullIn(0, blocks, hashes, numCached, numFull, run, blockSize)
}

// CacheFullIn is CacheFull in KV cache group g. A null block (a sliding window's freed slot)
// is skipped, its hash neither cached nor reported (:275-278, :319-322).
func (p *Pool) CacheFullIn(g uint32, blocks []*Block, hashes []BlockHash, numCached, numFull int, run StoredRun, blockSize int) {
	p.CacheFullMaskedIn(g, blocks, hashes, numCached, numFull, run, blockSize, nil)
}

// CacheFullMaskedIn is CacheFullIn with cache_full_blocks' block_mask (block_pool.py:264-278):
// a block whose mask entry is false is skipped like a null one. A nil mask caches every block.
func (p *Pool) CacheFullMaskedIn(g uint32, blocks []*Block, hashes []BlockHash, numCached, numFull int, run StoredRun,
	blockSize int, mask []bool) {
	var stored []BlockHash
	for i := numCached; i < numFull; i++ {
		b := blocks[i]
		if b.Null || mask != nil && !mask[i-numCached] {
			continue
		}
		if b.Hash != nil {
			p.evict(b)
		}
		p.insert(WithGroup(hashes[i], g), b, (i+1)*blockSize)
		stored = append(stored, hashes[i])
	}
	if p.events != nil {
		var parent BlockHash
		if numCached > 0 {
			parent = hashes[numCached-1]
		}
		*p.events = append(*p.events, Event{Hashes: stored, Parent: parent,
			Tokens:    append([]int64(nil), run.Tokens[numCached*blockSize:numFull*blockSize]...),
			BlockSize: blockSize, LoRA: run.LoRA, Group: g})
	}
}

// WithGroup is make_block_hash_with_group_id (kv_cache_utils.py:74-84): the hash followed by
// the group ID in 4 big-endian bytes.
func WithGroup(h BlockHash, group uint32) []byte {
	return append(append(make([]byte, 0, len(h)+4), h...), byte(group>>24), byte(group>>16), byte(group>>8), byte(group))
}

// insert is _insert_block_hash (block_pool.py:575-609@5df63a72): the first hash a block
// takes is its primary one, with the prefix length it covers; a later one is an extra key to
// the same block.
func (p *Pool) insert(h []byte, b *Block, tokens int) {
	if bytes.Equal(b.Hash, h) || p.holds(h, b) {
		return
	}
	if b.Hash == nil {
		b.Hash, b.HashTokens = h, tokens
	} else {
		b.extra = append(b.extra, h)
	}
	p.cached[string(h)] = append(p.cached[string(h)], b)
}

func (p *Pool) holds(h []byte, b *Block) bool {
	for _, x := range p.cached[string(h)] {
		if x == b {
			return true
		}
	}
	return false
}

// evict removes every hash that reaches b from the cache map and resets b's hash
// (_maybe_evict_cached_block, block_pool.py:729-750; _remove_cached_block_hashes, :549-573),
// queuing one BlockRemoved event per hash with events on (_emit_block_removed_events,
// :611-624). It reports whether b held a hash.
func (p *Pool) evict(b *Block) bool {
	if b.Hash == nil {
		return false
	}
	for _, h := range append([][]byte{b.Hash}, b.extra...) {
		k := string(h)
		bs := p.cached[k]
		found := false
		for i, x := range bs {
			if x == b {
				bs = append(bs[:i], bs[i+1:]...)
				found = true
				break
			}
		}
		if len(bs) == 0 {
			delete(p.cached, k)
		} else {
			p.cached[k] = bs
		}
		if found && p.events != nil {
			n := len(h) - 4
			*p.events = append(*p.events, Event{Removed: true, Hashes: []BlockHash{append(BlockHash(nil), h[:n]...)},
				Group: uint32(h[n])<<24 | uint32(h[n+1])<<16 | uint32(h[n+2])<<8 | uint32(h[n+3])})
		}
	}
	b.Hash, b.HashTokens, b.extra = nil, 0, nil
	return true
}

// MaybeEvictCached is _maybe_evict_cached_block (block_pool.py:729-750@5df63a72): it drops
// every cache entry that reaches b and reports whether there was one.
func (p *Pool) MaybeEvictCached(b *Block) bool { return p.evict(b) }

// IsBlockWritable is is_block_writable (block_pool.py:770-772@5df63a72): a sole owner may
// write a block that is not null and holds no cached hash.
func (p *Pool) IsBlockWritable(b *Block) bool { return !b.Null && b.RefCnt == 1 && b.Hash == nil }

// CachePartialBlock is cache_partial_block (block_pool.py:449-547@5df63a72) in KV cache group
// g: it makes block b reachable under the hash of the prefix that ends numTokens tokens in,
// without a new block. hashes are the request's block hashes at hashBlockSize, numTokens a
// positive multiple of it. With replace, b's earlier entries are removed first (its contents
// were replaced). It returns the hash key (hash plus group), or nil for a null block. With
// events on, a BlockStored event reports the prefix's last hash block unless b already
// held the entry.
func (p *Pool) CachePartialBlock(g uint32, b *Block, hashes []BlockHash, numTokens, hashBlockSize int,
	run StoredRun, replace bool) []byte {
	if b.Null {
		return nil
	}
	if numTokens%hashBlockSize != 0 || numTokens <= 0 || numTokens/hashBlockSize > len(hashes) {
		panic(fmt.Sprintf("kv: partial entry at %d tokens needs a positive multiple of the hash block size %d within %d hashes",
			numTokens, hashBlockSize, len(hashes)))
	}
	n := numTokens / hashBlockSize
	key := WithGroup(hashes[n-1], g)
	already := bytes.Equal(b.Hash, key) || p.holds(key, b)
	if replace {
		p.evict(b)
		already = false
	} else if !already && b.Hash != nil && b.HashTokens < n*hashBlockSize {
		p.evict(b)
	}
	p.insert(key, b, n*hashBlockSize)
	if p.events != nil && !already {
		var parent BlockHash
		if n > 1 {
			parent = hashes[n-2]
		}
		start := (n - 1) * hashBlockSize
		*p.events = append(*p.events, Event{Hashes: []BlockHash{hashes[n-1]}, Parent: parent,
			Tokens:    append([]int64(nil), run.Tokens[start:numTokens]...),
			BlockSize: numTokens - start, LoRA: run.LoRA, Group: g})
	}
	return key
}

// freeQueue is FreeKVCacheBlockQueue (kv_cache_utils.py:246-420): a doubly linked list of
// free blocks, LRU at the head, with O(1) removal from the middle. It starts ordered by
// block ID.
type freeQueue struct {
	head, tail Block // sentinels
	n          int
}

func (q *freeQueue) init(bs []*Block) {
	q.head.next, q.tail.prev = &q.tail, &q.head
	for _, b := range bs {
		q.pushBack(b)
	}
}

func (q *freeQueue) pushBack(b *Block) {
	last := q.tail.prev
	last.next, b.prev, b.next, q.tail.prev = b, last, &q.tail, b
	b.in = q
	q.n++
}

func (q *freeQueue) popLeft() *Block {
	b := q.head.next
	if b == &q.tail {
		panic("kv: no free blocks")
	}
	q.remove(b)
	return b
}

// remove unlinks b. It must be held by this queue, and its neighbours must link back to it,
// so a block in another queue or in none, or a queue whose links are corrupt, panics rather
// than splicing the wrong nodes or miscounting.
func (q *freeQueue) remove(b *Block) {
	if b.in != q || b.prev == nil || b.next == nil || b.prev.next != b || b.next.prev != b || q.n <= 0 {
		panic(fmt.Sprintf("kv: block %d is not in the free queue", b.ID))
	}
	b.prev.next, b.next.prev = b.next, b.prev
	b.prev, b.next, b.in = nil, nil, nil
	q.n--
}

// prependN puts bs at the head, keeping their order (prepend_n).
func (q *freeQueue) prependN(bs []*Block) {
	for i := len(bs) - 1; i >= 0; i-- {
		b, first := bs[i], q.head.next
		q.head.next, b.prev, b.next, first.prev = b, &q.head, first, b
		b.in = q
		q.n++
	}
}

// appendN puts bs at the tail, keeping their order (append_n).
func (q *freeQueue) appendN(bs []*Block) {
	for _, b := range bs {
		q.pushBack(b)
	}
}
