// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/block_pool.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/kv_cache_utils.py (KVCacheBlock, FreeKVCacheBlockQueue).

package kv

import "fmt"

// Block is one KV block: vLLM's KVCacheBlock (kv_cache_utils.py:178-235). A block's ID
// is an allocator detail; what a block holds is its hash and reference count.
type Block struct {
	ID     int
	RefCnt int
	// Hash is set while the block is a full, cached prefix block, nil otherwise. It is the
	// block hash with its KV cache group ID appended as 4 big-endian bytes, as vLLM keys
	// its cache (make_block_hash_with_group_id, kv_cache_utils.py:74-84).
	Hash []byte
	// Null marks the pool's reserved null block, which never holds KV.
	Null bool

	prev, next *Block
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
}

// EnableEvents makes the pool record KV cache events (enable_kv_cache_events).
func (p *Pool) EnableEvents() { p.events = &[]Event{} }

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
// (evicted first), one with a hash to its tail (free_blocks, block_pool.py:774-805).
func (p *Pool) Free(ordered []*Block) {
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
	var stored []BlockHash
	for i := numCached; i < numFull; i++ {
		b := blocks[i]
		if b.Null {
			continue
		}
		if b.Hash != nil {
			p.evict(b)
		}
		p.insert(WithGroup(hashes[i], g), b)
		stored = append(stored, hashes[i])
	}
	if p.events != nil {
		var parent BlockHash
		if numCached > 0 {
			parent = hashes[numCached-1]
		}
		*p.events = append(*p.events, Event{Hashes: stored, Parent: parent,
			Tokens:    append([]int64(nil), run.Tokens[numCached*blockSize:numFull*blockSize]...),
			BlockSize: blockSize, LoRA: run.LoRA})
	}
}

// WithGroup is make_block_hash_with_group_id (kv_cache_utils.py:74-84): the hash followed by
// the group ID in 4 big-endian bytes.
func WithGroup(h BlockHash, group uint32) []byte {
	return append(append(make([]byte, 0, len(h)+4), h...), byte(group>>24), byte(group>>16), byte(group>>8), byte(group))
}

func (p *Pool) insert(h []byte, b *Block) {
	for _, x := range p.cached[string(h)] {
		if x == b {
			return
		}
	}
	b.Hash = h
	p.cached[string(h)] = append(p.cached[string(h)], b)
}

// evict removes b's hash from the cache map (_maybe_evict_cached_block, block_pool.py:729-750),
// queuing a BlockRemoved event with events on (_emit_block_removed_events, :611-624).
func (p *Pool) evict(b *Block) {
	if b.Hash == nil {
		return
	}
	if p.events != nil {
		*p.events = append(*p.events, Event{Removed: true, Hashes: []BlockHash{append(BlockHash(nil), b.Hash[:len(b.Hash)-4]...)}})
	}
	k := string(b.Hash)
	bs := p.cached[k]
	for i, x := range bs {
		if x == b {
			bs = append(bs[:i], bs[i+1:]...)
			break
		}
	}
	if len(bs) == 0 {
		delete(p.cached, k)
	} else {
		p.cached[k] = bs
	}
	b.Hash = nil
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

func (q *freeQueue) remove(b *Block) {
	if b.prev == nil || b.next == nil {
		panic(fmt.Sprintf("kv: block %d is not in the free queue", b.ID))
	}
	b.prev.next, b.next.prev = b.next, b.prev
	b.prev, b.next = nil, nil
	q.n--
}

// prependN puts bs at the head, keeping their order (prepend_n).
func (q *freeQueue) prependN(bs []*Block) {
	for i := len(bs) - 1; i >= 0; i-- {
		b, first := bs[i], q.head.next
		q.head.next, b.prev, b.next, first.prev = b, &q.head, first, b
		q.n++
	}
}

// appendN puts bs at the tail, keeping their order (append_n).
func (q *freeQueue) appendN(bs []*Block) {
	for _, b := range bs {
		q.pushBack(b)
	}
}
