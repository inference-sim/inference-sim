// Original to blis-vllm.

package kv

import "testing"

// A partial entry (cache_partial_block, block_pool.py:449-547@5df63a72) makes a block with a
// primary hash reachable under a second key; evicting the block removes both
// (_remove_cached_block_hashes, :549-573), and each removal is reported.
func TestPoolCachePartialBlock_SecondKeyAndEviction(t *testing.T) {
	p := NewPool(6, true)
	p.EnableEvents()
	b := p.New(1)[0]
	hashes := []BlockHash{BlockHash("a"), BlockHash("b"), BlockHash("c")}
	run := StoredRun{Tokens: make([]int64, 12)}

	// Full hash at 8 tokens (primary), then a partial entry at 4 tokens.
	p.insert(WithGroup(hashes[1], 0), b, 8)
	key := p.CachePartialBlock(0, b, hashes, 4, 4, run, false)
	if string(key) != string(WithGroup(hashes[0], 0)) {
		t.Fatalf("partial key = %x, want the hash at 4 tokens", key)
	}
	if b.HashTokens != 8 {
		t.Errorf("primary hash covers %d tokens, want it kept at 8", b.HashTokens)
	}
	if p.CachedIn(hashes[0], 0) != b || p.CachedIn(hashes[1], 0) != b {
		t.Fatal("both keys should reach the block")
	}
	if ev := p.TakeEvents(); len(ev) != 1 || len(ev[0].Hashes) != 1 || string(ev[0].Hashes[0]) != "a" || ev[0].BlockSize != 4 {
		t.Errorf("events after a partial entry = %+v, want one BlockStored of the hash at 4 tokens", ev)
	}
	// Registering the same entry again stores nothing new.
	p.CachePartialBlock(0, b, hashes, 4, 4, run, false)
	if ev := p.TakeEvents(); len(ev) != 0 {
		t.Errorf("a repeated partial entry queued %d events", len(ev))
	}
	if !p.MaybeEvictCached(b) {
		t.Fatal("MaybeEvictCached reported no entry")
	}
	if p.CachedIn(hashes[0], 0) != nil || p.CachedIn(hashes[1], 0) != nil || b.Hash != nil || b.HashTokens != 0 {
		t.Error("eviction left an entry or the hash metadata")
	}
	if ev := p.TakeEvents(); len(ev) != 2 || !ev[0].Removed || !ev[1].Removed {
		t.Errorf("eviction events = %+v, want two BlockRemoved", ev)
	}
	if p.MaybeEvictCached(b) {
		t.Error("evicting a block without entries should report false")
	}
}

// replace_existing_hashes drops what the block was cached under, because its contents were
// replaced (:510-513); without it a shorter earlier prefix is replaced by a longer one
// (:514-519) but not the reverse.
func TestPoolCachePartialBlock_Replace(t *testing.T) {
	p := NewPool(6, true)
	b := p.New(1)[0]
	hashes := []BlockHash{BlockHash("a"), BlockHash("b"), BlockHash("c")}
	run := StoredRun{Tokens: make([]int64, 12)}
	p.CachePartialBlock(0, b, hashes, 8, 4, run, false)
	p.CachePartialBlock(0, b, hashes, 12, 4, run, false) // longer: replaces
	if p.CachedIn(hashes[1], 0) != nil || p.CachedIn(hashes[2], 0) != b || b.HashTokens != 12 {
		t.Error("a longer partial entry should replace the shorter one")
	}
	p.CachePartialBlock(0, b, hashes, 4, 4, run, true) // replace: only the new key remains
	if p.CachedIn(hashes[2], 0) != nil || p.CachedIn(hashes[0], 0) != b || b.HashTokens != 4 {
		t.Error("replace should leave only the new key")
	}
	if p.CachePartialBlock(0, p.null, hashes, 4, 4, run, false) != nil {
		t.Error("a null block takes no entry")
	}
}

// is_block_writable (:770-772): a sole owner may write a block with no hash.
func TestPoolIsBlockWritable(t *testing.T) {
	p := NewPool(4, true)
	b := p.New(1)[0]
	if !p.IsBlockWritable(b) {
		t.Error("an unshared unhashed block is writable")
	}
	b.RefCnt++
	if p.IsBlockWritable(b) {
		t.Error("a shared block is not writable")
	}
	b.RefCnt--
	p.insert(WithGroup(BlockHash("x"), 0), b, 4)
	if p.IsBlockWritable(b) || p.IsBlockWritable(p.null) {
		t.Error("a hashed or null block is not writable")
	}
}
