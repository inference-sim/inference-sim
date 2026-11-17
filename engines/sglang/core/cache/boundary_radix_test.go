// Original to blis-sglang.

package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"slices"
	"testing"
)

type radixRig struct {
	t     *testing.T
	cache *RadixCache
	alloc *TokenAllocator
	pool  *RequestPool
}

func newRig(t *testing.T, size int, cfg RadixConfig) *radixRig {
	t.Helper()
	cfg.PageSize = max(cfg.PageSize, 1)
	alloc := NewTokenAllocator(size, nil, false)
	pool := NewRequestPool(8, 64)
	c, err := NewRadixCache(cfg, alloc, pool)
	if err != nil {
		t.Fatal(err)
	}
	return &radixRig{t: t, cache: c, alloc: alloc, pool: pool}
}

// finished runs a request of the given prompt and output through the pool and checkpoints it
// into the tree, as the scheduler does when the request ends.
func (r *radixRig) finished(tokens []int64, promptLen int) *Req {
	r.t.Helper()
	req := &Req{ID: "r", OriginInputIDs: tokens[:promptLen], FillIDs: tokens}
	if _, ok := r.pool.Alloc([]*Req{req}); !ok {
		r.t.Fatal("no row")
	}
	slots, ok := r.alloc.Alloc(len(tokens))
	if !ok {
		r.t.Fatal("no slots")
	}
	r.pool.Write(*req.KV.ReqPoolIdx, 0, slots)
	req.KV.KVAllocatedLen = len(tokens)
	r.cache.Checkpoint(req, len(tokens))
	return req
}

// A checkpoint inserts the request's KV and splits the leaf at the prompt boundary, so eviction
// can drop the output and keep the prompt (unified_radix_cache.py:1250-1271@297e54fd); a second
// checkpoint with nothing new is accepted.
func TestCheckpointSplitsAtThePromptAndRepeats(t *testing.T) {
	r := newRig(t, 32, RadixConfig{})
	req := r.finished([]int64{1, 2, 3, 4, 5, 6, 7, 8}, 5)
	r.cache.Unlock(req.Lock)
	if got := r.cache.EvictableSize(); got != 8 {
		t.Fatalf("evictable %d, want the 8 inserted tokens", got)
	}
	if got := r.cache.Evict(EvictParams{NumTokens: 3}).NumTokensEvicted; got != 3 {
		t.Errorf("evicting 3 tokens evicted %d: the output must be a leaf apart from the prompt", got)
	}
	if m := r.cache.MatchPrefix(MatchPrefixParams{Key: Key{Tokens: []int64{1, 2, 3, 4, 5, 6, 7, 8}}}); m.DevicePrefixLen != 5 {
		t.Errorf("after the output went, %d tokens still match, want the prompt's 5", m.DevicePrefixLen)
	}

	r2 := newRig(t, 32, RadixConfig{})
	whole := r2.finished([]int64{1, 2, 3, 4}, 4) // prompt is the whole sequence
	r2.cache.Checkpoint(whole, 4)                // nothing new to insert: the protected prefix covers it
	r2.cache.Unlock(whole.Lock)
	if got := r2.cache.Evict(EvictParams{NumTokens: 4}).NumTokensEvicted; got != 4 {
		t.Errorf("a prompt that is the whole sequence: evicted %d of 4", got)
	}
}

// Evicting zero tokens evicts nothing, and an eviction for an allocation stops the moment the
// allocator can serve the shortfall, not one leaf later (unified_radix_cache.py:952-1058).
func TestEvictionStopsAtItsTarget(t *testing.T) {
	r := newRig(t, 20, RadixConfig{})
	for _, toks := range [][]int64{{1, 2, 3}, {10, 11, 12}, {20, 21, 22}} {
		req := r.finished(toks, len(toks))
		r.cache.Unlock(req.Lock)
	}
	if got := r.cache.Evict(EvictParams{NumTokens: 0}).NumTokensEvicted; got != 0 || r.cache.EvictableSize() != 9 {
		t.Errorf("evicting 0 tokens evicted %d, %d evictable remain", got, r.cache.EvictableSize())
	}
	before := r.alloc.AvailableSize() // 11
	if got := r.cache.EvictForAlloc(EvictParams{NumTokens: 3}).NumTokensEvicted; got != 3 {
		t.Errorf("an allocation short by 3 evicted %d tokens", got)
	}
	if got := r.alloc.AvailableSize(); got != before+3 {
		t.Errorf("available %d, want %d: the walk goes on past its target", got, before+3)
	}
	if got := r.cache.EvictForAlloc(EvictParams{NumTokens: 0}).NumTokensEvicted; got != 0 {
		t.Errorf("no shortfall evicted %d", got)
	}
}

// Freed row ranges that touch are freed as one segment; ranges with a gap stay apart.
func TestFreedRowRangesCoalesceOnlyWhenTheyTouch(t *testing.T) {
	if got := coalesce([][2]int{{0, 2}, {2, 5}, {7, 9}, {9, 10}}); !slices.Equal(got, [][2]int{{0, 5}, {7, 10}}) {
		t.Errorf("coalesced %v", got)
	}
	if got := coalesce([][2]int{{0, 2}, {3, 5}}); !slices.Equal(got, [][2]int{{0, 2}, {3, 5}}) {
		t.Errorf("a gap was closed: %v", got)
	}
	r := newRig(t, 16, RadixConfig{})
	req := &Req{}
	r.pool.Alloc([]*Req{req})
	slots, _ := r.alloc.Alloc(10)
	r.pool.Write(*req.KV.ReqPoolIdx, 0, slots)
	r.cache.FreeKVRow(req, [][2]int{{0, 3}, {3, 6}})
	if r.alloc.AvailableSize() != 6+6 {
		t.Errorf("available %d after freeing 6 of 10", r.alloc.AvailableSize())
	}
	r.cache.FreeKVRow(req, nil)
	if r.alloc.AvailableSize() != 12 {
		t.Errorf("an empty free changed the pool to %d", r.alloc.AvailableSize())
	}
}

// Events are queued only when asked for and not on a disabled cache; a node's stored block names
// its parent's last block as its parent, and a removal names the blocks that left (events.py:140-302).
func TestKVEventsFollowTheTreeAndTheirSwitch(t *testing.T) {
	off := newRig(t, 16, RadixConfig{})
	off.finished([]int64{1, 2, 3}, 3)
	if ev := off.cache.TakeEvents(); len(ev) != 0 {
		t.Errorf("events without being asked for: %v", ev)
	}
	disabled := newRig(t, 16, RadixConfig{KVCacheEvents: true, Disable: true})
	disabled.finished([]int64{1, 2, 3}, 3)
	for _, e := range disabled.cache.TakeEvents() {
		if e.Type == "BlockStored" {
			t.Errorf("a disabled cache stored a block: %v", e)
		}
	}

	r := newRig(t, 32, RadixConfig{KVCacheEvents: true, PageSize: 2})
	a := r.finished([]int64{1, 2, 3, 4}, 4)
	r.cache.Unlock(a.Lock)
	b := r.finished([]int64{1, 2, 9, 9}, 4) // splits the first node after its first page
	r.cache.Unlock(b.Lock)
	var stored []Event
	for _, e := range r.cache.TakeEvents() {
		if e.Type == "BlockStored" {
			stored = append(stored, e)
		}
	}
	if len(stored) != 2 {
		t.Fatalf("%d stored events, want two: %v", len(stored), stored)
	}
	if !slices.Equal(stored[0].TokenIDs, []int64{1, 2, 3, 4}) || stored[0].ParentBlockHash != nil || len(stored[0].BlockHashes) != 2 {
		t.Errorf("first event %+v", stored[0])
	}
	if !slices.Equal(stored[1].TokenIDs, []int64{9, 9}) || stored[1].ParentBlockHash == nil ||
		*stored[1].ParentBlockHash != stored[0].BlockHashes[0] {
		t.Errorf("second event %+v does not name the first page's block as its parent", stored[1])
	}
	r.cache.Evict(EvictParams{NumTokens: 2}) // the older leaf, tokens 3 and 4
	var removed []Event
	for _, e := range r.cache.TakeEvents() {
		if e.Type == "BlockRemoved" {
			removed = append(removed, e)
		}
	}
	if len(removed) != 1 || len(removed[0].BlockHashes) != 1 || removed[0].BlockHashes[0] != stored[0].BlockHashes[1] {
		t.Errorf("removal %v, want the one block of the evicted page", removed)
	}
}

// A page's hash is the chained SHA-256 of its tokens as 32-bit words, or, when any token needs 33
// bits or more, the 0xff tag and 64-bit words; a bigram page is its pairs
// (cpp_utils/hash_binding.cpp:224-275@297e54fd).
func TestPageHashesAreTheChainedSHA256(t *testing.T) {
	word := func(prior []byte, wide bool, ws ...uint64) []byte {
		h := sha256.New()
		h.Write(prior)
		if wide {
			h.Write([]byte{0xff})
		}
		for _, w := range ws {
			if wide {
				h.Write(binary.LittleEndian.AppendUint64(nil, w))
			} else {
				h.Write(binary.LittleEndian.AppendUint32(nil, uint32(w)))
			}
		}
		return h.Sum(nil)
	}
	p1 := word(nil, false, 1, 2)
	p2 := word(p1, false, 3)
	if got := pageHashes(Key{Tokens: []int64{1, 2, 3}}, nil, 2); !slices.Equal(got, []string{hex.EncodeToString(p1), hex.EncodeToString(p2)}) {
		t.Errorf("pages of 2 over 3 tokens: %v", got)
	}
	// 0xffffffff still fits in 32 bits; 0x100000000 does not, and one wide token widens its page.
	if got := pageHashes(Key{Tokens: []int64{0xffffffff, 7}}, nil, 2); got[0] != hex.EncodeToString(word(nil, false, 0xffffffff, 7)) {
		t.Errorf("a 32-bit token was hashed wide: %v", got)
	}
	if got := pageHashes(Key{Tokens: []int64{7, 0x100000000}}, nil, 2); got[0] != hex.EncodeToString(word(nil, true, 7, 0x100000000)) {
		t.Errorf("a 33-bit token was hashed narrow: %v", got)
	}
	// A bigram key of tokens 1, 2, 3 has two units, (1, 2) and (2, 3).
	if got := pageHashes(Key{Tokens: []int64{1, 2, 3}, Bigram: true}, nil, 2); got[0] != hex.EncodeToString(word(nil, false, 1, 2, 2, 3)) {
		t.Errorf("a bigram page: %v", got)
	}
	// Namespaces are equal only when both the extra key and the salt are.
	x, y := "x", "y"
	if !sameNamespace(Key{ExtraKey: &x, CacheSalt: &y}, Key{ExtraKey: &x, CacheSalt: &y}) ||
		sameNamespace(Key{ExtraKey: &x, CacheSalt: &y}, Key{ExtraKey: &x, CacheSalt: &x}) ||
		sameNamespace(Key{ExtraKey: &x}, Key{ExtraKey: &y}) || sameNamespace(Key{ExtraKey: &x}, Key{}) ||
		!sameNamespace(Key{}, Key{}) {
		t.Error("namespace equality is wrong")
	}
}

// A bigram (EAGLE) page is published as its pairs of tokens, flattened.
func TestBigramPagesAreStoredAsPairs(t *testing.T) {
	r := newRig(t, 16, RadixConfig{KVCacheEvents: true, IsEagle: true, PageSize: 2})
	req := r.finished([]int64{1, 2, 3}, 3)
	r.cache.Unlock(req.Lock)
	var got []int64
	for _, e := range r.cache.TakeEvents() {
		if e.Type == "BlockStored" {
			got = append(got, e.TokenIDs...)
		}
	}
	if !slices.Equal(got, []int64{1, 2, 2, 3}) {
		t.Errorf("stored tokens %v, want the pairs (1, 2) and (2, 3)", got)
	}
}

// A lock on the root, which a request that matched nothing holds, protects nothing and releases
// cleanly; a lock released twice is refused.
func TestLocksOnTheRootAndDoubleReleases(t *testing.T) {
	r := newRig(t, 16, RadixConfig{})
	m := r.cache.MatchPrefix(MatchPrefixParams{Key: Key{Tokens: []int64{5, 6}}})
	l := r.cache.Lock(m.LastDeviceNode)
	if l.Receipt.Node != m.LastDeviceNode || r.cache.ProtectedSize() != 0 {
		t.Errorf("the root lock protects %d tokens", r.cache.ProtectedSize())
	}
	r.cache.Unlock(l)
	req := r.finished([]int64{1, 2, 3}, 3)
	if r.cache.ProtectedSize() != 3 {
		t.Errorf("protected %d, want 3 under the request's lock", r.cache.ProtectedSize())
	}
	r.cache.Unlock(req.Lock)
	if r.cache.ProtectedSize() != 0 || r.cache.EvictableSize() != 3 {
		t.Errorf("protected %d, evictable %d after the release", r.cache.ProtectedSize(), r.cache.EvictableSize())
	}
	mustPanic(t, "a second release of one lock", func() { r.cache.Unlock(req.Lock) })
}

// A request with no prompt tokens checkpoints its output alone.
func TestCheckpointOfAnEmptyPrompt(t *testing.T) {
	r := newRig(t, 16, RadixConfig{})
	req := r.finished([]int64{1, 2, 3}, 0)
	r.cache.Unlock(req.Lock)
	if got := r.cache.Evict(EvictParams{NumTokens: 3}).NumTokensEvicted; got != 3 {
		t.Errorf("evicted %d of 3", got)
	}
}
