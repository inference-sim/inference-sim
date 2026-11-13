// Original to blis-vllm.

package kv

import (
	"fmt"
	"strings"
	"testing"
)

// The tests drive mambaAlign through the sequence vLLM's allocate_slots runs for one Mamba
// "align" group (vllm/v1/core/kv_cache_manager.py:353-617@5df63a72), and compare block tables
// with what the Python methods give when read line by line; each test cites the lines. Block
// size is 4 tokens, so a table reads at a glance. A table is shown as one letter a block:
// n is the null block, h a block that holds a cached hash, b a block with none.

const alignBS = 4

type alignRig struct {
	t    *testing.T
	pool *Pool
	g    *group
	a    *mambaAlign
	// lastReturned is what allocate_new_blocks returned in the last allocate.
	lastReturned []*Block
}

func newAlignRig(t *testing.T, numBlocks int, cfg mambaAlignConfig) *alignRig {
	t.Helper()
	if cfg.HashBlockSize == 0 {
		cfg.HashBlockSize = alignBS
	}
	if cfg.SchedulerBlockSize == 0 {
		cfg.SchedulerBlockSize = alignBS
	}
	pool := NewPool(numBlocks, true)
	g := &group{id: 0, spec: Group{Kind: Mamba, MambaCacheMode: "align"}, blockSize: alignBS,
		reqBlocks: map[string][]*Block{}, numCached: map[string]int{}}
	a, err := newMambaAlign(pool, g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &alignRig{t: t, pool: pool, g: g, a: a}
}

// simReq is a request: the token count it has (prompt plus output), the prompt's length, the
// tokens computed, and which of its block hashes are shared with other requests.
type simReq struct {
	id        string
	prompt    int
	numTokens int
	computed  int
	shared    int // leading blocks whose hashes are common to the requests that say so
}

func (r *simReq) hashes() []BlockHash {
	n := r.numTokens / alignBS
	out := make([]BlockHash, n)
	for i := range out {
		if i < r.shared {
			out[i] = BlockHash(fmt.Sprintf("shared/%d", i))
		} else {
			out[i] = BlockHash(fmt.Sprintf("%s/%d", r.id, i))
		}
	}
	return out
}

func (r *simReq) mreq() mambaReq {
	return mambaReq{ID: r.id, Hashes: r.hashes(), NumPromptTokens: r.prompt}
}

// shape renders a block table.
func shape(bs []*Block) string {
	var sb strings.Builder
	for _, b := range bs {
		switch {
		case b.Null:
			sb.WriteByte('n')
		case b.Hash != nil:
			sb.WriteByte('h')
		default:
			sb.WriteByte('b')
		}
	}
	return sb.String()
}

// hit is get_computed_blocks for a Mamba-only model (UnitaryKVCacheCoordinator,
// kv_cache_coordinator.py:574-591: alignment is the block size; kv_cache_manager.py:290-319:
// at most num_tokens-1 tokens).
func (r *alignRig) hit(q *simReq) ([]*Block, int) {
	blocks, n := mambaFindLongestHit(r.pool, q.hashes(), q.numTokens-1, []uint32{0}, alignBS, alignBS)
	return blocks[0], n
}

// allocate is allocate_slots (kv_cache_manager.py:371-618) with no lookahead, no connector,
// no watermark, nothing in flight and dense retention (prefill_end 0).
func (r *alignRig) allocate(q *simReq, numNew int, hit []*Block, hitTokens int, mustFit bool) bool {
	local := q.computed + hitTokens
	total := local
	if mustFit {
		full := q.numTokens
		if r.a.numBlocksToAllocate(q.id, hit, total, local, full, true, 0) > r.pool.NumFree() {
			return false
		}
	}
	main := total + numNew
	r.a.removeSkippedBlocks(q.id, max(0, total))
	if r.a.numBlocksToAllocate(q.id, hit, local, local, main, false, 0) > r.pool.NumFree() {
		return false
	}
	if len(hit) > 0 && !r.a.running(q.id) {
		r.a.addLocalComputedBlocks(q.id, hit, local, 0)
	}
	r.lastReturned = r.a.allocateNewBlocks(q.id, main)
	r.a.cacheBlocks(q.mreq(), min(total+numNew, q.numTokens))
	q.computed = total + numNew
	return true
}

func (r *alignRig) table(id string) string { return shape(r.g.reqBlocks[id]) }

func (r *alignRig) want(id, table string, free int) {
	r.t.Helper()
	if got := r.table(id); got != table {
		r.t.Errorf("request %s block table = %q, want %q", id, got, table)
	}
	if got := r.pool.NumFree(); got != free {
		r.t.Errorf("free blocks = %d, want %d", got, free)
	}
}

func mustAllocate(t *testing.T, ok bool, what string) {
	t.Helper()
	if !ok {
		t.Fatalf("%s: allocate_slots refused a request that fits", what)
	}
}

// A prompt of three blocks in one chunk, with no speculative blocks: allocate_new_blocks fills
// the first two slots with null blocks and takes one block (:1812-1893: num_skipped_blocks =
// 3 - 0 - 1 = 2; physical cap 1 + 0 + 0), the admission gate counts that one block
// (:1770-1774), and the block is cached under the prompt's last hash. Each decode step then
// retires the blocks below the last computed token (:1587-1625) and frees the state block
// the previous step ran in, which stays cached for a later hit (free_blocks sends a hashed
// block to the tail of the free queue).
func TestMambaAlign_PrefillDecodeRetiresStatesAndKeepsTheLastCached(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{})
	q := &simReq{id: "r1", prompt: 12, numTokens: 12, shared: 3}
	if got := r.a.numBlocksToAllocate("r1", nil, 0, 0, 12, true, 0); got != 1 {
		t.Fatalf("admission count for a 3-block prompt = %d, want 1 (physical cap)", got)
	}
	mustAllocate(t, r.allocate(q, 12, nil, 0, true), "prefill")
	r.want("r1", "nnh", 18)
	if got := shape(r.lastReturned); got != "nnh" {
		t.Errorf("returned blocks = %q, want the two nulls and the new block", got)
	}
	cached := r.g.reqBlocks["r1"][2]
	if cached.HashTokens != 12 {
		t.Errorf("state block covers %d tokens, want 12", cached.HashTokens)
	}

	// Decode, first token: block 3 is needed (cdiv(13,4) = 4), the table grows by one.
	q.numTokens = 13
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "decode 1")
	r.want("r1", "nnhb", 17)
	if got := r.a.lastStateBlockIdx["r1"]; got != 2 {
		t.Errorf("last_state_block_idx = %d, want 2 (prev_len 3 - 1 - 0)", got)
	}
	// Next step: the state of block 2 is two steps old, remove_skipped_blocks frees it.
	q.numTokens = 14
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "decode 2")
	r.want("r1", "nnnb", 18)
	if cached.RefCnt != 0 || r.pool.Cached(BlockHash("shared/2")) != cached {
		t.Errorf("the freed state block should stay cached with no reference: ref %d", cached.RefCnt)
	}
}

// A hit on a block cached earlier in the same step is refused with num_gpu_blocks + 1
// (:1685-1692), so the request is admitted next step; new_step_starts clears the set (:1971).
func TestMambaAlign_HitOnABlockCachedThisStepWaitsForTheNextStep(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{})
	q1 := &simReq{id: "r1", prompt: 12, numTokens: 12, shared: 3}
	mustAllocate(t, r.allocate(q1, 12, nil, 0, true), "r1")
	q2 := &simReq{id: "r2", prompt: 14, numTokens: 14, shared: 3}
	hit, n := r.hit(q2)
	if n != 12 || shape(hit) != "nnh" {
		t.Fatalf("hit = %q for %d tokens, want nnh for 12", shape(hit), n)
	}
	if got := r.a.numBlocksToAllocate("r2", hit, n, n, 14, false, 0); got != r.pool.NumBlocks()+1 {
		t.Fatalf("same-step hit count = %d, want num_gpu_blocks + 1 = %d", got, r.pool.NumBlocks()+1)
	}
	if r.allocate(q2, 2, hit, n, true) {
		t.Fatal("a request hitting a block cached this step was admitted")
	}
	if got := len(r.g.reqBlocks["r2"]); got != 0 || r.pool.NumFree() != 18 {
		t.Errorf("a refused request changed state: %d blocks, %d free", got, r.pool.NumFree())
	}

	r.a.newStepStarts()
	mustAllocate(t, r.allocate(q2, 2, hit, n, true), "r2 next step")
	r.want("r2", "nnhb", 17)
	if rc := hit[2].RefCnt; rc != 2 {
		t.Errorf("shared state block ref count = %d, want 2 (touched by the hit)", rc)
	}
}

// Free puts a request's blocks back and forgets its bookkeeping (:1935-1953); the same
// request id re-admitted after a preemption hits the prefix it cached and starts as a new
// request (blocks_allocated false), counting the speculative blocks again in its cap.
func TestMambaAlign_FreeAndReadmitHitsTheCachedState(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumSpeculativeBlocks: 2})
	q := &simReq{id: "r1", prompt: 12, numTokens: 12, shared: 3}
	mustAllocate(t, r.allocate(q, 12, nil, 0, true), "prefill")
	// 3 + 2 required, first-time cap 1 + 2: [n n h b b]
	r.want("r1", "nnhbb", 16)
	q.numTokens, q.computed = 13, 12
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "decode")
	r.a.free("r1")
	if len(r.a.allocated)+len(r.a.lastStateBlockIdx)+len(r.a.numRetired)+len(r.a.checkpoints) != 0 {
		t.Errorf("free left bookkeeping: %v %v %v %v", r.a.allocated, r.a.lastStateBlockIdx, r.a.numRetired, r.a.checkpoints)
	}
	if r.pool.NumFree() != 19 || len(r.g.reqBlocks) != 0 {
		t.Fatalf("after free: %d free blocks, %d tables; want 19 and 0", r.pool.NumFree(), len(r.g.reqBlocks))
	}

	r.a.newStepStarts()
	q2 := &simReq{id: "r1", prompt: 12, numTokens: 15, shared: 3}
	hit, n := r.hit(q2)
	if n != 12 || shape(hit) != "nnh" || hit[2].RefCnt != 0 {
		t.Fatalf("re-admission hit = %q for %d tokens (ref %d), want nnh for 12, free", shape(hit), n, hit[2].RefCnt)
	}
	// required = cdiv(15,4) + 2 = 6; 6 - 3 hit blocks = 3 new, capped at 1 + 2 (first allocation)
	// = 3; plus the free hit block: 4.
	if got := r.a.numBlocksToAllocate("r1", hit, n, n, 15, false, 0); got != 4 {
		t.Errorf("re-admission count = %d, want 4", got)
	}
	mustAllocate(t, r.allocate(q2, 3, hit, n, true), "re-admit")
	r.want("r1", "nnhbbb", 19-1-3)
}

// Two speculative blocks and a prompt of 10 tokens in one chunk: [n n b b b] (num_skipped_blocks
// = 5 - 2 - 1 = 2, three new blocks, the first-allocation cap 1 + 2); no block holds a state
// at a block end, so nothing is cached. Decode inside the allocated blocks takes none
// (:1805-1810). Crossing into a new block takes one and records the running state's index
// (:1812-1816); the next step retires the block it replaced.
func TestMambaAlign_SpeculativeBlocksAndDecodeGrowth(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumSpeculativeBlocks: 2})
	q := &simReq{id: "r", prompt: 10, numTokens: 10}
	if got := r.a.numBlocksToAllocate("r", nil, 0, 0, 10, true, 0); got != 3 {
		t.Fatalf("admission count = %d, want 3 (1 + 2 speculative)", got)
	}
	mustAllocate(t, r.allocate(q, 10, nil, 0, true), "prefill")
	r.want("r", "nnbbb", 16)
	if r.pool.NumCached() != 0 {
		t.Errorf("%d blocks cached after a prefill that ends mid-block, want 0", r.pool.NumCached())
	}
	// At 12 tokens the third block is full and is cached (cache_full_blocks, num_full_blocks 3).
	for _, step := range []struct {
		computed, tokens int
		table            string
	}{{10, 11, "nnbbb"}, {11, 12, "nnhbb"}} {
		q.computed, q.numTokens = step.computed, step.tokens
		mustAllocate(t, r.allocate(q, 1, nil, 0, false), "decode in place")
		r.want("r", step.table, 16)
		if len(r.lastReturned) != 0 {
			t.Errorf("decode inside the allocation returned %d blocks", len(r.lastReturned))
		}
	}
	old := append([]*Block(nil), r.g.reqBlocks["r"]...)
	q.computed, q.numTokens = 12, 13
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "new block")
	r.want("r", "nnhbbb", 15)
	got := r.g.reqBlocks["r"]
	for i := range old {
		if got[i] != old[i] {
			t.Errorf("block %d changed identity", i)
		}
	}
	if r.a.lastStateBlockIdx["r"] != 2 {
		t.Errorf("last_state_block_idx = %d, want 2 (prev_len 5 - 1 - 2)", r.a.lastStateBlockIdx["r"])
	}
	q.computed, q.numTokens = 13, 14
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "retire")
	r.want("r", "nnnbbb", 16)
	if old[2].RefCnt != 0 || old[2].Hash == nil {
		t.Errorf("the block at index 2 should be freed and stay cached, ref %d", old[2].RefCnt)
	}
}

// A second prefill chunk of two blocks with two speculative blocks moves a scratch block
// forward (:1836-1848): after chunk 1 the table is [n A B C] (prompt 16, 8 tokens, required
// 2 + 2 = 4, num_skipped 1); chunk 2 needs 4 + 2 = 6 slots, num_skipped 3; the scratch block
// at index 2 is relocated to the end, a null takes its place, one new block is taken (cap
// 1), and the block at index 3 holds the 16-token state and is cached.
func TestMambaAlign_ChunkedPrefillRelocatesAScratchBlock(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumSpeculativeBlocks: 2})
	q := &simReq{id: "r", prompt: 16, numTokens: 16}
	mustAllocate(t, r.allocate(q, 8, nil, 0, true), "chunk 1")
	r.want("r", "nhbb", 16)
	tbl := r.g.reqBlocks["r"]
	a, b, c := tbl[1], tbl[2], tbl[3]
	if a.HashTokens != 8 {
		t.Errorf("chunk 1 state covers %d tokens, want 8", a.HashTokens)
	}
	if got := r.a.numBlocksToAllocate("r", nil, 8, 8, 16, false, 0); got != 1 {
		t.Errorf("chunk 2 count = %d, want 1 (cap 1 for a running request)", got)
	}
	mustAllocate(t, r.allocate(q, 8, nil, 0, true), "chunk 2")
	r.want("r", "nhnhbb", 15)
	tbl = r.g.reqBlocks["r"]
	if tbl[3] != c || tbl[4] != b || tbl[1] != a {
		t.Error("expected the scratch block to move to index 4 and the others to stay")
	}
	if len(r.lastReturned) != 2 || r.lastReturned[0] != b || r.lastReturned[1] != tbl[5] {
		t.Errorf("returned blocks should be the relocated block then the new one, got %d", len(r.lastReturned))
	}
	if c.HashTokens != 16 {
		t.Errorf("block at index 3 covers %d tokens, want 16", c.HashTokens)
	}
	// Next step the old state (index 1, covering 8 tokens) is retired: it stays cached, free.
	q.numTokens = 17
	mustAllocate(t, r.allocate(q, 1, nil, 0, false), "decode")
	r.want("r", "nnnhbbb", 15)
	if a.RefCnt != 0 || a.Hash == nil {
		t.Errorf("retired state block: ref %d, hash set %v; want free and still cached", a.RefCnt, a.Hash != nil)
	}
}

// One prefill-checkpoint block with alignment 1: the prompt's last chunk reserves a block at
// column cdiv(16,4) - 2 = 2 for the state at checkpoint position 12 (:1727-1766; valid per
// kv_cache_interface.py:1124-1148). Nulls fill up to num_skipped - 1 = 2, so the table is
// [n n X Y], and both X (12 tokens) and Y (16) are cached. Without a checkpoint block there is
// one block, [n n n Y].
func TestMambaAlign_CheckpointBlockReservedOnThePromptsLastChunk(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 1})
	q := &simReq{id: "r", prompt: 16, numTokens: 16}
	if got := r.a.numBlocksToAllocate("r", nil, 0, 0, 16, true, 0); got != 2 {
		t.Fatalf("admission count = %d, want 2", got)
	}
	if len(r.a.checkpoints) != 0 {
		t.Fatal("the admission gate must not record a checkpoint")
	}
	mustAllocate(t, r.allocate(q, 16, nil, 0, true), "prefill")
	r.want("r", "nnhh", 17)
	if ck := r.a.checkpoints["r"]; ck != (mambaCheckpoint{12, 2}) {
		t.Errorf("checkpoint = %+v, want position 12 at column 2", ck)
	}
	tbl := r.g.reqBlocks["r"]
	if tbl[2].HashTokens != 12 || tbl[3].HashTokens != 16 {
		t.Errorf("states cover %d and %d tokens, want 12 and 16", tbl[2].HashTokens, tbl[3].HashTokens)
	}
	if r.pool.CachedIn(BlockHash("r/2"), 0) != tbl[2] {
		t.Error("the checkpoint state is not reachable under the hash at 12 tokens")
	}

	none := newAlignRig(t, 20, mambaAlignConfig{PrefillCheckpointAlignment: 1})
	q = &simReq{id: "r", prompt: 16, numTokens: 16}
	mustAllocate(t, none.allocate(q, 16, nil, 0, true), "prefill without checkpoint blocks")
	none.want("r", "nnnh", 18)
}

// With alignment 16 the checkpoint must sit a multiple of 16 tokens past the chunk start
// (is_mamba_prefill_checkpoint_valid): position 12 of a 16-token prompt does not, so there is
// none; position 16 of a 20-token prompt does, at column 3: [n n n X Y].
func TestMambaAlign_CheckpointAlignment16(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 16})
	q := &simReq{id: "short", prompt: 16, numTokens: 16}
	mustAllocate(t, r.allocate(q, 16, nil, 0, true), "16-token prompt")
	r.want("short", "nnnh", 18)
	if len(r.a.checkpoints) != 0 {
		t.Errorf("checkpoint recorded for an unaligned position: %v", r.a.checkpoints)
	}
	q = &simReq{id: "long", prompt: 20, numTokens: 20}
	mustAllocate(t, r.allocate(q, 20, nil, 0, true), "20-token prompt")
	r.want("long", "nnnhh", 16)
	if ck := r.a.checkpoints["long"]; ck != (mambaCheckpoint{16, 3}) {
		t.Errorf("checkpoint = %+v, want position 16 at column 3", ck)
	}
}

// Chunked prefill with a checkpoint: chunk 1 (8 of 20 tokens) reserves column 0 for position
// 4, so the table is [X Y]; chunk 2 first retires X (remove_skipped_blocks), then reserves
// column 3 for position 16: nulls up to num_skipped - 1 = 3 leave one gap at index 2, so the
// table is [n Y n X' Y'] (:1827-1834: prev_len 2, num_skipped_blocks 5 - 0 - 1 = 4).
func TestMambaAlign_ChunkedPrefillWithCheckpoint(t *testing.T) {
	r := newAlignRig(t, 20, mambaAlignConfig{NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 1})
	q := &simReq{id: "r", prompt: 20, numTokens: 20}
	mustAllocate(t, r.allocate(q, 8, nil, 0, true), "chunk 1")
	r.want("r", "hh", 17)
	if ck := r.a.checkpoints["r"]; ck != (mambaCheckpoint{4, 0}) {
		t.Fatalf("chunk 1 checkpoint = %+v, want position 4 at column 0", ck)
	}
	mustAllocate(t, r.allocate(q, 12, nil, 0, true), "chunk 2")
	r.want("r", "nhnhh", 16)
	if ck := r.a.checkpoints["r"]; ck != (mambaCheckpoint{16, 3}) {
		t.Errorf("chunk 2 checkpoint = %+v, want position 16 at column 3", ck)
	}
}

// get_mamba_prefill_checkpoint_position and is_mamba_prefill_checkpoint_valid
// (kv_cache_interface.py:1110-1148@5df63a72), read branch by branch.
func TestMambaPrefillCheckpointHelpers(t *testing.T) {
	for _, c := range []struct {
		n, hbs int
		eagle  bool
		want   int
	}{
		{16, 4, false, 12}, // (16-1)//4*4
		{16, 4, true, 12},  // 16//4*4 - 4
		{18, 4, false, 16},
		{18, 4, true, 12},
		{3, 4, false, 0},
		{4, 4, true, 0}, // 4 - 4
		{2, 4, true, 0}, // negative clamps to 0
	} {
		if got := mambaPrefillCheckpointPosition(c.n, c.hbs, c.eagle); got != c.want {
			t.Errorf("position(%d, %d, eagle=%v) = %d, want %d", c.n, c.hbs, c.eagle, got, c.want)
		}
	}
	for _, c := range []struct {
		name                      string
		qs, qe, pos, hbs, mbs, al int
		want                      bool
	}{
		{"valid", 0, 16, 12, 4, 4, 1, true},
		{"no alignment", 0, 16, 12, 4, 4, 0, false},
		{"unaligned to 16", 0, 16, 12, 4, 4, 16, false},
		{"start off the hash grid", 2, 16, 12, 4, 4, 1, false},
		{"column not past the initial state", 4, 12, 8, 4, 8, 1, false},
		{"column past the initial state", 4, 20, 8, 4, 8, 1, true},
		{"too close to the start", 8, 20, 8, 4, 4, 1, false},
		{"at the query end", 0, 12, 12, 4, 4, 1, false},
	} {
		if got := mambaPrefillCheckpointValid(c.qs, c.qe, c.pos, c.hbs, c.mbs, c.al); got != c.want {
			t.Errorf("%s: valid = %v, want %v", c.name, got, c.want)
		}
	}
}

func cachedAt(t *testing.T, p *Pool, group uint32, tag string, tokens int) *Block {
	t.Helper()
	b := p.New(1)[0]
	p.insert(WithGroup(BlockHash(tag), group), b, tokens)
	return b
}

// find_longest_cache_hit (:1450-1520): the last cached block that ends on the alignment, with
// null blocks in front, at most max_length tokens; all groups of the spec group must have it.
func TestMambaFindLongestHit(t *testing.T) {
	p := NewPool(20, true)
	h := []BlockHash{BlockHash("h0"), BlockHash("h1"), BlockHash("h2"), BlockHash("h3"), BlockHash("h4")}
	b1 := cachedAt(t, p, 0, "h1", 8)
	b3 := cachedAt(t, p, 0, "h3", 16)

	blocks, n := mambaFindLongestHit(p, h, 20, []uint32{0}, 4, 4)
	if n != 16 || shape(blocks[0]) != "nnnh" || blocks[0][3] != b3 {
		t.Errorf("max 20: hit %d tokens, table %q", n, shape(blocks[0]))
	}
	blocks, n = mambaFindLongestHit(p, h, 15, []uint32{0}, 4, 4)
	if n != 8 || shape(blocks[0]) != "nh" || blocks[0][1] != b1 {
		t.Errorf("max 15: hit %d tokens, table %q", n, shape(blocks[0]))
	}
	if blocks, n = mambaFindLongestHit(p, h, 7, []uint32{0}, 4, 4); n != 0 || len(blocks[0]) != 0 {
		t.Errorf("max 7: hit %d tokens, %d blocks, want a miss", n, len(blocks[0]))
	}
	// Hit lengths must be multiples of the alignment (8): the state at 12 tokens does not count.
	b2 := cachedAt(t, p, 0, "h2", 12)
	blocks, n = mambaFindLongestHit(p, h, 15, []uint32{0}, 4, 8)
	if n != 8 || blocks[0][1] != b1 {
		t.Errorf("alignment 8, max 15: hit %d tokens, want 8 (skipping the unaligned 12)", n)
	}
	_ = b2
	// Two groups look up together.
	if _, n = mambaFindLongestHit(p, h, 20, []uint32{0, 1}, 4, 4); n != 0 {
		t.Errorf("a hash cached in group 0 only hit %d tokens in groups 0 and 1", n)
	}
	cachedAt(t, p, 1, "h3", 16)
	blocks, n = mambaFindLongestHit(p, h, 20, []uint32{0, 1}, 4, 4)
	if n != 16 || len(blocks) != 2 || shape(blocks[1]) != "nnnh" {
		t.Errorf("both groups cached h3: hit %d tokens, %d lists", n, len(blocks))
	}
}

// Partial hash hits are refused by name, not half ported.
func TestMambaAlign_RefusesPartialHashHits(t *testing.T) {
	p := NewPool(8, true)
	g := &group{blockSize: 8, reqBlocks: map[string][]*Block{}, numCached: map[string]int{}}
	_, err := newMambaAlign(p, g, mambaAlignConfig{HashBlockSize: 4, SchedulerBlockSize: 8})
	if err == nil || !strings.Contains(err.Error(), "enable_partial_hash_hits") {
		// The hash block size (4) below the Mamba block (8) is the partial-hit setup.
		t.Fatalf("hash block below the mamba block: error %v, want one naming enable_partial_hash_hits", err)
	}
	g.blockSize = 4
	if _, err = newMambaAlign(p, g, mambaAlignConfig{HashBlockSize: 4, SchedulerBlockSize: 8}); err != nil {
		t.Errorf("a mamba block of 4 over a scheduler block of 8 is valid: %v", err)
	}
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(fmt.Sprint(r), "enable_partial_hash_hits") {
				t.Errorf("a find with alignment below the block size: recovered %v", r)
			}
		}()
		mambaFindLongestHit(p, nil, 8, []uint32{0}, 8, 4)
	}()
	a, _ := newMambaAlign(p, g, mambaAlignConfig{HashBlockSize: 4, SchedulerBlockSize: 4})
	defer func() {
		if r := recover(); r == nil || !strings.Contains(fmt.Sprint(r), "enable_partial_hash_hits") {
			t.Errorf("a hit ending inside a block: recovered %v", r)
		}
	}()
	a.addLocalComputedBlocks("x", []*Block{p.null, p.New(1)[0]}, 6, 0)
}
