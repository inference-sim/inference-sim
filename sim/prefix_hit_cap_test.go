package sim

import "testing"

// #1943: a prompt whose every block is a prefix-cache hit must still compute its last token,
// as vLLM v0.31.0 caps the hit at num_tokens - 1 (KVCacheManager.get_computed_blocks,
// vllm/v1/core/kv_cache_manager.py:295, "we must recompute the last token to obtain
// logits"). Otherwise the admission schedules no work, claims none of the cached blocks, and
// the request runs holding no KV. It bites both a request resumed after preemption (its own
// blocks, freed with their hashes) and a first admission whose block-aligned prompt another
// request left cached.
func TestAFullyCachedPromptRecomputesItsLastBlockAndHoldsItsKV(t *testing.T) {
	const blockSize = 4
	prompt := []TokenID{1, 2, 3, 4, 5, 6, 7, 8} // two whole blocks
	for _, tt := range []struct {
		name string
		id   string // the admitted request's id: the earlier holder's (resume) or a new one
	}{
		{"resumed after preemption", "a"},
		{"first admission over another request's prefix", "b"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			kv := MustNewKVCacheState(16, blockSize)
			first := &Request{ID: "a", InputTokens: append([]TokenID(nil), prompt...)}
			if !kv.AllocateKVBlocks(first, 0, int64(len(prompt)), nil) {
				t.Fatal("setup allocation refused")
			}
			kv.ReleaseKVBlocks(first) // both blocks free, hashed: a full prefix hit
			if got := len(kv.GetCachedBlocks(prompt)); got != 2 {
				t.Fatalf("setup: %d cached blocks, want 2", got)
			}

			req := &Request{ID: tt.id, InputTokens: append([]TokenID(nil), prompt...),
				OutputTokens: make([]TokenID, 3), State: StateQueued}
			wq := &WaitQueue{}
			wq.Enqueue(req)
			ctx := BatchContext{RunningBatch: &Batch{}, WaitQ: wq, KVCache: kv,
				MaxNumBatchedTokens: 64, MaxNumSeqs: 4, ComputedTokens: map[string]int64{}}
			res := NewBatchFormation("").FormBatch(ctx)
			if len(res.NewlyScheduled) != 1 {
				t.Fatalf("the request was not admitted")
			}
			// The hit is capped to the first block (7 tokens -> 1 whole block), so the last
			// block, 4 tokens, is computed again.
			if req.NumNewTokens != blockSize {
				t.Errorf("scheduled %d prompt tokens, want the last block's %d", req.NumNewTokens, blockSize)
			}
			if ctx.ComputedTokens[req.ID] != int64(len(prompt)) {
				t.Errorf("computed tokens %d after admission, want the whole prompt %d", ctx.ComputedTokens[req.ID], len(prompt))
			}
			if used := kv.UsedBlocks(); used != 2 {
				t.Errorf("the admitted request holds %d blocks, want its prompt's 2", used)
			}
		})
	}
}

// A prompt that is not block-aligned already leaves its tail uncached, so the cap changes
// nothing: the whole-block prefix is credited and only the tail is computed.
func TestAPartlyCachedPromptKeepsItsWholeBlockHit(t *testing.T) {
	kv := MustNewKVCacheState(16, 4)
	prompt := []TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	first := &Request{ID: "a", InputTokens: append([]TokenID(nil), prompt...)}
	if !kv.AllocateKVBlocks(first, 0, int64(len(prompt)), nil) {
		t.Fatal("setup allocation refused")
	}
	kv.ReleaseKVBlocks(first)
	req := &Request{ID: "b", InputTokens: append([]TokenID(nil), prompt...), OutputTokens: make([]TokenID, 2), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(req)
	NewBatchFormation("").FormBatch(BatchContext{RunningBatch: &Batch{}, WaitQ: wq, KVCache: kv,
		MaxNumBatchedTokens: 64, MaxNumSeqs: 4, ComputedTokens: map[string]int64{}})
	if req.NumNewTokens != 2 {
		t.Errorf("scheduled %d tokens, want only the 2-token uncached tail", req.NumNewTokens)
	}
	if used := kv.UsedBlocks(); used != 3 {
		t.Errorf("holds %d blocks, want 3", used)
	}
}

// DeepSeek-V4.1's SWA bounded replay computes a hit's last window again without storing it
// again (vLLM v0.31.0 sched/scheduler.py:1059-1079, 3038-3050): the admission bills the
// replayed tokens and the uncached tail, claims the whole hit, allocates only the tail, and
// still reports the whole hit. A step budget that ends inside the replay skips it.
func TestBoundedReplayRecomputesTheHitsLastWindow(t *testing.T) {
	const blockSize, replay = 4, 8
	prompt := make([]TokenID, 40)
	for i := range prompt {
		prompt[i] = TokenID(i + 1)
	}
	for _, tt := range []struct {
		name           string
		budget, billed int64
		wantComputed   int64
	}{
		{"budget covers the replay", 64, replay + 8, 48},
		{"budget ends inside the replay", replay, replay, 48},
	} {
		t.Run(tt.name, func(t *testing.T) {
			kv := NewKVStoreFromConfig(NewKVCacheConfig(32, blockSize, 0, 0, 0, 0,
				WithPrefixHit(KVPrefixHit{ReplayTokens: replay})), 0)
			first := &Request{ID: "a", InputTokens: append([]TokenID(nil), prompt...)}
			if !kv.AllocateKVBlocks(first, 0, 40, nil) {
				t.Fatal("setup allocation refused")
			}
			kv.ReleaseKVBlocks(first)
			in := append(append([]TokenID(nil), prompt...), 101, 102, 103, 104, 105, 106, 107, 108)
			req := &Request{ID: "b", InputTokens: in, OutputTokens: make([]TokenID, 2), State: StateQueued}
			wq := &WaitQueue{}
			wq.Enqueue(req)
			ctx := BatchContext{RunningBatch: &Batch{}, WaitQ: wq, KVCache: kv,
				MaxNumBatchedTokens: tt.budget, MaxNumSeqs: 4, ComputedTokens: map[string]int64{}}
			res := NewBatchFormation("").FormBatch(ctx)
			if len(res.NewlyScheduled) != 1 {
				t.Fatalf("the request was not admitted")
			}
			if int64(req.NumNewTokens) != tt.billed {
				t.Errorf("billed %d prompt tokens, want %d", req.NumNewTokens, tt.billed)
			}
			if got := ctx.ComputedTokens[req.ID]; got != tt.wantComputed {
				t.Errorf("computed tokens %d after admission, want %d", got, tt.wantComputed)
			}
			if got := res.NewlyScheduled[0].Lookup.HitTokens; got != 40 {
				t.Errorf("reported hit %d, want the whole 40-token hit", got)
			}
			if used := kv.UsedBlocks(); used != 12 {
				t.Errorf("holds %d blocks, want the prompt's 12 (the replay stores nothing)", used)
			}
		})
	}
}
