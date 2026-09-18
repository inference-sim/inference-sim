package sim

import "testing"

// fakeReloadKV is a KVStore + ReloadReportingKVStore double (#1699, #1706). It models an
// offload store whose same-step CPU->GPU reload would extend a new request's GPU-cached
// prefix to reloadEnd[reqID], reported by the PURE pre-allocation query ReloadablePrefixEnd
// (the analogue of vLLM's get_num_new_matched_tokens). GetCachedBlocks returns the
// PRE-reload GPU prefix (here: none), exactly as OffloadCache does — so batch formation
// folds the reloadable boundary into the computed baseline before sizing the chunk.
type fakeReloadKV struct {
	blockSize int64
	// reloadEnd[reqID] = the CPU-reloadable prefix boundary (token index). Reported by
	// the pure query ReloadablePrefixEnd; not consumed (matches the #1706 pre-query).
	reloadEnd map[string]int64
}

func newFakeReloadKV(blockSize int64) *fakeReloadKV {
	return &fakeReloadKV{blockSize: blockSize, reloadEnd: map[string]int64{}}
}

func (f *fakeReloadKV) AllocateKVBlocks(_ *Request, _, _ int64, _ []int64) bool { return true }
func (f *fakeReloadKV) GetCachedBlocks(_ []TokenID) []int64                     { return nil } // GPU prefix: none
func (f *fakeReloadKV) ReleaseKVBlocks(_ *Request)                              {}
func (f *fakeReloadKV) BlockSize() int64                                        { return f.blockSize }
func (f *fakeReloadKV) UsedBlocks() int64                                       { return 0 }
func (f *fakeReloadKV) TotalCapacity() int64                                    { return 1 << 30 }
func (f *fakeReloadKV) CacheHitRate() float64                                   { return 0 }
func (f *fakeReloadKV) PendingTransferLatency() int64                           { return 0 }
func (f *fakeReloadKV) ConsumePendingTransferLatency() int64                    { return 0 }
func (f *fakeReloadKV) KVThrashingRate() float64                                { return 0 }
func (f *fakeReloadKV) SetClock(_ int64)                                        {}
func (f *fakeReloadKV) MirrorToCPU(_ []*Request)                                {}

func (f *fakeReloadKV) ReloadablePrefixEnd(req *Request, startIndex int64) (int64, bool) {
	v, ok := f.reloadEnd[req.ID]
	if ok && v > startIndex {
		return v, true
	}
	return startIndex, false
}

var (
	_ KVStore                = (*fakeReloadKV)(nil)
	_ ReloadReportingKVStore = (*fakeReloadKV)(nil)
)

func reloadReq(id string, inputTokens int) *Request {
	return &Request{ID: id, InputTokens: make([]TokenID, inputTokens), OutputTokens: make([]TokenID, 4), State: StateQueued}
}

func reloadCtx(wq *WaitQueue, kv KVStore) BatchContext {
	return BatchContext{
		RunningBatch:        &Batch{},
		WaitQ:               wq,
		KVCache:             kv,
		MaxNumBatchedTokens: 100000,
		MaxNumSeqs:          10,
		Now:                 1000,
		StepCount:           1,
		ComputedTokens:      make(map[string]int64),
	}
}

// A CPU/secondary reload that extends the cached prefix must SHRINK the billed prefill
// work: NumNewTokens becomes InputLen-newStart, ComputedTokens becomes InputLen, and the
// token budget is debited by the smaller (re-billed) value (#1699).
func TestFormBatch_ReloadReducesNumNewTokens(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 48 // 3 of 4 blocks reloaded from CPU

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64)) // 64-token input, 4 blocks

	ctx := reloadCtx(wq, kv)
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	a := result.RunningBatch.Requests[0]
	if a.NumNewTokens != 16 {
		t.Fatalf("reload of 48 tokens must bill only the 16-token tail, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 64 {
		t.Fatalf("ComputedTokens must reach InputLen=64 (reloaded prefix + billed tail), got %d", ctx.ComputedTokens["A"])
	}
}

// A full-prefix reload (newStart == InputLen) is a valid zero-work admission: the
// request enters the batch billed 0 prefill tokens and transitions straight to decode.
func TestFormBatch_FullReloadZeroPrefill(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 64 // entire 64-token input reloaded

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("full-hit A must still be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 0 {
		t.Fatalf("a full-prefix reload must bill 0 prefill tokens, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 64 {
		t.Fatalf("full hit must set ComputedTokens=InputLen=64, got %d", ctx.ComputedTokens["A"])
	}
}

// MaxModelLen caps the uncached remainder after the reload is folded in (#1706): with
// computedStart=32 the remainder is InputLen-32=32, but MaxModelLen=40 caps it to
// 40-1-32=7, so billed=7 and progress reaches computedStart+7=39.
func TestFormBatch_ReloadRespectsMaxModelLen(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 32 // 2 blocks reloadable => computedStart=32

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	ctx.MaxModelLen = 40 // caps the remainder to 40-1-32=7 => endIndex=39
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 7 {
		t.Fatalf("billed remainder must be MaxModelLen-capped to 40-1-32=7, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 39 {
		t.Fatalf("progress must reach computedStart+billed=32+7=39, got ComputedTokens=%d", ctx.ComputedTokens["A"])
	}
}

// A reload that covers the WHOLE input must credit the whole reloaded prefix even when a
// chunk cap is set (#1706): folding the reloadable boundary into the computed baseline
// BEFORE the chunk clamp (vLLM ordering) means the chunk sizes on the uncached remainder,
// which is zero here — so the request is billed 0 prefill work and progress reaches
// InputLen in one step, NOT capped at the (now-irrelevant) chunk size. This replaces the
// pre-#1706 assertion (NumNewTokens=0 but progress capped at endIndex=16, which re-billed
// the remainder as recompute on later steps).
func TestFormBatch_ReloadBeyondCappedChunk(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 64 // whole 64-token input reloadable from CPU...

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	ctx.PrefillTokenThreshold = 16 // ...and a 16-token chunk cap must NOT cap the credit
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 0 {
		t.Fatalf("a fully-reloaded input must bill 0 prefill work regardless of the chunk cap, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 64 {
		t.Fatalf("the whole reloaded prefix must be credited: progress must reach InputLen=64, not the chunk cap, got ComputedTokens=%d", ctx.ComputedTokens["A"])
	}
}

// A reload extends the prefix PAST a capped chunk but leaves an uncached remainder
// (#1706): the chunk covers only the uncached tail beyond the reloadable boundary. Input
// 96 (6 blocks), 64 reloadable, chunk cap 16 => computedStart=64, remainder=32 capped to
// 16, so billed=16, progress reaches 64+16=80. The reloaded [0,64) is fully credited (not
// re-billed as recompute), and only the true uncached remainder is chunked.
func TestFormBatch_ReloadThenChunkedRemainder(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 64 // 4 of 6 blocks reloadable

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 96))

	ctx := reloadCtx(wq, kv)
	ctx.PrefillTokenThreshold = 16
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 16 {
		t.Fatalf("chunk must cover only the uncached remainder tail (cap 16), got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 80 {
		t.Fatalf("progress must reach computedStart+chunk = 64+16 = 80, got ComputedTokens=%d", ctx.ComputedTokens["A"])
	}
}

// A store that does NOT implement ReloadReportingKVStore (single-tier / legacy) leaves
// NumNewTokens at the pre-reload value — the INV-6 byte-identity no-op path.
func TestFormBatch_NonReloadingStoreUnchanged(t *testing.T) {
	// fakeDeferKV (from batch_formation_deferral_test.go) is a plain KVStore that does
	// not implement ReloadReportingKVStore.
	kv := newFakeDeferKV()
	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 64 {
		t.Fatalf("a non-reporting store must bill the full input (no re-bill), got NumNewTokens=%d", a.NumNewTokens)
	}
}
