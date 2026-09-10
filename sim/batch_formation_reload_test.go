package sim

import "testing"

// fakeReloadKV is a KVStore + ReloadReportingKVStore double (#1699). It models an
// offload store that, during AllocateKVBlocks, reloads a CPU/secondary-resident prefix
// onto the GPU and reports the enlarged boundary via ReloadedPrefixEnd. GetCachedBlocks
// returns the PRE-reload GPU prefix (here: none), exactly as OffloadCache does — so
// batch formation's first numNewTokens is the full input and must be re-billed down.
type fakeReloadKV struct {
	blockSize int64
	// reloadEnd[reqID] = post-reload cached prefix boundary (token index). The value is
	// returned once by ReloadedPrefixEnd (one-shot), matching OffloadCache.
	reloadEnd map[string]int64
	consumed  map[string]bool
}

func newFakeReloadKV(blockSize int64) *fakeReloadKV {
	return &fakeReloadKV{blockSize: blockSize, reloadEnd: map[string]int64{}, consumed: map[string]bool{}}
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

func (f *fakeReloadKV) ReloadedPrefixEnd(reqID string) (int64, bool) {
	if f.consumed[reqID] {
		return 0, false
	}
	v, ok := f.reloadEnd[reqID]
	if ok {
		f.consumed[reqID] = true
	}
	return v, ok
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

// Under a MaxModelLen-capped chunk, billing is the non-reloaded tail WITHIN the chunk:
// the cap sets endIndex, the reload covers [startIndex, newStart), so billed =
// endIndex-newStart. Progress (ComputedTokens) still reaches the capped endIndex.
func TestFormBatch_ReloadRespectsMaxModelLen(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 32 // 2 blocks reloaded, within the chunk

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	ctx.MaxModelLen = 40 // caps numNewTokens to 39 => endIndex=39; billed = 39-32 = 7
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 7 {
		t.Fatalf("billed tail within the capped chunk must be endIndex-newStart=39-32=7, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 39 {
		t.Fatalf("progress must reach the capped endIndex=39, got ComputedTokens=%d", ctx.ComputedTokens["A"])
	}
}

// A reload can extend the GPU prefix PAST a capped chunk (newStart > endIndex). Only
// [startIndex, endIndex) was committed this step, so the whole capped chunk is free
// this step (billed 0), progress reaches endIndex, and the tail beyond it is billed on
// a later step — never over-crediting progress to InputLen.
func TestFormBatch_ReloadBeyondCappedChunk(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 64 // whole input reloaded to GPU...

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))

	ctx := reloadCtx(wq, kv)
	ctx.PrefillTokenThreshold = 16 // ...but this step only processes a 16-token chunk
	bf := NewBatchFormation("")
	result := bf.FormBatch(ctx)

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("A must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if a := result.RunningBatch.Requests[0]; a.NumNewTokens != 0 {
		t.Fatalf("a chunk fully covered by the reload must bill 0, got NumNewTokens=%d", a.NumNewTokens)
	}
	if ctx.ComputedTokens["A"] != 16 {
		t.Fatalf("progress must advance only to the capped endIndex=16 (not InputLen), got ComputedTokens=%d", ctx.ComputedTokens["A"])
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
