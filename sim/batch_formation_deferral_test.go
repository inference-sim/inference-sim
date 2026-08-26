package sim

import (
	"sort"
	"testing"
)

// fakeDeferKV is a KVStore + DeferrableKVStore double that drives the FormBatch
// skip-scan deterministically: `pending` ids defer (AllocateKVBlocks registers them
// and returns false), `resolved` ids admit (and clear their deferral), `gpuFull`
// ids fail allocation WITHOUT being deferred (GPU pressure → head-of-line break).
type fakeDeferKV struct {
	pending  map[string]bool
	resolved map[string]bool
	gpuFull  map[string]bool
	deferred map[string]bool
	admitted []string
	polls    int
}

func newFakeDeferKV() *fakeDeferKV {
	return &fakeDeferKV{
		pending:  map[string]bool{},
		resolved: map[string]bool{},
		gpuFull:  map[string]bool{},
		deferred: map[string]bool{},
	}
}

func (f *fakeDeferKV) AllocateKVBlocks(req *Request, _, _ int64, _ []int64) bool {
	switch {
	case f.resolved[req.ID]:
		delete(f.deferred, req.ID) // admitted; episode over
		f.admitted = append(f.admitted, req.ID)
		return true
	case f.pending[req.ID]:
		f.deferred[req.ID] = true // register the deferral (still in flight)
		return false
	case f.gpuFull[req.ID]:
		return false // GPU pressure, NOT a deferral
	default:
		f.admitted = append(f.admitted, req.ID)
		return true
	}
}

// PollDeferred returns the still-pending (not resolved) tracked ids, sorted.
func (f *fakeDeferKV) PollDeferred(_ int64) []string {
	f.polls++
	out := make([]string, 0, len(f.deferred))
	for id := range f.deferred {
		if !f.resolved[id] {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// IsDeferred is true only while pending (resolved requests break like GPU pressure).
func (f *fakeDeferKV) IsDeferred(id string) bool { return f.deferred[id] && !f.resolved[id] }

func (f *fakeDeferKV) ClearDeferred(id string) { delete(f.deferred, id) }

// Inert KVStore stubs.
func (f *fakeDeferKV) GetCachedBlocks(_ []TokenID) []int64  { return nil }
func (f *fakeDeferKV) ReleaseKVBlocks(_ *Request)           {}
func (f *fakeDeferKV) BlockSize() int64                     { return 16 }
func (f *fakeDeferKV) UsedBlocks() int64                    { return 0 }
func (f *fakeDeferKV) TotalCapacity() int64                 { return 1 << 30 }
func (f *fakeDeferKV) CacheHitRate() float64                { return 0 }
func (f *fakeDeferKV) PendingTransferLatency() int64        { return 0 }
func (f *fakeDeferKV) ConsumePendingTransferLatency() int64 { return 0 }
func (f *fakeDeferKV) KVThrashingRate() float64             { return 0 }
func (f *fakeDeferKV) SetClock(_ int64)                     {}
func (f *fakeDeferKV) MirrorToCPU(_ []*Request)             {}

var _ DeferrableKVStore = (*fakeDeferKV)(nil)

func deferReq(id string) *Request {
	return &Request{ID: id, InputTokens: make([]TokenID, 16), OutputTokens: make([]TokenID, 4), State: StateQueued}
}

func deferCtx(wq *WaitQueue, kv KVStore) BatchContext {
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

// A still-deferred request at the queue head must NOT block admittable requests
// behind it — batch formation skips it and admits the rest (vLLM step_skipped_waiting
// + continue). The deferred request stays in the WaitQ, re-polled next step.
func TestFormBatch_DeferredRequestDoesNotBlockOthers(t *testing.T) {
	kv := newFakeDeferKV()
	kv.pending["A"] = true // A's secondary fetch is in flight
	wq := &WaitQueue{}
	wq.Enqueue(deferReq("A"))
	wq.Enqueue(deferReq("B"))
	wq.Enqueue(deferReq("C"))

	bf := NewBatchFormation("")
	result := bf.FormBatch(deferCtx(wq, kv))

	ids := map[string]bool{}
	for _, r := range result.RunningBatch.Requests {
		ids[r.ID] = true
	}
	if ids["A"] {
		t.Fatalf("A is still fetching from a secondary tier; it must not be admitted")
	}
	if !ids["B"] || !ids["C"] {
		t.Fatalf("B and C must be admitted past the deferred A, got %v", ids)
	}
	// A remains in the WaitQ (re-polled next step); B and C were removed.
	if wq.Len() != 1 || wq.Peek().ID != "A" {
		t.Fatalf("only the deferred A must remain in the WaitQ, got len=%d head=%v", wq.Len(), wq.Peek())
	}
}

// Once its fetch resolves, a formerly-deferred request is admitted on a later step.
func TestFormBatch_ResolvedDeferralIsAdmitted(t *testing.T) {
	kv := newFakeDeferKV()
	kv.pending["A"] = true
	wq := &WaitQueue{}
	wq.Enqueue(deferReq("A"))

	bf := NewBatchFormation("")
	// Step 1: A defers, nothing admitted.
	r1 := bf.FormBatch(deferCtx(wq, kv))
	if len(r1.RunningBatch.Requests) != 0 || wq.Len() != 1 {
		t.Fatalf("step 1: A must defer and remain queued, admitted=%d qlen=%d", len(r1.RunningBatch.Requests), wq.Len())
	}
	// Fetch completes.
	kv.resolved["A"] = true
	// Step 2: A is admitted and leaves the WaitQ.
	r2 := bf.FormBatch(deferCtx(wq, kv))
	if len(r2.RunningBatch.Requests) != 1 || r2.RunningBatch.Requests[0].ID != "A" {
		t.Fatalf("step 2: resolved A must be admitted, got %v", r2.RunningBatch.Requests)
	}
	if wq.Len() != 0 {
		t.Fatalf("step 2: admitted A must be removed from the WaitQ, qlen=%d", wq.Len())
	}
}

// GPU pressure (an allocation failure that is NOT a deferral) still breaks
// head-of-line — a full head stalls the requests behind it, as before H3.
func TestFormBatch_GPUPressureStillBreaksHeadOfLine(t *testing.T) {
	kv := newFakeDeferKV()
	kv.gpuFull["A"] = true // A cannot allocate (GPU full), and is not deferred
	wq := &WaitQueue{}
	wq.Enqueue(deferReq("A"))
	wq.Enqueue(deferReq("B"))

	bf := NewBatchFormation("")
	result := bf.FormBatch(deferCtx(wq, kv))

	if len(result.RunningBatch.Requests) != 0 {
		t.Fatalf("GPU-pressured head must block the batch (head-of-line), admitted=%d", len(result.RunningBatch.Requests))
	}
	if wq.Len() != 2 {
		t.Fatalf("both requests must remain queued behind the GPU-pressured head, qlen=%d", wq.Len())
	}
}

// A deferred request that is skipped, followed by a GPU-pressured request, admits
// nothing but leaves both queued — skip (A) then break (B).
func TestFormBatch_SkipThenBreak(t *testing.T) {
	kv := newFakeDeferKV()
	kv.pending["A"] = true
	kv.gpuFull["B"] = true
	wq := &WaitQueue{}
	wq.Enqueue(deferReq("A"))
	wq.Enqueue(deferReq("B"))
	wq.Enqueue(deferReq("C"))

	bf := NewBatchFormation("")
	result := bf.FormBatch(deferCtx(wq, kv))
	if len(result.RunningBatch.Requests) != 0 {
		t.Fatalf("A is skipped and B breaks head-of-line, so nothing is admitted, got %d", len(result.RunningBatch.Requests))
	}
	if wq.Len() != 3 {
		t.Fatalf("all three requests must remain queued (A deferred, B/C behind the break), qlen=%d", wq.Len())
	}
}
