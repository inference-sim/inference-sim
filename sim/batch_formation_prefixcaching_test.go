package sim

import "testing"

// prefixSharingPair builds two prompts with an identical block-aligned prefix and
// different tails, so only cross-request prefix caching can reduce the second prefill.
func prefixSharingPair(blockSize, sharedBlocks, tailBlocks int64) (*Request, *Request) {
	shared := make([]TokenID, 0, sharedBlocks*blockSize)
	for i := int64(0); i < sharedBlocks*blockSize; i++ {
		shared = append(shared, TokenID(1000+i))
	}
	build := func(id string, tailSeed TokenID) *Request {
		tokens := append([]TokenID(nil), shared...)
		for i := int64(0); i < tailBlocks*blockSize; i++ {
			tokens = append(tokens, tailSeed+TokenID(i))
		}
		return &Request{ID: id, InputTokens: tokens, State: StateQueued}
	}
	return build("first", 20000), build("second", 40000)
}

func cachePrefixFixture(t *testing.T, kv KVStore, req *Request) {
	t.Helper()
	if ok := kv.AllocateKVBlocks(req, 0, req.InputLen(), nil); !ok {
		t.Fatal("fixture: failed to allocate prefix request")
	}
	kv.ReleaseKVBlocks(req)
}

func chargedPrefill(t *testing.T, kv KVStore, disabled bool, req *Request) int {
	t.Helper()
	wq := &WaitQueue{}
	wq.Enqueue(req)
	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kv,
		MaxNumBatchedTokens:   1 << 20,
		MaxNumSeqs:            64,
		PrefillTokenThreshold: 0,
		Now:                   0,
		ComputedTokens:        make(map[string]int64),
		PrefixCachingDisabled: disabled,
	}
	result := NewBatchFormation("").FormBatch(ctx)
	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("expected one admitted request, got %d", len(result.RunningBatch.Requests))
	}
	return result.RunningBatch.Requests[0].NumNewTokens
}

// The engine-level switch must change actual scheduled work, not merely configuration.
// Removing the GetCachedBlocks gate makes the disabled arm incorrectly receive the same
// cross-request prefix credit as the enabled arm, so this test fails behaviorally.
func TestDisablingPrefixCachingChargesWholePrompt(t *testing.T) {
	const blockSize, sharedBlocks, tailBlocks int64 = 16, 4, 2
	wantWholePrompt := int((sharedBlocks + tailBlocks) * blockSize)

	kvOn := MustNewKVCacheState(4096, blockSize)
	first, second := prefixSharingPair(blockSize, sharedBlocks, tailBlocks)
	cachePrefixFixture(t, kvOn, first)
	withCaching := chargedPrefill(t, kvOn, false, second)

	kvOff := MustNewKVCacheState(4096, blockSize)
	firstOff, secondOff := prefixSharingPair(blockSize, sharedBlocks, tailBlocks)
	cachePrefixFixture(t, kvOff, firstOff)
	withoutCaching := chargedPrefill(t, kvOff, true, secondOff)

	if withoutCaching != wantWholePrompt {
		t.Fatalf("prefix caching disabled: charged %d tokens, want whole %d-token prompt",
			withoutCaching, wantWholePrompt)
	}
	wantCachedWork := int(tailBlocks * blockSize)
	if withCaching != wantCachedWork {
		t.Fatalf("default/enabled prefix caching should credit the %d-token shared prefix and charge %d tail tokens, got %d",
			sharedBlocks*blockSize, wantCachedWork, withCaching)
	}
	if withCaching >= withoutCaching {
		t.Fatalf("prefix caching enabled should reduce scheduled prefill work: enabled=%d disabled=%d",
			withCaching, withoutCaching)
	}
}

// The zero value is the compatibility path: when no prefix matches, enabling/disabling
// caching cannot alter scheduled work.
func TestDisablingPrefixCachingIsInertWithoutSharedPrefix(t *testing.T) {
	const blockSize int64 = 16
	build := func(id string, seed TokenID) *Request {
		tokens := make([]TokenID, 96)
		for i := range tokens {
			tokens[i] = seed + TokenID(i)
		}
		return &Request{ID: id, InputTokens: tokens, State: StateQueued}
	}

	kvOn := MustNewKVCacheState(4096, blockSize)
	cachePrefixFixture(t, kvOn, build("other", 10000))
	withCaching := chargedPrefill(t, kvOn, false, build("target", 50000))

	kvOff := MustNewKVCacheState(4096, blockSize)
	cachePrefixFixture(t, kvOff, build("other", 10000))
	withoutCaching := chargedPrefill(t, kvOff, true, build("target", 50000))

	if withCaching != withoutCaching || withCaching != 96 {
		t.Fatalf("non-sharing prompt should be charged identically: enabled=%d disabled=%d want=96",
			withCaching, withoutCaching)
	}
}

// A RUNNING request resumes from its own ProgressIndex in Phase 1. The engine switch only
// controls cross-request reuse for new admissions and must not restart chunked prefill.
func TestDisablingPrefixCachingDoesNotRestartChunkedPrefill(t *testing.T) {
	const blockSize int64 = 16
	run := func(disabled bool) int {
		kv := MustNewKVCacheState(4096, blockSize)
		req, _ := prefixSharingPair(blockSize, 4, 2)
		req.State = StateRunning
		req.ProgressIndex = 32
		if ok := kv.AllocateKVBlocks(req, 0, 32, nil); !ok {
			t.Fatal("fixture: failed to allocate running request")
		}
		ctx := BatchContext{
			RunningBatch:          &Batch{Requests: []*Request{req}},
			WaitQ:                 &WaitQueue{},
			KVCache:               kv,
			MaxNumBatchedTokens:   1 << 20,
			MaxNumSeqs:            64,
			Now:                   0,
			ComputedTokens:        make(map[string]int64),
			PrefixCachingDisabled: disabled,
		}
		result := NewBatchFormation("").FormBatch(ctx)
		if len(result.RunningBatch.Requests) != 1 {
			t.Fatalf("running request vanished from the batch (disabled=%v)", disabled)
		}
		return result.RunningBatch.Requests[0].NumNewTokens
	}

	withCaching, withoutCaching := run(false), run(true)
	if withCaching != withoutCaching {
		t.Fatalf("chunked prefill must resume identically: enabled=%d disabled=%d",
			withCaching, withoutCaching)
	}
	if want := 96 - 32; withCaching != want {
		t.Fatalf("request with 32/96 tokens computed should schedule %d more, got %d",
			want, withCaching)
	}
}

// GPU prefix reuse and offload-tier reload are separate mechanisms. Disabling the former
// must not silently disable a same-step CPU->GPU reload reported by ReloadablePrefixEnd.
func TestDisablingPrefixCachingDoesNotDisableOffloadReload(t *testing.T) {
	kv := newFakeReloadKV(16)
	kv.reloadEnd["A"] = 48

	wq := &WaitQueue{}
	wq.Enqueue(reloadReq("A", 64))
	ctx := reloadCtx(wq, kv)
	ctx.PrefixCachingDisabled = true

	result := NewBatchFormation("").FormBatch(ctx)
	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("reloadable request must be admitted, got %d", len(result.RunningBatch.Requests))
	}
	if got := result.RunningBatch.Requests[0].NumNewTokens; got != 16 {
		t.Fatalf("offload reload should still credit 48 tokens and bill the 16-token tail, got %d", got)
	}
	if got := ctx.ComputedTokens["A"]; got != 64 {
		t.Fatalf("offload reload should advance computed progress to 64, got %d", got)
	}
}
