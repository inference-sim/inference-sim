package sim

import "testing"

// prefixSharingPair builds two requests with an identical leading prefix and distinct
// tails, which is the shape prefix caching exists to exploit.
func prefixSharingPair(blockSize, sharedBlocks, tailBlocks int64) (*Request, *Request) {
	shared := make([]TokenID, 0, sharedBlocks*blockSize)
	for i := int64(0); i < sharedBlocks*blockSize; i++ {
		shared = append(shared, TokenID(1000+i))
	}
	build := func(id string, tailSeed TokenID) *Request {
		toks := append([]TokenID(nil), shared...)
		for i := int64(0); i < tailBlocks*blockSize; i++ {
			toks = append(toks, tailSeed+TokenID(i))
		}
		return &Request{ID: id, InputTokens: toks, State: StateQueued}
	}
	return build("first", 500000), build("second", 900000)
}

// formOneStep admits whatever it can from the wait queue and reports how many NEW tokens
// the named request was charged.
func formOneStep(t *testing.T, kv KVStore, blockSize int64, disabled bool,
	reqs ...*Request) map[string]int {
	t.Helper()
	wq := &WaitQueue{}
	for _, r := range reqs {
		wq.Enqueue(r)
	}
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
	res := NewBatchFormation("").FormBatch(ctx)
	charged := map[string]int{}
	for _, r := range res.RunningBatch.Requests {
		charged[r.ID] = r.NumNewTokens
	}
	return charged
}

// cacheThePrefixOf puts a request's blocks into the cache, the way the simulator does: one
// allocation cycle followed by a release, which retains the block hashes for a later match.
func cacheThePrefixOf(kv KVStore, req *Request) {
	kv.AllocateKVBlocks(req, 0, req.InputLen(), nil)
	kv.ReleaseKVBlocks(req)
}

// With prefix caching ON, a request whose leading blocks another request already placed in
// the cache is charged only for the remainder. With it OFF, it is charged for the whole
// prompt. This is the behaviour --no-enable-prefix-caching selects, and it changes the work
// a step does rather than only the memory it holds (#1867).
func TestDisablingPrefixCachingChargesTheWholePrompt(t *testing.T) {
	const blockSize, sharedBlocks, tailBlocks = 16, 4, 2
	total := int((sharedBlocks + tailBlocks) * blockSize)

	// Caching ON: admit the first request, let its blocks become cached, then admit the
	// second and see what it is charged.
	kvOn := MustNewKVCacheState(4096, blockSize)
	first, second := prefixSharingPair(blockSize, sharedBlocks, tailBlocks)
	cacheThePrefixOf(kvOn, first)
	onCharged := formOneStep(t, kvOn, blockSize, false, second)["second"]

	// Caching OFF: same sequence, same cache state.
	kvOff := MustNewKVCacheState(4096, blockSize)
	first2, second2 := prefixSharingPair(blockSize, sharedBlocks, tailBlocks)
	cacheThePrefixOf(kvOff, first2)
	offCharged := formOneStep(t, kvOff, blockSize, true, second2)["second"]

	if offCharged != total {
		t.Errorf("with caching DISABLED the second request should be charged its whole "+
			"%d-token prompt, got %d", total, offCharged)
	}
	if onCharged >= offCharged {
		t.Errorf("with caching ENABLED the second request should be charged LESS than the "+
			"whole prompt: on=%d off=%d. If these are equal the gate is not reachable, or "+
			"the fixture shares no prefix", onCharged, offCharged)
	}
	t.Logf("charged: caching on %d tokens, caching off %d tokens (prompt %d)",
		onCharged, offCharged, total)
}

// Disabling reuse must not disturb a request that shares nothing. This is the INV-6 half:
// a workload with no shared prefix has nothing to credit, so the two settings must charge
// identically and the gate must be invisible.
func TestDisablingPrefixCachingIsInertWithoutASharedPrefix(t *testing.T) {
	const blockSize int64 = 16
	build := func(id string, seed TokenID) *Request {
		toks := make([]TokenID, 0, 96)
		for i := 0; i < 96; i++ {
			toks = append(toks, seed+TokenID(i))
		}
		return &Request{ID: id, InputTokens: toks, State: StateQueued}
	}

	kvOn := MustNewKVCacheState(4096, blockSize)
	cacheThePrefixOf(kvOn, build("other", 111000))
	on := formOneStep(t, kvOn, blockSize, false, build("solo", 777000))["solo"]

	kvOff := MustNewKVCacheState(4096, blockSize)
	cacheThePrefixOf(kvOff, build("other", 111000))
	off := formOneStep(t, kvOff, blockSize, true, build("solo", 777000))["solo"]

	if on != off {
		t.Errorf("with no shared prefix the setting must not change what is charged: "+
			"on=%d off=%d", on, off)
	}
	if on != 96 {
		t.Errorf("a request sharing nothing should be charged its whole 96-token prompt, "+
			"got %d", on)
	}
}

// Phase 1 re-chunks a RUNNING request from its own ProgressIndex and is not reached by the
// gate, so a partially-prefilled request must advance identically either way. Without this,
// a plausible-looking gate placed one call earlier would silently restart every chunked
// prefill.
func TestDisablingPrefixCachingDoesNotRestartAChunkedPrefill(t *testing.T) {
	const blockSize int64 = 16
	run := func(disabled bool) int {
		kv := MustNewKVCacheState(4096, blockSize)
		req, _ := prefixSharingPair(blockSize, 4, 2)
		req.State = StateRunning
		req.ProgressIndex = 32
		kv.AllocateKVBlocks(req, 0, 32, nil)
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
		res := NewBatchFormation("").FormBatch(ctx)
		for _, r := range res.RunningBatch.Requests {
			if r.ID == req.ID {
				return r.NumNewTokens
			}
		}
		t.Fatalf("the running request vanished from the batch (disabled=%v)", disabled)
		return 0
	}
	on, off := run(false), run(true)
	if on != off {
		t.Errorf("a chunked prefill must resume identically: on=%d off=%d", on, off)
	}
	if want := 96 - 32; on != want {
		t.Errorf("a request with 32 of 96 tokens computed should be charged %d more, got %d",
			want, on)
	}
}
