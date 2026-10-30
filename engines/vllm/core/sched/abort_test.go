package sched

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// loadingConnector supplies every prompt but its last block asynchronously, so a new request
// waits in WAITING_FOR_REMOTE_KVS with its blocks allocated while the load is in flight.
type loadingConnector struct{ finished []string }

func (c *loadingConnector) OnNewRequest(*Request) {}
func (c *loadingConnector) MatchedTokens(r *Request, local int) (int, bool) {
	return len(r.Prompt) - 4 - local, true
}
func (c *loadingConnector) AfterAlloc(*Request, AllocatedBlocks, int) {}
func (c *loadingConnector) Finished(r *Request, _ []int) bool {
	c.finished = append(c.finished, r.ID)
	return false
}
func (c *loadingConnector) AfterOutput(TransferOutcome) {}

// An abort that names a request twice finishes it once, as vLLM's finish_requests takes the
// ids as a set (vllm/v1/core/sched/scheduler.py:2667-2668@5df63a72). Counted twice, the second
// copy would see the first's FINISHED_ABORTED status and free blocks the in-flight load is
// still writing, report the request finished twice and tell the connector twice.
func TestAbortNamingARequestTwiceFinishesItOnce(t *testing.T) {
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40,
		MaxModelLen: 128, EnableChunkedPrefill: true, Policy: FCFS, EnablePrefixCaching: true,
		ReserveFullSequence: true})
	if err != nil {
		t.Fatal(err)
	}
	c := &loadingConnector{}
	if err := s.SetConnector(c, ConnectorConfig{}); err != nil {
		t.Fatal(err)
	}
	prompt := make([]int64, 16)
	for i := range prompt {
		prompt[i] = int64(100 + i)
	}
	if err := s.Add(&Request{ID: "a", Prompt: prompt, MaxTokens: 4, EOS: eosToken, Transfer: &TransferParams{}}); err != nil {
		t.Fatal(err)
	}
	s.Schedule()
	if got := s.requests["a"].Status; got != WaitingForRemoteKVs {
		t.Fatalf("setup: the request is %v, want it waiting for its remote KV", got)
	}
	free := s.KV.Pool.NumFree()

	s.Abort([]string{"a", "a"})

	if got := s.KV.Pool.NumFree(); got != free {
		t.Errorf("free blocks %d -> %d: the blocks of an in-flight load were freed", free, got)
	}
	if len(c.finished) != 1 {
		t.Errorf("the connector was told the request finished %d times, want once", len(c.finished))
	}
	if s.finishedPending != 1 {
		t.Errorf("finishedPending = %d, want 1", s.finishedPending)
	}
}

// A request's block hashes are its blocks' keys exactly as kv.BlockHashes computes them over
// prompt and output together, LoRA key and first-block cache salt included, whether a block
// lies in the prompt, in the output, or across the boundary.
func TestRequestHashesAreTheBlockHashesOfItsTokens(t *testing.T) {
	keys := kv.BlockKeys{Extra: []kv.ExtraKey{{"lora", "a1", "a1"}}, CacheSalt: "salt"}
	prompt := []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	r := &Request{Prompt: prompt, Keys: keys}
	r.updateHashes(4)
	for _, tok := range []int64{11, 12, 13, 14, 15, 16, 17} {
		r.Output = append(r.Output, tok)
		r.updateHashes(4)
	}
	want := kv.BlockHashes(append(append([]int64(nil), prompt...), r.Output...), 4, keys)
	if len(r.hashes) != len(want) {
		t.Fatalf("%d hashes, want %d", len(r.hashes), len(want))
	}
	for i := range want {
		if string(r.hashes[i]) != string(want[i]) {
			t.Errorf("block %d: hash differs from kv.BlockHashes", i)
		}
	}
}

// hmaConnector is loadingConnector stating it supports the hybrid memory allocator and the
// given capabilities, recording the per-group block ids it is handed at finish.
type hmaConnector struct {
	loadingConnector
	caps    Capabilities
	groups  [][]int
	alloced AllocatedBlocks
}

func (c *hmaConnector) ConnectorModels() Capabilities           { return c.caps }
func (c *hmaConnector) MatchedTokens(*Request, int) (int, bool) { return 0, false }
func (c *hmaConnector) FinishedAllGroups(r *Request, ids [][]int) bool {
	c.groups = ids
	return c.Finished(r, nil)
}
func (c *hmaConnector) AfterAlloc(_ *Request, b AllocatedBlocks, _ int) { c.alloced = b }

// A connector is refused where it is not modelled for it, each refusal naming the issue that
// brings it, and installed where its Capabilities state it is diff-tested: several KV cache
// groups only for a SupportsHMA connector (vLLM turns the hybrid manager off for any other,
// vllm/config/vllm.py:2170-2192@5df63a72), drafters and the batch queue only where stated, and
// never beside a group kind the connector oracle has not run (a bounded-replay window, a ring,
// a group of its own block size).
func TestSetConnectorRefusesWhatItDoesNotModel(t *testing.T) {
	base := Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40, MaxModelLen: 128,
		EnableChunkedPrefill: true, Policy: FCFS, EnablePrefixCaching: true, ReserveFullSequence: true}
	hybrid := func(c *Config) {
		c.KVGroups = kv.ResolveGroups([]kv.Group{{Kind: kv.FullAttention}, {Kind: kv.SlidingWindow, Window: 8}}, false)
	}
	drafter := func(c *Config) { c.Spec = ResolveSpec(SpecOptions{Method: "dspark", NumTokens: 7}) }
	async := func(c *Config) { c.Async, c.MaxConcurrentBatches = true, 2 }
	nixlLike := func() Connector { return &hmaConnector{caps: Capabilities{HybridGroups: true, Drafters: true}} }
	plain := func() Connector { return &loadingConnector{} }
	for _, c := range []struct {
		name string
		conn func() Connector
		edit func(*Config)
		want string
	}{
		{"one full-attention group, synchronous, no drafter", plain, func(*Config) {}, ""},
		{"several KV cache groups, a connector without HMA", plain, hybrid, "KV cache groups"},
		{"several KV cache groups, SupportsHMA without the capability",
			func() Connector { return &hmaConnector{} }, hybrid, "KV cache groups"},
		{"several KV cache groups, SupportsHMA and diff-tested", nixlLike, hybrid, ""},
		{"a bounded-replay window", nixlLike, func(c *Config) {
			c.KVGroups = []kv.Group{{Kind: kv.FullAttention}, {Kind: kv.SlidingWindowMLA, Window: 8, BoundedReplay: true}}
		}, "is not modelled"},
		{"a ring", nixlLike, func(c *Config) {
			c.KVGroups = []kv.Group{{Kind: kv.FullAttention}, {Kind: kv.CircularBuffer}}
		}, "is not modelled"},
		{"a drafter, a connector without the capability", plain, drafter, "speculative decoding"},
		{"a drafter, diff-tested", nixlLike, drafter, ""},
		{"a drafter beside several groups, diff-tested", nixlLike, func(c *Config) { hybrid(c); drafter(c) }, ""},
		{"async scheduling", nixlLike, async, "async scheduling"},
		{"concurrent batches", nixlLike, func(c *Config) { c.MaxConcurrentBatches = 2 }, "concurrent batches"},
		{"async scheduling, diff-tested",
			func() Connector { return &hmaConnector{caps: Capabilities{ConcurrentBatches: true}} }, async, ""},
	} {
		cfg := base
		c.edit(&cfg)
		s, err := New(cfg)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		err = s.SetConnector(c.conn(), ConnectorConfig{})
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: refused: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want) ||
			!strings.Contains(err.Error(), "https://github.com/inference-sim/blis-vllm/issues/34")):
			t.Errorf("%s: %v, want a refusal naming %q and blis-vllm#34", c.name, err, c.want)
		}
	}
}

// With several KV cache groups the connector sees every group's block ids: at admission
// (update_state_after_alloc's KVCacheBlocks, AllocatedBlocks.Groups), and at finish through
// request_finished_all_groups, after each sliding window freed its blocks below the window
// (_connector_finished, vllm/v1/core/sched/scheduler.py:3024-3058@5df63a72), each group clipped
// to the blocks covering the computed tokens, a freed slot as the null block's id (0).
func TestAnHMAConnectorSeesEveryGroupsBlocks(t *testing.T) {
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40, MaxModelLen: 128,
		EnableChunkedPrefill: true, Policy: FCFS, EnablePrefixCaching: true, ReserveFullSequence: true,
		KVGroups: kv.ResolveGroups([]kv.Group{{Kind: kv.FullAttention}, {Kind: kv.SlidingWindow, Window: 4}}, false)})
	if err != nil {
		t.Fatal(err)
	}
	c := &hmaConnector{caps: Capabilities{HybridGroups: true}}
	if err := s.SetConnector(c, ConnectorConfig{}); err != nil {
		t.Fatal(err)
	}
	prompt := make([]int64, 16)
	for i := range prompt {
		prompt[i] = int64(200 + i)
	}
	if err := s.Add(&Request{ID: "a", Prompt: prompt, MaxTokens: 4, EOS: eosToken}); err != nil {
		t.Fatal(err)
	}
	out := s.Schedule()
	if got := len(c.alloced.Groups); got != 2 {
		t.Fatalf("update_state_after_alloc saw %d groups' block ids, want 2", got)
	}
	if len(c.alloced.Groups[0]) != len(c.alloced.IDs) || c.alloced.Groups[0][0] != c.alloced.IDs[0] {
		t.Errorf("AllocatedBlocks.Groups[0] %v is not group 0's IDs %v", c.alloced.Groups[0], c.alloced.IDs)
	}
	s.Update(out, map[string][]int64{"a": {7}})
	s.Abort([]string{"a"})
	if len(c.groups) != 2 {
		t.Fatalf("request_finished_all_groups got %d groups, want 2", len(c.groups))
	}
	// 16 computed tokens: full attention holds 4 blocks; the window of 4 keeps only the blocks
	// for its last 3 tokens, so blocks 0-2 are freed to null slots before the hand-over.
	if len(c.groups[0]) != 4 || len(c.groups[1]) != 4 {
		t.Fatalf("block ids %v, want 4 per group", c.groups)
	}
	for i, id := range c.groups[1][:3] {
		if id != 0 {
			t.Errorf("window block %d is %d, want the null block (0): the window was not trimmed first", i, id)
		}
	}
	if c.groups[1][3] == 0 {
		t.Errorf("the window's last block was freed: %v", c.groups[1])
	}
}
