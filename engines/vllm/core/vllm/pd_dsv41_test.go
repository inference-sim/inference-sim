package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// A DeepSeek V4.1-shaped layer stack: a 128-token bounded-replay window in 32-token blocks, the
// compressed cache in 256-token blocks (a group of its own block size) and the compressor's ring.
const dsv41Layers = "kv_layers:\n" +
	"- {count: 1, spec: sliding_window_mla, sliding_window: 128, num_kv_heads: 1, head_size: 512, block_size: 32, state_bytes: 584, alignment: 576, bounded_replay: true}\n" +
	"- {count: 1, spec: mla, num_kv_heads: 1, head_size: 512, block_size: 256, tokens_per_state: 128, state_bytes: 584, alignment: 576}\n" +
	"- {count: 1, spec: circular_buffer, num_kv_heads: 1, head_size: 64, block_size: 8, elem_bytes: 4}\n"

// A full-attention group beside a sliding window, no bounded replay and no ring.
const swaLayers = "kv_layers:\n" +
	"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
	"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n"

func cdiv(a, b int) int { return (a + b - 1) / b }

// connectorBlocks is the blocks NIXL's pull moves for a prompt of tokens tokens when the decoder
// holds held of them already, summed over the engine's KV cache groups: every block of a
// full-attention or MLA group, a ring's one, and of a sliding window the blocks that cover its
// last window-1 tokens (the only ones the manager allocates, the rest being null), at most
// cdiv(window, block)+1 (nixl/base_scheduler.py:262-300 and pull_scheduler.py:149-155 at vLLM
// v0.32.0). A window of 128 tokens in 32-token blocks moves 4 blocks when the prompt ends on a
// block boundary and 5 when it does not.
func connectorBlocks(e *Engine, tokens, held int) int {
	n := 0
	for g := 0; g < e.s.KV.NumGroups(); g++ {
		bs := e.s.KV.GroupBlockSize(g)
		sp := e.s.KV.GroupSpec(g)
		blocks := cdiv(tokens, bs) - held/bs
		switch sp.Kind {
		case kv.SlidingWindow, kv.SlidingWindowMLA:
			first := max(tokens-(sp.Window-1), 0) / bs
			blocks = min(cdiv(tokens, bs)-first, cdiv(sp.Window, bs)+1)
		case kv.CircularBuffer:
			blocks = 1
		}
		n += max(blocks, 0)
	}
	return n
}

// pull runs one remote-prefilled request of prompt tokens on a consumer and returns the load the
// engine reported for it and the engine.
func pull(t *testing.T, extra string, prompt int) (engine.Transfer, *Engine) {
	t.Helper()
	d := mustEngine(t, extra)
	r := req("a", prompt, 4)
	r.RemotePrefill, r.Peer = true, peer
	if err := d.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	_, started, _ := runOne(t, d, 100)
	if len(started) != 1 {
		t.Fatalf("started %+v, want one load", started)
	}
	return started[0], d
}

// A decoder's pull reports the connector's clipped blocks, summed over the KV cache groups: a long
// prompt over a 128-token window in 32-token blocks moves five of that group's blocks, not all of
// its prompt's; and a model with one group moves every block of it, as before.
func TestPullBlocksFollowTheConnectorsClipRulePerGroup(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		prompt      int
	}{
		{"deepseek v4.1 long prompt", "block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers, 640},
		{"deepseek v4.1 window off a block boundary", "block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers, 651},
		{"deepseek v4.1 short prompt", "block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers, 70},
		{"full attention beside a window", "block_size: 16\nnum_gpu_blocks: 200\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + swaLayers, 300},
		{"one full-attention group", "", 90},
	} {
		t.Run(c.name, func(t *testing.T) {
			tr, d := pull(t, consumer+c.extra, c.prompt)
			want := connectorBlocks(d, tr.Tokens, 0)
			if tr.Blocks != want {
				t.Fatalf("a pull of %d tokens reports %d blocks, want %d (the connector's clipped count over %d groups)",
					tr.Tokens, tr.Blocks, want, d.s.KV.NumGroups())
			}
		})
	}
	// A prompt ending off a block boundary moves five blocks of the window (the bound), one more
	// than the four a boundary-aligned one needs: 5 + 3 of the compressed cache + the ring's 1.
	if tr, _ := pull(t, consumer+"block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n"+dsv41Layers, 651); tr.Blocks != 9 {
		t.Fatalf("a 650-token pull reports %d blocks, want 9", tr.Blocks)
	}
	// The clip is visible: the long V4.1 prompt moves fewer blocks than its prompt holds.
	tr, d := pull(t, consumer+"block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n"+dsv41Layers, 640)
	if all := cdiv(tr.Tokens, 32) * d.s.KV.NumGroups(); tr.Blocks >= all {
		t.Fatalf("a pull reports %d blocks, as many as the %d its groups hold in full", tr.Blocks, all)
	}
}

// A decoder that already holds a prompt's leading blocks (a local prefix hit) pulls only the rest:
// fewer tokens and fewer blocks than the same prompt on a cold decoder.
func TestPullBlocksSkipWhatTheDecoderHolds(t *testing.T) {
	extra := consumer + "block_size: 16\nnum_gpu_blocks: 200\nmax_model_len: 1024\nmax_num_batched_tokens: 256\nenable_prefix_caching: true\n" +
		"kv_layers:\n- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n"
	cold, _ := pull(t, extra, 160)
	d := mustEngine(t, extra)
	local := req("l", 96, 2) // computes the first 96 tokens of the same text
	copy(local.Prompt, req("a", 96, 1).Prompt)
	if err := d.Add(local, 0); err != nil {
		t.Fatal(err)
	}
	runOne(t, d, 0)
	r := req("a", 160, 4)
	r.RemotePrefill, r.Peer = true, peer
	if err := d.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	_, started, _ := runOne(t, d, 100)
	if len(started) != 1 {
		t.Fatalf("started %+v, want one load", started)
	}
	warm := started[0]
	if warm.Tokens >= cold.Tokens || warm.Blocks >= cold.Blocks {
		t.Fatalf("a decoder holding 96 tokens pulls %d tokens in %d blocks; a cold one %d in %d", warm.Tokens, warm.Blocks, cold.Tokens, cold.Blocks)
	}
	if want := connectorBlocks(d, warm.Tokens, 0); warm.Blocks != want {
		t.Fatalf("held case: %d blocks, want %d", warm.Blocks, want)
	}
}

// The engine with DeepSeek V4.1-shaped KV cache groups passes the engine contract as a decoder
// and as a prefiller beside the connector.
func TestPDConformanceDeepSeekV41Groups(t *testing.T) {
	const shape = "block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers
	for _, c := range []struct {
		name, extra string
		mark        func(*engine.Request)
	}{
		{"decoder", consumer + shape, func(r *engine.Request) { r.RemotePrefill, r.Peer = true, peer }},
		{"prefiller", producer + shape, func(r *engine.Request) { r.RemoteDecode, r.MaxOutputTokens = true, 1 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			reqs := []engine.Request{req("a", 300, 6), req("b", 700, 3), req("c", 90, 9), req("d", 400, 4)}
			for i := range reqs {
				if i%2 == 0 {
					c.mark(&reqs[i])
				}
			}
			enginetest.TestEngine(t, build(t, c.extra), arrivals(reqs...), enginetest.Options{
				Boundary:     boundary,
				TransferTime: func(engine.Transfer) engine.Duration { return 250 },
			})
		})
	}
}

// What vLLM refuses stays refused beside V4.1's layout, each with the reason it gives: the NIXL
// pull worker does not take a hybrid layout (this one needs the hybrid manager) under pipeline
// parallelism (nixl/base_worker.py:760-765@5df63a72).
func TestPDRefusalsBesideDeepSeekV41Layers(t *testing.T) {
	for _, c := range []struct{ name, extra, want string }{
		{"producer at pp 2", producer + dsv41Layers + "pipeline_parallel_size: 2\nasync_scheduling: true\nuse_v2_model_runner: true\n", ""},
	} {
		_, err := build(t, c.extra)()
		if err == nil || (c.want != "" && !strings.Contains(err.Error(), c.want)) {
			t.Errorf("%s: %v, want a refusal naming %q", c.name, err, c.want)
		}
	}
}

// handoffs runs prefill legs through a producer and returns each one's handoff.
func handoffs(t *testing.T, extra string, reqs ...engine.Request) (map[string][]int64, *Engine) {
	t.Helper()
	p := mustEngine(t, extra)
	for i, r := range reqs {
		r.RemoteDecode, r.MaxOutputTokens = true, 1
		if err := p.Add(r, engine.Time(i)); err != nil {
			t.Fatal(err)
		}
	}
	results, _, _ := runOne(t, p, 0)
	out := map[string][]int64{}
	for _, res := range results {
		for _, h := range res.Handoffs {
			out[h.ID] = h.Blocks
		}
	}
	return out, p
}

// A V4.1-shaped prefiller holds every transfer group's clipped blocks for the read, not one
// group's: the handoff names as many blocks as the connector's rule gives over the groups (the
// prefiller computed the prompt less the one backoff token).
func TestProducerHandoffHoldsEveryTransferGroup(t *testing.T) {
	h, p := handoffs(t, producer+dsv41Layers, req("a", 400, 1))
	got := h["a"]
	if want := connectorBlocks(p, 399, 0); len(got) != want {
		t.Errorf("handoff holds %d blocks, the connector's rule over the groups gives %d", len(got), want)
	}
	seen := map[int64]bool{}
	for _, b := range got {
		if seen[b] {
			t.Errorf("block %d held twice in one handoff", b)
		}
		seen[b] = true
	}
}

// Two prompts that share a prefix hold the shared blocks once between them in the pool, and each
// handoff names them: the prefix's blocks appear in both, so a count of distinct blocks across
// handoffs is smaller than the sum.
func TestHandoffsOfSharedPrefixesNameTheSharedBlocks(t *testing.T) {
	a := req("a", 400, 1)
	b := req("b", 400, 1)
	copy(b.Prompt[:256], a.Prompt[:256])
	h, _ := handoffs(t, producer+dsv41Layers, a, b)
	distinct := map[int64]bool{}
	sum := 0
	for _, blocks := range h {
		sum += len(blocks)
		for _, id := range blocks {
			distinct[id] = true
		}
	}
	if len(h) != 2 || len(distinct) >= sum {
		t.Errorf("handoffs %v: %d distinct blocks across %d listed, want the shared prefix named by both", h, len(distinct), sum)
	}
}

// A finished prefill's advertised block counts are dropped once the decoder has read it: the
// connector keeps nothing for the engine's lifetime.
func TestAFinishedProducerRequestLeavesNothingInTheConnector(t *testing.T) {
	p := mustEngine(t, producer+dsv41Layers)
	r := req("a", 400, 1)
	r.RemoteDecode = true
	if err := p.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	results, _, now := runOne(t, p, 0)
	var h []engine.Handoff
	for _, res := range results {
		h = append(h, res.Handoffs...)
	}
	if len(h) != 1 {
		t.Fatalf("handoffs %+v, want one", h)
	}
	if _, held := p.pd.nixl.Sent("a"); !held {
		t.Fatal("the connector advertised nothing for a held request")
	}
	if err := p.TransferDone(h[0].Transfer, true, now); err != nil {
		t.Fatal(err)
	}
	runOne(t, p, 0)
	if _, held := p.pd.nixl.Sent("a"); held {
		t.Error("the connector still holds the advertised blocks of a request whose read finished")
	}
}
