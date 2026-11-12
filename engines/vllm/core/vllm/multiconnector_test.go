package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// multiBlock states vLLM's MultiConnector over a NIXL connector and the OffloadingConnector, listing the
// children in the order vLLM tries them: the first that reports matched tokens loads a request's
// KV (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:434-456@5df63a72).
func multiBlock(order string) string {
	return "kv_transfer: {connector: MultiConnector, kv_role: kv_both, connectors: [" + order + "]}\n" + cpuTier
}

const (
	nixlFirst    = "NixlConnector, OffloadingConnector"
	offloadFirst = "OffloadingConnector, NixlConnector"
)

// A deployment that runs a NIXL handoff and a CPU tier together is the MultiConnector llm-d
// builds, and it used to be refused. A prefill leg on it keeps its blocks for the decoder as
// NIXL's producer does, and stores its prompt to the CPU tier as the offloading connector does;
// ending the hold frees the blocks, whichever transfer ID it carries.
func TestMultiConnectorPrefillHoldsForTheDecoderAndStoresToTheTier(t *testing.T) {
	for _, order := range []string{nixlFirst, offloadFirst} {
		t.Run(order, func(t *testing.T) {
			e := mustEngine(t, multiBlock(order))
			r := req("a", 80, 6)
			r.RemoteDecode, r.MaxOutputTokens = true, 1
			if err := e.Add(r, 0); err != nil {
				t.Fatal(err)
			}
			results, started, _ := runOne(t, e, 10)
			var h []engine.Handoff
			for _, res := range results {
				h = append(h, res.Handoffs...)
			}
			if len(h) != 1 || h[0].ID != "a" || len(h[0].Blocks) == 0 {
				t.Fatalf("handoffs %+v, want a's blocks held for the decoder", h)
			}
			stores := 0
			for _, tr := range started {
				if tr.Kind == engine.TransferStore && tr.Tier == "cpu" {
					stores++
				}
			}
			if stores == 0 {
				t.Fatalf("transfers %+v: the prompt was not stored to the CPU tier", started)
			}
			// The hold is a transfer of its own: no store's ID is it, and it is the only one that frees blocks.
			for _, tr := range started {
				if tr.ID == h[0].Transfer {
					t.Fatalf("the hold's transfer ID %d is also the ID of %+v", tr.ID, tr)
				}
			}
			free := e.s.KV.Pool.NumFree()
			if err := e.TransferDone(h[0].Transfer, true, 1_000_000); err != nil {
				t.Fatal(err)
			}
			runOne(t, e, 0)
			if got := e.s.KV.Pool.NumFree(); got <= free {
				t.Fatalf("free blocks %d after the hold ended, %d before: the held blocks were not freed", got, free)
			}
		})
	}
}

// decodeAfterEviction runs a request that stores its prompt to the CPU tier, evicts it from the
// GPU's prefix cache with another, and then sends a decode leg (RemotePrefill) whose prompt
// repeats the first one's and four more tokens. It returns the transfers that leg started, the
// tokens it emitted, and the engine.
func decodeAfterEviction(t *testing.T, order string) (leg []engine.Transfer, emitted int, e *Engine) {
	t.Helper()
	e = mustEngine(t, multiBlock(order)+tightGPU)
	a := req("a", 96, 2)
	if err := e.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	_, _, now := runOne(t, e, 5000)
	if err := e.Add(req("xx", 140, 2), now); err != nil {
		t.Fatal(err)
	}
	_, _, now = runOne(t, e, 10)
	d := req("d", 0, 6)
	d.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	d.RemotePrefill, d.Peer = true, peer
	if err := e.Add(d, now); err != nil {
		t.Fatal(err)
	}
	results, started, _ := runOne(t, e, 500)
	for _, tr := range started {
		if tr.Request == "d" {
			leg = append(leg, tr)
		}
	}
	for _, res := range results {
		emitted += res.Emitted["d"]
	}
	return leg, emitted, e
}

// Which child loads a decode leg's KV is the first in the configured order that reports tokens.
// Listed first, NIXL pulls the prompt from the prefiller, and the CPU tier loads nothing; listed
// second, the offloading connector reports its hit first and loads it from the CPU tier, and
// NIXL, given no external tokens, only tells the prefiller to release its blocks: a pull of
// nothing, whose end does not stand for the load landing.
func TestMultiConnectorLoadsFromTheFirstChildThatMatches(t *testing.T) {
	var cpuLoads = func(ts []engine.Transfer) (n, tokens int) {
		for _, tr := range ts {
			if tr.Kind == engine.TransferLoad && tr.Tier == "cpu" {
				n, tokens = n+1, tokens+tr.Tokens
			}
		}
		return
	}
	pulls := func(ts []engine.Transfer) (n, tokens int) {
		for _, tr := range ts {
			if tr.Kind == engine.TransferLoad && tr.Tier == "" {
				n, tokens = n+1, tokens+tr.Tokens
			}
		}
		return
	}

	leg, emitted, _ := decodeAfterEviction(t, nixlFirst)
	if n, _ := cpuLoads(leg); n != 0 {
		t.Errorf("NIXL listed first: the CPU tier loaded for the decode leg: %+v", leg)
	}
	if n, tok := pulls(leg); n != 1 || tok == 0 || (100-tok)%16 != 0 {
		t.Errorf("NIXL listed first: pulls %+v, want one pull of the 100-token prompt past its block-aligned local hit", leg)
	}
	if emitted != 6 {
		t.Errorf("NIXL listed first: the decode leg emitted %d tokens, want 6", emitted)
	}

	leg, emitted, _ = decodeAfterEviction(t, offloadFirst)
	if n, tok := cpuLoads(leg); n != 1 || tok == 0 || tok%16 != 0 {
		t.Errorf("offload listed first: transfers %+v, want one CPU load of the stored prompt blocks", leg)
	}
	if n, tok := pulls(leg); n != 1 || tok != 0 {
		t.Errorf("offload listed first: transfers %+v, want one pull of no tokens (the prefiller's release)", leg)
	}
	if emitted != 6 {
		t.Errorf("offload listed first: the decode leg emitted %d tokens, want 6", emitted)
	}
}

// Every transfer the MultiConnector's engine starts is one the cost model can price: it validates,
// its move is against a device of the pool's plan (the CPU tier's) or between pools, and no two
// transfers share an ID, so a TransferDone finds the right connector's.
func TestMultiConnectorTransfersPriceAgainstThePlan(t *testing.T) {
	for _, order := range []string{nixlFirst, offloadFirst} {
		leg, _, e := decodeAfterEviction(t, order)
		seen := map[engine.TransferID]bool{}
		for _, tr := range leg {
			if err := tr.Validate(); err != nil {
				t.Errorf("%s: %+v: %v", order, tr, err)
			}
			if err := e.Plan().CheckTransfer(tr.Move); err != nil {
				t.Errorf("%s: %+v is not priced by the plan: %v", order, tr, err)
			}
			if seen[tr.ID] {
				t.Errorf("%s: transfer ID %d started twice", order, tr.ID)
			}
			seen[tr.ID] = true
		}
		if len(leg) == 0 {
			t.Errorf("%s: the decode leg started no transfer", order)
		}
	}
}

// The engine contract's laws hold with both connectors behind one: the same requests as the
// offloading engine's conformance run, a tight GPU cache so the CPU tier serves prompts back, one
// full-attention group, a hybrid model's full and sliding-window groups, and the batch queue
// (async scheduling).
func TestMultiConnectorConformance(t *testing.T) {
	shapes := map[string]string{
		"one group": tightGPU,
		"hybrid": "num_gpu_blocks: 24\nmax_model_len: 160\nkv_layers:\n" +
			"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
			"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n",
		"async": tightGPU + "async_scheduling: true\n",
	}
	for _, order := range []string{nixlFirst, offloadFirst} {
		for shape, extra := range shapes {
			t.Run(order+"/"+shape, func(t *testing.T) { conformance(t, multiBlock(order)+extra) })
		}
	}
}

func conformance(t *testing.T, extra string) {
	t.Helper()
	var arr []enginetest.Arrival
	at := engine.Time(0)
	for i, id := range []string{"a", "b", "a2", "c", "a3"} {
		base, n := 100000*(i+1), 100+4*i
		if id[0] == 'a' {
			base, n = 1000, 96
		}
		r := req(id, 0, 3)
		for k := 0; k < n; k++ {
			r.Prompt = append(r.Prompt, int32(base+k))
		}
		r.Prompt = append(r.Prompt, int32(i))
		arr = append(arr, enginetest.Arrival{At: at, Req: r})
		at += 3000
	}
	kinds := map[engine.TransferKind]int{}
	enginetest.TestEngine(t, build(t, extra), arr, enginetest.Options{
		Boundary: boundary,
		TransferTime: func(tr engine.Transfer) engine.Duration {
			kinds[tr.Kind]++
			return engine.Duration(50 + 10*tr.Blocks)
		},
	})
	if kinds[engine.TransferStore] == 0 || kinds[engine.TransferLoad] == 0 {
		t.Errorf("transfers %v: the CPU tier was not exercised", kinds)
	}
}

// A connector combination vLLM refuses, or that is not modelled, stays a named refusal.
func TestMultiConnectorRefusals(t *testing.T) {
	for _, c := range []struct{ name, extra, want string }{
		{"NIXL beside an offload tier without MultiConnector", producer + cpuTier, "MultiConnector"},
		{"children on a connector that has none", "kv_transfer: {connector: NixlConnector, kv_role: kv_both, connectors: [NixlConnector]}\n", "connectors"},
		{"MultiConnector without the offload tier", "kv_transfer: {connector: MultiConnector, kv_role: kv_both, connectors: [NixlConnector, OffloadingConnector]}\n", "without kv_offload"},
		{"a child that is not modelled", multiBlock("NixlConnector, MooncakeConnector"), "MooncakeConnector"},
		{"the same child twice", multiBlock("NixlConnector, NixlConnector"), "OffloadingConnector"},
		{"no children listed", "kv_transfer: {connector: MultiConnector, kv_role: kv_both}\n" + cpuTier, "connectors"},
		{"three children", multiBlock("NixlConnector, OffloadingConnector, NixlPullConnector"), "connectors"},
		{"at data parallelism 2", multiBlock(nixlFirst) + "data_parallel_size: 2\n", "data parallelism"},
	} {
		_, err := build(t, c.extra)()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want a refusal naming %q", c.name, err, c.want)
		}
	}
}
