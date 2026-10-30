package vllm

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

const producer = "kv_transfer: {connector: NixlConnector, kv_role: kv_producer}\n"
const consumer = "kv_transfer: {connector: NixlConnector, kv_role: kv_consumer}\n"

// runOne drives one engine through e's batches until it has no requests, completing each batch
// at its step time and reporting each transfer it starts done after delay; it returns every
// result and every transfer started.
func runOne(t *testing.T, e *Engine, delay engine.Duration) ([]engine.Result, []engine.Transfer, engine.Time) {
	t.Helper()
	var results []engine.Result
	var started []engine.Transfer
	now := engine.Time(0)
	type due struct {
		at engine.Time
		id engine.TransferID
	}
	var transfers []due
	for steps := 0; steps < 10000; steps++ {
		for len(transfers) > 0 && transfers[0].at <= now {
			if err := e.TransferDone(transfers[0].id, true, now); err != nil {
				t.Fatal(err)
			}
			transfers = transfers[1:]
		}
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			started = append(started, tr)
			transfers = append(transfers, due{now + delay, tr.ID})
		}
		if b.Empty() {
			if !e.s.HasUnfinished() && len(transfers) == 0 {
				return results, started, now
			}
			next := now + 1000
			if len(transfers) > 0 {
				next = min(next, transfers[0].at)
			}
			if w, ok := e.NextWake(now); ok {
				next = min(next, w)
			}
			now = max(next, now+1)
			continue
		}
		now += e.StepTime(b)
		res, err := e.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		results = append(results, res)
	}
	t.Fatal("no progress")
	return nil, nil, 0
}

func mustEngine(t *testing.T, extra string) *Engine {
	t.Helper()
	eng, err := build(t, extra)()
	if err != nil {
		t.Fatal(err)
	}
	return eng.(*Engine)
}

// A disaggregated prefill on a kv_producer samples one token, finishes, and keeps its blocks for
// the decode side: they are reported as a handoff, held until the simulator ends it, and then
// freed. A disaggregated decode on a kv_consumer pulls the prompt's KV (one TransferLoad sized by
// the tokens NIXL matched), runs once the pull lands, and computes none of the prompt itself.
func TestPDThroughTheEngine(t *testing.T) {
	p := mustEngine(t, producer)
	r := req("a", 80, 6)
	// A P/D proxy sends the prefill leg with max_tokens 1 and do_remote_decode.
	r.RemoteDecode, r.MaxOutputTokens = true, 1
	if err := p.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	results, started, _ := runOne(t, p, 10)
	if len(started) != 0 {
		t.Fatalf("the prefiller started transfers %v", started)
	}
	var h []engine.Handoff
	var fin []engine.Finished
	for _, res := range results {
		h = append(h, res.Handoffs...)
		fin = append(fin, res.Finished...)
	}
	if len(fin) != 1 || fin[0].OutputTokens != 1 {
		t.Fatalf("the prefill finished %+v, want one token", fin)
	}
	if len(h) != 1 || h[0].ID != "a" || len(h[0].Blocks) == 0 {
		t.Fatalf("handoffs %+v, want a's blocks held", h)
	}
	free := p.s.KV.Pool.NumFree()
	if err := p.TransferDone(h[0].Transfer, true, 1_000_000); err != nil {
		t.Fatal(err)
	}
	runOne(t, p, 0)
	if got := p.s.KV.Pool.NumFree(); got <= free {
		t.Fatalf("free blocks %d after the hold ended, %d before: the held blocks were not freed", got, free)
	}
	if p.s.NumRequests() != 0 {
		t.Fatalf("the prefiller still holds %d requests", p.s.NumRequests())
	}

	d := mustEngine(t, consumer)
	r = req("a", 80, 6)
	r.RemotePrefill = true
	if err := d.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	results, started, _ = runOne(t, d, 500)
	if len(started) != 1 || started[0].Kind != engine.TransferLoad || started[0].Request != "a" || started[0].Tokens != 80 {
		t.Fatalf("the decoder started %+v, want one load of a's 80 prompt tokens", started)
	}
	emitted := 0
	for _, res := range results {
		emitted += res.Emitted["a"]
	}
	if emitted != 6 {
		t.Fatalf("the decoder emitted %d tokens, want 6", emitted)
	}
}

// A pull the simulator reports failed ends as vLLM's kv_load_failure_policy says
// (scheduler.py:2057-2061, 3132-3153, 3341-3354@5df63a72): under "fail", the default, the request
// finishes as an error; under "recompute" the decoder computes the prompt itself and the request
// completes. An error finish the step that applied the outcome made, which ran nothing, is
// reported in that schedule's Batch.Finished. Either way the transfer is then unknown and every
// block is free again.
func TestPDFailedPullFollowsTheLoadFailurePolicy(t *testing.T) {
	for _, c := range []struct {
		policy string
		want   engine.FinishReason
		out    int
	}{{"", engine.FinishError, 0}, {"recompute", engine.FinishStop, 6}} {
		extra := consumer
		if c.policy != "" {
			extra = "kv_transfer: {connector: NixlConnector, kv_role: kv_consumer, kv_load_failure_policy: " + c.policy + "}\n"
		}
		d := mustEngine(t, extra)
		r := req("a", 80, 6)
		r.RemotePrefill = true
		if err := d.Add(r, 0); err != nil {
			t.Fatal(err)
		}
		b, err := d.Schedule(0)
		if err != nil || len(b.TransfersStarted) != 1 {
			t.Fatalf("%q: schedule: %v, transfers %v", c.policy, err, b.TransfersStarted)
		}
		if err := d.TransferDone(b.TransfersStarted[0].ID, false, 100); err != nil {
			t.Fatal(err)
		}
		if err := d.TransferDone(b.TransfersStarted[0].ID, true, 100); err != engine.ErrUnknownTransfer {
			t.Fatalf("%q: a second TransferDone of the same pull: %v, want ErrUnknownTransfer", c.policy, err)
		}
		results, batches, _ := runOneBatches(t, d)
		var fin []engine.Finished
		for _, res := range results {
			fin = append(fin, res.Finished...)
		}
		for _, b := range batches {
			fin = append(fin, b.Finished...)
		}
		if len(fin) != 1 || fin[0].Reason != c.want || fin[0].OutputTokens != c.out {
			t.Fatalf("%q: finished %+v, want a as %s with %d tokens", c.policy, fin, c.want, c.out)
		}
		if d.s.NumRequests() != 0 || d.s.KV.Pool.NumFree() != d.s.KV.Pool.NumBlocks()-1 {
			t.Fatalf("%q: %d requests and %d of %d blocks free after the failed pull", c.policy, d.s.NumRequests(),
				d.s.KV.Pool.NumFree(), d.s.KV.Pool.NumBlocks()-1)
		}
	}
}

// runOneBatches is runOne with no transfers, also returning every batch scheduled.
func runOneBatches(t *testing.T, e *Engine) ([]engine.Result, []engine.Batch, engine.Time) {
	t.Helper()
	var results []engine.Result
	var batches []engine.Batch
	now := engine.Time(0)
	for steps := 0; steps < 10000; steps++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		batches = append(batches, b)
		if b.Empty() {
			if !e.s.HasUnfinished() {
				return results, batches, now
			}
			now += 1000
			continue
		}
		now += e.StepTime(b)
		res, err := e.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		results = append(results, res)
	}
	t.Fatal("no progress")
	return nil, nil, 0
}

// Every pull failing keeps the engine contract's laws: no request is lost (enginetest's
// conservation), whatever the policy.
func TestPDConformanceWhenEveryPullFails(t *testing.T) {
	for _, extra := range []string{consumer, "kv_transfer: {connector: NixlConnector, kv_role: kv_consumer, kv_load_failure_policy: recompute}\n"} {
		// A lone failed pull, whose error finish no result carries, and a mix.
		lone := req("z", 40, 6)
		lone.RemotePrefill = true
		enginetest.TestEngine(t, build(t, extra), arrivals(lone), enginetest.Options{
			TransferFails: func(engine.Transfer) bool { return true }})
		reqs := []engine.Request{req("a", 40, 6), req("b", 70, 3), req("c", 20, 9)}
		reqs[0].RemotePrefill, reqs[2].RemotePrefill = true, true
		enginetest.TestEngine(t, build(t, extra), arrivals(reqs...), enginetest.Options{
			TransferTime:  func(engine.Transfer) engine.Duration { return 250 },
			TransferFails: func(engine.Transfer) bool { return true },
		})
	}
}

// The P/D engines keep the engine contract's laws (blis-schemas enginetest) with a mix of
// disaggregated and ordinary requests.
func TestPDConformance(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		mark        func(*engine.Request)
	}{
		{"prefiller", producer, func(r *engine.Request) { r.RemoteDecode = true }},
		{"decoder", consumer, func(r *engine.Request) { r.RemotePrefill = true }},
		// The measured Muse P/D arm runs vLLM's default async scheduling: two batches in flight.
		{"prefiller async", producer + "async_scheduling: true\nuse_v2_model_runner: true\n", func(r *engine.Request) { r.RemoteDecode = true }},
		{"decoder async", consumer + "async_scheduling: true\nuse_v2_model_runner: true\n", func(r *engine.Request) { r.RemotePrefill = true }},
		{"decoder hybrid async", consumer + "async_scheduling: true\nuse_v2_model_runner: true\nkv_layers:\n" +
			"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
			"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n",
			func(r *engine.Request) { r.RemotePrefill = true }},
	} {
		t.Run(c.name, func(t *testing.T) {
			reqs := []engine.Request{req("a", 40, 6), req("b", 70, 3), req("c", 20, 9), req("d", 90, 4)}
			for i := range reqs {
				if i%2 == 0 {
					c.mark(&reqs[i])
					if reqs[i].RemoteDecode {
						reqs[i].MaxOutputTokens = 1
					}
				}
			}
			enginetest.TestEngine(t, build(t, c.extra), arrivals(reqs...), enginetest.Options{
				TransferTime: func(engine.Transfer) engine.Duration { return 250 },
			})
		})
	}
}

// What the engine refuses: a request marked for a role the engine is not, a connector it does not
// model, and what vLLM's NIXL worker refuses for the parallel layout.
func TestPDRefusals(t *testing.T) {
	for _, c := range []struct{ name, extra string }{
		{"unknown connector", "kv_transfer: {connector: LMCacheConnectorV1, kv_role: kv_producer}\n"},
		{"kv_both", "kv_transfer: {connector: NixlConnector, kv_role: kv_both}\n"},
		{"unknown role", "kv_transfer: {connector: NixlConnector, kv_role: decoder}\n"},
		{"consumer at pp 2", consumer + "pipeline_parallel_size: 2\nasync_scheduling: true\nuse_v2_model_runner: true\n"},
		{"producer at dp 2", producer + "data_parallel_size: 2\n"},
		{"consumer at dp 2", consumer + "data_parallel_size: 2\n"},
	} {
		if _, err := build(t, c.extra)(); err == nil {
			t.Errorf("%s: built, want refused", c.name)
		}
	}
	plain := mustEngine(t, "")
	r := req("a", 40, 2)
	r.RemotePrefill = true
	if err := plain.Add(r, 0); err == nil {
		t.Error("an engine without a connector accepted a remote prefill")
	}
	p := mustEngine(t, producer)
	if err := p.Add(r, 0); err == nil {
		t.Error("a kv_producer accepted a remote prefill")
	}
}

// A prefill leg is the proxy's max_tokens 1 request: one token sampled, which finishes it at its
// length cap, whatever output the simulator assigned beyond it; a leg with none assigned is
// refused, as any request is, rather than given a token the simulator never assigned.
func TestPDPrefillLegSamplesOneToken(t *testing.T) {
	p := mustEngine(t, producer)
	r := req("a", 40, 9)
	r.RemoteDecode = true
	if err := p.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	results, _, _ := runOne(t, p, 0)
	var fin []engine.Finished
	for _, res := range results {
		fin = append(fin, res.Finished...)
	}
	if len(fin) != 1 || fin[0].OutputTokens != 1 || fin[0].Reason != engine.FinishLength {
		t.Fatalf("finished %+v, want one token at the length cap", fin)
	}
	none := req("b", 40, 1)
	none.Output, none.RemoteDecode = nil, true
	if err := mustEngine(t, producer).Add(none, 0); err == nil {
		t.Fatal("a prefill leg with no assigned output was accepted")
	}
	// The contract's laws hold for prefill legs: none emits more than it was assigned.
	legs := []engine.Request{req("c", 40, 1), req("d", 70, 5)}
	for i := range legs {
		legs[i].RemoteDecode = true
	}
	enginetest.TestEngine(t, build(t, producer), arrivals(legs...), enginetest.Options{})
}

// A hybrid model's handoff holds the prompt's full-attention blocks: every one of its tokens,
// even where group 0 is a sliding window whose early slots are freed.
func TestPDHybridHandoffHoldsThePrompt(t *testing.T) {
	const windowFirst = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n"
	p := mustEngine(t, producer+windowFirst)
	r := req("a", 160, 1)
	r.RemoteDecode = true
	if err := p.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	results, _, _ := runOne(t, p, 0)
	var h []engine.Handoff
	for _, res := range results {
		h = append(h, res.Handoffs...)
	}
	if len(h) != 1 || len(h[0].Blocks) != 10 {
		t.Fatalf("handoffs %+v, want the 160-token prompt's 10 blocks", h)
	}
}
