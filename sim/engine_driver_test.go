package sim

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// newFIFODriver is an EngineDriver over blis-schemas' fake FIFO engine, which prices a step as
// base + per-token x tokens and needs no kernel.
func newFIFODriver(t *testing.T, cfg enginetest.FIFOConfig, horizon int64) *EngineDriver {
	t.Helper()
	eng, err := enginetest.NewFIFO(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return NewEngineDriver(eng, EngineDriverConfig{Horizon: horizon})
}

func tokens(n int) []TokenID {
	out := make([]TokenID, n)
	for i := range out {
		out[i] = TokenID(i + 1)
	}
	return out
}

// One request through the contract: it is admitted, prefilled in one step, decoded one token a
// step, and completes. The host's report records its time to first token, its inter-token
// latencies and its completion, as the compatibility engine's recorder would.
func TestEngineDriver_OneRequestCompletesWithTTFTAndITL(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1}, 1<<40)
	var firstAt, doneAt int64 = -1, -1
	d.OnFirstToken = func(_ *Request, at int64) { firstAt = at }
	d.OnRequestDone = func(_ *Request, at int64) []*Request { doneAt = at; return nil }
	req := &Request{ID: "r0", InputTokens: tokens(10), OutputTokens: tokens(4), ArrivalTime: 0}
	d.InjectArrival(req)
	d.Run()

	m := d.Metrics
	if m.CompletedRequests != 1 {
		t.Fatalf("completed = %d, want 1", m.CompletedRequests)
	}
	// Prefill of 10 tokens: one step of 100 + 10 us, which samples the first token.
	if req.FirstTokenTime != 110 {
		t.Errorf("FirstTokenTime = %d, want 110", req.FirstTokenTime)
	}
	if firstAt != 110 {
		t.Errorf("OnFirstToken at %d, want 110", firstAt)
	}
	// Three decode steps of 101 us each emit the other three tokens.
	if got := len(req.ITL); got != 3 {
		t.Fatalf("ITL entries = %d, want 3", got)
	}
	for i, v := range req.ITL {
		if v != 101 {
			t.Errorf("ITL[%d] = %d, want 101", i, v)
		}
	}
	if m.TotalOutputTokens != 4 {
		t.Errorf("TotalOutputTokens = %d, want 4", m.TotalOutputTokens)
	}
	if got, want := m.RequestE2Es["r0"], float64(110+3*101); got != want {
		t.Errorf("E2E = %v, want %v", got, want)
	}
	if got, want := m.RequestITLs["r0"], 101.0; got != want {
		t.Errorf("mean ITL = %v, want %v", got, want)
	}
	if doneAt != 110+3*101 {
		t.Errorf("OnRequestDone at %d, want %d", doneAt, 110+3*101)
	}
	if req.State != StateCompleted {
		t.Errorf("state = %v, want completed", req.State)
	}
}

// Requests that arrive while a step runs wait for it: the GPU is serial, so the next batch
// starts no earlier than the running one ends, and both complete.
func TestEngineDriver_ArrivalDuringAStepWaitsForIt(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 0}, 1<<40)
	a := &Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(2), ArrivalTime: 0}
	b := &Request{ID: "b", InputTokens: tokens(8), OutputTokens: tokens(2), ArrivalTime: 50}
	d.InjectArrival(a)
	d.InjectArrival(b)
	d.Run()
	if d.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed = %d, want 2", d.Metrics.CompletedRequests)
	}
	// a's prefill runs 0-100. b arrives at 50 and is admitted at the next schedule, at 100.
	if a.FirstTokenTime != 100 {
		t.Errorf("a FirstTokenTime = %d, want 100", a.FirstTokenTime)
	}
	if b.FirstTokenTime != 200-50 {
		t.Errorf("b FirstTokenTime = %d, want %d (admitted when a's step ended)", b.FirstTokenTime, 150)
	}
}

// The driver steps batches back to back while work remains, and stops when none does: every
// step's start is the previous step's end, and the run ends at the last completion.
func TestEngineDriver_StepsBackToBackAndStopsWhenIdle(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 16, StepBaseUs: 10, StepPerTokenUs: 0}, 1<<40)
	for i, id := range []string{"x", "y", "z"} {
		d.InjectArrival(&Request{ID: id, InputTokens: tokens(4), OutputTokens: tokens(3), ArrivalTime: int64(i)})
	}
	d.Run()
	if d.Metrics.CompletedRequests != 3 {
		t.Fatalf("completed = %d, want 3", d.Metrics.CompletedRequests)
	}
	if d.HasPendingEvents() {
		t.Errorf("events still pending after the run")
	}
	if d.Metrics.SimEndedTime <= 0 {
		t.Errorf("SimEndedTime = %d, want the last completion", d.Metrics.SimEndedTime)
	}
}

// A request the engine refuses at Add is dropped as unservable, as the compatibility engine
// drops one at enqueue, and its done callback fires.
func TestEngineDriver_RefusedRequestIsDropped(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 16, StepBaseUs: 10}, 1<<40)
	done := 0
	d.OnRequestDone = func(*Request, int64) []*Request { done++; return nil }
	// FIFO refuses a request with an empty prompt.
	d.InjectArrival(&Request{ID: "bad", OutputTokens: tokens(2), ArrivalTime: 0})
	d.Run()
	if d.Metrics.DroppedUnservable != 1 || done != 1 {
		t.Errorf("dropped = %d, done callbacks = %d, want 1 and 1", d.Metrics.DroppedUnservable, done)
	}
}

// The host's front end renders a request before the engine sees it (INV-R1), as on the
// compatibility path: with one render thread, two requests arriving together render one after
// the other, so the second reaches the engine a render later; with the queue off
// (RendererNumWorkers 0) or one API server each, both render at once.
func TestEngineDriver_RendersOnTheFrontEndBeforeTheEngine(t *testing.T) {
	const render = 1000
	run := func(f FrontendConfig) (a, b *Request) {
		eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100})
		if err != nil {
			t.Fatal(err)
		}
		d, err := NewEngineDriverWithFrontend(eng, EngineDriverConfig{Horizon: 1 << 40,
			QueueingTime: func(*Request) int64 { return render }}, f)
		if err != nil {
			t.Fatal(err)
		}
		a = &Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
		b = &Request{ID: "b", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
		d.InjectArrival(a)
		d.InjectArrival(b)
		d.Run()
		if d.Metrics.CompletedRequests != 2 {
			t.Fatalf("completed = %d, want 2", d.Metrics.CompletedRequests)
		}
		return a, b
	}
	// One thread: a renders 0-1000 and prefills 1000-1100; b renders 1000-2000, then prefills.
	a, b := run(FrontendConfig{RendererNumWorkers: 1})
	if a.FirstTokenTime != render+100 || b.FirstTokenTime != 2*render+100 {
		t.Errorf("one render thread: first tokens at %d and %d, want %d and %d (b renders after a)",
			a.FirstTokenTime, b.FirstTokenTime, render+100, 2*render+100)
	}
	for _, f := range []FrontendConfig{{}, {RendererNumWorkers: 1, APIServerCount: 2}} {
		a, b := run(f)
		if a.FirstTokenTime != render+100 || b.FirstTokenTime != render+100 {
			t.Errorf("%+v: first tokens at %d and %d, want both %d (no render contention)",
				f, a.FirstTokenTime, b.FirstTokenTime, render+100)
		}
	}
}

// With two batches in flight (vLLM's async scheduling), the GPU still runs one at a time: the
// second batch's step starts when the first ends, so its tokens arrive a step later, not at
// once.
func TestEngineDriver_SecondBatchInFlightWaitsForTheGPU(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 8, InFlight: 2, StepBaseUs: 100}, 1<<40)
	a := &Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
	b := &Request{ID: "b", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
	d.InjectArrival(a)
	d.InjectArrival(b)
	d.Run()
	if d.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed = %d, want 2", d.Metrics.CompletedRequests)
	}
	// A token budget of 8 fits one prefill a batch: a runs 0-100, b's batch is scheduled at 0
	// too (two may be in flight) but starts when a's ends, 100-200.
	if a.FirstTokenTime != 100 || b.FirstTokenTime != 200 {
		t.Errorf("first tokens at %d and %d, want 100 and 200 (the GPU is serial)", a.FirstTokenTime, b.FirstTokenTime)
	}
}

// Requests that reach the engine at one instant share its next batch: the contract orders
// arrivals before steps at one instant, so the driver schedules once after every arrival of the
// instant rather than inside the first (as the compatibility engine's scheduleStepBy does).
func TestEngineDriver_SameInstantArrivalsShareOneBatch(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 8, TokenBudget: 1024, StepBaseUs: 100, StepPerTokenUs: 1}, 1<<40)
	var reqs []*Request
	for i := 0; i < 4; i++ {
		r := &Request{ID: string(rune('a' + i)), InputTokens: tokens(10), OutputTokens: tokens(2), ArrivalTime: 0}
		reqs = append(reqs, r)
		d.InjectArrival(r)
	}
	d.Run()
	// One prefill step of 100 + 4 x 10 us samples every first token at once.
	for _, r := range reqs {
		if r.FirstTokenTime != 140 {
			t.Errorf("%s: FirstTokenTime = %d, want 140 (one shared prefill step)", r.ID, r.FirstTokenTime)
		}
	}
}

// Each ITL entry is the step's advance since the request's last emitting step plus the output
// processing of the tokens it emits, as the compatibility engine records it (currStepAdvance +
// NumNewTokens x OutputTokenProcessingTime, blis-vllm compat/engine/engine.go:996); TTFT carries
// the first token's own output processing.
func TestEngineDriver_ITLIncludesOutputProcessingAsTheCompatibilityEngine(t *testing.T) {
	eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1})
	if err != nil {
		t.Fatal(err)
	}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, OutputTokenProcessingTime: 5})
	req := &Request{ID: "r0", InputTokens: tokens(10), OutputTokens: tokens(4), ArrivalTime: 0}
	d.InjectArrival(req)
	d.Run()
	// Prefill 110 us, then the first token's 5 us of output processing.
	if req.FirstTokenTime != 115 {
		t.Errorf("FirstTokenTime = %d, want 115", req.FirstTokenTime)
	}
	// Three decode steps of 101 us, each emitting one token with 5 us of output processing.
	if len(req.ITL) != 3 {
		t.Fatalf("ITL entries = %d, want 3", len(req.ITL))
	}
	for i, v := range req.ITL {
		if v != 106 {
			t.Errorf("ITL[%d] = %d, want 106 (101 us step + 5 us output processing)", i, v)
		}
	}
	if got, want := d.Metrics.RequestE2Es["r0"], float64(115+3*106); got != want {
		t.Errorf("E2E = %v, want %v", got, want)
	}
}

// Acceptance is an input the host supplies (#2021 B3): a request verifying K drafts a step is
// told α x K of them were accepted, by the compatibility engine's deterministic carry. With K = 3
// and α = 0.5, the accepted counts alternate 1, 2, so the request finishes in fewer decode steps
// than it would accepting none.
func TestEngineDriver_SuppliesTheRunsAcceptanceToTheEngine(t *testing.T) {
	run := func(alpha float64) (*Request, *EngineDriver) {
		eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, Drafts: 3})
		if err != nil {
			t.Fatal(err)
		}
		d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Acceptance: alpha})
		req := &Request{ID: "r0", InputTokens: tokens(8), OutputTokens: tokens(13), ArrivalTime: 0}
		d.InjectArrival(req)
		d.Run()
		return req, d
	}
	none, _ := run(0)
	half, d := run(0.5)
	if d.Metrics.CompletedRequests != 1 {
		t.Fatalf("completed = %d, want 1", d.Metrics.CompletedRequests)
	}
	// 12 decode tokens after the first: accepting none takes 12 decode steps; accepting
	// 1, 2, 1, 2, ... (plus the bonus token each step) takes 5.
	if got := len(none.ITL); got != 12 {
		t.Errorf("α 0: ITL entries = %d, want 12", got)
	}
	if got := len(half.ITL); got != 5 {
		t.Errorf("α 0.5: ITL entries = %d, want 5 (accepted 1, 2, 1, 2 then the rest)", got)
	}
}
