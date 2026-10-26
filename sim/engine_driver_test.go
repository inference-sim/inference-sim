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
