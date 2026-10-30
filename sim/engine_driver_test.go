package sim

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
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

// Through an engine whose steps run on pipeline stages (engine.Pipelined), two batches in
// flight overlap one per stage: the second enters the first stage as the first batch leaves it,
// instead of waiting for the whole step. Same engine and requests as the serial test above,
// split into two stages of 50 us each.
func TestEngineDriver_PipelinedBatchesOverlapOnePerStage(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 8, InFlight: 2,
		StepBaseUs: 100, Stages: 2}, 1<<40)
	a := &Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
	b := &Request{ID: "b", InputTokens: tokens(8), OutputTokens: tokens(1), ArrivalTime: 0}
	d.InjectArrival(a)
	d.InjectArrival(b)
	d.Run()
	if d.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed = %d, want 2", d.Metrics.CompletedRequests)
	}
	// a runs stage 0 at 0-50 and stage 1 at 50-100; b enters stage 0 at 50 and stage 1 at 100.
	if a.FirstTokenTime != 100 || b.FirstTokenTime != 150 {
		t.Errorf("first tokens at %d and %d, want 100 and 150 (one batch a stage)", a.FirstTokenTime, b.FirstTokenTime)
	}
	if d.BusyUntil() < b.FirstTokenTime {
		t.Errorf("BusyUntil %d before the pipeline drains at %d", d.BusyUntil(), b.FirstTokenTime)
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

// lookupEngine is an engine.Engine that reports a scripted prefix-cache lookup at each admission
// of a request: a first admission's, and, after the driver is told the request was preempted, a
// resumption's. It wraps the FIFO fake, which never preempts: the batch after "resumed"'s second
// reports it preempted and, the fake having kept it, readmitted on that same batch.
type lookupEngine struct {
	engine.Engine
	waiting map[string]bool // not running since their last admission
	seen    map[string]int  // batches each request ran in
	resumes int
	preempt bool // preempt "resumed" on the next batch
}

func (e *lookupEngine) Add(r engine.Request, now engine.Time) error {
	e.waiting[r.ID] = true
	return e.Engine.Add(r, now)
}

func (e *lookupEngine) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.Engine.Schedule(now)
	if err != nil || b.Empty() {
		return b, err
	}
	if e.preempt {
		// The schedule preempts "resumed": it is reported preempted, and, the fake having kept
		// it, readmitted on this same batch, as an engine that frees and readmits it at once.
		e.preempt = false
		b.Preempted = append(b.Preempted, "resumed")
		e.waiting["resumed"] = true
	}
	for i, s := range b.Reqs {
		e.seen[s.ID]++
		switch {
		case e.waiting[s.ID] && s.ID == "skipped":
			// The engine ran no lookup for it (its prefix cache is off for it).
		case e.waiting[s.ID] && e.seen[s.ID] == 1:
			b.Reqs[i].PrefixLookup, b.Reqs[i].CachedTokens = true, 8
		case e.waiting[s.ID]:
			// A resumption's lookup: reported, and not counted again by the host.
			b.Reqs[i].PrefixLookup, b.Reqs[i].CachedTokens = true, 16
			e.resumes++
		}
		e.waiting[s.ID] = false
		if s.ID == "resumed" && e.seen[s.ID] == 2 {
			e.preempt = true
		}
	}
	return b, nil
}

// The report's prefix-cache hit rate is over each request's first admission's lookup, as the
// engine reports it (Scheduled.PrefixLookup): a request the engine did not look up counts
// neither its prompt nor a hit, and a resumption's lookup is not counted again, as vLLM's
// prefix-cache stats count only a request's first (PrefixCacheStats.record,
// vllm/v1/metrics/stats.py:138-155@5df63a72).
func TestEngineDriver_RecordsEachRequestsFirstPrefixLookup(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100})
	if err != nil {
		t.Fatal(err)
	}
	eng := &lookupEngine{Engine: fifo, waiting: map[string]bool{}, seen: map[string]int{}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
	for _, id := range []string{"resumed", "skipped"} {
		d.InjectArrival(&Request{ID: id, InputTokens: tokens(32), OutputTokens: tokens(4)})
	}
	d.Run()
	m := d.Metrics
	if m.CompletedRequests != 2 || eng.resumes != 1 {
		t.Fatalf("completed = %d, resumptions = %d; want 2 and 1", m.CompletedRequests, eng.resumes)
	}
	if m.PrefixCacheQueryTokens != 32 || m.PrefixCacheHitTokens != 8 {
		t.Errorf("queries %d, hits %d; want 32 and 8: only the looked-up request's first admission",
			m.PrefixCacheQueryTokens, m.PrefixCacheHitTokens)
	}
}

// A request that reached the host but has not reached the engine when the horizon ends the run
// -- its render still an event in the pipeline -- is still queued (INV-1): it is in the host's
// records and in no other terminal bucket, so leaving it out of still_queued loses it from the
// conservation ledger. A request whose client deadline already passed in the pipeline is timed
// out, its own bucket, and is not also still queued.
func TestEngineDriver_ARequestStillInThePipelineAtTheHorizonIsStillQueued(t *testing.T) {
	for _, tc := range []struct {
		name             string
		deadline         int64 // the late request's client deadline (0 = none)
		queued, timedOut int
	}{
		{"render still running at the horizon", 0, 1, 0},
		{"timed out during its render", 950, 0, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100})
			if err != nil {
				t.Fatal(err)
			}
			// A 500 us render, so a request arriving at 900 is still rendering at the 1000 us horizon.
			d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1000, QueueingTime: func(*Request) int64 { return 500 }})
			early := &Request{ID: "early", InputTokens: tokens(8), OutputTokens: tokens(2), ArrivalTime: 0}
			late := &Request{ID: "late", InputTokens: tokens(8), OutputTokens: tokens(2), ArrivalTime: 900, Deadline: tc.deadline}
			d.InjectArrival(early)
			d.InjectArrivalAt(late, late.ArrivalTime)
			if tc.deadline > 0 {
				// The host's client timeout: it fires before the render ends, marking the request.
				d.schedule(&driverTimeoutEvent{at: tc.deadline, req: late})
			}
			d.Run()
			m := d.Metrics
			if _, ok := m.Requests["late"]; !ok {
				t.Fatal("non-vacuity: the late request is not in the host's records")
			}
			if m.CompletedRequests != 1 {
				t.Fatalf("non-vacuity: completed %d, want the early request only", m.CompletedRequests)
			}
			if m.StillQueued != tc.queued || m.TimedOutRequests != tc.timedOut {
				t.Errorf("still queued %d, timed out %d; want %d and %d", m.StillQueued, m.TimedOutRequests, tc.queued, tc.timedOut)
			}
			if accounted := inv1Accounted(m); accounted != len(m.Requests) {
				t.Errorf("INV-1: %d requests registered, %d accounted for", len(m.Requests), accounted)
			}
		})
	}
}

// unevenStages is a FIFO whose batches split by stage as listed, in order: batch i takes
// splits[i] (the last entry repeats).
type unevenStages struct {
	*enginetest.FIFO
	splits [][]engine.Duration
	n      int
}

func (u *unevenStages) StepTime(b engine.Batch) engine.Duration {
	var t engine.Duration
	for _, s := range u.split() {
		t += s
	}
	return t
}

func (u *unevenStages) split() []engine.Duration { return u.splits[min(u.n, len(u.splits)-1)] }

func (u *unevenStages) StageTimes(b engine.Batch) []engine.Duration {
	s := u.split()
	u.n++
	return append([]engine.Duration(nil), s...)
}

// A decode under pipeline parallelism samples from the step pp batches back: vLLM's non-last rank
// waits on that step's sampled tokens before its forward (pp_utils.py:125-152@5df63a72). With
// stage splits [8,20], [1,8], [20,8] and two stages, batch 2 runs batch 0's request and so may
// not start before batch 0 leaves the last stage at 28: it ends at 56, not 44.
func TestEngineDriver_PipelinedDecodeWaitsForItsPreviousToken(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 3, StepBaseUs: 1, Stages: 2})
	if err != nil {
		t.Fatal(err)
	}
	eng := &unevenStages{FIFO: fifo, splits: [][]engine.Duration{{8, 20}, {1, 8}, {20, 8}}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
	var ends []int64
	for _, id := range []string{"a", "b", "a"} {
		_, e := d.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: id, Tokens: 1}}}, 0)
		ends = append(ends, e)
	}
	if want := []int64{28, 36, 56}; ends[0] != want[0] || ends[1] != want[1] || ends[2] != want[2] {
		t.Fatalf("batch ends %v, want %v (batch 2 waits for batch 0's token)", ends, want)
	}
	// A batch with none of the earlier batch's requests is held back too: the non-last rank
	// waits on step T-pp's broadcast whenever it sampled for a live request
	// (TestEngineDriver_PipelinedStepWaitsOnTheSampledStepPPBack).
	d2 := NewEngineDriver(&unevenStages{FIFO: fifo, splits: [][]engine.Duration{{8, 20}, {1, 8}, {20, 8}}},
		EngineDriverConfig{Horizon: 1 << 40})
	var e2 []int64
	for _, id := range []string{"a", "b", "c"} {
		_, e := d2.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: id, Tokens: 1}}}, 0)
		e2 = append(e2, e)
	}
	if e2[2] != 56 {
		t.Fatalf("an unrelated third batch ends at %d, want 56", e2[2])
	}
}

// A batch whose engine prices it as one stage (core/vllm's StageTimes when the pricer's split is
// not pp long) after pipelined batches runs behind them: it may not end before the batch ahead of
// it leaves the last stage, since batches complete in the order they were scheduled (the driver
// completes inFlight[0] at each step end). Splits [10,90] then [20]: batch 1 ends at 100, so
// batch 2 ends no earlier.
func TestEngineDriver_OneStageBatchDoesNotOvertakeThePipeline(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 8, InFlight: 3, StepBaseUs: 1, Stages: 2})
	if err != nil {
		t.Fatal(err)
	}
	eng := &unevenStages{FIFO: fifo, splits: [][]engine.Duration{{10, 90}, {20}, {5, 5}}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
	var ends []int64
	for _, id := range []string{"a", "b", "c"} {
		_, e := d.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: id, Tokens: 1}}}, 0)
		ends = append(ends, e)
	}
	for i := 1; i < len(ends); i++ {
		if ends[i] < ends[i-1] {
			t.Fatalf("batch ends %v: batch %d ends before batch %d", ends, i+1, i)
		}
	}
	if d.BusyUntil() < ends[len(ends)-1] {
		t.Fatalf("BusyUntil %d before the last batch ends at %d", d.BusyUntil(), ends[len(ends)-1])
	}
}

// vLLM's non-last pipeline rank, before step T's forward, waits on step T-pp's sampled-token
// broadcast whenever that step sampled for a request still live, whatever step T runs
// (PPHandler.get_prev_sampled_outputs: the slot pp steps back, minus freed requests and those
// that sampled nothing, vllm/v1/worker/gpu/pp_utils.py:125-152, 37-53; model_runner.py:1732
// @5df63a72). A non-final prefill chunk samples nothing, so it holds no later step back, and a
// request preempted after step T-pp is freed before step T (finish_requests, model_runner.py:
// 1152-1166). Splits [8,20], [1,8], [20,8] at pp 2:
//   - a decode of a at batch 0 holds batch 2 back even though batch 2 runs c: ends 28, 36, 56;
//   - a non-final chunk of a at batch 0 does not hold a's next chunk at batch 2: 28, 36, 44;
//   - a decode of a at batch 0, preempted at batch 1, does not hold batch 2: 28, 36, 44.
//
// Whether step T-pp sampled for a request is computed + scheduled >= prefill_len, less a final
// prefill chunk whose one token reaches the length cap (compute_need_sampled_mask, pp_utils.py:
// 37-51@5df63a72). prefill_len is the prompt plus the outputs before the request's admission
// (states.py:104-110@5df63a72), and the cap is prompt + max_tokens (states.py:104):
//   - a final chunk that reaches prefill_len exactly samples, so it holds batch 2: 56;
//   - a final chunk of a max_tokens-1 request (every P/D prefill leg) samples nothing: 44;
//   - a resumed request's recompute of its prompt alone, short of prompt + outputs, samples
//     nothing: 44;
//   - a request finished (aborted or timed out) before batch 1's schedule is freed by it
//     (finished_req_ids, model_runner.py:1152-1166@5df63a72), so batch 2 does not wait: 44;
//   - finished before batch 2's own schedule, it is freed only after batch 2 consumed batch 0's
//     slot (update_pp_decode_requests runs before finish_requests, model_runner.py:1732-1733): 56.
func TestEngineDriver_PipelinedStepWaitsOnTheSampledStepPPBack(t *testing.T) {
	decode := engine.Scheduled{ID: "a", Tokens: 1, Computed: 8, PromptLen: 8}
	chunk := engine.Scheduled{ID: "a", Tokens: 4, Computed: 0, PromptLen: 12}
	final := engine.Scheduled{ID: "a", Tokens: 4, Computed: 8, PromptLen: 12}
	b := engine.Batch{Reqs: []engine.Scheduled{{ID: "b", Tokens: 1, Computed: 8, PromptLen: 8}}}
	other := engine.Batch{Reqs: []engine.Scheduled{{ID: "c", Tokens: 1, Computed: 8, PromptLen: 8}}}
	queued := func(maxOut, emitted int) map[string]*driverReq {
		return map[string]*driverReq{"a": {req: &Request{ID: "a", MaxOutputLen: maxOut, State: StateQueued}, emitted: emitted}}
	}
	for _, c := range []struct {
		name    string
		batches []engine.Batch
		reqs    map[string]*driverReq
		freeAt  int // the a is finished just before batch freeAt is timed (0: never)
		want    int64
	}{
		{"final chunk at prefill_len", []engine.Batch{{Reqs: []engine.Scheduled{final}}, b, other}, nil, 0, 56},
		{"length-capped final chunk", []engine.Batch{{Reqs: []engine.Scheduled{final}}, b, other}, queued(1, 0), 0, 44},
		{"resumed recompute", []engine.Batch{{Reqs: []engine.Scheduled{{ID: "a", Tokens: 8, Computed: 0, PromptLen: 8}}}, b, other},
			queued(0, 3), 0, 44},
		{"finished before batch 1", []engine.Batch{{Reqs: []engine.Scheduled{decode}}, b, other}, nil, 1, 44},
		{"finished before batch 2", []engine.Batch{{Reqs: []engine.Scheduled{decode}}, b, other}, nil, 2, 56},
		{"unshared decode", []engine.Batch{{Reqs: []engine.Scheduled{decode}}, b,
			{Reqs: []engine.Scheduled{{ID: "c", Tokens: 1, Computed: 8, PromptLen: 8}}}}, nil, 0, 56},
		{"prefill chunk", []engine.Batch{{Reqs: []engine.Scheduled{chunk}}, b,
			{Reqs: []engine.Scheduled{{ID: "a", Tokens: 4, Computed: 4, PromptLen: 12}}}}, nil, 0, 44},
		{"preempted", []engine.Batch{{Reqs: []engine.Scheduled{decode}},
			{Reqs: b.Reqs, Preempted: []string{"a"}},
			{Reqs: []engine.Scheduled{{ID: "a", Tokens: 9, Computed: 0, PromptLen: 8}}}}, nil, 0, 44},
	} {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 3, StepBaseUs: 1, Stages: 2})
		if err != nil {
			t.Fatal(err)
		}
		d := NewEngineDriver(&unevenStages{FIFO: fifo, splits: [][]engine.Duration{{8, 20}, {1, 8}, {20, 8}}},
			EngineDriverConfig{Horizon: 1 << 40})
		if c.reqs != nil {
			d.reqs = c.reqs
		}
		var end int64
		for i, bt := range c.batches {
			if c.freeAt > 0 && i == c.freeAt {
				d.noteFreed("a")
			}
			_, end = d.timeBatch(bt, 0)
		}
		if end != c.want {
			t.Errorf("%s: batch 2 ends at %d, want %d", c.name, end, c.want)
		}
	}
}

// ringStages is unevenStages through an engine that counts its runner steps (RunnerSteps), as
// blis-vllm core/vllm does: steps is set by the test, one per schedule.
type ringStages struct {
	*unevenStages
	steps int
}

func (r *ringStages) RunnerSteps() int { return r.steps }

// vLLM's slot ring advances on every execute_model call, a schedule that ran nothing included:
// update_pp_decode_requests pops the slot pp steps back and reserves an empty one before the early
// return (model_runner.py:1729-1747; pp_utils.py:125-137@5df63a72). So at pp 2, with batch 0 (a
// decode), an empty schedule, batch 1 and batch 2, batch 1 is runner step 2 and waits on batch 0,
// and batch 2 is runner step 3 and waits on the empty step's slot, which is nothing. Splits [8,20],
// [1,8], [20,8]: ends 28, 37, 57. An engine that counts no runner steps keeps the driver's own
// count, the batches it timed: 28, 36, 56.
func TestEngineDriver_PipelinedStepGatesOnTheRunnerStepPPBack(t *testing.T) {
	decode := func(id string) engine.Batch {
		return engine.Batch{Reqs: []engine.Scheduled{{ID: id, Tokens: 1, Computed: 8, PromptLen: 8}}}
	}
	for _, c := range []struct {
		name string
		ring bool
		want []int64
	}{{"runner steps", true, []int64{28, 37, 57}}, {"batches timed", false, []int64{28, 36, 56}}} {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 3, StepBaseUs: 1, Stages: 2})
		if err != nil {
			t.Fatal(err)
		}
		u := &unevenStages{FIFO: fifo, splits: [][]engine.Duration{{8, 20}, {1, 8}, {20, 8}}}
		r := &ringStages{unevenStages: u}
		var eng engine.Engine = u
		if c.ring {
			eng = r
		}
		d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
		var ends []int64
		r.steps++
		_, e := d.timeBatch(decode("a"), 0)
		ends = append(ends, e)
		r.steps++
		d.noteEmptyStep(engine.Batch{})
		for _, id := range []string{"b", "c"} {
			r.steps++
			_, e := d.timeBatch(decode(id), 0)
			ends = append(ends, e)
		}
		if !reflect.DeepEqual(ends, c.want) {
			t.Errorf("%s: batch ends %v, want %v", c.name, ends, c.want)
		}
	}
}

// An empty runner step still waits on its slot pp steps back (the wait_event on the main stream,
// pp_utils.py:152@5df63a72), so the next forward starts no earlier. Batch 0 a decode [8,20] ending
// at 28, batch 1 a non-final chunk [1,8] ending at 36, then an empty step that waits on batch 0,
// then batch 2 [20,8], whose own slot (batch 1) sampled nothing: batch 2 starts at 28, ends at 56,
// not 44.
func TestEngineDriver_EmptyRunnerStepWaitHoldsTheNextForward(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 3, StepBaseUs: 1, Stages: 2})
	if err != nil {
		t.Fatal(err)
	}
	r := &ringStages{unevenStages: &unevenStages{FIFO: fifo, splits: [][]engine.Duration{{8, 20}, {1, 8}, {20, 8}}}}
	d := NewEngineDriver(r, EngineDriverConfig{Horizon: 1 << 40})
	r.steps++
	d.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: "a", Tokens: 1, Computed: 8, PromptLen: 8}}}, 0)
	r.steps++
	d.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: "b", Tokens: 4, Computed: 0, PromptLen: 12}}}, 0)
	r.steps++
	d.noteEmptyStep(engine.Batch{})
	r.steps++
	start, end := d.timeBatch(engine.Batch{Reqs: []engine.Scheduled{{ID: "c", Tokens: 1, Computed: 8, PromptLen: 8}}}, 0)
	if start != 28 || end != 56 {
		t.Fatalf("batch 2 runs %d-%d, want 28-56", start, end)
	}
}
