package harness

import (
	"slices"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// The trace and pool flags a point runs reproduce the client's concurrency gate, checked on the
// code `blis replay` runs for them (cmd/replay.go closed-loop pool mode:
// LoadTraceV2SessionBlueprints, then BuildSessionPool, then SessionPoolDriver.OnComplete per
// completion).
//
// The client (benchmark_serving.py benchmark()) creates one task per main-phase request, in
// order, each acquiring a semaphore of max_concurrency before sending; at --request-rate inf
// get_request yields without waiting. So the first CONC requests are sent at once and, each
// time one finishes, the next in request order is sent. The pool must therefore inject records
// 0..CONC-1 at t=0 and admit record k at the completion that frees its slot, in record order,
// with no think time.
func TestThePoolIsTheClientsSemaphore(t *testing.T) {
	const conc, n = 3, 8
	pl := PointLengths{ISL: 64, OSL: 64, Concurrency: conc}
	for i := 0; i < n; i++ {
		pl.InputLens = append(pl.InputLens, 51+i)
		pl.OutputLens = append(pl.OutputLens, 60+i)
	}
	header, records := pointTrace(pl)
	trace := &workload.TraceV2{Header: *header, Records: records}
	r0, blueprints, err := workload.LoadTraceV2SessionBlueprints(trace, 42, nil, 1<<62)
	if err != nil {
		t.Fatal(err)
	}
	if len(r0) != n || len(blueprints) != n {
		t.Fatalf("%d round-0 requests and %d blueprints from %d records; every record must be its own session",
			len(r0), len(blueprints), n)
	}
	driver, initial, err := workload.BuildSessionPool(blueprints, r0, conc, n, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != conc {
		t.Fatalf("%d requests start at t=0, want the semaphore's %d", len(initial), conc)
	}
	for i, req := range initial {
		if req.ID != "request_"+strconv.Itoa(i) || req.ArrivalTime != 0 {
			t.Errorf("initial request %d is %s at %d; want request_%d at 0", i, req.ID, req.ArrivalTime, i)
		}
	}
	// Each completion admits exactly the next record, at the completion tick: no think time,
	// no reordering, no clone. Lengths are the record's.
	inFlight := append([]string(nil), "request_0", "request_1", "request_2")
	for k, tick := conc, int64(1000); k < n; k, tick = k+1, tick+1000 {
		done := initialOrAdmitted(initial, inFlight[0])
		done.State = sim.StateCompleted // as the simulator hands a completion to the pool
		inFlight = inFlight[1:]
		next := driver.OnComplete(done, tick)
		if len(next) != 1 {
			t.Fatalf("completion %d admitted %d requests, want 1", k-conc, len(next))
		}
		got := next[0]
		if got.ID != "request_"+strconv.Itoa(k) || got.ArrivalTime != tick {
			t.Errorf("completion at %d admitted %s at %d; want request_%d at %d", tick, got.ID, got.ArrivalTime, k, tick)
		}
		if int(got.InputLen()) != pl.InputLens[k] || len(got.OutputTokens) != pl.OutputLens[k] {
			t.Errorf("request_%d runs %d/%d tokens, want %d/%d", k, got.InputLen(), len(got.OutputTokens),
				pl.InputLens[k], pl.OutputLens[k])
		}
		initial = append(initial, got)
		inFlight = append(inFlight, got.ID)
	}
	// With every record started, a completion admits nothing more.
	last := initialOrAdmitted(initial, inFlight[0])
	last.State = sim.StateCompleted
	if more := driver.OnComplete(last, 99999); len(more) != 0 {
		t.Errorf("after the last record a completion admitted %d more", len(more))
	}
	if driver.Unstarted() != 0 || driver.TotalSessions() != n {
		t.Errorf("pool of %d with %d unstarted; want %d and 0", driver.TotalSessions(), driver.Unstarted(), n)
	}
}

func initialOrAdmitted(reqs []*sim.Request, id string) *sim.Request {
	for _, r := range reqs {
		if r.ID == id {
			return r
		}
	}
	return nil
}

// End to end through the binary: one point replayed exactly as the scorers replay it runs its
// 10*CONC requests at the fixture's lengths, starts the first CONC at t=0, and starts every
// later one, in request order, after t=0.
func TestAReplayedPointIsTheBenchmarksMainPhase(t *testing.T) {
	ws, err := LoadBenchmarkWorkloads(DefaultWorkloadsPath(0))
	if err != nil {
		t.Fatal(err)
	}
	const conc = 4
	pl, err := ws.At(1024, 1024, conc)
	if err != nil {
		t.Fatal(err)
	}
	sw := committedSweep("vllm")
	m, err := kernelmodel.Open(sw.Scenario, testRepos())
	if err != nil {
		t.Fatal(err)
	}
	eng, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	disabled := false
	ps := &PointSettings{Passed: PassedSettings{EnablePrefixCaching: &disabled}}
	set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: sw.Scenario, Label: sw.Label,
		ByConcurrency: map[string]*PointSettings{"4": ps}}}}
	pc := resolveAdmission(sw, conc, eng, m.Deployment(), set)
	cfg := Config{Repos: testRepos(), Blis: testBlis(t), Seed: 42}
	out, err := runBlis(cfg, sw.Scenario, pc, mainPhase(pl))
	if err != nil {
		t.Fatal(err)
	}
	if out.CompletedRequests != 10*conc || len(out.Requests) != 10*conc {
		t.Fatalf("completed %d with %d request records, want %d", out.CompletedRequests, len(out.Requests), 10*conc)
	}
	arrival := make([]float64, 10*conc)
	for _, r := range out.Requests {
		i, err := requestIndex(r.ID)
		if err != nil {
			t.Fatal(err)
		}
		arrival[i] = r.ArrivedAt
	}
	for i, a := range arrival {
		switch {
		case i < conc && a != 0:
			t.Errorf("request %d of the first %d arrived at %v s, not 0", i, conc, a)
		case i >= conc && a <= 0:
			t.Errorf("request %d arrived at %v s; only the first %d start at 0", i, a, conc)
		case i > 0 && a < arrival[i-1]:
			t.Errorf("request %d arrived at %v s, before request %d at %v s: trace order lost", i, a, i-1, arrival[i-1])
		}
	}
	if _, err := summarize(out.Requests, pl, nil); err != nil {
		t.Errorf("summarize: %v", err)
	}
}

// The primed trace states request 0's prompt cached, with existing trace features only: each
// warm-up copy's prompt is token-for-token main request 0's; the pool starts the copies at t=0
// and the first CONC main requests together, in order, at the main-phase start; and a copy's
// completion admits no main request (its session holds the slot for a far-later holder round).
func TestThePrimedTraceCachesRequestZerosPromptAndNothingElse(t *testing.T) {
	const conc, n, copies = 3, 7, 2
	pl := PointLengths{ISL: 64, OSL: 64, Concurrency: conc}
	for i := 0; i < n; i++ {
		pl.InputLens = append(pl.InputLens, 51+i)
		pl.OutputLens = append(pl.OutputLens, 60+i)
	}
	run := primedPhase(pl, copies)
	if run.pool != conc+copies || run.sessions != n+copies {
		t.Fatalf("pool %d of %d sessions, want %d of %d", run.pool, run.sessions, conc+copies, n+copies)
	}
	r0, bps, err := workload.LoadTraceV2SessionBlueprints(&workload.TraceV2{Header: *run.header, Records: run.records},
		42, nil, 1<<62)
	if err != nil {
		t.Fatal(err)
	}
	driver, initial, err := workload.BuildSessionPool(bps, r0, run.pool, run.sessions, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != conc+copies {
		t.Fatalf("%d initial requests, want %d", len(initial), conc+copies)
	}
	main0 := initial[copies]
	if main0.ID != "request_0" || int(main0.InputLen()) != pl.InputLens[0] {
		t.Fatalf("first main request is %s with %d tokens", main0.ID, main0.InputLen())
	}
	for j := 0; j < copies; j++ {
		w := initial[j]
		if w.ArrivalTime != 0 || !slices.Equal(w.InputTokens, main0.InputTokens) || len(w.OutputTokens) != pl.OutputLens[0] {
			t.Errorf("warm-up copy %d (%s) at %d is not request 0's prompt and output", j, w.ID, w.ArrivalTime)
		}
	}
	for i, r := range initial[copies:] {
		if r.ID != "request_"+strconv.Itoa(i) || r.ArrivalTime != primeMainStartUs {
			t.Errorf("main request %d is %s at %d, want request_%d at %d", i, r.ID, r.ArrivalTime, i, primeMainStartUs)
		}
		if i > 0 && slices.Equal(r.InputTokens[:16], main0.InputTokens[:16]) {
			t.Errorf("main request %d shares request 0's first block", i)
		}
	}
	// A warm-up copy completing yields only its holder round, far after the main phase, and
	// admits no main request. (A completion reaches the pool in state completed; a queued
	// request reaching it means a drop, which cancels the session.)
	initial[0].State = sim.StateCompleted
	next := driver.OnComplete(initial[0], 5000)
	if len(next) != 1 || next[0].SessionID != "warmup-0" || next[0].ArrivalTime != 5000+primeHoldUs {
		t.Fatalf("a warm-up copy's completion produced %+v, want only its holder round at %d", next, 5000+primeHoldUs)
	}
	// A main completion admits the next main request, in order.
	main0.State = sim.StateCompleted
	if got := driver.OnComplete(main0, primeMainStartUs+7); len(got) != 1 || got[0].ID != "request_"+strconv.Itoa(conc) {
		t.Fatalf("a main completion admitted %+v, want request_%d", got, conc)
	}
}

// End to end: with prefix caching on, the primed run's main request 0 is served from the
// warm-up copy's cached prompt, so its first token comes sooner than in the cold run; the run
// still completes the point's 10*CONC main requests, and the bracket reports both.
func TestThePrimedRunServesRequestZeroFromTheCache(t *testing.T) {
	sw := committedSweep("vllm")
	sw.Points = sw.Points[:1]
	enabled := true
	set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: sw.Scenario, Label: sw.Label,
		ByConcurrency: map[string]*PointSettings{"4": {Passed: PassedSettings{EnablePrefixCaching: &enabled}}}}}}
	cfg := Config{Repos: testRepos(), Blis: testBlis(t), Seed: 42, EngineSettings: set}
	cold, err := Run(sw, 4, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := RunWarmupBracket(sw, 4, cfg, cold, nil)
	if err != nil {
		t.Fatal(err)
	}
	if b.Primed.Completed != 40 || b.Primed.MeanTTFTUs >= cold.MeanTTFTUs {
		t.Errorf("primed run completed %d, mean TTFT %.0f us against the cold %.0f us; request 0's cached "+
			"prompt should lower it", b.Primed.Completed, b.Primed.MeanTTFTUs, cold.MeanTTFTUs)
	}
	if b.Bounded {
		t.Error("a bracket with no noise draws cannot be bounded")
	}
	t.Logf("cold TTFT %.0f us TPOT %.1f us; primed TTFT %.0f us (%+.2f%%) TPOT %.1f us (%+.3f%%)",
		cold.MeanTTFTUs, cold.MeanTPOTUs, b.Primed.MeanTTFTUs, b.DTTFTPct, b.Primed.MeanTPOTUs, b.DTPOTPct)
}
