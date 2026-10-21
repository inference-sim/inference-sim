package harness

import (
	"slices"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	kvcache "github.com/inference-sim/inference-sim/sim/kv"
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

// prefixPoint is the committed scenario at concurrency 4 with its run's prefix caching set.
func prefixPoint(t *testing.T, enabled bool) (Sweep, Config) {
	t.Helper()
	sw := committedSweep("vllm")
	sw.Points = sw.Points[:1]
	set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: sw.Scenario, Label: sw.Label,
		ByConcurrency: map[string]*PointSettings{"4": {Passed: PassedSettings{EnablePrefixCaching: &enabled}}}}}}
	return sw, Config{Repos: testRepos(), Blis: testBlis(t), Seed: 42, EngineSettings: set}
}

// Run scores the faithful protocol: on a prefix-caching point the primed main phase -- main
// request 0 served from the cached prompt, so its first token comes sooner than from an empty
// engine -- and on a point with caching off the cold one, which is then the same thing. RunCold
// is the information-only empty-engine run.
//
// The mechanism is asserted on request 0 itself. A full prefix hit computes only the prompt's
// last block (vLLM v0.31.0 KVCacheManager.get_computed_blocks, max_cache_hit_length =
// num_tokens - 1, vllm/v1/core/kv_cache_manager.py:289-299), so its prefill is shorter. The
// main-phase MEAN is not a measure of it: the client is closed-loop, each completion sends the
// next request, so request 0 finishing sooner shifts every later arrival, and a shifted
// arrival can land on another's prefill. At this point request 5 then queues ~19 ms longer and
// the primed mean exceeds the cold one by 0.17%, though request 0 is 15 ms faster.
func TestRunScoresThePrimedMainPhaseWhereThePrefixIsCached(t *testing.T) {
	sw, cfg := prefixPoint(t, true)
	scored, err := Run(sw, 4, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cold, err := RunCold(sw, 4, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !scored.Primed || cold.Primed || scored.Completed != 40 || cold.Completed != 40 {
		t.Fatalf("scored primed=%v completed=%d, cold primed=%v completed=%d", scored.Primed, scored.Completed,
			cold.Primed, cold.Completed)
	}
	if scored.FirstTTFTUs <= 0 || scored.FirstTTFTUs >= cold.FirstTTFTUs {
		t.Errorf("primed request 0 TTFT %.0f us is not below the cold %.0f us: its cached prompt did not "+
			"shorten its prefill", scored.FirstTTFTUs, cold.FirstTTFTUs)
	}
	d, err := RunWarmupDifference(sw, 4, cfg, scored)
	if err != nil {
		t.Fatal(err)
	}
	if d.Scored != scored || d.Cold != cold || d.DTTFTPct != 100*(scored.MeanTTFTUs/cold.MeanTTFTUs-1) {
		t.Errorf("the reported difference is not scored against cold: %+v", d)
	}
	t.Logf("scored (primed) TTFT %.0f us TPOT %.1f us, request 0 %.0f us; cold TTFT %.0f us (%+.2f%%) TPOT %.1f us (%+.3f%%), request 0 %.0f us",
		scored.MeanTTFTUs, scored.MeanTPOTUs, scored.FirstTTFTUs, cold.MeanTTFTUs, d.DTTFTPct, cold.MeanTPOTUs, d.DTPOTPct, cold.FirstTTFTUs)

	off, cfgOff := prefixPoint(t, false)
	a, err := Run(off, 4, cfgOff)
	if err != nil {
		t.Fatal(err)
	}
	b, err := RunCold(off, 4, cfgOff)
	if err != nil {
		t.Fatal(err)
	}
	if a.Primed || a != b {
		t.Errorf("with prefix caching off Run must be the cold main phase: %+v against %+v", a, b)
	}
}

// The primed main phase is the drained warm-up's state, not merely a state with request 0's
// prompt cached: it must not matter how many copies of request 0 ran before it. The benchmark
// runs 2*CONC copies; Run runs dp. Both leave every replica holding the same prompt, an idle
// engine and a router counter a multiple of dp along, so the main phase is identical -- down to
// the last microsecond -- whichever ran. A difference here would mean the primed trace leaves
// some other state behind.
func TestThePrimedMainPhaseDoesNotDependOnHowManyCopiesRan(t *testing.T) {
	sw, cfg := prefixPoint(t, true)
	ws, err := cfg.workloads()
	if err != nil {
		t.Fatal(err)
	}
	pl, err := ws.At(1024, 1024, 4)
	if err != nil {
		t.Fatal(err)
	}
	m, err := kernelmodel.Open(sw.Scenario, cfg.Repos)
	if err != nil {
		t.Fatal(err)
	}
	eng, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	pc := resolveAdmission(sw, 4, eng, m.Deployment(), cfg.EngineSettings)
	if pc.PrefixCachingDisabled || m.Deployment().DP != 1 {
		t.Fatalf("the reference point must cache prefixes on one replica: %+v dp=%d", pc, m.Deployment().DP)
	}
	ignore := func(id string) bool { i, err := requestIndex(id); return err != nil || i >= len(pl.InputLens) }
	var obs []Observation
	for _, copies := range []int{1, InferenceXWarmupsPerConcurrency * 4} {
		out, err := runBlis(cfg, sw.Scenario, pc, primedPhase(pl, copies))
		if err != nil {
			t.Fatalf("%d copies: %v", copies, err)
		}
		if err := checkPrimed(out.Requests, len(pl.InputLens)); err != nil {
			t.Fatalf("%d copies: %v", copies, err)
		}
		o, err := summarize(out.Requests, pl, ignore)
		if err != nil {
			t.Fatal(err)
		}
		obs = append(obs, o)
	}
	if obs[0] != obs[1] {
		t.Errorf("the main phase after 1 copy of request 0 (%+v) differs from after %d copies (%+v)",
			obs[0], InferenceXWarmupsPerConcurrency*4, obs[1])
	}
}

// What a drained warm-up leaves in the KV cache, on the KV cache itself: once a copy of request
// 0 has run and released its blocks, no block is in use, request 0's prompt is readable in full
// blocks, and no other main request's prompt finds anything.
func TestADrainedCopyLeavesOnlyRequestZerosPromptReadable(t *testing.T) {
	ws, err := LoadBenchmarkWorkloads(DefaultWorkloadsPath(0))
	if err != nil {
		t.Fatal(err)
	}
	pl, err := ws.At(1024, 1024, 4)
	if err != nil {
		t.Fatal(err)
	}
	run := primedPhase(pl, 1)
	r0, _, err := workload.LoadTraceV2SessionBlueprints(&workload.TraceV2{Header: *run.header, Records: run.records},
		42, nil, 1<<62)
	if err != nil {
		t.Fatal(err)
	}
	const block = 16
	kv := kvcache.NewKVCacheState(4096, block)
	copyOfZero := r0[0] // warm-up session 0's round 0
	if !kv.AllocateKVBlocks(copyOfZero, 0, copyOfZero.InputLen(), nil) {
		t.Fatal("could not allocate the copy's prompt")
	}
	kv.ReleaseKVBlocks(copyOfZero)
	if kv.UsedBlocks() != 0 {
		t.Errorf("%d blocks in use after the copy drained", kv.UsedBlocks())
	}
	main := r0[1:] // main requests 0..n-1, in order
	if got, want := len(kv.GetCachedBlocks(main[0].InputTokens)), int(main[0].InputLen())/block; got != want {
		t.Errorf("main request 0 finds %d cached blocks of its prompt, want all %d full blocks", got, want)
	}
	for i, r := range main[1:] {
		if n := len(kv.GetCachedBlocks(r.InputTokens)); n != 0 {
			t.Errorf("main request %d finds %d cached blocks; only request 0's prompt was cached", i+1, n)
		}
	}
}

// checkPrimed refuses each way a primed run could fail to be "the main phase after a drained
// warm-up": a copy still running when the main phase starts, main request 0 not starting with
// the main phase, or a holder round entering the engine before the main phase has finished.
func TestCheckPrimedRefusesAnUndrainedWarmup(t *testing.T) {
	const n = 2
	T := float64(primeMainStartUs) / 1e6 // seconds, as arrived_at is reported
	good := []sim.RequestMetrics{
		{ID: "request_2", ArrivedAt: 0, E2E: 5000},                                     // the copy: ends at 5 s
		{ID: "request_0", ArrivedAt: T, E2E: 3000},                                     // main 0
		{ID: "request_1", ArrivedAt: T, E2E: 4000},                                     // main 1
		{ID: "session_warmup-0_round_1_1", ArrivedAt: T + 3600, E2E: 1, RoundIndex: 1}, // holder
	}
	if err := checkPrimed(good, n); err != nil {
		t.Fatalf("a drained warm-up is refused: %v", err)
	}
	for name, mutate := range map[string]func([]sim.RequestMetrics){
		"a copy still running at the main-phase start": func(r []sim.RequestMetrics) { r[0].E2E = T*1e3 + 1 },
		"main request 0 starting late":                 func(r []sim.RequestMetrics) { r[1].ArrivedAt = T + 1 },
		"a holder round inside the main phase":         func(r []sim.RequestMetrics) { r[3].ArrivedAt = T + 2 },
	} {
		r := append([]sim.RequestMetrics(nil), good...)
		mutate(r)
		if checkPrimed(r, n) == nil {
			t.Errorf("checkPrimed accepted %s", name)
		}
	}
}

// On a data-parallel point (dp=2) the harness's router places like vLLM's DP balancer: the
// prefix-caching warm-up leaves request 0's prompt on every replica (Run checks it is served
// from the cache), the main phase spreads across the replicas instead of herding onto one, and
// the score is deterministic.
func TestADataParallelPointPlacesLikeVLLMsBalancer(t *testing.T) {
	const scenario = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"
	enabled := true
	sw := Sweep{Scenario: scenario, Label: "1k1k", Workload: "1024:1024", Framework: "vllm", Serving: "aggregated",
		SpecMethod: "none", Points: []Point{{Concurrency: 8}}}
	set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: scenario, Label: sw.Label,
		ByConcurrency: map[string]*PointSettings{"8": {Passed: PassedSettings{EnablePrefixCaching: &enabled}}}}}}
	cfg := Config{Repos: testRepos(), Blis: testBlis(t), Seed: 42, EngineSettings: set}

	m, err := kernelmodel.Open(scenario, cfg.Repos)
	if err != nil {
		t.Fatal(err)
	}
	if m.Deployment().DP != 2 {
		t.Fatalf("%s: dp=%d, want 2", scenario, m.Deployment().DP)
	}
	a, err := Run(sw, 8, cfg)
	if err != nil {
		t.Fatal(err) // includes checkPrimed: request 0 must be served from its cached prompt
	}
	b, err := Run(sw, 8, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Primed || a.Completed != 80 || a != b {
		t.Fatalf("primed=%v completed=%d, repeat equal=%v", a.Primed, a.Completed, a == b)
	}

	ws, err := cfg.workloads()
	if err != nil {
		t.Fatal(err)
	}
	pl, err := ws.At(1024, 1024, 8)
	if err != nil {
		t.Fatal(err)
	}
	eng, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	out, err := runBlis(cfg, scenario, resolveAdmission(sw, 8, eng, m.Deployment(), cfg.EngineSettings), mainPhase(pl))
	if err != nil {
		t.Fatal(err)
	}
	per := map[string]int{}
	for _, r := range out.Requests {
		per[r.HandledBy]++
	}
	if len(per) != 2 {
		t.Fatalf("main phase used replicas %v, want both", per)
	}
	for id, n := range per {
		if n < 30 || n > 50 {
			t.Errorf("replica %s served %d of 80 requests; balancing by requests in flight keeps them near 40 (%v)", id, n, per)
		}
	}
}
