package harness

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"slices"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/build"
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	kvcache "github.com/inference-sim/inference-sim/engines/vllm/core/kv"
	"github.com/inference-sim/inference-sim/sim"
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
	eng, err := build.StatedRunSettings(m)
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
	if out.Requests.Completed != 10*conc || len(out.RequestRows) != 10*conc {
		t.Fatalf("completed %d with %d request rows, want %d", out.Requests.Completed, len(out.RequestRows), 10*conc)
	}
	arrival := make([]float64, 10*conc)
	for _, r := range out.RequestRows {
		i, err := requestIndex(r.ID)
		if err != nil {
			t.Fatal(err)
		}
		arrival[i] = float64(r.ArrivalUs) / 1e6
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
	if _, err := summarize(out.RequestRows, pl, nil); err != nil {
		t.Errorf("summarize: %v", err)
	}
}

// The primed trace states request 0's prompt cached with two drain-gated phases: each warm-up copy's
// prompt is token-for-token main request 0's; the copies are sent at t=0; the main phase is held until
// every copy has ended, then its first CONC requests are sent together, in order, and each later one
// when a main request ends (the client's semaphore of CONC).
func TestThePrimedTraceCachesRequestZerosPromptAndNothingElse(t *testing.T) {
	const conc, n, copies = 3, 7, 2
	pl := PointLengths{ISL: 64, OSL: 64, Concurrency: conc}
	for i := 0; i < n; i++ {
		pl.InputLens = append(pl.InputLens, 51+i)
		pl.OutputLens = append(pl.OutputLens, 60+i)
	}
	run := primedPhase(pl, copies)
	if !run.phased || len(run.records) != n+copies {
		t.Fatalf("phased=%v with %d records, want a phased trace of %d", run.phased, len(run.records), n+copies)
	}
	reqs, err := workload.LoadTraceV2Requests(&workload.TraceV2{Header: *run.header, Records: run.records}, 42)
	if err != nil {
		t.Fatal(err)
	}
	gate, initial, err := workload.NewPhaseGate(run.header.Phases, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != copies {
		t.Fatalf("%d requests sent at the start, want only the %d warm-up copies", len(initial), copies)
	}
	var main0 *sim.Request
	for _, r := range reqs {
		if r.ID == "request_0" {
			main0 = r
		}
	}
	if main0 == nil || int(main0.InputLen()) != pl.InputLens[0] {
		t.Fatalf("main request 0 is %+v", main0)
	}
	for _, w := range initial {
		if w.ArrivalTime != 0 || !slices.Equal(w.InputTokens, main0.InputTokens) || len(w.OutputTokens) != pl.OutputLens[0] {
			t.Errorf("warm-up copy %s at %d is not request 0's prompt and output", w.ID, w.ArrivalTime)
		}
	}
	// The first copy ending releases nothing; the last one ending (at 5000) sends the first CONC main
	// requests together, in order, at that clock.
	if got := gate.Ended(initial[0], 4000); len(got) != 0 {
		t.Fatalf("one copy ending sent %d request(s) before the warm-up drained", len(got))
	}
	sent := gate.Ended(initial[1], 5000)
	if len(sent) != conc {
		t.Fatalf("the drained warm-up sent %d main requests, want %d", len(sent), conc)
	}
	for i, r := range sent {
		if r.ID != "request_"+strconv.Itoa(i) || r.ArrivalTime != 5000 {
			t.Errorf("main request %d is %s at %d, want request_%d at 5000", i, r.ID, r.ArrivalTime, i)
		}
		if i > 0 && slices.Equal(r.InputTokens[:16], main0.InputTokens[:16]) {
			t.Errorf("main request %d shares request 0's first block", i)
		}
	}
	// A main completion sends the next main request, in order.
	if got := gate.Ended(sent[0], 5007); len(got) != 1 || got[0].ID != "request_"+strconv.Itoa(conc) || got[0].ArrivalTime != 5007 {
		t.Fatalf("a main completion sent %+v, want request_%d at 5007", got, conc)
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
	eng, err := build.StatedRunSettings(m)
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
		if err := checkPrimed(out.RequestRows); err != nil {
			t.Fatalf("%d copies: %v", copies, err)
		}
		o, err := summarize(out.RequestRows, pl, ignore)
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
	cache := kvcache.NewManager(4096, block, 4096*block, true, 0)
	hashesOf := func(in []sim.TokenID) []kvcache.BlockHash {
		return kvcache.BlockHashes(engineTokens(in), block, kvcache.BlockKeys{})
	}
	cachedBlocks := func(in []sim.TokenID) int {
		blocks, _ := cache.ComputedBlocks(hashesOf(in), len(in)+1) // +1: the last token is always recomputed
		return len(blocks[0])
	}
	copyOfZero := r0[0] // warm-up session 0's round 0
	if !cache.Allocate(copyOfZero.ID, kvcache.Alloc{New: len(copyOfZero.InputTokens), Hashes: hashesOf(copyOfZero.InputTokens),
		NumTokens: len(copyOfZero.InputTokens)}) {
		t.Fatal("could not allocate the copy's prompt")
	}
	cache.Free(copyOfZero.ID)
	main := r0[1:] // main requests 0..n-1, in order
	if got, want := cachedBlocks(main[0].InputTokens), int(main[0].InputLen())/block; got != want {
		t.Errorf("main request 0 finds %d cached blocks of its prompt, want all %d full blocks", got, want)
	}
	for i, r := range main[1:] {
		if n := cachedBlocks(r.InputTokens); n != 0 {
			t.Errorf("main request %d finds %d cached blocks; only request 0's prompt was cached", i+1, n)
		}
	}
}

// checkPrimed refuses each way a primed run could fail to be "the main phase after a drained
// warm-up": a copy that did not complete, a main request sent before the last copy ended, or a row
// in neither phase.
func TestCheckPrimedRefusesAnUndrainedWarmup(t *testing.T) {
	const T = int64(10_000_000) // the clock the warm-up drained at, microseconds, as the rows state them
	e2e := func(v int64) *int64 { return &v }
	good := []result.RequestRow{
		{ID: "request_2", Phase: primeWarmupPhase, Fate: result.FateCompleted, ArrivalUs: 0, E2EUs: e2e(T)}, // the copy
		{ID: "request_0", Phase: primeMainPhase, Fate: result.FateCompleted, ArrivalUs: T, E2EUs: e2e(3_000_000)},
		{ID: "request_1", Phase: primeMainPhase, Fate: result.FateCompleted, ArrivalUs: T + 10, E2EUs: e2e(4_000_000)},
	}
	if err := checkPrimed(good); err != nil {
		t.Fatalf("a drained warm-up is refused: %v", err)
	}
	for name, mutate := range map[string]func([]result.RequestRow){
		"a copy still running when the main phase is sent": func(r []result.RequestRow) { r[0].E2EUs = e2e(T + 1) },
		"a copy that did not complete":                     func(r []result.RequestRow) { r[0].Fate = result.FateTimedOut },
		"a row in no phase of the trace":                   func(r []result.RequestRow) { r[1].Phase = "" },
	} {
		r := append([]result.RequestRow(nil), good...)
		mutate(r)
		if checkPrimed(r) == nil {
			t.Errorf("checkPrimed accepted %s", name)
		}
	}
}

// engineTokens is a host token sequence as the engine's KV cache hashes it.
func engineTokens(in []sim.TokenID) []int64 {
	out := make([]int64, len(in))
	for i, t := range in {
		out[i] = int64(t)
	}
	return out
}
