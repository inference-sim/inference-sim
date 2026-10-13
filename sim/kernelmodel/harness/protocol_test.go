package harness

import (
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
	if more := driver.OnComplete(initialOrAdmitted(initial, inFlight[0]), 99999); len(more) != 0 {
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
	out, err := runBlis(cfg, sw.Scenario, pc, pl)
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
	if _, err := summarize(out.Requests, pl); err != nil {
		t.Errorf("summarize: %v", err)
	}
}
