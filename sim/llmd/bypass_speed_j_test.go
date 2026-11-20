package llmd

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// A request skips the router only when every input of the router's decision is at its default: flipping any one of them turns the
// bypass off.
func TestBypassNeedsEveryInputAtItsDefault(t *testing.T) {
	ok := bypassInputs{Enabled: true, DefaultRouter: true, OneEndpoint: true, EndpointReady: true}
	if !ok.canBypass() {
		t.Fatal("a plain request to the only ready endpoint of the default router must bypass")
	}
	v := reflect.ValueOf(&ok).Elem()
	for i := 0; i < v.NumField(); i++ {
		in := ok
		f := reflect.ValueOf(&in).Elem().Field(i)
		f.SetBool(!f.Bool())
		if in.canBypass() {
			t.Errorf("bypass stays on when %s is %v", v.Type().Field(i).Name, f.Bool())
		}
	}
}

func runBypass(t *testing.T, instances int, skip bool, mutate func(*sim.Request)) (*Gateway, []*sim.Request, string) {
	t.Helper()
	reqs := testRequests(80, 5, 700, 16)
	for _, r := range reqs {
		if mutate != nil {
			mutate(r)
		}
	}
	cs := newCluster(testDeployment(instances), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", SkipRouterForOneEndpoint: skip})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	m, _ := json.Marshal(cs.AggregatedMetrics())
	type rec struct {
		ID, Inst        string
		Arr, First, End int64
		Gen             int
	}
	var recs []rec
	for _, r := range reqs {
		recs = append(recs, rec{r.ID, r.AssignedInstance, r.ArrivalTime, r.FirstTokenTime, r.DepartureTime, r.GeneratedTokens})
	}
	b, _ := json.Marshal(recs)
	return gw, reqs, string(m) + string(b)
}

// With one endpoint the bypass is taken, and the run is the one the router gives: same placements, same times, same metrics.
func TestBypassRunEqualsRouterRun(t *testing.T) {
	_, _, want := runBypass(t, 1, false, nil)
	gw, _, got := runBypass(t, 1, true, nil)
	if gw.bypassed == 0 {
		t.Fatal("the bypass was never taken with one endpoint")
	}
	if got != want {
		t.Fatal("the run with the bypass differs from the run through the router")
	}
	if gw.LiveCoroutines() != 0 || gw.flightsDispatched != gw.flightsEnded {
		t.Errorf("coroutines %d, dispatched %d, ended %d", gw.LiveCoroutines(), gw.flightsDispatched, gw.flightsEnded)
	}
}

// Several endpoints, an objective on the request, or a client deadline that passed: the router decides.
func TestBypassNotTakenWhenTheRouterDecides(t *testing.T) {
	if gw, _, _ := runBypass(t, 3, true, nil); gw.bypassed != 0 {
		t.Errorf("%d requests bypassed the router in a pool of three endpoints", gw.bypassed)
	}
	if gw, _, _ := runBypass(t, 1, true, func(r *sim.Request) { r.SLOClass = "critical" }); gw.bypassed != 0 {
		t.Errorf("%d requests with an objective bypassed the router", gw.bypassed)
	}
	if gw, _, _ := runBypass(t, 1, true, func(r *sim.Request) { r.TenantID = "t1" }); gw.bypassed != 0 {
		t.Errorf("%d requests with a tenant bypassed the router", gw.bypassed)
	}
}

// With a deadline on each request (every real run has one) the bypassed run is still the routed run.
func TestBypassRunEqualsRouterRunWithDeadlines(t *testing.T) {
	dl := func(r *sim.Request) { r.Deadline = r.ArrivalTime + 40_000 }
	_, _, want := runBypass(t, 1, false, dl)
	gw, _, got := runBypass(t, 1, true, dl)
	if gw.bypassed == 0 || got != want {
		t.Fatalf("bypassed %d; equal %v", gw.bypassed, got == want)
	}
}
