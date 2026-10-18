package cluster

import (
	"reflect"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// A prefill engine whose KV is all held for pulls to a slow decode side can admit nothing
// until a pull completes. It must wait for that release rather than poll an empty step per
// StepTime(empty), and the run must be the one polling gives: same timings, same metrics.
func TestIdlePark_HeldKVPrefillWaitsForTheReleaseNotByPolling(t *testing.T) {
	const transfer = 2_000_000 // 2 s pulls, ~10^5 empty polls per pull
	run := func(polling bool) (*ClusterSimulator, int64) {
		cfg := holdConfig(true, transfer)
		blocks := int64(15) // one 10-block request fits, two do not
		cfg.PrefillOverrides.TotalKVBlocks = &blocks
		reqs := []*sim.Request{}
		for i, id := range []string{"a", "b", "c", "d"} {
			reqs = append(reqs, holdTestRequest(id, i+1, int64(i), 160, 4))
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
		for _, inst := range cs.Instances() {
			inst.sim.SetIdlePolling(polling)
		}
		mustRun(t, cs)
		var events int64
		for _, inst := range cs.Instances() {
			events += inst.sim.EventsProcessed()
		}
		return cs, events
	}
	polled, pollEvents := run(true)
	parked, parkEvents := run(false)

	t.Logf("events: polling %d, parked %d", pollEvents, parkEvents)
	if parkEvents > 400 {
		t.Errorf("parked run took %d events (polling %d); want at most 400", parkEvents, pollEvents)
	}
	if pollEvents < 10*parkEvents {
		t.Fatalf("the polling reference took only %d events vs %d: the config does not stall", pollEvents, parkEvents)
	}
	timings := func(cs *ClusterSimulator) map[string][4]int64 {
		m := map[string][4]int64{}
		for _, p := range cs.ParentRequests() {
			m[p.ID] = [4]int64{p.PrefillCompleteTime, p.TransferStartTime, p.TransferCompleteTime, p.CompletionTime}
		}
		return m
	}
	if a, b := timings(polled), timings(parked); !reflect.DeepEqual(a, b) {
		t.Errorf("handoff timings differ:\npolling %v\nparked  %v", a, b)
	}
	for i := range polled.Instances() {
		a, b := polled.Instances()[i].Metrics(), parked.Instances()[i].Metrics()
		if !reflect.DeepEqual(a, b) {
			t.Errorf("instance %d metrics differ:\npolling %+v\nparked  %+v", i, *a, *b)
		}
	}
}
