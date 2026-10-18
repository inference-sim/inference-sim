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

// With a real idle-loop period (sim.IdlePollEnv) a parked prefill engine wakes at the end of
// the poll period its release falls in, not on the StepTime(empty) grid: the fake model's
// empty step is 1000 us, so a 250 us loop starts the next prefill earlier, by less than one
// grid step, and never before the release.
func TestIdlePark_PollPeriodWakesOnTheNextPollBoundary(t *testing.T) {
	const transfer, period, gridStep = 2_000_357, 250, 1000
	run := func() *ClusterSimulator {
		cfg := holdConfig(true, transfer)
		blocks := int64(15)
		cfg.PrefillOverrides.TotalKVBlocks = &blocks
		reqs := []*sim.Request{holdTestRequest("a", 1, 0, 160, 4), holdTestRequest("b", 2, 1, 160, 4)}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
		mustRun(t, cs)
		return cs
	}
	grid := run()
	sim.SetIdlePollPeriod(period)
	defer sim.SetIdlePollPeriod(-1)
	polled := run()
	gb, pb := parentsByID(grid)["b"], parentsByID(polled)["b"]
	if gain := gb.PrefillCompleteTime - pb.PrefillCompleteTime; gain <= 0 || gain >= gridStep {
		t.Errorf("b's prefill ends %d us earlier than on the grid; want in (0, %d)", gain, gridStep)
	}
	// The step is the same length either way, so its start moved by the same gain: it starts
	// at the poll boundary at or after the release.
	if release := parentsByID(polled)["a"].TransferCompleteTime; pb.PrefillCompleteTime <= release {
		t.Errorf("b prefilled at %d, before a's pull released its KV at %d", pb.PrefillCompleteTime, release)
	}
}
