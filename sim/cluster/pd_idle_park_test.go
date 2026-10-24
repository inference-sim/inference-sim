package cluster

import (
	"fmt"
	"reflect"
	"testing"

	"pgregory.net/rapid"

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

// parkedAndPolled runs one saturated P/D draw (saturatedPD's fixture, without its per-step
// checks) twice: with idle parking and with the polling it must reproduce.
func parkedAndPolled(prefills, n, in, out int, decodeBlocks int64) (parked, polled *ClusterSimulator) {
	run := func(polling bool) *ClusterSimulator {
		cfg := newTestDisaggDeploymentConfig(prefills+1, prefills, 1)
		cfg.PDPrefillKVHold = true
		cfg.DecodeOverrides.TotalKVBlocks = &decodeBlocks
		reqs := make([]*sim.Request, n)
		for i := range reqs {
			reqs[i] = &sim.Request{ID: fmt.Sprintf("r%02d", i), InputTokens: make([]sim.TokenID, in),
				OutputTokens: make([]sim.TokenID, out), State: sim.StateQueued}
			for k := range reqs[i].InputTokens {
				reqs[i].InputTokens[k] = sim.TokenID(i*100000 + k) // distinct prompts: no prefix sharing
			}
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
		for _, inst := range cs.Instances() {
			inst.sim.SetIdlePolling(polling)
		}
		if err := cs.Run(); err != nil {
			panic(err)
		}
		return cs
	}
	return run(false), run(true)
}

// samePDRun reports how a parked run differs from the polled one: its handoff timings or an
// instance's metrics. Empty when they are the same run.
func samePDRun(parked, polled *ClusterSimulator) string {
	timings := func(cs *ClusterSimulator) map[string][4]int64 {
		m := map[string][4]int64{}
		for _, p := range cs.ParentRequests() {
			m[p.ID] = [4]int64{p.PrefillCompleteTime, p.TransferStartTime, p.TransferCompleteTime, p.CompletionTime}
		}
		return m
	}
	if a, b := timings(parked), timings(polled); !reflect.DeepEqual(a, b) {
		return fmt.Sprintf("handoff timings differ:\nparked  %v\npolling %v", a, b)
	}
	for i := range polled.Instances() {
		if a, b := parked.Instances()[i].Metrics(), polled.Instances()[i].Metrics(); !reflect.DeepEqual(a, b) {
			return fmt.Sprintf("instance %d metrics differ:\nparked  %+v\npolling %+v", i, *a, *b)
		}
	}
	return ""
}

// A decode step that preempts its only running request, to fit the request's next block
// beside the blocks its handoffs hold, schedules nothing: vLLM skips the WAITING phase in a
// step that preempted (vllm/v1/core/sched/scheduler.py:900@5df63a72). The next poll does run
// it, with the victim's blocks free, so the engine must not park on that empty step: no
// external event would wake it, and the run would end with every request stuck. This is the
// draw rapid found for TestPDDecodeAdmission_LawsHoldUnderRandomSaturation (3P1D, 5 requests
// of 269+20 tokens, 34 decode blocks), where the parked run completed none.
func TestIdlePark_AStepThatPreemptedDoesNotPark(t *testing.T) {
	parked, polled := parkedAndPolled(3, 5, 269, 20, 34)
	if got := polled.AggregatedMetrics().CompletedRequests; got != 5 {
		t.Fatalf("the polling reference completed %d of 5", got)
	}
	if decodePreemptions(polled) == 0 {
		t.Fatalf("the polling reference never preempted on the decode side: the fixture must")
	}
	if got := parked.AggregatedMetrics().CompletedRequests; got != 5 {
		t.Fatalf("parked run completed %d of 5 (polling completes all): the engine parked on a step that preempted", got)
	}
	if d := samePDRun(parked, polled); d != "" {
		t.Error(d)
	}
}

// The idle park's contract (sim/idle_park.go): a parked engine runs the run polling gives,
// with the same timings and metrics. Checked over random saturated P/D runs, the setting
// where decode engines both park and preempt.
func TestIdlePark_ParkedRunIsThePolledRun(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		prefills := rapid.IntRange(1, 3).Draw(rt, "prefills")
		n := rapid.IntRange(4, 40).Draw(rt, "requests")
		in := rapid.IntRange(16, 400).Draw(rt, "in")
		out := rapid.IntRange(2, 60).Draw(rt, "out")
		need := int64((in+out+15)/16) + 2
		blocks := rapid.Int64Range(need, 4*need).Draw(rt, "decodeBlocks")
		parked, polled := parkedAndPolled(prefills, n, in, out, blocks)
		if d := samePDRun(parked, polled); d != "" {
			rt.Fatalf("%s", d)
		}
	})
}
