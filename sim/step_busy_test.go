package sim

import (
	"math"
	"testing"

	"pgregory.net/rapid"
)

// The engine runs one forward pass at a time. A step whose requests all finish empties the
// running batch at the step's START (processCompletions runs inside Step), so before #1934's
// fix an arrival during that step started the next step at the arrival, overlapping the step
// still executing: every P/D prefill sub-request finishes in its one prefill step, so a
// prefill pool ran its steps in parallel and its throughput was unbounded. Property: for
// requests that each finish in one step, no request is scheduled before the previous one
// departs.
func TestStep_NoStepStartsBeforeThePreviousEnds(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 12).Draw(rt, "n")
		cfg := SimConfig{
			Horizon:             math.MaxInt64,
			KVCacheConfig:       NewKVCacheConfig(100000, 16, 0, 0, 0, 0),
			BatchConfig:         NewBatchConfig(1, 100000, 0), // one request per step
			ModelHardwareConfig: NewModelHardwareConfig(testModelConfig(), "m", "H100", 1, 1, false, 0),
		}
		s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), fakeLatencyFor(cfg))
		if err != nil {
			rt.Fatal(err)
		}
		reqs := make([]*Request, n)
		var at int64
		for i := range reqs {
			at += rapid.Int64Range(0, 3000).Draw(rt, "gap")
			reqs[i] = &Request{ID: string(rune('a' + i)), ArrivalTime: at, State: StateQueued,
				InputTokens: make([]TokenID, rapid.IntRange(1, 4000).Draw(rt, "in")), OutputTokens: make([]TokenID, 1), MaxOutputLen: 1}
			for j := range reqs[i].InputTokens {
				reqs[i].InputTokens[j] = TokenID(i*10_000 + j + 1)
			}
		}
		injectRequests(s, reqs)
		s.Run()
		var prevDeparture int64
		for _, r := range reqs {
			if r.State != StateCompleted {
				rt.Fatalf("%s did not complete", r.ID)
			}
			scheduled := r.ArrivalTime + s.Metrics.RequestSchedulingDelays[r.ID]
			if scheduled < prevDeparture {
				rt.Fatalf("%s scheduled at %d, before the previous step ended at %d", r.ID, scheduled, prevDeparture)
			}
			prevDeparture = r.DepartureTime
		}
	})
}
