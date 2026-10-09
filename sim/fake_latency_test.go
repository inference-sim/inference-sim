package sim

import "github.com/inference-sim/inference-sim/sim/internal/testutil"

// fakeLatencyModel adapts testutil.FakeLatency -- the deterministic, stateless fake
// every sim/ and sim/cluster/ behavior test prices steps with -- to LatencyModel for
// package sim's internal tests. (Other packages use sim/internal/testutil/fakelatency,
// which package sim cannot import without a cycle.) The two adapters share the
// arithmetic in testutil, so they price every batch identically.
type fakeLatencyModel struct {
	c testutil.FakeLatency
}

// newFakeLatency returns the fake with testutil's documented default coefficients.
func newFakeLatency() LatencyModel {
	return fakeLatencyModel{c: testutil.DefaultFakeLatency()}
}

// newFakeLatencyWith returns the fake with caller-chosen coefficients. It panics on a
// negative coefficient.
func newFakeLatencyWith(c testutil.FakeLatency) LatencyModel {
	if err := c.Validate(); err != nil {
		panic("newFakeLatencyWith: " + err.Error())
	}
	return fakeLatencyModel{c: c}
}

// fakeZeroQueueing is the default fake with no arrival-to-queue delay, for tests that
// need a request to enter the wait queue at its arrival tick.
func fakeZeroQueueing() LatencyModel {
	c := testutil.DefaultFakeLatency()
	c.QueueingTicks = 0
	return newFakeLatencyWith(c)
}

func (m fakeLatencyModel) StepTime(batch []*Request) int64 {
	return m.c.StepTicks(len(batch), func(i int) testutil.FakeStepEntry {
		r := batch[i]
		return testutil.FakeStepEntry{
			NumNewTokens:  int64(r.NumNewTokens),
			ProgressIndex: r.ProgressIndex,
			InputLen:      r.InputLen(),
		}
	})
}

func (m fakeLatencyModel) QueueingTime(r *Request) int64    { return m.c.QueueingTicksFor(r.InputLen()) }
func (m fakeLatencyModel) OutputTokenProcessingTime() int64 { return m.c.OutputTokenProcessingTicks }
func (m fakeLatencyModel) PostDecodeFixedOverhead() int64   { return m.c.PostDecodeOverheadTicks }

// fakeLatencyFor returns cfg.LatencyModelOverride when a test set one (package sim
// never reads that field itself; the tests use it to pick a fake variant inline in the
// config literal), else the default fake.
func fakeLatencyFor(cfg SimConfig) LatencyModel {
	if cfg.LatencyModelOverride != nil {
		return cfg.LatencyModelOverride
	}
	return newFakeLatency()
}

// fakeTokenCost prices a step at exactly one tick per scheduled token (prefill chunk
// or decode), with no fixed, queueing or completion overhead.
func fakeTokenCost() LatencyModel {
	return newFakeLatencyWith(testutil.FakeLatency{PerScheduledTokenTicks: 1})
}

// fakeFixedStep prices every step at exactly ticks, with no queueing or completion
// overhead.
func fakeFixedStep(ticks int64) LatencyModel {
	return newFakeLatencyWith(testutil.FakeLatency{BaseTicks: ticks})
}
