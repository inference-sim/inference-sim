package cluster

import (
	"fmt"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// fakeLatency is the deterministic, stateless fake latency model (testutil.FakeLatency) as the
// compatibility engine's LatencyModel, which this package's tests hand the engines they build.
// It was sim/internal/testutil/fakelatency's Model, moved into each test package's files when
// the engine left the host (phase 2c), so no non-test file under sim/ imports blis-vllm.
type fakeLatency struct{ c testutil.FakeLatency }

var _ cengine.LatencyModel = fakeLatency{}

// newFakeLatencyModel returns the fake with testutil's documented default coefficients.
func newFakeLatencyModel() fakeLatency { return fakeLatency{c: testutil.DefaultFakeLatency()} }

// fakeLatencyWithCoeffs returns the fake with caller-chosen coefficients. It panics on a
// negative coefficient (a bad fake is a bug in the test).
func fakeLatencyWithCoeffs(c testutil.FakeLatency) fakeLatency {
	if err := c.Validate(); err != nil {
		panic(fmt.Sprintf("fakeLatencyWithCoeffs: %v", err))
	}
	return fakeLatency{c: c}
}

// StepTime prices the batch; always >= 1.
func (m fakeLatency) StepTime(batch []*cengine.Request) int64 {
	return m.c.StepTicks(len(batch), func(i int) testutil.FakeStepEntry {
		r := batch[i]
		return testutil.FakeStepEntry{
			NumNewTokens:  int64(r.NumNewTokens),
			ProgressIndex: r.ProgressIndex,
			InputLen:      r.InputLen(),
		}
	})
}

// QueueingTime returns the per-request arrival-to-queue delay.
func (m fakeLatency) QueueingTime(r *cengine.Request) int64 {
	return m.c.QueueingTicksFor(r.InputLen())
}

// OutputTokenProcessingTime returns the constant per-token post-processing time.
func (m fakeLatency) OutputTokenProcessingTime() int64 { return m.c.OutputTokenProcessingTicks }

// PostDecodeFixedOverhead returns the constant per-request completion overhead.
func (m fakeLatency) PostDecodeFixedOverhead() int64 { return m.c.PostDecodeOverheadTicks }
