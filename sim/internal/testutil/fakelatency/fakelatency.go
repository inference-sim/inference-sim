// Package fakelatency provides a deterministic fake sim.LatencyModel for tests that
// exercise simulation behavior rather than pricing. The arithmetic lives in
// testutil.FakeLatency (which cannot import sim); this package adapts it to
// sim.LatencyModel for every package other than sim itself.
package fakelatency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// Model is a stateless, deterministic sim.LatencyModel. See testutil.FakeLatency for
// the step-time formula. A Model value is safe to share across instances.
type Model struct {
	c testutil.FakeLatency
}

var _ sim.LatencyModel = Model{}

// New returns a Model with the documented default coefficients
// (testutil.DefaultFakeLatency).
func New() Model { return Model{c: testutil.DefaultFakeLatency()} }

// WithCoeffs returns a Model with caller-chosen coefficients. It panics on a negative
// coefficient (test-only code; a bad fake is a bug in the test).
func WithCoeffs(c testutil.FakeLatency) Model {
	if err := c.Validate(); err != nil {
		panic(fmt.Sprintf("fakelatency.WithCoeffs: %v", err))
	}
	return Model{c: c}
}

// Coeffs returns the model's coefficients.
func (m Model) Coeffs() testutil.FakeLatency { return m.c }

// StepTime prices the batch; always >= 1.
func (m Model) StepTime(batch []*sim.Request) int64 {
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
func (m Model) QueueingTime(r *sim.Request) int64 { return m.c.QueueingTicksFor(r.InputLen()) }

// OutputTokenProcessingTime returns the constant per-token post-processing time.
func (m Model) OutputTokenProcessingTime() int64 { return m.c.OutputTokenProcessingTicks }

// PostDecodeFixedOverhead returns the constant per-request completion overhead.
func (m Model) PostDecodeFixedOverhead() int64 { return m.c.PostDecodeOverheadTicks }
