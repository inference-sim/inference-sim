package testutil

import "fmt"

// FakeLatency is the arithmetic of the deterministic fake latency model that sim/
// and sim/cluster/ tests use in place of a real pricing backend. Tests there exercise
// simulation behavior (scheduling, KV, routing, PD, admission, conservation), not
// pricing, so they need a step time that is cheap, deterministic, monotone in work and
// identical on every platform -- nothing more.
//
// The model is integer-only (no float, so no FMA or rounding drift across
// architectures) and stateless: every method is a pure function of its arguments, so
// one value may be shared across instances and goroutines.
//
// Step time for a batch:
//
//	BaseTicks
//	  + PerScheduledTokenTicks    * Σ NumNewTokens
//	  + PerDecodeRequestTicks     * #(requests with ProgressIndex >= InputLen)
//	  + PerContextTokenMilliTicks * Σ ProgressIndex / 1000
//
// clamped to >= 1 tick (INV-3: a zero step would stall the clock). QueueingTime is
// QueueingTicks + QueueingPerInputTokenTicks * InputLen; the other two hooks are the
// constants OutputTokenProcessingTicks and PostDecodeOverheadTicks.
//
// This package cannot import sim (sim's own internal tests import it), so the
// sim.LatencyModel adapters live in sim/internal/testutil/fakelatency (for every
// package other than sim) and in sim's test files (for package sim).
type FakeLatency struct {
	BaseTicks                  int64 // fixed per-step cost
	PerScheduledTokenTicks     int64 // per token scheduled this step (prefill chunk or decode)
	PerDecodeRequestTicks      int64 // per request in the decode phase
	PerContextTokenMilliTicks  int64 // per KV-resident context token, in 1/1000 tick
	QueueingTicks              int64 // QueueingTime: fixed part, per request
	QueueingPerInputTokenTicks int64 // QueueingTime: per prompt token (default 0, i.e. constant)
	OutputTokenProcessingTicks int64 // OutputTokenProcessingTime
	PostDecodeOverheadTicks    int64 // PostDecodeFixedOverhead
}

// Default fake coefficients. The magnitudes are loosely those the small 8B-shaped test
// model got from a real backend (a ~1 ms step floor, a few µs per scheduled token), so
// tests whose horizons and deadlines were sized for a real backend keep their meaning.
// They are not a model of any hardware and must never be asserted on as such.
const (
	DefaultFakeBaseTicks                  = 1000
	DefaultFakePerScheduledTokenTicks     = 5
	DefaultFakePerDecodeRequestTicks      = 20
	DefaultFakePerContextTokenMilliTicks  = 10
	DefaultFakeQueueingTicks              = 100
	DefaultFakeOutputTokenProcessingTicks = 10
	DefaultFakePostDecodeOverheadTicks    = 50
)

// DefaultFakeLatency returns the fake with the documented default coefficients.
func DefaultFakeLatency() FakeLatency {
	return FakeLatency{
		BaseTicks:                  DefaultFakeBaseTicks,
		PerScheduledTokenTicks:     DefaultFakePerScheduledTokenTicks,
		PerDecodeRequestTicks:      DefaultFakePerDecodeRequestTicks,
		PerContextTokenMilliTicks:  DefaultFakePerContextTokenMilliTicks,
		QueueingTicks:              DefaultFakeQueueingTicks,
		OutputTokenProcessingTicks: DefaultFakeOutputTokenProcessingTicks,
		PostDecodeOverheadTicks:    DefaultFakePostDecodeOverheadTicks,
	}
}

// Validate rejects negative coefficients; a negative term could make the step time
// non-monotone in work.
func (f FakeLatency) Validate() error {
	for name, v := range map[string]int64{
		"BaseTicks":                  f.BaseTicks,
		"PerScheduledTokenTicks":     f.PerScheduledTokenTicks,
		"PerDecodeRequestTicks":      f.PerDecodeRequestTicks,
		"PerContextTokenMilliTicks":  f.PerContextTokenMilliTicks,
		"QueueingTicks":              f.QueueingTicks,
		"QueueingPerInputTokenTicks": f.QueueingPerInputTokenTicks,
		"OutputTokenProcessingTicks": f.OutputTokenProcessingTicks,
		"PostDecodeOverheadTicks":    f.PostDecodeOverheadTicks,
	} {
		if v < 0 {
			return fmt.Errorf("FakeLatency.%s must be >= 0, got %d", name, v)
		}
	}
	return nil
}

// QueueingTicksFor returns the arrival-to-queue delay for a prompt of inputLen tokens.
func (f FakeLatency) QueueingTicksFor(inputLen int64) int64 {
	return f.QueueingTicks + f.QueueingPerInputTokenTicks*inputLen
}

// FakeStepEntry is the part of one scheduled request the fake prices.
type FakeStepEntry struct {
	NumNewTokens  int64 // tokens scheduled this step
	ProgressIndex int64 // tokens already resident in KV
	InputLen      int64 // prompt length; ProgressIndex >= InputLen means decode
}

// StepTicks prices a batch of n entries, read through entry(i). The accessor form lets
// the adapters price a []*sim.Request without allocating.
func (f FakeLatency) StepTicks(n int, entry func(i int) FakeStepEntry) int64 {
	var scheduled, decodes, context int64
	for i := 0; i < n; i++ {
		e := entry(i)
		scheduled += e.NumNewTokens
		context += e.ProgressIndex
		if e.ProgressIndex >= e.InputLen {
			decodes++
		}
	}
	t := f.BaseTicks +
		f.PerScheduledTokenTicks*scheduled +
		f.PerDecodeRequestTicks*decodes +
		f.PerContextTokenMilliTicks*context/1000
	if t < 1 {
		return 1
	}
	return t
}
