package fakelatency

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

func genCoeffs(t *rapid.T) testutil.FakeLatency {
	return testutil.FakeLatency{
		BaseTicks:                  rapid.Int64Range(0, 10_000).Draw(t, "base"),
		PerScheduledTokenTicks:     rapid.Int64Range(0, 100).Draw(t, "perTok"),
		PerDecodeRequestTicks:      rapid.Int64Range(0, 100).Draw(t, "perDec"),
		PerContextTokenMilliTicks:  rapid.Int64Range(0, 1000).Draw(t, "perCtx"),
		QueueingTicks:              rapid.Int64Range(0, 1000).Draw(t, "q"),
		OutputTokenProcessingTicks: rapid.Int64Range(0, 1000).Draw(t, "otp"),
		PostDecodeOverheadTicks:    rapid.Int64Range(0, 1000).Draw(t, "pdo"),
	}
}

func genRequest(t *rapid.T, label string) *sim.Request {
	inputLen := rapid.IntRange(1, 2048).Draw(t, label+"_in")
	progress := rapid.Int64Range(0, int64(inputLen)+512).Draw(t, label+"_pi")
	return &sim.Request{
		InputTokens:   make([]sim.TokenID, inputLen),
		ProgressIndex: progress,
		NumNewTokens:  rapid.IntRange(0, 2048).Draw(t, label+"_new"),
	}
}

func genBatch(t *rapid.T) []*sim.Request {
	n := rapid.IntRange(0, 16).Draw(t, "n")
	b := make([]*sim.Request, n)
	for i := range b {
		b[i] = genRequest(t, "r")
	}
	return b
}

// Law: StepTime >= 1 for every batch and every non-negative coefficient set,
// including the all-zero fake and the empty batch (INV-3).
func TestFake_StepTimeAtLeastOneTick(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		m := WithCoeffs(genCoeffs(t))
		if got := m.StepTime(genBatch(t)); got < 1 {
			t.Fatalf("StepTime = %d, want >= 1", got)
		}
	})
	if got := WithCoeffs(testutil.FakeLatency{}).StepTime(nil); got != 1 {
		t.Fatalf("all-zero fake, empty batch: StepTime = %d, want 1", got)
	}
}

// Law: adding a request, or more scheduled tokens / context to a request, never makes
// a step cheaper.
func TestFake_StepTimeMonotoneInWork(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		m := WithCoeffs(genCoeffs(t))
		batch := genBatch(t)
		base := m.StepTime(batch)

		if more := m.StepTime(append(append([]*sim.Request{}, batch...), genRequest(t, "extra"))); more < base {
			t.Fatalf("adding a request lowered StepTime: %d -> %d", base, more)
		}
		if len(batch) == 0 {
			return
		}
		i := rapid.IntRange(0, len(batch)-1).Draw(t, "i")
		grown := *batch[i]
		grown.NumNewTokens += rapid.IntRange(0, 512).Draw(t, "dNew")
		grown.ProgressIndex += rapid.Int64Range(0, 512).Draw(t, "dPI")
		b2 := append([]*sim.Request{}, batch...)
		b2[i] = &grown
		if got := m.StepTime(b2); got < base {
			t.Fatalf("growing request %d lowered StepTime: %d -> %d", i, base, got)
		}
	})
}

// Law: the fake is stateless -- StepTime depends only on the multiset of requests
// (order-independent) and repeated or interleaved calls give the same answer.
func TestFake_StatelessAndOrderIndependent(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		m := WithCoeffs(genCoeffs(t))
		batch := genBatch(t)
		first := m.StepTime(batch)
		_ = m.StepTime(genBatch(t)) // an unrelated call in between must not matter
		perm := rapid.Permutation(batch).Draw(t, "perm")
		if got := m.StepTime(perm); got != first {
			t.Fatalf("permuted batch: StepTime = %d, want %d", got, first)
		}
		if got := m.StepTime(batch); got != first {
			t.Fatalf("repeated call: StepTime = %d, want %d", got, first)
		}
	})
}

// The constant terms come straight from the coefficients; the defaults are positive.
func TestFake_ConstantTerms(t *testing.T) {
	m := New()
	c := testutil.DefaultFakeLatency()
	if m.QueueingTime(&sim.Request{}) != c.QueueingTicks || c.QueueingTicks <= 0 {
		t.Errorf("QueueingTime = %d, want default %d > 0", m.QueueingTime(&sim.Request{}), c.QueueingTicks)
	}
	if m.OutputTokenProcessingTime() != c.OutputTokenProcessingTicks || c.OutputTokenProcessingTicks <= 0 {
		t.Errorf("OutputTokenProcessingTime = %d, want default %d > 0", m.OutputTokenProcessingTime(), c.OutputTokenProcessingTicks)
	}
	if m.PostDecodeFixedOverhead() != c.PostDecodeOverheadTicks || c.PostDecodeOverheadTicks <= 0 {
		t.Errorf("PostDecodeFixedOverhead = %d, want default %d > 0", m.PostDecodeFixedOverhead(), c.PostDecodeOverheadTicks)
	}
}

func TestWithCoeffs_RejectsNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("WithCoeffs with a negative coefficient did not panic")
		}
	}()
	WithCoeffs(testutil.FakeLatency{BaseTicks: -1})
}
