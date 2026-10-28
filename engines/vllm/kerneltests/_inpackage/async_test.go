// Moved from blis-latency-kernel/async_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// withScheduler builds the fixture with async_scheduling as stated (nil leaves it to the
// engine) and a per-request scheduler cost of perReq.
func withScheduler(t *testing.T, async *bool, spec *engineconfig.Speculative,
	perReq time.Duration) *Kernel {
	t.Helper()
	in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
	e := mustDecode(t, in.Deployment.Pools[0])
	e.AsyncScheduling, e.Speculative = async, spec
	setEngine(t, &in.Deployment.Pools[0], e)
	k := build(t, in)
	k.schedPerRequest = perReq
	return k
}

func ptr(b bool) *bool { return &b }

// Without async scheduling the engine schedules, executes and updates in series
// (vllm/v1/engine/core.py:630-668), so every request's scheduler cost lands on the step.
func TestSyncSchedulingChargesTheSchedulerInSeries(t *testing.T) {
	b := decodeBatch(32, 1, 4000)
	base := withScheduler(t, ptr(false), nil, 0).StepTime(b).NoOverlap
	got := withScheduler(t, ptr(false), nil, 10*time.Microsecond).StepTime(b).NoOverlap - base
	if want := 32 * 10 * time.Microsecond; got < want-time.Microsecond || got > want+time.Microsecond {
		t.Errorf("32 requests at 10 us moved a synchronous step by %v, want %v", got, want)
	}
}

// With async scheduling the next batch is scheduled while this one runs on the device
// (core.py:670-760): a scheduler cost shorter than the step is hidden, and only the part
// longer than the device work shows.
func TestAsyncSchedulingHidesTheSchedulerBehindTheDevice(t *testing.T) {
	b := decodeBatch(32, 1, 4000)
	base := withScheduler(t, ptr(true), nil, 0).StepTime(b).NoOverlap
	if d := withScheduler(t, ptr(true), nil, 10*time.Microsecond).StepTime(b).NoOverlap -
		base; d != 0 {
		t.Errorf("a 320 us scheduler cost behind a %v step moved it by %v, want 0", base, d)
	}
	// A scheduler cost far longer than the step shows by its excess over the step.
	long := withScheduler(t, ptr(true), nil, base).StepTime(b).NoOverlap
	if want := time.Duration(32) * base; long < want-time.Microsecond ||
		long > want+time.Microsecond {
		t.Errorf("a scheduler cost of 32 steps gave a %v step, want max(host, device) = %v",
			long, want)
	}
}

// Left to the engine, async scheduling resolves on unless the draft method is outside
// the set it supports (vllm/config/vllm.py:1576-1635), and requesting it with such a
// method is refused at startup (:1553-1567).
func TestAsyncSchedulingResolvesAsVLLMDoes(t *testing.T) {
	if !withScheduler(t, nil, nil, 0).Resolved().AsyncScheduling {
		t.Error("left to the engine with no drafter, async scheduling must resolve on")
	}
	ngram := &engineconfig.Speculative{Method: "ngram", NumSpecTokens: 3}
	if withScheduler(t, nil, ngram, 0).Resolved().AsyncScheduling {
		t.Error("left to the engine with an n-gram drafter, async scheduling must resolve off")
	}
	if withScheduler(t, ptr(false), nil, 0).Resolved().AsyncScheduling {
		t.Error("a stated false must stand")
	}
	in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
	e := mustDecode(t, in.Deployment.Pools[0])
	e.AsyncScheduling, e.Speculative = ptr(true), ngram
	setEngine(t, &in.Deployment.Pools[0], e)
	if _, err := newKernel(in); err == nil || !strings.Contains(err.Error(), "async_scheduling") {
		t.Errorf("async scheduling requested with an n-gram drafter: got %v, want a refusal", err)
	}
}
