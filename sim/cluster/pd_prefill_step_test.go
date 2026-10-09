package cluster

import (
	"testing"

	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
	"github.com/inference-sim/inference-sim/sim/internal/testutil/fakelatency"
)

// slowPrefill prices every step that still prefills some request extra ticks above its base
// model, and leaves every other step untouched.
type slowPrefill struct {
	sim.LatencyModel
	extra int64
}

func (m slowPrefill) StepTime(batch []*sim.Request) int64 {
	t := m.LatencyModel.StepTime(batch)
	for _, r := range batch {
		if r.ProgressIndex < r.InputLen() {
			return t + m.extra
		}
	}
	return t
}

// pdSingleRequest runs one request through a 1P1D cluster priced by model and returns its
// parent and client-visible TTFT.
func pdSingleRequest(rt *rapid.T, model sim.LatencyModel, inputLen, outputLen int) (*ParentRequest, float64) {
	cfg := newTestDisaggDeploymentConfig(2, 1, 1)
	cfg.LatencyModelOverride = model
	req := &sim.Request{
		ID: "request_0", InputTokens: make([]sim.TokenID, inputLen),
		OutputTokens: make([]sim.TokenID, outputLen), State: sim.StateQueued,
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{req}), nil)
	mustRun(rt, cs)
	for _, p := range cs.ParentRequests() {
		if p.CompletionTime > 0 && p.DecodeInstanceID != "" {
			ttft, ok := cs.AggregatedMetrics().RequestTTFTs[p.ID]
			if !ok {
				rt.Fatalf("parent %s completed but reports no TTFT", p.ID)
			}
			return p, ttft
		}
	}
	rt.Fatalf("the request did not complete through the P/D path")
	return nil, 0
}

// The KV handoff moves the KV the prefill step computes, so it cannot start before that
// step ends, and the client waits for the whole prefill step (#1903). Two laws, for any
// step model and request shape:
//
//   - causality: the handoff starts no earlier than the prefill sub-request leaves its
//     instance, and the parent completes no earlier than the decode sub-request does;
//   - metamorphic: making every prefill step extra ticks slower raises a lone request's
//     TTFT by at least extra. With the handoff starting at the prefill step's START, the
//     prefill step's cost never reached TTFT and the difference was zero.
func TestPD_HandoffWaitsForThePrefillStep(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		base := fakelatency.WithCoeffs(testutil.FakeLatency{
			BaseTicks:                  rapid.Int64Range(1, 5000).Draw(rt, "base"),
			PerScheduledTokenTicks:     rapid.Int64Range(0, 200).Draw(rt, "perToken"),
			PerDecodeRequestTicks:      rapid.Int64Range(0, 200).Draw(rt, "perDecode"),
			OutputTokenProcessingTicks: rapid.Int64Range(0, 200).Draw(rt, "otpt"),
			PostDecodeOverheadTicks:    rapid.Int64Range(0, 500).Draw(rt, "postDecode"),
		})
		inputLen := rapid.IntRange(1, 512).Draw(rt, "inputLen")
		outputLen := rapid.IntRange(1, 8).Draw(rt, "outputLen")
		extra := rapid.Int64Range(1, 10000).Draw(rt, "extra")

		p, ttft := pdSingleRequest(rt, base, inputLen, outputLen)
		if p.TransferStartTime < p.PrefillSubReq.DepartureTime {
			rt.Errorf("handoff starts at %d, before the prefill step ends at %d",
				p.TransferStartTime, p.PrefillSubReq.DepartureTime)
		}
		if p.CompletionTime < p.DecodeSubReq.DepartureTime {
			rt.Errorf("parent completes at %d, before its final decode step ends at %d",
				p.CompletionTime, p.DecodeSubReq.DepartureTime)
		}

		_, slowTTFT := pdSingleRequest(rt, slowPrefill{base, extra}, inputLen, outputLen)
		if slowTTFT-ttft < float64(extra) {
			rt.Errorf("prefill steps %d ticks slower raised TTFT by %.0f (%.0f -> %.0f); "+
				"the client must wait for the whole prefill step", extra, slowTTFT-ttft, ttft, slowTTFT)
		}
	})
}
