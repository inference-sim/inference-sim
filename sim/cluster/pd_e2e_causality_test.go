package cluster

import (
	"fmt"
	"math"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// Issue #1513: the PD-disaggregated parent (client-visible) E2E under-counted
// the decode sub-request's own step advance. projectPDMetrics reported
// parent.CompletionTime − ArrivalTime, where parent.CompletionTime is stamped on
// the cluster clock at the completion-DETECTION tick and omits the decode step's
// own advance. For short outputs the reported parent E2E fell below a single
// decode step (ITL[0]) — and below the parent TTFT — violating INV-5.
//
// The correct client-visible E2E is the arrival→last-token span. The decode
// sub-request's own per-instance E2E (FirstTokenTime + Σ ITL + PostDecodeOverhead)
// already captures the execution span correctly but is measured from the decode
// SCHEDULE instant (decode sub-requests never set FirstTokenTime because they
// start at ProgressIndex == InputLen). Adding the decode scheduling delay
// (arrival→decode-schedule) reconstitutes the full arrival→completion span:
//
//	parentE2E = decodeSchedulingDelay + decodeOwnE2E
//
// This is the E2E-analog of the #1510/#1512 TTFT fix
// (TTFT = decodeSchedulingDelay + firstDecodeStep), and it guarantees INV-5 by
// construction: decodeOwnE2E ≥ ITL[0] = firstDecodeStep, so E2E ≥ TTFT.

// pdDecodeOwnE2E returns the decode sub-request's own per-instance E2E and
// scheduling delay for the given parent, read from PER-INSTANCE metrics BEFORE
// projection. These are the independent oracle the fix is built from; the tests
// below never recompute the production formula from the same aggregated inputs
// (which would be tautological — the refactor-survival trap of the pre-fix
// E2E==CompletionTime−ArrivalTime assertion).
func pdDecodeOwnE2E(cs *ClusterSimulator, decodeSubReqID string) (e2e float64, schedDelay int64, ok bool) {
	for _, inst := range cs.PerInstanceMetricsByID() {
		if v, present := inst.RequestE2Es[decodeSubReqID]; present {
			return v, inst.RequestSchedulingDelays[decodeSubReqID], true
		}
	}
	return 0, 0, false
}

// runShortOutputPD runs a PD cluster with the given per-request output length and
// returns the aggregated metrics plus the cluster (for parent inspection).
func runShortOutputPD(t *testing.T, outTokens, numReqs int) (*sim.Metrics, *ClusterSimulator) {
	t.Helper()
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := make([]*sim.Request, numReqs)
	for i := 0; i < numReqs; i++ {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			InputTokens:  make([]sim.TokenID, 20),
			OutputTokens: make([]sim.TokenID, outTokens),
			State:        sim.StateQueued,
			ArrivalTime:  int64(i * 100),
		}
	}
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)
	return cs.AggregatedMetrics(), cs
}

// TestPDParentE2E_INV5_ShortOutputs is the regression test for issue #1513: the
// reported parent E2E must never fall below the parent TTFT (INV-5 causality),
// including for 1–3 output-token requests that previously fell inside the
// violation band.
func TestPDParentE2E_INV5_ShortOutputs(t *testing.T) {
	for _, outTokens := range []int{1, 2, 3, 5, 10} {
		outTokens := outTokens
		t.Run(fmt.Sprintf("out=%d", outTokens), func(t *testing.T) {
			m, cs := runShortOutputPD(t, outTokens, 5)
			checked := 0
			for _, parent := range cs.ParentRequests() {
				if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
					continue
				}
				pid := parent.ID
				ttft, hasTTFT := m.RequestTTFTs[pid]
				e2e, hasE2E := m.RequestE2Es[pid]
				if !hasTTFT || !hasE2E {
					t.Fatalf("parent %s: missing TTFT (%v) or E2E (%v) for completed parent", pid, hasTTFT, hasE2E)
				}
				// INV-5: arrival→first-token ≤ arrival→last-token.
				if ttft > e2e {
					t.Errorf("parent %s: TTFT (%.1f) > E2E (%.1f) — INV-5 violated (E2E under-counts decode step)",
						pid, ttft, e2e)
				}
				checked++
			}
			if checked == 0 {
				t.Fatal("no completed parents checked — config or reproduction drifted")
			}
		})
	}
}

// TestPDParentE2E_GeqSingleDecodeStep asserts the parent E2E is at least one
// decode step (ITL[0]). The pre-fix bug reported an E2E (152) smaller than a
// single decode step (ITL[0]=300) — a physically impossible client-visible
// latency for a request that emitted at least one token.
func TestPDParentE2E_GeqSingleDecodeStep(t *testing.T) {
	m, cs := runShortOutputPD(t, 1, 5)
	checked := 0
	for _, parent := range cs.ParentRequests() {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" || parent.DecodeSubReq == nil {
			continue
		}
		if len(parent.DecodeSubReq.ITL) == 0 {
			continue
		}
		pid := parent.ID
		e2e := m.RequestE2Es[pid]
		firstStep := float64(parent.DecodeSubReq.ITL[0])
		if e2e < firstStep {
			t.Errorf("parent %s: E2E (%.1f) < one decode step ITL[0] (%.1f) — client-visible latency cannot be below a single step",
				pid, e2e, firstStep)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no completed parents checked — config or reproduction drifted")
	}
}

// TestPDParentE2E_GeqDecodeOwnE2E asserts the parent E2E is at least the decode
// sub-request's own recorded per-instance E2E. The pre-fix bug DISCARDED the
// decode sub-request's correct own E2E (301) in favor of a
// parent.CompletionTime-based value (152) that omits the step advance. The
// parent (client-visible) span spans arrival→completion and therefore contains
// the decode sub-request's own execution span as a sub-interval.
func TestPDParentE2E_GeqDecodeOwnE2E(t *testing.T) {
	for _, outTokens := range []int{1, 2, 3, 10} {
		outTokens := outTokens
		t.Run(fmt.Sprintf("out=%d", outTokens), func(t *testing.T) {
			m, cs := runShortOutputPD(t, outTokens, 3)
			checked := 0
			for _, parent := range cs.ParentRequests() {
				if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
					continue
				}
				decodeOwnE2E, _, ok := pdDecodeOwnE2E(cs, parent.DecodeSubReqID)
				if !ok {
					continue
				}
				pid := parent.ID
				e2e := m.RequestE2Es[pid]
				if e2e < decodeOwnE2E {
					t.Errorf("parent %s: parent E2E (%.1f) < decode sub-request own E2E (%.1f) — parent must not under-count the decode execution span",
						pid, e2e, decodeOwnE2E)
				}
				checked++
			}
			if checked == 0 {
				t.Fatal("no completed parents checked — config or reproduction drifted")
			}
		})
	}
}

// TestPDParentE2E_ReconstructsArrivalToCompletionSpan verifies the E2E value
// against an INDEPENDENT oracle: the arrival→decode-schedule delay plus the
// decode sub-request's own execution E2E. Both operands are read from
// per-instance metrics (RequestSchedulingDelays / RequestE2Es of the decode
// sub-request), a different mechanism than the aggregated parent E2E the fix
// writes. This is the E2E-analog of the #1512 TTFT reconstruction guard.
func TestPDParentE2E_ReconstructsArrivalToCompletionSpan(t *testing.T) {
	for _, outTokens := range []int{1, 3, 10} {
		outTokens := outTokens
		t.Run(fmt.Sprintf("out=%d", outTokens), func(t *testing.T) {
			m, cs := runShortOutputPD(t, outTokens, 3)
			checked := 0
			for _, parent := range cs.ParentRequests() {
				if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
					continue
				}
				decodeOwnE2E, decodeDelay, ok := pdDecodeOwnE2E(cs, parent.DecodeSubReqID)
				if !ok {
					continue
				}
				pid := parent.ID
				e2e := m.RequestE2Es[pid]
				want := float64(decodeDelay) + decodeOwnE2E
				if math.Abs(e2e-want) > 1e-9 {
					t.Errorf("parent %s: E2E = %.1f, want %.1f (decodeSchedulingDelay %d + decodeOwnE2E %.1f)",
						pid, e2e, want, decodeDelay, decodeOwnE2E)
				}
				checked++
			}
			if checked == 0 {
				t.Fatal("no completed parents checked — config or reproduction drifted")
			}
		})
	}
}

// TestPDParentE2E_CompletionTimeMetricConsistency verifies that the projected
// aggregated RequestCompletionTimes metric satisfies the non-PD identity
// completion_metric == ArrivalTime + E2E. The pre-fix code left
// RequestCompletionTimes[pid] = parent.CompletionTime (cluster-clock,
// under-counted), so completion_metric − ArrivalTime disagreed with the E2E
// metric. This consistency keeps the session-duration metric
// (metrics.go: computeSessionMetrics reads RequestCompletionTimes) from
// inheriting the same under-count.
//
// NOTE: this asserts consistency of the METRIC, not the lifecycle field
// parent.CompletionTime, which is intentionally left untouched (it drives
// session follow-up scheduling; INV-10).
func TestPDParentE2E_CompletionTimeMetricConsistency(t *testing.T) {
	m, cs := runShortOutputPD(t, 1, 5)
	checked := 0
	for _, parent := range cs.ParentRequests() {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		pid := parent.ID
		e2e, hasE2E := m.RequestE2Es[pid]
		ct, hasCT := m.RequestCompletionTimes[pid]
		if !hasE2E || !hasCT {
			t.Fatalf("parent %s: missing E2E (%v) or completion-time metric (%v)", pid, hasE2E, hasCT)
		}
		want := float64(parent.ArrivalTime) + e2e
		if math.Abs(ct-want) > 1e-9 {
			t.Errorf("parent %s: completion-time metric = %.1f, want %.1f (ArrivalTime %d + E2E %.1f)",
				pid, ct, want, parent.ArrivalTime, e2e)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no completed parents checked — config or reproduction drifted")
	}
}

// newTestColocatedTrainedPhysicsConfig builds a single-instance (non-PD)
// deployment whose latency parameters MATCH newTestDisaggDeploymentConfig
// (same betas/alphas/model/hardware). It is the parity baseline: a PD request
// must not report a SMALLER client-visible E2E than the identical request served
// co-located, because PD adds a real KV-transfer cost on top of the same
// prefill+decode work.
func newTestColocatedTrainedPhysicsConfig() DeploymentConfig {
	modelCfg := sim.ModelConfig{NumLayers: 2, NumHeads: 4, HiddenDim: 64, IntermediateDim: 128, BytesPerParam: 2.0}
	hwCfg := sim.HardwareCalib{TFlopsPeak: 1.0, BwPeakTBs: 0.001}
	betas := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0}
	alphas := []float64{100, 1, 100}
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betas, alphas),
			ModelHardwareConfig: sim.NewModelHardwareConfig(modelCfg, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 0),
		},
		NumInstances:  1,
		RoutingPolicy: "round-robin",
	}
}

// TestPDParentE2E_GeqNonPDBaseline_OneToken asserts PD/non-PD E2E parity for the
// 1-output-token request the issue cites directly. Before the fix the PD path
// reported a SMALLER E2E than non-PD (152 vs 401), so the aggregated PD E2E
// distribution was under-reported independently of TTFT; the fix must make PD ≥
// non-PD, with the surplus being exactly the KV-transfer cost PD adds on top of
// the same prefill+decode work.
//
// Scoped to 1 token deliberately: for multi-token outputs the PD decode
// sub-request emits N−1 decode steps (the first token is produced during prefill
// and carried across the KV transfer), while the co-located path folds all N
// tokens' work into one instance-local timeline. That step-count difference is a
// separate modeling choice (issue #1510/#1511 territory) orthogonal to the E2E
// under-count fixed here, so an absolute multi-token PD-vs-non-PD comparison
// would confound the two. The out=1 case has no such confound (0 extra decode
// steps either way), giving a clean, exact decomposition.
func TestPDParentE2E_GeqNonPDBaseline_OneToken(t *testing.T) {
	// PD run (single request so there is no queueing skew vs the baseline).
	mPD, csPD := runShortOutputPD(t, 1, 1)
	var pdE2E, transferCost float64
	var found bool
	for _, parent := range csPD.ParentRequests() {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		pdE2E = mPD.RequestE2Es[parent.ID]
		transferCost = float64(parent.TransferCompleteTime - parent.TransferStartTime)
		found = true
	}
	if !found {
		t.Fatal("PD run produced no completed parent")
	}

	// Non-PD baseline: identical single request, co-located instance.
	nreq := &sim.Request{
		ID: "request_0", InputTokens: make([]sim.TokenID, 20),
		OutputTokens: make([]sim.TokenID, 1), State: sim.StateQueued, ArrivalTime: 0,
	}
	ncs := NewClusterSimulator(newTestColocatedTrainedPhysicsConfig(), NewSliceRequestSource([]*sim.Request{nreq}), nil)
	mustRun(t, ncs)
	nonPDE2E, ok := ncs.AggregatedMetrics().RequestE2Es["request_0"]
	if !ok {
		t.Fatal("non-PD baseline produced no E2E")
	}

	// Law 1: PD must not under-report vs co-located.
	if pdE2E < nonPDE2E {
		t.Errorf("PD E2E (%.1f) < non-PD baseline E2E (%.1f) — PD must not under-report vs co-located (it adds KV-transfer cost)",
			pdE2E, nonPDE2E)
	}
	// Law 2: the entire surplus is the KV-transfer cost (the only extra work PD does
	// for a 1-token request). Independent oracle: transferCost is read from parent
	// phase timestamps, not from either E2E value.
	if transferCost <= 0 {
		t.Fatalf("expected a positive KV-transfer cost, got %.1f", transferCost)
	}
	if diff := pdE2E - nonPDE2E; math.Abs(diff-transferCost) > 1e-9 {
		t.Errorf("PD − non-PD E2E surplus = %.1f, want %.1f (KV-transfer cost); PD=%.1f nonPD=%.1f",
			diff, transferCost, pdE2E, nonPDE2E)
	}
}
