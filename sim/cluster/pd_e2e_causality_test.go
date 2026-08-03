package cluster

import (
	"fmt"
	"math"
	"reflect"
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
// scheduling delay for the given parent, read from the per-instance metric maps
// (`PerInstanceMetricsByID()`) rather than the projected parent-level aggregate.
// This checks the PROJECTION logic (that projectPDMetrics recombines the
// sub-request values into the parent E2E correctly) without re-deriving the
// production formula from the aggregate itself — that would be the refactor-
// survival trap of the pre-fix `E2E == CompletionTime − ArrivalTime` assertion.
//
// Caveat: the per-instance maps are the same accumulation path that feeds
// `c.aggregatedMetrics`, so this reconstruction cannot catch a bug in how a
// decode sub-request's own E2E is accumulated at the instance level. The
// genuinely independent laws — which validate the E2E value against a separate
// physical quantity — are `TestPDParentE2E_INV5_ShortOutputs` (E2E ≥ TTFT) and
// `TestPDParentE2E_GeqNonPDBaseline_OneToken` (PD surplus ≥ KV-transfer cost,
// with the transfer cost read from parent phase timestamps).
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
// against the arrival→decode-schedule delay plus the decode sub-request's own
// execution E2E, both read from the per-instance metric maps (a different map
// reference than the projected parent aggregate). This checks the projection
// recombination, not the instance-level accumulation (see pdDecodeOwnE2E for the
// caveat, and the INV-5 / non-PD-baseline tests for the physically-independent
// laws). It is the E2E-analog of the #1512 TTFT reconstruction guard.
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
	// Guard the parity premise: the PD and co-located configs must use identical
	// latency parameters, otherwise a PD-vs-non-PD E2E comparison is meaningless.
	// This enforces the "same betas/alphas/model/hardware" claim in
	// newTestColocatedTrainedPhysicsConfig's doc comment rather than trusting it,
	// so the two helpers cannot silently drift apart.
	pdCfg := newTestDisaggDeploymentConfig(4, 2, 2)
	coloCfg := newTestColocatedTrainedPhysicsConfig()
	if !reflect.DeepEqual(pdCfg.LatencyCoeffs, coloCfg.LatencyCoeffs) {
		t.Fatalf("PD and co-located configs have diverging latency coefficients (%+v vs %+v) — parity comparison invalid",
			pdCfg.LatencyCoeffs, coloCfg.LatencyCoeffs)
	}
	if !reflect.DeepEqual(pdCfg.ModelHardwareConfig, coloCfg.ModelHardwareConfig) {
		t.Fatalf("PD and co-located configs have diverging model/hardware config — parity comparison invalid")
	}

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
	// Law 2: the PD surplus over co-located is at least the KV-transfer cost — the
	// extra work PD does for a 1-token request. Independent oracle: transferCost is
	// read from parent phase timestamps, not from either E2E value. Uses `>=` rather
	// than exact equality so a future model that adds any extra per-path step does
	// not break a physically-correct E2E (the surplus can only grow, never shrink
	// below the transfer cost).
	if transferCost <= 0 {
		t.Fatalf("expected a positive KV-transfer cost, got %.1f", transferCost)
	}
	if diff := pdE2E - nonPDE2E; diff < transferCost-1e-9 {
		t.Errorf("PD − non-PD E2E surplus = %.1f, want >= %.1f (KV-transfer cost); PD=%.1f nonPD=%.1f",
			diff, transferCost, pdE2E, nonPDE2E)
	}
}

// TestPDParentE2E_ProjectionBranches directly drives projectPDMetrics on stub
// parents to cover every E2E branch at unit granularity (analogous to
// TestDisaggregation_TTFT_ProjectionBranches for TTFT). It pins the preemption
// discriminator (issue #1513 review finding #1) and the fallback branches, which
// the integration tests cannot easily reach:
//
//   - PRIMARY (normal): FirstTokenTime == 0 ⇒ E2E = decodeDelay + decodeOwnE2E.
//   - PRIMARY (preempted): FirstTokenTime != 0 ⇒ decodeOwnE2E is already
//     arrival-relative, so E2E = decodeOwnE2E (decodeDelay NOT added — the guard
//     that prevents the double-count regression vs main).
//   - FALLBACK: decode-side metrics absent (nil DecodeSubReq / no recorded own E2E
//     or scheduling delay) ⇒ E2E = parent.CompletionTime − ArrivalTime.
//   - NEGATIVE GUARD: a negative reconstructed E2E is skipped (no entry emitted).
func TestPDParentE2E_ProjectionBranches(t *testing.T) {
	origReq := &sim.Request{ID: "orig", ArrivalTime: 0}

	tests := []struct {
		name          string
		parent        *ParentRequest
		setDecodeE2E  bool    // set RequestE2Es[dec] (decode sub-request's own E2E)
		decodeOwnE2E  float64 // value for RequestE2Es[dec]
		setDelay      bool    // set RequestSchedulingDelays[dec]
		decodeDelay   int64   // value for the decode scheduling delay
		wantEntry     bool    // whether a parent-keyed E2E entry is expected
		wantE2E       float64 // expected projected E2E (when wantEntry)
		wantCompletion float64 // expected RequestCompletionTimes[pid] (== ArrivalTime + E2E)
	}{
		{
			// PRIMARY normal: FirstTokenTime == 0 ⇒ add decodeDelay.
			name: "primary normal: decodeDelay + decodeOwnE2E",
			parent: &ParentRequest{
				ID: "p0", PrefillSubReqID: "p0_prefill", DecodeSubReqID: "p0_decode",
				OriginalRequest: origReq, ArrivalTime: 0, CompletionTime: 5000,
				DecodeInstanceID: "inst-0",
				DecodeSubReq:     &sim.Request{FirstTokenTime: 0, ITL: []int64{300}},
			},
			setDecodeE2E: true, decodeOwnE2E: 301,
			setDelay: true, decodeDelay: 151,
			wantEntry: true, wantE2E: 151 + 301, wantCompletion: 452,
		},
		{
			// PRIMARY preempted: FirstTokenTime != 0 ⇒ decodeOwnE2E already spans
			// arrival→completion; decodeDelay must NOT be added (double-count guard).
			// Without the guard this would report 151 + 900 = 1051 (regression).
			name: "primary preempted: decodeOwnE2E only (no double-count)",
			parent: &ParentRequest{
				ID: "p1", PrefillSubReqID: "p1_prefill", DecodeSubReqID: "p1_decode",
				OriginalRequest: origReq, ArrivalTime: 0, CompletionTime: 5000,
				DecodeInstanceID: "inst-0",
				DecodeSubReq:     &sim.Request{FirstTokenTime: 600, ITL: []int64{300}},
			},
			setDecodeE2E: true, decodeOwnE2E: 900, // = FirstTokenTime(600) + ITL(300) arrival-relative
			setDelay: true, decodeDelay: 151,
			wantEntry: true, wantE2E: 900, wantCompletion: 900,
		},
		{
			// FALLBACK: nil DecodeSubReq (drop-at-transfer-start / late-drop, #1511) ⇒
			// no decode own E2E ⇒ parent.CompletionTime − ArrivalTime.
			name: "fallback: nil DecodeSubReq",
			parent: &ParentRequest{
				ID: "p2", PrefillSubReqID: "p2_prefill", DecodeSubReqID: "p2_decode",
				OriginalRequest: origReq, ArrivalTime: 100, CompletionTime: 5000,
				DecodeInstanceID: "inst-0", DecodeSubReq: nil,
			},
			setDecodeE2E: false, setDelay: false,
			wantEntry: true, wantE2E: 4900, wantCompletion: 5000, // 5000 − 100 = 4900; completion = 100 + 4900
		},
		{
			// FALLBACK: decode own E2E present but scheduling delay absent (defensive:
			// both are required for the primary path) ⇒ CompletionTime-based value.
			name: "fallback: missing decode scheduling delay",
			parent: &ParentRequest{
				ID: "p3", PrefillSubReqID: "p3_prefill", DecodeSubReqID: "p3_decode",
				OriginalRequest: origReq, ArrivalTime: 0, CompletionTime: 5000,
				DecodeInstanceID: "inst-0",
				DecodeSubReq:     &sim.Request{FirstTokenTime: 0, ITL: []int64{300}},
			},
			setDecodeE2E: true, decodeOwnE2E: 301,
			setDelay: false,
			wantEntry: true, wantE2E: 5000, wantCompletion: 5000,
		},
		{
			// NEGATIVE GUARD: a negative reconstructed E2E (only reachable via a
			// hypothetical shared-clock regression) is skipped — no entry emitted.
			name: "negative guard: negative reconstructed E2E ⇒ no entry",
			parent: &ParentRequest{
				ID: "p4", PrefillSubReqID: "p4_prefill", DecodeSubReqID: "p4_decode",
				OriginalRequest: origReq, ArrivalTime: 0, CompletionTime: 5000,
				DecodeInstanceID: "inst-0",
				DecodeSubReq:     &sim.Request{FirstTokenTime: 0, ITL: []int64{300}},
			},
			setDecodeE2E: true, decodeOwnE2E: -1000,
			setDelay: true, decodeDelay: 500, // 500 + (−1000) = −500 < 0
			wantEntry: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := sim.NewMetrics()
			// Seed sub-request keys that projectPDMetrics must consume/delete.
			m.RequestTTFTs[tc.parent.PrefillSubReqID] = 2500.0
			if tc.setDecodeE2E {
				m.RequestE2Es[tc.parent.DecodeSubReqID] = tc.decodeOwnE2E
			}
			if tc.setDelay {
				m.RequestSchedulingDelays[tc.parent.DecodeSubReqID] = tc.decodeDelay
			}

			cs := &ClusterSimulator{
				aggregatedMetrics: m,
				parentRequests:    map[string]*ParentRequest{tc.parent.ID: tc.parent},
			}
			cs.projectPDMetrics()

			got, ok := m.RequestE2Es[tc.parent.ID]
			if tc.wantEntry {
				if !ok {
					t.Fatalf("parent %s: E2E entry missing after projection", tc.parent.ID)
				}
				if math.Abs(got-tc.wantE2E) > 1e-9 {
					t.Errorf("parent %s: E2E = %.1f, want %.1f", tc.parent.ID, got, tc.wantE2E)
				}
				// Completion-time metric consistency (== ArrivalTime + E2E).
				ct, ctOK := m.RequestCompletionTimes[tc.parent.ID]
				if !ctOK {
					t.Fatalf("parent %s: completion-time entry missing after projection", tc.parent.ID)
				}
				if math.Abs(ct-tc.wantCompletion) > 1e-9 {
					t.Errorf("parent %s: completion-time metric = %.1f, want %.1f", tc.parent.ID, ct, tc.wantCompletion)
				}
			} else {
				if ok {
					t.Errorf("parent %s: unexpected E2E entry %.1f (negative-guard branch should skip)", tc.parent.ID, got)
				}
				if _, ctOK := m.RequestCompletionTimes[tc.parent.ID]; ctOK {
					t.Errorf("parent %s: unexpected completion-time entry (should skip when E2E skipped)", tc.parent.ID)
				}
			}

			// INV-PD-6: sub-request keys must be deleted regardless of branch.
			if _, exists := m.RequestE2Es[tc.parent.DecodeSubReqID]; exists {
				t.Errorf("INV-PD-6: decode sub-request E2E key %s still present", tc.parent.DecodeSubReqID)
			}
		})
	}
}
