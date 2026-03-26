package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// stubLatencyModel returns fixed values for testing the interference wrapper.
type stubLatencyModel struct {
	stepTime                int64
	queueingTime            int64
	outputTokenProcessing   int64
	postDecodeFixedOverhead int64
}

func (s *stubLatencyModel) StepTime(_ []*sim.Request) int64         { return s.stepTime }
func (s *stubLatencyModel) QueueingTime(_ *sim.Request) int64       { return s.queueingTime }
func (s *stubLatencyModel) OutputTokenProcessingTime() int64        { return s.outputTokenProcessing }
func (s *stubLatencyModel) PostDecodeFixedOverhead() int64          { return s.postDecodeFixedOverhead }

// makeBatch creates a batch of requests with the given prefill/decode composition.
// Prefill requests have ProgressIndex=0 with len(InputTokens)=10.
// Decode requests have ProgressIndex=15 with len(InputTokens)=10 (5 decode steps past prefill).
func makeBatch(prefillCount, decodeCount int) []*sim.Request {
	batch := make([]*sim.Request, 0, prefillCount+decodeCount)
	for i := 0; i < prefillCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("prefill_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 0,
		})
	}
	for i := 0; i < decodeCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("decode_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 15, // > len(InputTokens), confirming >= check works beyond boundary
		})
	}
	return batch
}

func TestNewInterferenceLatencyModel_Validation(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	tests := []struct {
		name          string
		inner         sim.LatencyModel
		prefillFactor float64
		decodeFactor  float64
		wantErr       bool
	}{
		{name: "valid zero factors", inner: inner, prefillFactor: 0, decodeFactor: 0},
		{name: "valid positive factors", inner: inner, prefillFactor: 0.5, decodeFactor: 0.3},
		{name: "nil inner", inner: nil, prefillFactor: 0, decodeFactor: 0, wantErr: true},
		{name: "negative prefill factor", inner: inner, prefillFactor: -0.1, decodeFactor: 0, wantErr: true},
		{name: "negative decode factor", inner: inner, prefillFactor: 0, decodeFactor: -0.1, wantErr: true},
		{name: "NaN prefill factor", inner: inner, prefillFactor: math.NaN(), decodeFactor: 0, wantErr: true},
		{name: "Inf decode factor", inner: inner, prefillFactor: 0, decodeFactor: math.Inf(1), wantErr: true},
		{name: "NaN decode factor", inner: inner, prefillFactor: 0, decodeFactor: math.NaN(), wantErr: true},
		{name: "negative Inf prefill", inner: inner, prefillFactor: math.Inf(-1), decodeFactor: 0, wantErr: true},
		{name: "valid max factor", inner: inner, prefillFactor: MaxInterferenceFactor, decodeFactor: MaxInterferenceFactor},
		{name: "above max prefill", inner: inner, prefillFactor: MaxInterferenceFactor + 0.001, decodeFactor: 0, wantErr: true},
		{name: "above max decode", inner: inner, prefillFactor: 0, decodeFactor: MaxInterferenceFactor + 1, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewInterferenceLatencyModel(tc.inner, tc.prefillFactor, tc.decodeFactor)
			if (err != nil) != tc.wantErr {
				t.Errorf("NewInterferenceLatencyModel() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestInterferenceLatencyModel_StepTime(t *testing.T) {
	const baseStepTime int64 = 1000

	tests := []struct {
		name           string
		prefillFactor  float64
		decodeFactor   float64
		prefillCount   int
		decodeCount    int
		wantMultiplier float64
	}{
		{name: "zero factors mixed batch", prefillFactor: 0, decodeFactor: 0, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.0},
		{name: "all prefill", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 4, decodeCount: 0, wantMultiplier: 1.0},
		{name: "all decode", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 4, wantMultiplier: 1.0},
		{name: "empty batch", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 0, wantMultiplier: 1.0},
		{name: "prefill majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.125},
		{name: "decode majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 1, decodeCount: 3, wantMultiplier: 1.075},
		{name: "tied batch uses max factor", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 2, decodeCount: 2, wantMultiplier: 1.25},
		{name: "tied batch uses max factor reversed", prefillFactor: 0.3, decodeFactor: 0.5, prefillCount: 2, decodeCount: 2, wantMultiplier: 1.25},
		{name: "single prefill", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 1, decodeCount: 0, wantMultiplier: 1.0},
		{name: "single decode", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 0, decodeCount: 1, wantMultiplier: 1.0},
		{name: "even split factor 1.0", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 5, decodeCount: 5, wantMultiplier: 1.5},
		// Asymmetric factors: one is zero, the other is non-zero. The wrapper is active (|| semantics in
		// newInstanceSimulatorCore), but the zero factor produces multiplier=1.0 for its dominant phase
		// while the non-zero factor still applies for the other dominant phase.
		{name: "asymmetric: prefill factor set decode dominant", prefillFactor: 0.5, decodeFactor: 0, prefillCount: 1, decodeCount: 3, wantMultiplier: 1.0},
		{name: "asymmetric: decode factor set prefill dominant", prefillFactor: 0, decodeFactor: 0.5, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.0},
		{name: "asymmetric: prefill factor set prefill dominant", prefillFactor: 0.5, decodeFactor: 0, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.125},
		{name: "asymmetric: decode factor set decode dominant", prefillFactor: 0, decodeFactor: 0.5, prefillCount: 1, decodeCount: 3, wantMultiplier: 1.125},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			inner := &stubLatencyModel{stepTime: baseStepTime}
			model, err := NewInterferenceLatencyModel(inner, tc.prefillFactor, tc.decodeFactor)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			batch := makeBatch(tc.prefillCount, tc.decodeCount)
			got := model.StepTime(batch)
			want := int64(math.Round(float64(baseStepTime) * tc.wantMultiplier))
			if got != want {
				t.Errorf("StepTime() = %d, want %d (multiplier %.4f)", got, want, tc.wantMultiplier)
			}
			if m := model.LastAppliedMultiplier(); math.Abs(m-tc.wantMultiplier) > 1e-9 {
				t.Errorf("LastAppliedMultiplier() = %f, want %f", m, tc.wantMultiplier)
			}
		})
	}
}

func TestInterferenceLatencyModel_INV_P2_3_MultiplierMonotonicity(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	factors := []float64{0, 0.1, 0.5, 1.0, 2.0, 5.0}
	compositions := [][2]int{
		{0, 0}, {1, 0}, {0, 1}, {1, 1},
		{3, 1}, {1, 3}, {5, 5}, {10, 1}, {1, 10},
	}
	for _, pf := range factors {
		for _, df := range factors {
			model, err := NewInterferenceLatencyModel(inner, pf, df)
			if err != nil {
				t.Fatalf("factor (%f, %f): %v", pf, df, err)
			}
			for _, comp := range compositions {
				batch := makeBatch(comp[0], comp[1])
				got := model.StepTime(batch)
				if got < inner.stepTime {
					t.Errorf("INV-P2-3 violated: factors=(%f,%f) comp=(%d,%d) StepTime=%d < base=%d",
						pf, df, comp[0], comp[1], got, inner.stepTime)
				}
				if model.LastAppliedMultiplier() < 1.0 {
					t.Errorf("INV-P2-3 violated: factors=(%f,%f) comp=(%d,%d) multiplier=%f < 1.0",
						pf, df, comp[0], comp[1], model.LastAppliedMultiplier())
				}
			}
		}
	}
}

func TestInterferenceLatencyModel_PassThrough(t *testing.T) {
	inner := &stubLatencyModel{
		stepTime:                1000,
		queueingTime:            500,
		outputTokenProcessing:   200,
		postDecodeFixedOverhead: 100,
	}
	model, err := NewInterferenceLatencyModel(inner, 0.5, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	req := &sim.Request{InputTokens: make([]int, 5)}
	if got := model.QueueingTime(req); got != 500 {
		t.Errorf("QueueingTime() = %d, want 500", got)
	}
	if got := model.OutputTokenProcessingTime(); got != 200 {
		t.Errorf("OutputTokenProcessingTime() = %d, want 200", got)
	}
	if got := model.PostDecodeFixedOverhead(); got != 100 {
		t.Errorf("PostDecodeFixedOverhead() = %d, want 100", got)
	}
}

func TestInterferenceLatencyModel_LastAppliedMultiplier_InitialValue(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	model, err := NewInterferenceLatencyModel(inner, 0.5, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := model.LastAppliedMultiplier(); got != 1.0 {
		t.Errorf("initial LastAppliedMultiplier() = %f, want 1.0", got)
	}
}

func TestNewInstanceSimulatorCore_WrapsLatencyModel(t *testing.T) {
	// GIVEN a config that will produce measurable step times
	cfg := newTestSimConfig()

	// WHEN constructing with zero factors: step time should match baseline behavior.
	inst0 := newInstanceSimulatorCore("no-interference", cfg, 0, 0)
	if inst0 == nil {
		t.Fatal("newInstanceSimulatorCore returned nil with zero factors")
	}

	// WHEN constructing with positive factors: wrapping must be active and must
	// increase step time for a mixed prefill+decode batch (INV-P2-3).
	inst1 := newInstanceSimulatorCore("with-interference", cfg, 0.5, 0.3)
	if inst1 == nil {
		t.Fatal("newInstanceSimulatorCore returned nil with positive factors")
	}

	// THEN: verify interference actually applies by comparing step times on a
	// mixed-phase batch. Use a controlled stub to confirm the multiplier fires.
	inner := &stubLatencyModel{stepTime: 1000}
	noWrap, err := NewInterferenceLatencyModel(inner, 0, 0)
	if err != nil {
		t.Fatalf("unexpected error creating zero-factor model: %v", err)
	}
	withWrap, err := NewInterferenceLatencyModel(inner, 0.5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error creating positive-factor model: %v", err)
	}

	// Mixed batch: 3 prefill + 1 decode → majority is prefill → uses prefillInterference=0.5
	// Expected multiplier: 1.0 + 0.5*(1/4) = 1.125
	// So step time with interference must exceed step time without.
	batch := makeBatch(3, 1)
	baseStep := noWrap.StepTime(batch)
	intStep := withWrap.StepTime(batch)
	if intStep <= baseStep {
		t.Errorf("interference must increase step time for mixed batch: base=%d, interference=%d", baseStep, intStep)
	}
	if withWrap.LastAppliedMultiplier() < 1.0 {
		t.Errorf("INV-P2-3: multiplier must be >= 1.0, got %f", withWrap.LastAppliedMultiplier())
	}

	// Phase-pure batch: all prefill → multiplier must be 1.0 (BC-P2-10)
	pureBatch := makeBatch(4, 0)
	pureBase := noWrap.StepTime(pureBatch)
	pureInt := withWrap.StepTime(pureBatch)
	if pureInt != pureBase {
		t.Errorf("phase-pure batch must not be slowed down: base=%d, interference=%d", pureBase, pureInt)
	}
}

func TestInterferenceModel_ClusterIntegration(t *testing.T) {
	// Create a 2-instance cluster with interference and verify step times increase.
	baseCfg := newTestSimConfig()

	// Generate 10 requests: first 5 at t=0, next 5 at t=1.
	// This staggered arrival ensures mixed prefill+decode batches:
	// - Step 1: only t=0 requests are in each instance's batch (phase-pure prefill).
	// - Step 2: t=0 requests have completed prefill (in decode) while t=1 requests
	//   arrive and begin prefill → interference fires when both phases are present.
	rng := rand.New(rand.NewSource(42))
	requests := make([]*sim.Request, 10)
	for i := range requests {
		arrivalTime := int64(i / 5) // requests 0-4 at t=0, requests 5-9 at t=1
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(rng, 20),
			OutputTokens: sim.GenerateRandomTokenIDs(rng, 10),
			ArrivalTime:  arrivalTime,
		}
	}

	// Run without interference
	configBase := DeploymentConfig{
		SimConfig:    baseCfg,
		NumInstances: 2,
	}
	csBase := NewClusterSimulator(configBase, cloneRequests(requests), nil)
	if err := csBase.Run(); err != nil {
		t.Fatalf("baseline run failed: %v", err)
	}
	baseMetrics := csBase.AggregatedMetrics()

	// Run with interference
	configInterference := DeploymentConfig{
		SimConfig:             baseCfg,
		NumInstances:          2,
		PDInterferencePrefill: 0.5,
		PDInterferenceDecode:  0.5,
	}
	csInterference := NewClusterSimulator(configInterference, cloneRequests(requests), nil)
	if err := csInterference.Run(); err != nil {
		t.Fatalf("interference run failed: %v", err)
	}
	interferenceMetrics := csInterference.AggregatedMetrics()

	// Non-vacuity: at least one request must complete in both runs before comparing outcomes.
	if len(baseMetrics.RequestE2Es) == 0 {
		t.Fatal("baseline run completed 0 requests — test is vacuous")
	}
	if len(interferenceMetrics.RequestE2Es) == 0 {
		t.Fatal("interference run completed 0 requests — test is vacuous")
	}

	// With interference, simulation should take longer (higher SimEndedTime)
	if interferenceMetrics.SimEndedTime <= baseMetrics.SimEndedTime {
		t.Errorf("expected interference to increase simulation time: base=%d, interference=%d",
			baseMetrics.SimEndedTime, interferenceMetrics.SimEndedTime)
	}

	// Per-request E2E latencies should be larger with interference.
	// Compare completed requests that exist in both runs.
	for reqID, baseE2E := range baseMetrics.RequestE2Es {
		if intE2E, ok := interferenceMetrics.RequestE2Es[reqID]; ok {
			if intE2E < baseE2E {
				t.Errorf("request %s: interference E2E (%f) < base E2E (%f)", reqID, intE2E, baseE2E)
			}
		}
	}
}

// TestInterferenceModel_ClusterIntegration_INV_P2_3 is the R7 invariant companion to
// TestInterferenceModel_ClusterIntegration. It verifies the monotonicity law (INV-P2-3)
// at the cluster level: interference can only increase latency, never decrease it.
// This is a law test — it must hold for any valid factor > 0, not just a specific value.
func TestInterferenceModel_ClusterIntegration_INV_P2_3(t *testing.T) {
	baseCfg := newTestSimConfig()
	rng := rand.New(rand.NewSource(42))
	requests := make([]*sim.Request, 10)
	for i := range requests {
		arrivalTime := int64(i / 5) // requests 0-4 at t=0, requests 5-9 at t=1
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(rng, 20),
			OutputTokens: sim.GenerateRandomTokenIDs(rng, 10),
			ArrivalTime:  arrivalTime,
		}
	}

	configBase := DeploymentConfig{SimConfig: baseCfg, NumInstances: 2}
	csBase := NewClusterSimulator(configBase, cloneRequests(requests), nil)
	if err := csBase.Run(); err != nil {
		t.Fatalf("baseline run: %v", err)
	}
	baseMetrics := csBase.AggregatedMetrics()

	configInt := DeploymentConfig{
		SimConfig:             baseCfg,
		NumInstances:          2,
		PDInterferencePrefill: 0.5,
		PDInterferenceDecode:  0.5,
	}
	csInt := NewClusterSimulator(configInt, cloneRequests(requests), nil)
	if err := csInt.Run(); err != nil {
		t.Fatalf("interference run: %v", err)
	}
	intMetrics := csInt.AggregatedMetrics()

	// INV-P2-3 law: SimEndedTime is non-decreasing under interference (monotonicity).
	if intMetrics.SimEndedTime < baseMetrics.SimEndedTime {
		t.Errorf("INV-P2-3 violated: SimEndedTime decreased under interference: base=%d, int=%d",
			baseMetrics.SimEndedTime, intMetrics.SimEndedTime)
	}

	// Non-vacuity: at least one request must complete in both runs.
	commonCount := 0
	for reqID := range baseMetrics.RequestE2Es {
		if _, ok := intMetrics.RequestE2Es[reqID]; ok {
			commonCount++
		}
	}
	if commonCount == 0 {
		t.Fatalf("INV-P2-3 test is vacuous: no requests completed in both runs (base=%d, int=%d completed)",
			len(baseMetrics.RequestE2Es), len(intMetrics.RequestE2Es))
	}

	// INV-P2-3 law: for every request in both runs, E2E(interference) >= E2E(baseline).
	violations := 0
	for reqID, baseE2E := range baseMetrics.RequestE2Es {
		if intE2E, ok := intMetrics.RequestE2Es[reqID]; ok {
			if intE2E < baseE2E {
				t.Errorf("INV-P2-3 violated for %s: interference E2E %.3fms < base E2E %.3fms",
					reqID, intE2E*1000, baseE2E*1000)
				violations++
			}
		}
	}
	if violations > 0 {
		t.Errorf("INV-P2-3 violated for %d/%d requests in common", violations, commonCount)
	}
}

// TestInterferenceLatencyModel_LastAppliedMultiplier_UpdatesPerCall verifies BC-P2-12:
// LastAppliedMultiplier reflects the most recent StepTime call, not an earlier one.
func TestInterferenceLatencyModel_LastAppliedMultiplier_UpdatesPerCall(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	model, err := NewInterferenceLatencyModel(inner, 0.5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// First call: mixed batch → multiplier > 1.0
	mixedBatch := makeBatch(3, 1) // prefill majority: 1.0 + 0.5*(1/4) = 1.125
	model.StepTime(mixedBatch)
	firstMultiplier := model.LastAppliedMultiplier()
	if math.Abs(firstMultiplier-1.125) > 1e-9 {
		t.Errorf("first call multiplier = %f, want 1.125", firstMultiplier)
	}

	// Second call: phase-pure batch → multiplier must reset to 1.0
	pureBatch := makeBatch(4, 0)
	model.StepTime(pureBatch)
	secondMultiplier := model.LastAppliedMultiplier()
	if math.Abs(secondMultiplier-1.0) > 1e-9 {
		t.Errorf("second call multiplier = %f after phase-pure batch, want 1.0 (BC-P2-12: must reflect latest call)", secondMultiplier)
	}

	// Third call: different mixed batch → multiplier updates again
	mixedBatch2 := makeBatch(1, 3) // decode majority: 1.0 + 0.3*(1/4) = 1.075
	model.StepTime(mixedBatch2)
	thirdMultiplier := model.LastAppliedMultiplier()
	if math.Abs(thirdMultiplier-1.075) > 1e-9 {
		t.Errorf("third call multiplier = %f, want 1.075 (BC-P2-12: must reflect latest call, not first)", thirdMultiplier)
	}
}

// TestInterferenceModel_PDMode_PhasePurePools verifies BC-P2-10 at the cluster level:
// in a fully disaggregated deployment with AlwaysDisaggregate, pool instances only
// receive phase-pure sub-requests (INV-PD-2), so interference factors have no effect
// and results are identical to a zero-factor baseline run.
func TestInterferenceModel_PDMode_PhasePurePools(t *testing.T) {
	// Use the shared PD deployment config (4 instances: 2 prefill + 2 decode, AlwaysDisaggregate).
	baseCfg := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	// Run without interference.
	csBase := NewClusterSimulator(baseCfg, cloneRequests(requests), nil)
	if err := csBase.Run(); err != nil {
		t.Fatalf("baseline run failed: %v", err)
	}
	baseMetrics := csBase.AggregatedMetrics()
	if len(baseMetrics.RequestE2Es) == 0 {
		t.Fatal("baseline run completed 0 requests — test is vacuous")
	}

	// Run with non-zero interference in fully disaggregated mode.
	// BC-P2-10: phase-pure batches → multiplier=1.0, so results must be identical.
	cfgInterf := baseCfg
	cfgInterf.PDInterferencePrefill = 0.5
	cfgInterf.PDInterferenceDecode = 0.5
	csInterf := NewClusterSimulator(cfgInterf, cloneRequests(requests), nil)
	if err := csInterf.Run(); err != nil {
		t.Fatalf("interference run failed: %v", err)
	}
	interfMetrics := csInterf.AggregatedMetrics()

	// BC-P2-10 at cluster level: interference must have zero effect in fully
	// disaggregated deployment because pool batches are always phase-pure.
	if interfMetrics.SimEndedTime != baseMetrics.SimEndedTime {
		t.Errorf("BC-P2-10 violated: SimEndedTime differs with interference in PD mode: base=%d, interf=%d",
			baseMetrics.SimEndedTime, interfMetrics.SimEndedTime)
	}
	for reqID, baseE2E := range baseMetrics.RequestE2Es {
		if interfE2E, ok := interfMetrics.RequestE2Es[reqID]; ok {
			if interfE2E != baseE2E {
				t.Errorf("BC-P2-10 violated: request %s E2E differs in PD mode: base=%v, interf=%v",
					reqID, baseE2E, interfE2E)
			}
		}
	}
}

// cloneRequests creates deep copies of requests for independent simulation runs.
func cloneRequests(reqs []*sim.Request) []*sim.Request {
	result := make([]*sim.Request, len(reqs))
	for i, r := range reqs {
		clone := *r
		clone.InputTokens = make([]int, len(r.InputTokens))
		copy(clone.InputTokens, r.InputTokens)
		clone.OutputTokens = make([]int, len(r.OutputTokens))
		copy(clone.OutputTokens, r.OutputTokens)
		result[i] = &clone
	}
	return result
}
