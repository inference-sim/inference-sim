# PD Interference Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameterized interference model that applies multiplicative slowdown to step times when prefill and decode phases co-locate, enabling break-even analysis against disaggregation transfer cost.

**Architecture:** A `LatencyModel` wrapper (`InterferenceLatencyModel`) in `sim/cluster/interference.go` applies a multiplier to `StepTime()` based on batch phase composition. The existing `NewInstanceSimulator` public API is unchanged; an unexported `newInstanceSimulatorCore` variant accepts interference factors. CLI flags `--pd-interference-prefill` and `--pd-interference-decode` (float64, default 0) wire through `DeploymentConfig`.

**Tech Stack:** Go, table-driven tests, `math` package for float operations.

**Spec:** `docs/superpowers/specs/2026-03-13-pd-interference-model-design.md`
**Issue:** #635

---

## Task 1: InterferenceLatencyModel — constructor and StepTime

**Files:**
- Create: `sim/cluster/interference.go`
- Create: `sim/cluster/interference_test.go`

### Step 1.1: Write the failing constructor test

- [ ] Add `sim/cluster/interference_test.go` with constructor validation tests.

```go
package cluster

import (
	"math"
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
```

- [ ] Run test to verify it fails.

Run: `go test ./sim/cluster/ -run TestNewInterferenceLatencyModel_Validation -v`
Expected: FAIL — `NewInterferenceLatencyModel` undefined.

### Step 1.2: Write the failing StepTime tests

- [ ] Add StepTime test cases to `interference_test.go`.

```go
// makeBatch creates a batch of requests with the given prefill/decode composition.
// Prefill requests have ProgressIndex=0 with len(InputTokens)=10.
// Decode requests have ProgressIndex=10 with len(InputTokens)=10 (past prefill).
func makeBatch(prefillCount, decodeCount int) []*sim.Request {
	batch := make([]*sim.Request, 0, prefillCount+decodeCount)
	for i := 0; i < prefillCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("prefill_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 0, // still in prefill phase
		})
	}
	for i := 0; i < decodeCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("decode_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 10, // past prefill, in decode phase
		})
	}
	return batch
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
		// BC-P2-9: zero factors → identity
		{name: "zero factors mixed batch", prefillFactor: 0, decodeFactor: 0, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.0},
		// BC-P2-10: phase-pure → 1.0
		{name: "all prefill", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 4, decodeCount: 0, wantMultiplier: 1.0},
		{name: "all decode", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 4, wantMultiplier: 1.0},
		// Empty batch → 1.0
		{name: "empty batch", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 0, wantMultiplier: 1.0},
		// Prefill majority: minority is decode, use prefillFactor
		// 3 prefill + 1 decode: minority_fraction = 1/4 = 0.25, multiplier = 1.0 + 0.5 * 0.25 = 1.125
		{name: "prefill majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.125},
		// Decode majority: minority is prefill, use decodeFactor
		// 1 prefill + 3 decode: minority_fraction = 1/4 = 0.25, multiplier = 1.0 + 0.3 * 0.25 = 1.075
		{name: "decode majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 1, decodeCount: 3, wantMultiplier: 1.075},
		// Tied batch: use max factor
		// 2 prefill + 2 decode: minority_fraction = 2/4 = 0.5, multiplier = 1.0 + max(0.5, 0.3) * 0.5 = 1.25
		{name: "tied batch uses max factor", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 2, decodeCount: 2, wantMultiplier: 1.25},
		// Single request → phase-pure
		{name: "single prefill", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 1, decodeCount: 0, wantMultiplier: 1.0},
		{name: "single decode", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 0, decodeCount: 1, wantMultiplier: 1.0},
		// Max interference at 50/50 split with factor 1.0
		{name: "even split factor 1.0", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 5, decodeCount: 5, wantMultiplier: 1.5},
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
			// BC-P2-12: verify LastAppliedMultiplier
			if m := model.LastAppliedMultiplier(); math.Abs(m-tc.wantMultiplier) > 1e-9 {
				t.Errorf("LastAppliedMultiplier() = %f, want %f", m, tc.wantMultiplier)
			}
		})
	}
}
```

Add `"fmt"` to the imports block.

- [ ] Run test to verify it fails.

Run: `go test ./sim/cluster/ -run TestInterferenceLatencyModel_StepTime -v`
Expected: FAIL — `NewInterferenceLatencyModel` undefined.

### Step 1.3: Write the INV-P2-3 invariant test

- [ ] Add the invariant property test to `interference_test.go`.

```go
func TestInterferenceLatencyModel_INV_P2_3_MultiplierMonotonicity(t *testing.T) {
	// INV-P2-3: multiplier >= 1.0 for all valid factor/composition combinations.
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
```

- [ ] Run test to verify it fails.

Run: `go test ./sim/cluster/ -run TestInterferenceLatencyModel_INV_P2_3 -v`
Expected: FAIL — `NewInterferenceLatencyModel` undefined.

### Step 1.4: Write the pass-through delegation test

- [ ] Add pass-through test to `interference_test.go`.

```go
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
	// Before any StepTime call, LastAppliedMultiplier should return 1.0
	if got := model.LastAppliedMultiplier(); got != 1.0 {
		t.Errorf("initial LastAppliedMultiplier() = %f, want 1.0", got)
	}
}
```

### Step 1.5: Implement InterferenceLatencyModel

- [ ] Create `sim/cluster/interference.go` with the full implementation.

```go
package cluster

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// InterferenceLatencyModel wraps a LatencyModel to apply a multiplicative slowdown
// when prefill and decode phases co-locate in the same batch. This enables break-even
// analysis between disaggregation transfer cost and co-location interference cost.
//
// Extension type: tier composition (wraps LatencyModel).
//
// Multiplier formula: 1.0 + factor * (minority_count / total_count)
// where minority_count is the count of requests in the less-common phase.
// Effective range: factor of 1.0 produces at most 50% slowdown at even split.
//
// Behavioral guarantees:
//   - BC-P2-9:  factors=0 → step time identical to inner model
//   - BC-P2-10: phase-pure batch → multiplier=1.0
//   - BC-P2-11/INV-P2-3: multiplier >= 1.0 always
//   - BC-P2-12: LastAppliedMultiplier() records per-call multiplier
type InterferenceLatencyModel struct {
	inner               sim.LatencyModel
	prefillInterference float64 // slowdown for prefill-dominant batches (minority is decode)
	decodeInterference  float64 // slowdown for decode-dominant batches (minority is prefill)
	lastMultiplier      float64
}

// NewInterferenceLatencyModel creates an interference wrapper around the given LatencyModel.
// prefillFactor is the interference factor when prefill is the majority phase.
// decodeFactor is the interference factor when decode is the majority phase.
// Both factors must be >= 0 and finite (R3).
func NewInterferenceLatencyModel(inner sim.LatencyModel, prefillFactor, decodeFactor float64) (*InterferenceLatencyModel, error) {
	if inner == nil {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: inner must not be nil")
	}
	if prefillFactor < 0 || math.IsNaN(prefillFactor) || math.IsInf(prefillFactor, 0) {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: prefillFactor must be a finite non-negative number, got %f", prefillFactor)
	}
	if decodeFactor < 0 || math.IsNaN(decodeFactor) || math.IsInf(decodeFactor, 0) {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: decodeFactor must be a finite non-negative number, got %f", decodeFactor)
	}
	return &InterferenceLatencyModel{
		inner:               inner,
		prefillInterference: prefillFactor,
		decodeInterference:  decodeFactor,
		lastMultiplier:      1.0, // initialized to 1.0 (no interference before first call)
	}, nil
}

// StepTime applies the interference multiplier to the inner model's step time.
// Classifies each request as prefill (ProgressIndex < len(InputTokens)) or decode,
// then applies: multiplier = 1.0 + factor * (minority_count / total_count).
func (m *InterferenceLatencyModel) StepTime(batch []*sim.Request) int64 {
	baseTime := m.inner.StepTime(batch)

	multiplier := m.computeMultiplier(batch)
	m.lastMultiplier = multiplier

	result := int64(math.Round(float64(baseTime) * multiplier))
	// INV-3: guarantee clock advancement.
	if result < 1 {
		result = 1
	}
	return result
}

// computeMultiplier determines the interference multiplier from batch composition.
func (m *InterferenceLatencyModel) computeMultiplier(batch []*sim.Request) float64 {
	total := len(batch)
	if total == 0 {
		return 1.0
	}

	prefillCount := 0
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			prefillCount++
		}
	}
	decodeCount := total - prefillCount

	minorityCount := min(prefillCount, decodeCount)
	if minorityCount == 0 {
		return 1.0 // phase-pure batch (BC-P2-10)
	}

	// Select factor based on majority phase; tie → max (conservative)
	var factor float64
	switch {
	case prefillCount > decodeCount:
		factor = m.prefillInterference
	case decodeCount > prefillCount:
		factor = m.decodeInterference
	default: // tied
		factor = max(m.prefillInterference, m.decodeInterference)
	}

	return 1.0 + factor*(float64(minorityCount)/float64(total))
}

// QueueingTime delegates to inner model (interference does not affect queueing).
func (m *InterferenceLatencyModel) QueueingTime(req *sim.Request) int64 {
	return m.inner.QueueingTime(req)
}

// OutputTokenProcessingTime delegates to inner model.
func (m *InterferenceLatencyModel) OutputTokenProcessingTime() int64 {
	return m.inner.OutputTokenProcessingTime()
}

// PostDecodeFixedOverhead delegates to inner model.
func (m *InterferenceLatencyModel) PostDecodeFixedOverhead() int64 {
	return m.inner.PostDecodeFixedOverhead()
}

// LastAppliedMultiplier returns the multiplier applied in the most recent StepTime call.
// Returns 1.0 before any StepTime call (BC-P2-12).
func (m *InterferenceLatencyModel) LastAppliedMultiplier() float64 {
	return m.lastMultiplier
}
```

- [ ] Run all interference tests to verify they pass.

Run: `go test ./sim/cluster/ -run TestInterferenceLatencyModel -v && go test ./sim/cluster/ -run TestNewInterferenceLatencyModel -v`
Expected: ALL PASS.

### Step 1.6: Commit

- [ ] Commit the wrapper and tests.

```bash
git add sim/cluster/interference.go sim/cluster/interference_test.go
git commit -m "feat(sim/cluster): add InterferenceLatencyModel wrapper (#635)

Tier composition wrapper that applies multiplicative slowdown to StepTime()
based on batch phase composition. Satisfies BC-P2-9 through BC-P2-12 and
INV-P2-3 (multiplier >= 1.0)."
```

---

## Task 2: DeploymentConfig + instance.go injection

**Files:**
- Modify: `sim/cluster/deployment.go:7-55` (add fields)
- Modify: `sim/cluster/instance.go:29-48` (extract core constructor)
- Modify: `sim/cluster/cluster.go:82-91` (switch to core constructor)

### Step 2.1: Write failing test for instance construction with interference

- [ ] Add integration test to `interference_test.go` verifying the injection path.

```go
func TestNewInstanceSimulatorCore_WrapsLatencyModel(t *testing.T) {
	cfg := newTestSimConfig()
	// With zero factors: no wrapping, baseline behavior
	inst0 := newInstanceSimulatorCore("no-interference", cfg, 0, 0)
	if inst0 == nil {
		t.Fatal("newInstanceSimulatorCore returned nil with zero factors")
	}

	// With positive factors: wrapping active
	inst1 := newInstanceSimulatorCore("with-interference", cfg, 0.5, 0.3)
	if inst1 == nil {
		t.Fatal("newInstanceSimulatorCore returned nil with positive factors")
	}
}
```

Note: `newTestSimConfig()` is a helper used in existing test files. Check `instance_test.go` for the definition — it returns a valid `sim.SimConfig` with blackbox latency model defaults. If it's not accessible from the test file, use the same pattern from existing tests.

- [ ] Run test to verify it fails.

Run: `go test ./sim/cluster/ -run TestNewInstanceSimulatorCore -v`
Expected: FAIL — `newInstanceSimulatorCore` undefined.

### Step 2.2: Add DeploymentConfig fields

**R4 analysis:** `DeploymentConfig` is constructed in `cmd/root.go:1145` (covered by Task 3) and in ~15 test files (`cluster_test.go`, `disaggregation_test.go`, `transfer_contention_test.go`, `resolve_test.go`, `inflight_requests_test.go`, `pd_traces_test.go`, `evaluation_test.go`, `cluster_trace_test.go`, `metrics_substrate_test.go`, `prefix_routing_test.go`). Both new fields are `float64` with zero-value 0.0, which means "no interference" (BC-P2-9). All existing construction sites that omit these fields get the correct default behavior. No test changes required.

- [ ] Add interference fields to `DeploymentConfig` in `sim/cluster/deployment.go`.

Insert after the `PDTransferContention` field (line 44) and before the per-pool routing scorer fields:

```go
	// PD interference model (PR3, INV-P2-3)
	// Multiplicative slowdown applied to StepTime when prefill and decode co-locate.
	// 0 = no interference (default, BC-P2-9). Both must be >= 0.
	PDInterferencePrefill float64 // interference factor for prefill-dominant batches
	PDInterferenceDecode  float64 // interference factor for decode-dominant batches
```

### Step 2.3: Extract newInstanceSimulatorCore in instance.go

- [ ] Refactor `sim/cluster/instance.go` to extract an unexported constructor.

Replace the body of `NewInstanceSimulator` with a delegation call:

```go
// NewInstanceSimulator creates an InstanceSimulator from a SimConfig struct.
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
// Failure modes: Panics if internal Simulator creation fails (matches existing behavior).
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
	return newInstanceSimulatorCore(id, cfg, 0, 0)
}

// newInstanceSimulatorCore is the internal constructor that optionally wraps the
// latency model with InterferenceLatencyModel when interference factors are non-zero.
// Used by NewClusterSimulator to pass deployment-level interference config.
func newInstanceSimulatorCore(id InstanceID, cfg sim.SimConfig, prefillInterference, decodeInterference float64) *InstanceSimulator {
	kvStore := kv.NewKVStore(cfg.KVCacheConfig)
	latencyModel, err := latency.NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		panic(fmt.Sprintf("newInstanceSimulatorCore(%s): NewLatencyModel: %v", id, err))
	}
	// Wrap with interference model when factors are configured (BC-P2-9: no-op at zero)
	if prefillInterference > 0 || decodeInterference > 0 {
		wrapped, wrapErr := NewInterferenceLatencyModel(latencyModel, prefillInterference, decodeInterference)
		if wrapErr != nil {
			panic(fmt.Sprintf("newInstanceSimulatorCore(%s): NewInterferenceLatencyModel: %v", id, wrapErr))
		}
		latencyModel = wrapped
	}
	s, err := sim.NewSimulator(cfg, kvStore, latencyModel)
	if err != nil {
		panic(fmt.Sprintf("newInstanceSimulatorCore(%s): %v", id, err))
	}
	return &InstanceSimulator{
		id:  id,
		sim: s,
	}
}
```

### Step 2.4: Update cluster.go to use newInstanceSimulatorCore

- [ ] In `sim/cluster/cluster.go`, update the instance construction loop (~line 90).

Replace:
```go
instances[idx] = NewInstanceSimulator(id, simCfg)
```

With:
```go
instances[idx] = newInstanceSimulatorCore(id, simCfg, config.PDInterferencePrefill, config.PDInterferenceDecode)
```

### Step 2.5: Run tests

- [ ] Run all cluster tests to verify nothing is broken.

Run: `go test ./sim/cluster/ -v -count=1`
Expected: ALL PASS (existing tests use `NewInstanceSimulator` which delegates with 0,0).

### Step 2.6: Commit

- [ ] Commit the injection plumbing.

```bash
git add sim/cluster/deployment.go sim/cluster/instance.go sim/cluster/cluster.go
git commit -m "feat(sim/cluster): wire interference model into instance construction (#635)

Add PDInterferencePrefill/Decode to DeploymentConfig. Extract
newInstanceSimulatorCore to wrap latency model when factors are non-zero.
Public NewInstanceSimulator API unchanged (R4: 23 test call sites unaffected)."
```

---

## Task 3: CLI flags

**Files:**
- Modify: `cmd/root.go:107-113` (variable declarations)
- Modify: `cmd/root.go:1174` (DeploymentConfig wiring)
- Modify: `cmd/root.go:1467-1470` (flag registration)

### Step 3.1: Add CLI flag variables

- [ ] In `cmd/root.go`, add two variables after the `pdTransferContention` declaration (line 111).

```go
	// PD interference config (PR3)
	pdInterferencePrefill float64 // interference factor for prefill-dominant batches
	pdInterferenceDecode  float64 // interference factor for decode-dominant batches
```

### Step 3.2: Register flags

- [ ] In `cmd/root.go`, add flag registrations after the `pd-transfer-contention` flag (line 1470).

```go
	runCmd.Flags().Float64Var(&pdInterferencePrefill, "pd-interference-prefill", 0, "Interference slowdown factor for prefill-dominant batches when decode co-locates (0 = disabled)")
	runCmd.Flags().Float64Var(&pdInterferenceDecode, "pd-interference-decode", 0, "Interference slowdown factor for decode-dominant batches when prefill co-locates (0 = disabled)")
```

### Step 3.3: Add CLI validation

- [ ] In `cmd/root.go`, add validation after the existing PD transfer parameter validation block (~line 1078). The interference flags are valid even without PD disaggregation (they model co-location cost in non-PD mode), so validate unconditionally.

```go
		// PD interference parameter validation (R3)
		if pdInterferencePrefill < 0 || math.IsNaN(pdInterferencePrefill) || math.IsInf(pdInterferencePrefill, 0) {
			logrus.Fatalf("--pd-interference-prefill must be a finite non-negative number, got %f", pdInterferencePrefill)
		}
		if pdInterferenceDecode < 0 || math.IsNaN(pdInterferenceDecode) || math.IsInf(pdInterferenceDecode, 0) {
			logrus.Fatalf("--pd-interference-decode must be a finite non-negative number, got %f", pdInterferenceDecode)
		}
```

### Step 3.4: Wire to DeploymentConfig

- [ ] In `cmd/root.go`, add fields to the `DeploymentConfig` struct literal after `PDTransferContention` (line 1174).

```go
			PDInterferencePrefill:   pdInterferencePrefill,
			PDInterferenceDecode:    pdInterferenceDecode,
```

### Step 3.5: Verify build and run

- [ ] Build and test the CLI flag is recognized.

Run: `go build -o blis main.go && ./blis run --help | grep -A1 interference`
Expected: Both `--pd-interference-prefill` and `--pd-interference-decode` appear in help output.

- [ ] Run full test suite.

Run: `go test ./... -count=1`
Expected: ALL PASS.

### Step 3.6: Commit

- [ ] Commit CLI integration.

```bash
git add cmd/root.go
git commit -m "feat(cmd): add --pd-interference-prefill/decode CLI flags (#635)

Wire interference factors through CLI → DeploymentConfig → instance construction.
Validated as finite non-negative (R3). Default 0 = no interference (BC-P2-9)."
```

---

## Task 4: Integration test

**Files:**
- Modify: `sim/cluster/interference_test.go`

### Step 4.1: Write integration test

- [ ] Add a cluster-level integration test to `interference_test.go`.

The test file is in `package cluster`, so no `cluster.` prefix on types. Uses 2 instances and 10 requests per spec. Asserts both total simulation time and per-request E2E latencies increase.

```go
func TestInterferenceModel_ClusterIntegration(t *testing.T) {
	// Create a 2-instance cluster with interference and verify step times increase.
	baseCfg := newTestSimConfig()

	// Generate 10 requests arriving at t=0 to create mixed prefill/decode batches.
	rng := rand.New(rand.NewSource(42))
	requests := make([]*sim.Request, 10)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(rng, 20),
			OutputTokens: sim.GenerateRandomTokenIDs(rng, 10),
			ArrivalTime:  0,
		}
	}

	// Run without interference
	configBase := DeploymentConfig{
		SimConfig:    baseCfg,
		NumInstances: 2,
	}
	csBase := NewClusterSimulator(configBase, cloneRequests(requests))
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
	csInterference := NewClusterSimulator(configInterference, cloneRequests(requests))
	if err := csInterference.Run(); err != nil {
		t.Fatalf("interference run failed: %v", err)
	}
	interferenceMetrics := csInterference.AggregatedMetrics()

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
```

Add `"math/rand"` to the imports block.

Note: If `newTestSimConfig()` is not accessible from this test file, use the same config pattern from `instance_test.go` to build a valid SimConfig with blackbox latency coefficients.

- [ ] Run integration test.

Run: `go test ./sim/cluster/ -run TestInterferenceModel_ClusterIntegration -v`
Expected: PASS — interference increases simulation time.

### Step 4.2: Commit

- [ ] Commit integration test.

```bash
git add sim/cluster/interference_test.go
git commit -m "test(sim/cluster): add cluster integration test for interference model (#635)

Verifies that non-zero interference factors produce longer simulation times
compared to the zero-interference baseline."
```

---

## Task 5: Documentation updates

**Files:**
- Modify: `CLAUDE.md:103-107` (invariant table)
- Modify: `CLAUDE.md:399-404` (CLI flags in disaggregated data flow)
- Modify: `CLAUDE.md:22` (file organization — instance.go description)
- Modify: `docs/contributing/templates/design-guidelines.md:242` (module map)

### Step 5.1: Add INV-P2-3 to CLAUDE.md invariants

- [ ] In `CLAUDE.md`, add INV-P2-3 after INV-P2-2 (line 107).

```
- **INV-P2-3 Interference monotonicity**: Multiplier >= 1.0 (interference never speeds up execution). Multiplier = 1.0 when batch is phase-pure.
```

### Step 5.2: Add CLI flags to CLAUDE.md disaggregated data flow

- [ ] In `CLAUDE.md`, append the new flags to the CLI flags line (line 403, after `--decode-routing-scorers`).

Add to the end of the CLI flags line:
```
, `--pd-interference-prefill` (float64, default 0 — prefill-dominant batch slowdown), `--pd-interference-decode` (float64, default 0 — decode-dominant batch slowdown)
```

### Step 5.3: Update file organization

- [ ] In `CLAUDE.md`, add `interference.go` entry to the `sim/cluster/` section of the file tree.

Add after the `deployment.go` entry:
```
│   ├── interference.go        # InterferenceLatencyModel: tier-composition wrapper applying co-location slowdown to StepTime (INV-P2-3)
```

### Step 5.4: Update CLAUDE.md root.go description

- [ ] In `CLAUDE.md`, update the `cmd/root.go` file description in the file tree to include the new flags.

Find the line containing `root.go` and append `--pd-interference-prefill, --pd-interference-decode` to the flags list in the parenthetical.

### Step 5.5: Update design guidelines module map (acceptance criterion)

- [ ] In `docs/contributing/templates/design-guidelines.md`, add an interference model row to the Section 4.2 module map table (after the Latency Model row, line 243).

```
| **Interference Model** | Apply co-location slowdown when prefill/decode share an instance | `InterferenceLatencyModel` (wraps `LatencyModel`) | Implemented — PR3 |
```

### Step 5.6: Run lint

- [ ] Verify no lint issues.

Run: `golangci-lint run ./...`
Expected: No new issues.

### Step 5.7: Run full test suite

- [ ] Final verification.

Run: `go test ./... -count=1`
Expected: ALL PASS.

### Step 5.8: Commit

- [ ] Commit documentation updates.

```bash
git add CLAUDE.md docs/contributing/templates/design-guidelines.md
git commit -m "docs: update CLAUDE.md and design guidelines for interference model (#635)

Add INV-P2-3 (interference monotonicity), CLI flags, file organization
entry, and module map entry for InterferenceLatencyModel."
```
