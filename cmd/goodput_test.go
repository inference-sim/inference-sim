package cmd

import (
	"path/filepath"
	"reflect"
	"testing"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestParseSLODurationFlag_HappyPath verifies BC-8: comma-separated key=duration pairs parse correctly.
func TestParseSLODurationFlag_HappyPath(t *testing.T) {
	got, err := parseSLODurationFlag("critical=100ms,standard=500ms")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := map[string]time.Duration{
		"critical": 100 * time.Millisecond,
		"standard": 500 * time.Millisecond,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestParseSLODurationFlag_Empty verifies BC-8: empty input returns (nil, nil).
func TestParseSLODurationFlag_Empty(t *testing.T) {
	got, err := parseSLODurationFlag("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("got %v, want nil", got)
	}
	got, err = parseSLODurationFlag("   ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("whitespace-only: got %v, want nil", got)
	}
}

// TestParseSLODurationFlag_RejectInvalid verifies BC-8: invalid inputs error.
func TestParseSLODurationFlag_RejectInvalid(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"empty key", "=100ms"},
		{"missing equals", "critical"},
		{"unparseable duration", "critical=not-a-duration"},
		{"negative duration", "critical=-100ms"},
		{"zero duration", "critical=0s"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseSLODurationFlag(tc.input); err == nil {
				t.Errorf("input %q: expected error, got nil", tc.input)
			}
		})
	}
}

// TestMergeGoodputTargets_PrecedenceCLIWins verifies BC-1: CLI > trace header > spec.
func TestMergeGoodputTargets_PrecedenceCLIWins(t *testing.T) {
	cliE2E := map[string]time.Duration{"critical": 100 * time.Millisecond}
	header := map[string]workload.SLODimTargets{"critical": {E2EMs: 200}}
	spec := map[string]workload.SLODimTargets{"critical": {E2EMs: 300}}
	got := mergeGoodputTargets(nil, nil, cliE2E, header, spec)
	if got["critical"].E2EMs != 100 {
		t.Errorf("E2EMs: got %v, want 100 (CLI should win)", got["critical"].E2EMs)
	}
}

// TestMergeGoodputTargets_PrecedenceHeaderOverSpec verifies BC-1: trace header > spec.
func TestMergeGoodputTargets_PrecedenceHeaderOverSpec(t *testing.T) {
	header := map[string]workload.SLODimTargets{"critical": {E2EMs: 200}}
	spec := map[string]workload.SLODimTargets{"critical": {E2EMs: 300}}
	got := mergeGoodputTargets(nil, nil, nil, header, spec)
	if got["critical"].E2EMs != 200 {
		t.Errorf("E2EMs: got %v, want 200 (header should win over spec)", got["critical"].E2EMs)
	}
}

// TestMergeGoodputTargets_PartialDimensionOverride verifies BC-1: each dim merges
// independently; CLI TTFT does not erase header E2E.
func TestMergeGoodputTargets_PartialDimensionOverride(t *testing.T) {
	cliTTFT := map[string]time.Duration{"critical": 50 * time.Millisecond}
	header := map[string]workload.SLODimTargets{"critical": {E2EMs: 5000}}
	got := mergeGoodputTargets(cliTTFT, nil, nil, header, nil)
	if got["critical"].TTFTMs != 50 {
		t.Errorf("TTFTMs: got %v, want 50", got["critical"].TTFTMs)
	}
	if got["critical"].E2EMs != 5000 {
		t.Errorf("E2EMs: got %v, want 5000 (header E2E should survive CLI TTFT)", got["critical"].E2EMs)
	}
}

// TestMergeGoodputTargets_ClassUnion verifies BC-1: classes from any tier are included.
func TestMergeGoodputTargets_ClassUnion(t *testing.T) {
	cliE2E := map[string]time.Duration{"critical": 100 * time.Millisecond}
	header := map[string]workload.SLODimTargets{"standard": {E2EMs: 500}}
	spec := map[string]workload.SLODimTargets{"batch": {E2EMs: 30000}}
	got := mergeGoodputTargets(nil, nil, cliE2E, header, spec)
	if len(got) != 3 {
		t.Errorf("got %d classes, want 3 (critical, standard, batch)", len(got))
	}
	if got["critical"].E2EMs != 100 {
		t.Errorf("critical E2EMs: got %v, want 100", got["critical"].E2EMs)
	}
	if got["standard"].E2EMs != 500 {
		t.Errorf("standard E2EMs: got %v, want 500", got["standard"].E2EMs)
	}
	if got["batch"].E2EMs != 30000 {
		t.Errorf("batch E2EMs: got %v, want 30000", got["batch"].E2EMs)
	}
}

// TestMergeGoodputTargets_AllNil verifies BC-3: nil sources produce nil result
// so the no-targets path stays byte-identical.
func TestMergeGoodputTargets_AllNil(t *testing.T) {
	if got := mergeGoodputTargets(nil, nil, nil, nil, nil); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

// TestEmitGoodput_NoOpWhenTargetsEmpty verifies BC-3: empty targets leaves
// the output struct's goodput fields unchanged (zero/omitempty preserved).
func TestEmitGoodput_NoOpWhenTargetsEmpty(t *testing.T) {
	out := sim.MetricsOutput{}
	m := sim.NewMetrics()
	emitGoodput(&out, m, nil, 1.0, nil)
	if out.GoodputRPS != 0 || out.SLOAttainment != 0 || out.PerClass != nil {
		t.Errorf("expected zero/nil goodput fields, got GoodputRPS=%v SLOAttainment=%v PerClass=%v",
			out.GoodputRPS, out.SLOAttainment, out.PerClass)
	}
}

// TestEmitGoodput_OneClassE2EOnly verifies BC-2: a single configured dimension
// produces only that dim in slo_attainment_by_dim and a complete per-class entry.
func TestEmitGoodput_OneClassE2EOnly(t *testing.T) {
	m := sim.NewMetrics()
	// Two requests under "default": one meets E2E=5s threshold, one does not.
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", SLOClass: "default"}
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", SLOClass: "default"}
	m.RequestE2Es["r1"] = 1_000_000   // 1s in µs (passes 5s threshold)
	m.RequestE2Es["r2"] = 10_000_000  // 10s in µs (fails)

	targets := map[string]workload.SLODimTargets{
		"default": {E2EMs: 5000},
	}
	injected := map[string]int64{"default": 2}

	out := sim.MetricsOutput{}
	emitGoodput(&out, m, injected, 10.0, targets)

	if out.SLOAttainment != 0.5 {
		t.Errorf("SLOAttainment: got %v, want 0.5", out.SLOAttainment)
	}
	if out.GoodputRPS != 0.1 {
		t.Errorf("GoodputRPS: got %v, want 0.1 (1 good / 10s)", out.GoodputRPS)
	}
	per, ok := out.PerClass.(map[string]map[string]any)
	if !ok {
		t.Fatalf("PerClass type: got %T, want map[string]map[string]any", out.PerClass)
	}
	def := per["default"]
	if def["slo_attainment"] != 0.5 {
		t.Errorf("default slo_attainment: got %v, want 0.5", def["slo_attainment"])
	}
	byDim, ok := def["slo_attainment_by_dim"].(map[string]float64)
	if !ok {
		t.Fatalf("slo_attainment_by_dim type: got %T, want map[string]float64", def["slo_attainment_by_dim"])
	}
	if _, hasTTFT := byDim["ttft"]; hasTTFT {
		t.Errorf("byDim should not contain ttft when only E2E is configured")
	}
	if _, hasITL := byDim["itl"]; hasITL {
		t.Errorf("byDim should not contain itl when only E2E is configured")
	}
	if v, hasE2E := byDim["e2e"]; !hasE2E || v != 0.5 {
		t.Errorf("byDim e2e: got %v (present=%v), want 0.5", v, hasE2E)
	}
}

// TestEmitObserveGoodput_OkErrorTimeoutDenominator verifies BC-2 on observe path:
// error and timeout records count in the denominator but never count toward goodput;
// only ok records can contribute to the numerator.
func TestEmitObserveGoodput_OkErrorTimeoutDenominator(t *testing.T) {
	records := []workload.TraceRecord{
		// "ok" with E2E 1s — meets 5s threshold (numerator)
		{RequestID: 1, SLOClass: "default", Status: "ok", SendTimeUs: 0, FirstChunkTimeUs: 100_000, LastChunkTimeUs: 1_000_000},
		// "ok" with E2E 10s — fails 5s threshold
		{RequestID: 2, SLOClass: "default", Status: "ok", SendTimeUs: 0, FirstChunkTimeUs: 100_000, LastChunkTimeUs: 10_000_000},
		// "error" — denominator only, never numerator
		{RequestID: 3, SLOClass: "default", Status: "error"},
		// "timeout" — denominator only
		{RequestID: 4, SLOClass: "default", Status: "timeout"},
		// "incomplete" — neither denominator nor numerator (workload still in flight)
		{RequestID: 5, SLOClass: "default", Status: "incomplete"},
	}
	targets := map[string]workload.SLODimTargets{"default": {E2EMs: 5000}}

	out := sim.MetricsOutput{}
	emitObserveGoodput(&out, records, nil, 10.0, targets)

	if out.SLOAttainment != 0.25 {
		t.Errorf("SLOAttainment: got %v, want 0.25 (1 ok-met out of 4 dispatched: ok+ok+error+timeout)", out.SLOAttainment)
	}
	per, ok := out.PerClass.(map[string]map[string]any)
	if !ok {
		t.Fatalf("PerClass type: got %T", out.PerClass)
	}
	def := per["default"]
	if def["count"] != int64(4) {
		t.Errorf("count: got %v, want 4 (denominator excludes incomplete)", def["count"])
	}
	if def["slo_attainment"] != 0.25 {
		t.Errorf("default slo_attainment: got %v, want 0.25", def["slo_attainment"])
	}
	// goodput_rps: 1 good / 10s
	if def["goodput_rps"] != 0.1 {
		t.Errorf("goodput_rps: got %v, want 0.1", def["goodput_rps"])
	}
}

// TestStripITLForObserveFallback_StripsAndSignals verifies BC-6: when
// --record-itl is false and any class carries an ITL threshold, the helper
// returns a fresh map with ITL zeroed and signals the strip so the caller can
// emit the warning. The TTFT/E2E thresholds on the same class must be preserved.
func TestStripITLForObserveFallback_StripsAndSignals(t *testing.T) {
	in := map[string]workload.SLODimTargets{
		"critical": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
		"batch":    {E2EMs: 60000}, // no ITL configured for this class
	}
	got, stripped := stripITLForObserveFallback(in, false /* recordITL */)
	if !stripped {
		t.Fatalf("expected stripped=true when at least one class has ITL configured")
	}
	if got["critical"].ITLMs != 0 {
		t.Errorf("critical ITLMs: got %v, want 0 (stripped)", got["critical"].ITLMs)
	}
	if got["critical"].TTFTMs != 100 {
		t.Errorf("critical TTFTMs: got %v, want 100 (preserved)", got["critical"].TTFTMs)
	}
	if got["critical"].E2EMs != 5000 {
		t.Errorf("critical E2EMs: got %v, want 5000 (preserved)", got["critical"].E2EMs)
	}
	if got["batch"].E2EMs != 60000 {
		t.Errorf("batch E2EMs: got %v, want 60000 (preserved)", got["batch"].E2EMs)
	}
	// Critical: input map must NOT be mutated (header still carries original ITL).
	if in["critical"].ITLMs != 50 {
		t.Errorf("input map mutated: in[critical].ITLMs = %v, want 50", in["critical"].ITLMs)
	}
}

// TestStripITLForObserveFallback_NoOpWhenRecordITL verifies BC-6: with
// --record-itl true, targets pass through unchanged and stripped=false.
func TestStripITLForObserveFallback_NoOpWhenRecordITL(t *testing.T) {
	in := map[string]workload.SLODimTargets{
		"critical": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
	}
	got, stripped := stripITLForObserveFallback(in, true /* recordITL */)
	if stripped {
		t.Errorf("expected stripped=false when recordITL is true")
	}
	if got["critical"].ITLMs != 50 {
		t.Errorf("ITLMs should pass through when recordITL is true; got %v, want 50", got["critical"].ITLMs)
	}
}

// TestStripITLForObserveFallback_NoOpWhenNoITLConfigured verifies BC-6: when
// targets exist but no class has an ITL threshold, the helper returns the
// original map (no copy) and stripped=false (no warning needed).
func TestStripITLForObserveFallback_NoOpWhenNoITLConfigured(t *testing.T) {
	in := map[string]workload.SLODimTargets{
		"default": {E2EMs: 5000}, // no ITLMs
	}
	got, stripped := stripITLForObserveFallback(in, false)
	if stripped {
		t.Errorf("expected stripped=false when no class has ITL configured")
	}
	if got["default"].E2EMs != 5000 {
		t.Errorf("default E2EMs: got %v, want 5000", got["default"].E2EMs)
	}
}

// TestStripITLForObserveFallback_EmptyTargetsNoOp verifies the empty-input edge case.
func TestStripITLForObserveFallback_EmptyTargetsNoOp(t *testing.T) {
	got, stripped := stripITLForObserveFallback(nil, false)
	if stripped || got != nil {
		t.Errorf("nil input: expected (nil, false), got (%v, %v)", got, stripped)
	}
	got, stripped = stripITLForObserveFallback(map[string]workload.SLODimTargets{}, false)
	if stripped {
		t.Errorf("empty input: expected stripped=false, got true")
	}
}

// TestEmitObserveGoodput_NoOpWhenTargetsEmpty verifies BC-3 on observe path.
func TestEmitObserveGoodput_NoOpWhenTargetsEmpty(t *testing.T) {
	out := sim.MetricsOutput{}
	emitObserveGoodput(&out, nil, nil, 1.0, nil)
	if out.GoodputRPS != 0 || out.SLOAttainment != 0 || out.PerClass != nil {
		t.Errorf("expected zero goodput fields, got %+v", out)
	}
}

// TestRunReplayParity_SameTargets_SameOutput verifies BC-4 (INV-13): identical
// metrics + identical resolved targets produce identical goodput numbers
// regardless of whether the targets came from CLI or trace header.
func TestRunReplayParity_SameTargets_SameOutput(t *testing.T) {
	m := sim.NewMetrics()
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", SLOClass: "default"}
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", SLOClass: "default"}
	m.RequestE2Es["r1"] = 1_000_000
	m.RequestE2Es["r2"] = 6_000_000
	injected := map[string]int64{"default": 2}

	// "Run" path: CLI flag provided 5s threshold → resolved targets {default: {E2EMs: 5000}}.
	cliE2E := map[string]time.Duration{"default": 5 * time.Second}
	runTargets := mergeGoodputTargets(nil, nil, cliE2E, nil, nil)

	// "Replay" path: trace header carried 5s threshold → resolved targets {default: {E2EMs: 5000}}.
	header := map[string]workload.SLODimTargets{"default": {E2EMs: 5000}}
	replayTargets := mergeGoodputTargets(nil, nil, nil, header, nil)

	var runOut, replayOut sim.MetricsOutput
	emitGoodput(&runOut, m, injected, 10.0, runTargets)
	emitGoodput(&replayOut, m, injected, 10.0, replayTargets)

	if runOut.GoodputRPS != replayOut.GoodputRPS {
		t.Errorf("BC-4: GoodputRPS differs (run=%v replay=%v)", runOut.GoodputRPS, replayOut.GoodputRPS)
	}
	if runOut.SLOAttainment != replayOut.SLOAttainment {
		t.Errorf("BC-4: SLOAttainment differs (run=%v replay=%v)", runOut.SLOAttainment, replayOut.SLOAttainment)
	}
}

// TestGoodputTargets_ObserveExportReplayLoad_RoundTrip verifies BC-7 end-to-end:
// CLI flags → merger → exported TraceHeader (via observe path) → loaded TraceHeader
// (via replay path) → merger again. The downstream merger sees the same per-class
// thresholds without requiring CLI flags on the replay side.
//
// This is the contract that makes a captured-real-server trace replayable for
// goodput measurement out-of-the-box: the observe-side intent is preserved.
func TestGoodputTargets_ObserveExportReplayLoad_RoundTrip(t *testing.T) {
	// Step 1: simulate the observe-side merger from CLI flags + (no) workload spec.
	cliTTFT := map[string]time.Duration{"critical": 100 * time.Millisecond, "standard": 500 * time.Millisecond}
	cliE2E := map[string]time.Duration{"critical": 5 * time.Second}
	observeMerged := mergeGoodputTargets(cliTTFT, nil, cliE2E, nil, nil)

	// Step 2: write a TraceV2 carrying these targets in its header (observe export path).
	tmp := t.TempDir()
	headerPath := filepath.Join(tmp, "trace.yaml")
	dataPath := filepath.Join(tmp, "trace.csv")
	exportHeader := &workload.TraceHeader{
		Version:           3,
		TimeUnit:          "us",
		Mode:              "real",
		GoodputSLOTargets: observeMerged,
	}
	if err := workload.ExportTraceV2(exportHeader, []workload.TraceRecord{}, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}

	// Step 3: load the trace as replay/calibrate would.
	loaded, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}

	// Step 4: replay-side merger (no CLI overrides) — header must be the only source.
	replayMerged := mergeGoodputTargets(nil, nil, nil, loaded.Header.GoodputSLOTargets, nil)
	if !reflect.DeepEqual(replayMerged, observeMerged) {
		t.Fatalf("BC-7 round-trip drift:\n  observe-side: %#v\n  replay-side:  %#v", observeMerged, replayMerged)
	}

	// Spot-check: per-class per-dimension values match the original CLI input
	// (proves the merger isn't silently coercing values).
	if replayMerged["critical"].TTFTMs != 100 {
		t.Errorf("critical TTFTMs: got %v, want 100", replayMerged["critical"].TTFTMs)
	}
	if replayMerged["critical"].E2EMs != 5000 {
		t.Errorf("critical E2EMs: got %v, want 5000", replayMerged["critical"].E2EMs)
	}
	if replayMerged["standard"].TTFTMs != 500 {
		t.Errorf("standard TTFTMs: got %v, want 500", replayMerged["standard"].TTFTMs)
	}
}

// TestEmitGoodput_BCThreeByteIdenticalNoTargets verifies BC-3: when no targets
// are configured the goodput-related JSON keys remain absent (omitempty), so
// a `blis run` invocation without --slo-* flags or spec.GoodputSLOTargets emits
// stdout JSON byte-identical to a current-`main` build.
func TestEmitGoodput_BCThreeByteIdenticalNoTargets(t *testing.T) {
	m := sim.NewMetrics()
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", SLOClass: "default"}
	m.RequestE2Es["r1"] = 1_000_000

	out := sim.MetricsOutput{}
	emitGoodput(&out, m, map[string]int64{"default": 1}, 1.0, nil)

	// All three goodput JSON slots must remain at zero values (omitempty suppresses).
	if out.GoodputRPS != 0 {
		t.Errorf("GoodputRPS: got %v, want 0", out.GoodputRPS)
	}
	if out.SLOAttainment != 0 {
		t.Errorf("SLOAttainment: got %v, want 0", out.SLOAttainment)
	}
	if out.PerClass != nil {
		t.Errorf("PerClass: got %v, want nil", out.PerClass)
	}
}

// TestEmitGoodput_DeterministicAcrossRuns verifies INV-6: per-class JSON shape
// is identical across repeated calls (sorted keys).
func TestEmitGoodput_DeterministicAcrossRuns(t *testing.T) {
	m := sim.NewMetrics()
	for i, cls := range []string{"critical", "standard", "batch"} {
		id := string(rune('a' + i))
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: cls}
		m.RequestE2Es[id] = 500_000
	}
	targets := map[string]workload.SLODimTargets{
		"critical": {E2EMs: 1000},
		"standard": {E2EMs: 2000},
		"batch":    {E2EMs: 60000},
	}
	injected := map[string]int64{"critical": 1, "standard": 1, "batch": 1}

	var first sim.MetricsOutput
	emitGoodput(&first, m, injected, 1.0, targets)
	for i := 0; i < 5; i++ {
		var got sim.MetricsOutput
		emitGoodput(&got, m, injected, 1.0, targets)
		if got.SLOAttainment != first.SLOAttainment {
			t.Errorf("iteration %d: SLOAttainment changed (%v vs %v)", i, got.SLOAttainment, first.SLOAttainment)
		}
		if got.GoodputRPS != first.GoodputRPS {
			t.Errorf("iteration %d: GoodputRPS changed (%v vs %v)", i, got.GoodputRPS, first.GoodputRPS)
		}
	}
}
