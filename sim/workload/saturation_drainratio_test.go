// sim/workload/saturation_drainratio_test.go
//
// Behavioral tests for drainRatioClassifier (#1392). These complement the
// classifier-agnostic property tests in saturation_properties_test.go by
// pinning specific outputs for canonical inputs.
package workload

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	sim "blis/sim"
)

// makeWindow constructs a WindowMetrics with the given counts and a non-NaN DrainRatio.
// Used by the table-driven tests below.
func makeWindow(start, end int64, entered, left int) WindowMetrics {
	w := WindowMetrics{
		StartUs:    start,
		EndUs:      end,
		NumEntered: entered,
		NumLeft:    left,
	}
	if entered > 0 {
		w.DrainRatio = float64(left) / float64(entered)
	}
	return w
}

func TestDrainRatioClassifier_Unsaturated(t *testing.T) {
	// GIVEN steady-state windows with DrainRatio ≈ 1.0 (engine drains all arrivals)
	// WHEN drain-ratio classifier runs
	// THEN classifies UNSATURATED with low ρ.
	cfg := DefaultBacklogDriftConfig()
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 50),  // warmup window 0
		makeWindow(10_000_000, 20_000_000, 100, 100), // warmup window 1
		makeWindow(20_000_000, 30_000_000, 100, 100), // steady
		makeWindow(30_000_000, 40_000_000, 100, 100), // steady
		makeWindow(40_000_000, 50_000_000, 100, 100), // steady
	}
	c := drainRatioClassifier{}
	classification, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if classification != "UNSATURATED" {
		t.Errorf("Expected UNSATURATED, got %s. Note: %s", classification, note)
	}
}

func TestDrainRatioClassifier_PersistentlySaturated(t *testing.T) {
	// GIVEN steady-state windows with DrainRatio = 0.866 (μ/λ ≈ 0.87, ρ ≈ 1.15)
	// WHEN drain-ratio classifier runs
	// THEN classifies PERSISTENTLY_SATURATED with ρ ≈ 1.15 in note.
	cfg := DefaultBacklogDriftConfig()
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 800, 500),         // warmup
		makeWindow(10_000_000, 20_000_000, 800, 680), // warmup
		makeWindow(20_000_000, 30_000_000, 800, 692), // steady (DrainRatio=0.865)
		makeWindow(30_000_000, 40_000_000, 800, 692),
		makeWindow(40_000_000, 50_000_000, 800, 692),
		makeWindow(50_000_000, 60_000_000, 800, 692),
	}
	c := drainRatioClassifier{}
	classification, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if classification != "PERSISTENTLY_SATURATED" {
		t.Errorf("Expected PERSISTENTLY_SATURATED, got %s. Note: %s", classification, note)
	}
	// Verify the note reports a ρ ≈ 1.15
	if !strings.Contains(note, "ρ") {
		t.Errorf("Expected ρ in note, got: %s", note)
	}
}

func TestDrainRatioClassifier_TransientBacklog(t *testing.T) {
	// GIVEN steady-state windows with DrainRatio just under TransientDrainRatio (0.96 < 0.98)
	// WHEN drain-ratio classifier runs
	// THEN classifies TRANSIENT_BACKLOG.
	cfg := DefaultBacklogDriftConfig() // SaturatedDrainRatio=0.95, TransientDrainRatio=0.98
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 50),  // warmup
		makeWindow(10_000_000, 20_000_000, 100, 96), // warmup
		makeWindow(20_000_000, 30_000_000, 100, 96), // steady DrainRatio=0.96
		makeWindow(30_000_000, 40_000_000, 100, 96),
		makeWindow(40_000_000, 50_000_000, 100, 96),
	}
	c := drainRatioClassifier{}
	classification, _, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if classification != "TRANSIENT_BACKLOG" {
		t.Errorf("Expected TRANSIENT_BACKLOG, got %s", classification)
	}
}

func TestDrainRatioClassifier_NoArrivals_Unsaturated(t *testing.T) {
	// GIVEN windows where every NumEntered == 0 (degenerate — no workload)
	// WHEN drain-ratio classifier runs
	// THEN classifies UNSATURATED with explanatory note.
	cfg := DefaultBacklogDriftConfig()
	windows := []WindowMetrics{
		{StartUs: 0, EndUs: 10_000_000, NumEntered: 0, NumLeft: 0},
		{StartUs: 10_000_000, EndUs: 20_000_000, NumEntered: 0, NumLeft: 0},
	}
	c := drainRatioClassifier{}
	classification, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if classification != "UNSATURATED" {
		t.Errorf("Expected UNSATURATED with no arrivals, got %s", classification)
	}
	if !strings.Contains(note, "No arrivals") {
		t.Errorf("Expected 'No arrivals' in note, got: %s", note)
	}
}

func TestDrainRatioClassifier_AllWarmup_Unsaturated(t *testing.T) {
	// GIVEN fewer inject windows than WarmupWindows
	// WHEN drain-ratio classifier runs
	// THEN classifies UNSATURATED with insufficient-data note.
	cfg := DefaultBacklogDriftConfig() // WarmupWindows=2
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 50),
		makeWindow(10_000_000, 20_000_000, 100, 100),
	}
	c := drainRatioClassifier{}
	classification, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if classification != "UNSATURATED" {
		t.Errorf("Expected UNSATURATED with all-warmup, got %s", classification)
	}
	if !strings.Contains(note, "Insufficient") {
		t.Errorf("Expected 'Insufficient' in note, got: %s", note)
	}
}

func TestDrainRatioClassifier_LastArrivalWindow_TrailingDrainExcluded(t *testing.T) {
	// GIVEN windows where the last 3 have NumEntered=0 (drain phase)
	// WHEN drain-ratio classifier runs
	// THEN drain windows are excluded; only inject windows contribute to mean.
	// Verifies the last_arrival_window predicate.
	cfg := DefaultBacklogDriftConfig()
	cfg.WarmupWindows = 0 // skip warmup so we test the inject-phase truncation directly
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 50),  // inject DrainRatio=0.50
		makeWindow(10_000_000, 20_000_000, 100, 100), // inject DrainRatio=1.0
		// drain phase (NumEntered=0)
		{StartUs: 20_000_000, EndUs: 30_000_000, NumEntered: 0, NumLeft: 80},
		{StartUs: 30_000_000, EndUs: 40_000_000, NumEntered: 0, NumLeft: 70},
		{StartUs: 40_000_000, EndUs: 50_000_000, NumEntered: 0, NumLeft: 0},
	}
	c := drainRatioClassifier{}
	classification, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	// Mean DrainRatio over inject = (0.50 + 1.0) / 2 = 0.75 → PERSISTENTLY_SATURATED
	if classification != "PERSISTENTLY_SATURATED" {
		t.Errorf("Expected PERSISTENTLY_SATURATED (drain excluded → mean=0.75), got %s. Note: %s", classification, note)
	}
}

func TestDrainRatioClassifier_InteriorEmptyWindow_RemainsInject(t *testing.T) {
	// GIVEN windows with an interior NumEntered=0 (closed-loop think-time gap)
	// followed by more arrivals
	// WHEN drain-ratio classifier runs
	// THEN the interior empty window is part of inject phase (last_arrival_window
	// predicate), and is skipped only because DrainRatio is NaN — the surrounding
	// windows still contribute correctly.
	cfg := DefaultBacklogDriftConfig()
	cfg.WarmupWindows = 0
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 95),       // DrainRatio=0.95
		// interior empty window — DrainRatio NaN matches what computeWindowMetrics sets when NumEntered==0
		{StartUs: 10_000_000, EndUs: 20_000_000, NumEntered: 0, NumLeft: 5, DrainRatio: math.NaN()},
		makeWindow(20_000_000, 30_000_000, 100, 95), // DrainRatio=0.95
	}
	c := drainRatioClassifier{}
	classification, _, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	// Mean over n=2 valid windows = 0.95; threshold for PERSISTENTLY_SATURATED is < 0.95.
	// 0.95 is NOT < 0.95, so → TRANSIENT_BACKLOG (mean < TransientDrainRatio=0.98).
	if classification != "TRANSIENT_BACKLOG" {
		t.Errorf("Expected TRANSIENT_BACKLOG (interior empty window doesn't truncate inject phase), got %s", classification)
	}
}

func TestNewBacklogClassifier_DefaultIsDrainRatio(t *testing.T) {
	// BC-5 (#1392): empty-string name and "drain-ratio" both return drainRatioClassifier.
	c1 := NewBacklogClassifier("")
	c2 := NewBacklogClassifier("drain-ratio")
	if _, ok := c1.(drainRatioClassifier); !ok {
		t.Errorf("Empty string should default to drainRatioClassifier, got %T", c1)
	}
	if _, ok := c2.(drainRatioClassifier); !ok {
		t.Errorf("drain-ratio should return drainRatioClassifier, got %T", c2)
	}
}

func TestNewBacklogClassifier_SlopeBased(t *testing.T) {
	// BC-3 (#1391): "slope-based" returns slopeBasedClassifier (preserved opt-in).
	c := NewBacklogClassifier("slope-based")
	if _, ok := c.(slopeBasedClassifier); !ok {
		t.Errorf("slope-based should return slopeBasedClassifier, got %T", c)
	}
}

func TestNewBacklogClassifier_PanicsOnUnknown(t *testing.T) {
	// BC-3 (#1391): unknown name panics. CLI is expected to validate upstream.
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for unknown classifier name")
		}
	}()
	_ = NewBacklogClassifier("nonexistent-classifier")
}

func TestBacklogDriftConfig_NewBacklogDriftConfig_ValidatesDrainRatioRange(t *testing.T) {
	// GIVEN SaturatedDrainRatio > TransientDrainRatio (regions would overlap)
	// WHEN constructing config
	// THEN panics with descriptive message.
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for SaturatedDrainRatio > TransientDrainRatio")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.99, 0.95)
}

// TestBacklogDriftConfig_NewBacklogDriftConfig_DrainRatioParamValidation table-tests
// every panic branch for the new drain-ratio classifier knobs (#1392). Each row
// constructs an otherwise-valid config with one parameter forced into an invalid
// value and asserts the constructor panics with a descriptive message.
func TestBacklogDriftConfig_NewBacklogDriftConfig_DrainRatioParamValidation(t *testing.T) {
	cases := []struct {
		name           string
		warmup, tail   int
		sat, transient float64
		wantSubstr     string
	}{
		{"warmup_negative", -1, 1, 0.95, 0.98, "WarmupWindows must be >= 0"},
		{"tail_negative", 2, -1, 0.95, 0.98, "TailWindows must be >= 0"},
		{"saturated_zero", 2, 1, 0, 0.98, "SaturatedDrainRatio must be in (0, 1]"},
		{"saturated_negative", 2, 1, -0.5, 0.98, "SaturatedDrainRatio must be in (0, 1]"},
		{"saturated_above_one", 2, 1, 1.5, 1.6, "SaturatedDrainRatio must be in (0, 1]"},
		{"saturated_nan", 2, 1, math.NaN(), 0.98, "SaturatedDrainRatio must be in (0, 1]"},
		{"saturated_inf", 2, 1, math.Inf(1), 0.98, "SaturatedDrainRatio must be in (0, 1]"},
		{"transient_zero", 2, 1, 0.95, 0, "TransientDrainRatio must be in (0, 1]"},
		{"transient_above_one", 2, 1, 0.95, 1.2, "TransientDrainRatio must be in (0, 1]"},
		{"transient_nan", 2, 1, 0.95, math.NaN(), "TransientDrainRatio must be in (0, 1]"},
		{"saturated_gt_transient", 2, 1, 0.99, 0.95, "SaturatedDrainRatio"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("Expected panic for %s", tc.name)
				}
				msg := fmt.Sprint(r)
				if !strings.Contains(msg, tc.wantSubstr) {
					t.Errorf("Panic message %q did not contain %q", msg, tc.wantSubstr)
				}
			}()
			_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, tc.warmup, tc.tail, tc.sat, tc.transient)
		})
	}
}

// TestNewBacklogClassifier_FactoryRegistryAgreement enforces that every name in
// the validBacklogClassifiers registry is constructible via NewBacklogClassifier.
// Catches drift where a name is added to the registry but the factory's switch
// statement isn't updated (or vice versa).
func TestNewBacklogClassifier_FactoryRegistryAgreement(t *testing.T) {
	for _, name := range sim.ValidBacklogClassifierNames() {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("NewBacklogClassifier(%q) panicked: %v", name, r)
				}
			}()
			c := NewBacklogClassifier(name)
			if c == nil {
				t.Fatalf("NewBacklogClassifier(%q) returned nil", name)
			}
		})
	}
}

// TestAnalyzeBacklogDriftWithClassifier_NilClassifier_DefaultsToSlopeBased verifies
// the nil-defense at the orchestrator level. Calling with classifier=nil should
// not panic and should produce a report identical to passing slopeBasedClassifier{}
// explicitly (preserves the documented backward-compat shim semantics).
func TestAnalyzeBacklogDriftWithClassifier_NilClassifier_DefaultsToSlopeBased(t *testing.T) {
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	requests := []*sim.Request{
		{ArrivalTime: 0, FirstTokenTime: 100, ITL: []int64{500}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 5_000_000, FirstTokenTime: 100, ITL: []int64{500}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 10_000_000, FirstTokenTime: 100, ITL: []int64{500}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 15_000_000, FirstTokenTime: 100, ITL: []int64{500}, TTFTSet: true, State: sim.StateCompleted},
	}
	simEndUs := int64(20_000_000)

	rNil := AnalyzeBacklogDriftWithClassifier(requests, simEndUs, cfg, nil)
	rExplicit := AnalyzeBacklogDriftWithClassifier(requests, simEndUs, cfg, slopeBasedClassifier{})

	if rNil.Classification != rExplicit.Classification {
		t.Errorf("Nil classifier produced %s, explicit slope-based produced %s — defense regression",
			rNil.Classification, rExplicit.Classification)
	}
	if rNil.Note != rExplicit.Note {
		t.Errorf("Nil classifier note differs from explicit slope-based note (defense regression)")
	}
}

// TestIsValidBacklogClassifier_RegistryContents asserts the public registry
// contract used by all three CLI commands (run/replay/observe) for upstream
// validation before reaching the panic-style factory.
func TestIsValidBacklogClassifier_RegistryContents(t *testing.T) {
	tests := []struct {
		name    string
		isValid bool
	}{
		{"", true},
		{"drain-ratio", true},
		{"slope-based", true},
		{"unknown", false},
		{"DrainRatio", false}, // case-sensitive
		{"drain_ratio", false}, // underscore not hyphen
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := sim.IsValidBacklogClassifier(tc.name); got != tc.isValid {
				t.Errorf("IsValidBacklogClassifier(%q) = %v, want %v", tc.name, got, tc.isValid)
			}
		})
	}

	// ValidBacklogClassifierNames excludes the empty string and is sorted.
	names := sim.ValidBacklogClassifierNames()
	for _, name := range names {
		if name == "" {
			t.Errorf("ValidBacklogClassifierNames() returned empty-string entry")
		}
	}
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Errorf("ValidBacklogClassifierNames() not sorted: %v", names)
			break
		}
	}
}

// TestAnalyzeBacklogDriftWithClassifier_OrchestratorFallback_AllInjectWindows verifies
// the orchestrator-level fallback path: when WarmupWindows + TailWindows >= number of
// inject windows, the slope-regression sample set falls back to all inject windows
// (with a logrus.Warnf surfacing the bias). The drain-ratio classifier handles the
// same regime by returning UNSATURATED with an explicit note. This test asserts both
// classifiers handle the regime without panic and produce a defensible verdict.
func TestAnalyzeBacklogDriftWithClassifier_OrchestratorFallback_AllInjectWindows(t *testing.T) {
	// Synthetic input: 4 inject windows, 0 drain. Engine drains at 100/s, arrivals at 50/s.
	// → ρ = 0.5, all windows have NumLeft >= NumEntered.
	// With cfg.WarmupWindows=2 + cfg.TailWindows=2 = 4 (== injectWindows), trim collapses.
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3
	cfg.WarmupWindows = 2
	cfg.TailWindows = 2

	requests := []*sim.Request{
		{ArrivalTime: 0, FirstTokenTime: 100, ITL: []int64{1000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 5_000_000, FirstTokenTime: 100, ITL: []int64{1000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 10_000_000, FirstTokenTime: 100, ITL: []int64{1000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 15_000_000, FirstTokenTime: 100, ITL: []int64{1000}, TTFTSet: true, State: sim.StateCompleted},
	}
	simEndUs := int64(20_000_000)

	// Both classifiers should produce a verdict without panicking. The exact verdict can
	// differ between classifiers in this fallback regime — we only assert no-panic and
	// non-empty output, since the meaningful behavior is the orchestrator's logrus.Warnf.
	for _, name := range []string{"drain-ratio", "slope-based"} {
		t.Run(name, func(t *testing.T) {
			c := NewBacklogClassifier(name)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("classifier=%s panicked in orchestrator fallback regime: %v", name, r)
				}
			}()
			r := AnalyzeBacklogDriftWithClassifier(requests, simEndUs, cfg, c)
			if r.Classification == "" {
				t.Errorf("classifier=%s produced empty classification", name)
			}
			// At ρ=0.5 and warmup+tail collapse, neither classifier should fire PERSISTENTLY_SATURATED.
			if r.Classification == "PERSISTENTLY_SATURATED" {
				t.Errorf("classifier=%s incorrectly fired PERSISTENTLY_SATURATED at ρ=0.5: %s", name, r.Note)
			}
		})
	}
}

// TestDrainRatioClassifier_NaNSkipCount_SurfacesInNote verifies the post-review
// fix that the count of skipped NaN/Inf DrainRatio windows is reported in the
// classifier's note (R1: no silent discard). Asserts BC-1 behavior on closed-loop
// gaps that surface as NaN per WindowMetrics convention.
func TestDrainRatioClassifier_NaNSkipCount_SurfacesInNote(t *testing.T) {
	cfg := DefaultBacklogDriftConfig()
	cfg.WarmupWindows = 0
	cfg.TailWindows = 0
	windows := []WindowMetrics{
		makeWindow(0, 10_000_000, 100, 95),                                                        // valid
		{StartUs: 10_000_000, EndUs: 20_000_000, NumEntered: 0, NumLeft: 0, DrainRatio: math.NaN()}, // skipped
		makeWindow(20_000_000, 30_000_000, 100, 95),                                               // valid
		{StartUs: 30_000_000, EndUs: 40_000_000, NumEntered: 0, NumLeft: 0, DrainRatio: math.NaN()}, // skipped
		makeWindow(40_000_000, 50_000_000, 100, 95),                                               // valid
	}
	c := drainRatioClassifier{}
	_, note, _ := c.Classify(windows, SlopeStats{}, 0, 0, 0, 0, cfg)
	if !strings.Contains(note, "skipped 2 windows") {
		t.Errorf("Expected note to surface skip count of 2, got: %s", note)
	}
}
