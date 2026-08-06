// sim/saturation/bank_test.go
package saturation

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// countingDetector is a test spy that streams like a real detector but only
// counts Observe/Detect/Reset calls. It exists to prove the bank actually drives
// each detector (vacuous-pass guard, #1519).
type countingDetector struct {
	name     string
	observed int
	detected int
	resets   int
}

func (c *countingDetector) Name() string   { return c.name }
func (c *countingDetector) Observe(Event)  { c.observed++ }
func (c *countingDetector) Detect() Result { c.detected++; return Result{} }
func (c *countingDetector) Reset()         { c.resets++ }

func threeRequests() []sim.RequestMetrics {
	return []sim.RequestMetrics{
		{ID: "request_1", ArrivedAt: 2.0, E2E: 100}, // out of order to exercise sort
		{ID: "request_0", ArrivedAt: 1.0, E2E: 100},
		{ID: "request_2", ArrivedAt: 0.5, E2E: 50},
	}
}

// TestBank_ClassifyActuallyReplaysEvents is the vacuous-pass guard the issue
// mandates: a streaming detector in the bank must observe exactly 2*N events
// (one arrival + one completion per request) — proving Classify really replays
// rather than silently no-op'ing. It also checks Reset ran (fresh state) and one
// Detect per Observe (one verdict per event).
func TestBank_ClassifyActuallyReplaysEvents(t *testing.T) {
	spy := &countingDetector{name: "composite"}
	bank := &Bank{detectors: []Detector{spy}, sink: NewNoOpSink()}

	reqs := threeRequests()
	bank.Classify(reqs, len(reqs))

	wantEvents := 2 * len(reqs)
	if spy.observed != wantEvents {
		t.Errorf("detector observed %d events, want %d (2 per request)", spy.observed, wantEvents)
	}
	if spy.detected != wantEvents {
		t.Errorf("detector produced %d verdicts, want %d (one per observed event)", spy.detected, wantEvents)
	}
	if spy.resets != 1 {
		t.Errorf("detector Reset called %d times, want exactly 1", spy.resets)
	}
}

// TestNewBank_UnknownNameErrors verifies an unknown name is a hard error naming
// it and listing the valid roster (R1), never a silent drop.
func TestNewBank_UnknownNameErrors(t *testing.T) {
	_, err := NewBank([]string{"composite", "bogus"}, SaturationConfig{}, NewNoOpSink())
	if err == nil {
		t.Fatal("expected error for unknown detector name")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error should name the offending detector, got: %v", err)
	}
	for _, name := range rosterOrder {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error should list valid detector %q, got: %v", name, err)
		}
	}
}

// TestNewBank_EmptySelectionErrors verifies the bank refuses to run with no
// detectors — it must drive at least one.
func TestNewBank_EmptySelectionErrors(t *testing.T) {
	if _, err := NewBank(nil, SaturationConfig{}, NewNoOpSink()); err == nil {
		t.Error("expected error for empty detector selection")
	}
}

// TestNewBank_CanonicalOrderAndDedup verifies the roster is re-ordered into
// canonical order and de-duplicated regardless of CLI argument order (INV-6).
func TestNewBank_CanonicalOrderAndDedup(t *testing.T) {
	// Supplied reversed and with a duplicate.
	bank, err := NewBank([]string{"backlog-drift", "composite", "composite"}, SaturationConfig{}, NewNoOpSink())
	if err != nil {
		t.Fatalf("NewBank: %v", err)
	}
	got := make([]string, 0, len(bank.detectors))
	for _, d := range bank.detectors {
		got = append(got, d.Name())
	}
	want := []string{"composite", "backlog-drift"} // canonical order, dup removed
	if len(got) != len(want) {
		t.Fatalf("roster = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("roster[%d] = %q, want %q (canonical order)", i, got[i], want[i])
		}
	}
}

// TestBank_AllEqualsExplicitList verifies `all` (AllDetectorNames) produces a
// byte-identical trace to the explicit full comma-list (INV-6): selection order
// never changes how detectors see traffic.
func TestBank_AllEqualsExplicitList(t *testing.T) {
	reqs := threeRequests()

	run := func(names []string) []TraceRecord {
		c := NewInMemoryCollector()
		bank, err := NewBank(names, SaturationConfig{}, c)
		if err != nil {
			t.Fatalf("NewBank(%v): %v", names, err)
		}
		bank.Classify(reqs, len(reqs))
		bank.Close()
		return c.Records()
	}

	all := run(AllDetectorNames())
	explicit := run([]string{"threshold", "backlog-drift", "composite"}) // scrambled order

	assertRecordsEqual(t, all, explicit)
}

// TestBank_SubsetMatchesRecordsUnderAll verifies a subset detector's records are
// byte-identical to the same detector's records under `all` — selection filters
// WHICH detectors run, never HOW they see traffic (INV-6 / INV-13).
func TestBank_SubsetMatchesRecordsUnderAll(t *testing.T) {
	reqs := threeRequests()

	collectAll := NewInMemoryCollector()
	bankAll, err := NewBank(AllDetectorNames(), SaturationConfig{}, collectAll)
	if err != nil {
		t.Fatalf("NewBank(all): %v", err)
	}
	bankAll.Classify(reqs, len(reqs))
	bankAll.Close()

	collectSubset := NewInMemoryCollector()
	bankSubset, err := NewBank([]string{"composite"}, SaturationConfig{}, collectSubset)
	if err != nil {
		t.Fatalf("NewBank(composite): %v", err)
	}
	bankSubset.Classify(reqs, len(reqs))
	bankSubset.Close()

	// Filter the `all` trace down to composite records and compare.
	compositeUnderAll := make([]TraceRecord, 0)
	for _, r := range collectAll.Records() {
		if r.Detector == "composite" {
			compositeUnderAll = append(compositeUnderAll, r)
		}
	}
	assertRecordsEqual(t, compositeUnderAll, collectSubset.Records())
}

// TestBank_Deterministic verifies two identical runs yield byte-identical
// records (INV-6).
func TestBank_Deterministic(t *testing.T) {
	reqs := threeRequests()
	run := func() []TraceRecord {
		c := NewInMemoryCollector()
		bank, err := NewBank(AllDetectorNames(), SaturationConfig{}, c)
		if err != nil {
			t.Fatalf("NewBank: %v", err)
		}
		bank.Classify(reqs, len(reqs))
		bank.Close()
		return c.Records()
	}
	assertRecordsEqual(t, run(), run())
}

// TestNewBank_ForeignBlockIgnored verifies a config block for a detector NOT in
// the selection is ignored (not an error) — several detectors share one config.
// A threshold: block is present but only composite is selected.
func TestNewBank_ForeignBlockIgnored(t *testing.T) {
	thr := 1234.0
	cfg := SaturationConfig{Threshold: &ThresholdBlock{ThresholdMs: &thr}}
	if _, err := NewBank([]string{"composite"}, cfg, NewNoOpSink()); err != nil {
		t.Errorf("threshold block should be ignored when composite-only, got: %v", err)
	}
}

// TestNewBank_SelectedBlockValueErrorSurfaces verifies an out-of-range value in a
// SELECTED detector's own block still errors (never panics — R6).
func TestNewBank_SelectedBlockValueErrorSurfaces(t *testing.T) {
	bad := -1.0
	cfg := SaturationConfig{Threshold: &ThresholdBlock{ThresholdMs: &bad}}
	if _, err := NewBank([]string{"threshold"}, cfg, NewNoOpSink()); err == nil {
		t.Error("expected error for negative threshold_ms in a selected detector")
	}
}

// assertRecordsEqual fails the test if two record slices differ in any observable
// field (the fields WriteCombinedReport serializes).
func assertRecordsEqual(t *testing.T, a, b []TraceRecord) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("record count differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Timestamp != b[i].Timestamp || a[i].Detector != b[i].Detector ||
			a[i].Result.Level != b[i].Result.Level || a[i].Result.Score != b[i].Result.Score ||
			a[i].Result.Confidence != b[i].Result.Confidence {
			t.Errorf("record %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}
