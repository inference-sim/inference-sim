package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// resetSaturationGlobals clears the three shared saturation flag globals so each
// test starts from "off". Callers set what they need afterwards.
func resetSaturationGlobals() {
	detectorName = ""
	saturationConfigPath = ""
	saturationReport = ""
}

// twoRequests is the shared fixture for tracer trace() assertions.
func twoRequests() []sim.RequestMetrics {
	return []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 100},
		{ID: "request_1", ArrivedAt: 1, E2E: 200},
	}
}

// TestResolveSaturation_Off verifies that with no --detectors and no config/report,
// resolveSaturation returns a nil tracer — saturation is off.
func TestResolveSaturation_Off(t *testing.T) {
	resetSaturationGlobals()
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tracer != nil {
		t.Errorf("expected nil tracer when off, got %v", tracer)
	}
}

// TestResolveSaturation_ConfigOrReportWithoutDetectors verifies the hard-error
// contract: --saturation-config or --saturation-report given without --detectors.
func TestResolveSaturation_ConfigOrReportWithoutDetectors(t *testing.T) {
	t.Run("config without detectors", func(t *testing.T) {
		resetSaturationGlobals()
		saturationConfigPath = "some.yaml"
		if _, err := resolveSaturation(); err == nil || !strings.Contains(err.Error(), "--saturation-config requires --detectors") {
			t.Errorf("expected 'requires --detectors' error, got: %v", err)
		}
	})
	t.Run("report without detectors", func(t *testing.T) {
		resetSaturationGlobals()
		saturationReport = "some.json"
		if _, err := resolveSaturation(); err == nil || !strings.Contains(err.Error(), "--saturation-report requires --detectors") {
			t.Errorf("expected 'requires --detectors' error, got: %v", err)
		}
	})
}

// TestResolveSaturation_UnknownName verifies an unknown single name errors listing
// the valid names.
func TestResolveSaturation_UnknownName(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "bogus"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	_, err := resolveSaturation()
	if err == nil {
		t.Fatal("expected error for unknown detector name")
	}
	for _, name := range []string{"composite", "threshold", "backlog-drift"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error should list %q, got: %v", name, err)
		}
	}
}

// TestResolveSaturation_UnknownNameInList verifies an unknown name inside a
// comma-list is a hard error naming it (R1) — routed through the bank.
func TestResolveSaturation_UnknownNameInList(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite,bogus"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	_, err := resolveSaturation()
	if err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Errorf("expected error naming 'bogus', got: %v", err)
	}
}

// TestResolveSaturation_EmptySelection verifies a comma-only value (no real
// names) is a hard error rather than a silent off.
func TestResolveSaturation_EmptySelection(t *testing.T) {
	resetSaturationGlobals()
	detectorName = ", ,"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	if _, err := resolveSaturation(); err == nil {
		t.Error("expected error for a selection with no detector names")
	}
}

// TestResolveSaturation_UnwritableReportPath verifies the report path is validated
// up front (fast-fail before the run), for both single and bank selections.
func TestResolveSaturation_UnwritableReportPath(t *testing.T) {
	for _, sel := range []string{"composite", "all"} {
		resetSaturationGlobals()
		detectorName = sel
		saturationReport = filepath.Join(t.TempDir(), "nonexistent-dir", "x.json")
		if _, err := resolveSaturation(); err == nil {
			t.Errorf("detectors=%q: expected error for unwritable report path", sel)
		}
	}
}

// TestResolveSaturation_ValidSingleDetector verifies the single-detector happy
// path returns a tracer whose detector (not bank) is set.
func TestResolveSaturation_ValidSingleDetector(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tracer == nil || tracer.detector == nil || tracer.bank != nil {
		t.Fatalf("expected single-detector tracer, got %+v", tracer)
	}
	if tracer.detector.Name() != "composite" {
		t.Errorf("expected composite, got %q", tracer.detector.Name())
	}
}

// TestResolveSaturation_AllUsesBank verifies "all" routes through the bank.
func TestResolveSaturation_AllUsesBank(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "all"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tracer == nil || tracer.bank == nil || tracer.detector != nil {
		t.Fatalf("expected bank tracer for --detectors all, got %+v", tracer)
	}
}

// TestResolveSaturation_SubsetListUsesBank verifies a comma-list routes through
// the bank.
func TestResolveSaturation_SubsetListUsesBank(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite,threshold"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tracer == nil || tracer.bank == nil || tracer.detector != nil {
		t.Fatalf("expected bank tracer for a comma-list, got %+v", tracer)
	}
}

// TestResolveSaturation_SingleDetectorBlockOwnership verifies the single-detector
// path still enforces block↔detector ownership (a threshold: block errors when
// composite is selected) — #1516 behavior preserved.
func TestResolveSaturation_SingleDetectorBlockOwnership(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(cfgPath, []byte("threshold:\n  threshold_ms: 1234\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	resetSaturationGlobals()
	detectorName = "composite"
	saturationConfigPath = cfgPath
	saturationReport = filepath.Join(dir, "x.json")
	if _, err := resolveSaturation(); err == nil {
		t.Error("expected block-ownership error for threshold block with composite selected")
	}
}

// TestResolveSaturation_BankUnselectedBlockErrors verifies the bank path enforces
// block ownership over the SELECTED SET: a threshold: block errors when the
// selection excludes threshold, matching the single-detector path (R1 — no silent
// drop of user tuning). Here composite,backlog-drift omits threshold, so the
// threshold block is invalid.
func TestResolveSaturation_BankUnselectedBlockErrors(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(cfgPath, []byte("threshold:\n  threshold_ms: 1234\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	resetSaturationGlobals()
	detectorName = "composite,backlog-drift" // bank path; threshold NOT selected
	saturationConfigPath = cfgPath
	saturationReport = filepath.Join(dir, "x.json")
	if _, err := resolveSaturation(); err == nil {
		t.Error("expected error: threshold block supplied but threshold not among selected detectors")
	}
}

// TestResolveSaturation_BankTrailingCommaEnforcesOwnership verifies the trailing-
// comma edge case can no longer bypass ownership: `--detectors composite,` still
// routes through the bank, and a foreign threshold: block errors just as it does
// for the bare `--detectors composite` single-detector spelling.
func TestResolveSaturation_BankTrailingCommaEnforcesOwnership(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(cfgPath, []byte("threshold:\n  threshold_ms: 1234\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	resetSaturationGlobals()
	detectorName = "composite," // trailing comma → bank path, effective selection {composite}
	saturationConfigPath = cfgPath
	saturationReport = filepath.Join(dir, "x.json")
	if _, err := resolveSaturation(); err == nil {
		t.Error("expected error: trailing-comma selection must still enforce block ownership")
	}
}

// TestResolveSaturation_BankSelectedBlockAccepted verifies a block IS accepted
// when its owner is in the selection.
func TestResolveSaturation_BankSelectedBlockAccepted(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(cfgPath, []byte("threshold:\n  threshold_ms: 1234\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	resetSaturationGlobals()
	detectorName = "composite,threshold" // threshold selected → block valid
	saturationConfigPath = cfgPath
	saturationReport = filepath.Join(dir, "x.json")
	if _, err := resolveSaturation(); err != nil {
		t.Errorf("threshold block should be valid when threshold is selected, got: %v", err)
	}
}

// TestSaturationTracer_TraceNoOpWhenNoReport verifies trace() writes nothing when
// no report path is set (the trace would be discarded anyway).
func TestSaturationTracer_TraceNoOpWhenNoReport(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite" // triggers the no-report warning path
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation: %v", err)
	}
	if tracer == nil {
		t.Fatal("expected non-nil tracer")
	}
	if err := tracer.trace(twoRequests()); err != nil {
		t.Fatalf("trace: %v", err)
	}
	// saturationReport is "" — nothing to check on disk; the assertion is that no
	// error occurs and the call is inert.
}

// TestSaturationTracer_DecoupledFromGlobals is the regression guard for the
// field-capture refactor: once resolveSaturation returns, the tracer must be
// self-contained and NOT re-read the saturationReport/detectorName flag globals in
// trace(). We resolve, then CLEAR the globals, then trace — the trace must still be
// written to the report path captured at construction. If a future edit reverted
// trace() to read the globals, it would see "" and silently no-op, and this test
// would fail (no file written).
func TestSaturationTracer_DecoupledFromGlobals(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "all"
	capturedReport := filepath.Join(t.TempDir(), "captured.json")
	saturationReport = capturedReport
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation: %v", err)
	}

	// Simulate a future cobra-lifecycle change that clears the globals between
	// construction and use. The tracer must not depend on them any more.
	resetSaturationGlobals()

	if err := tracer.trace(twoRequests()); err != nil {
		t.Fatalf("trace after clearing globals: %v", err)
	}
	// The trace must have been written to the path captured at construction, even
	// though saturationReport is now "".
	report := readReport(t, capturedReport)
	if len(report.Trace) == 0 {
		t.Error("expected records written to the captured report path after globals were cleared; got empty trace (trace() may be re-reading the globals)")
	}
}

// TestSaturationTracer_SingleWritesTrace verifies the single-detector happy path
// writes a {"trace":[...]} file with one record per event.
func TestSaturationTracer_SingleWritesTrace(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite"
	saturationReport = filepath.Join(t.TempDir(), "trace.json")
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation: %v", err)
	}
	if err := tracer.trace(twoRequests()); err != nil {
		t.Fatalf("trace: %v", err)
	}
	report := readReport(t, saturationReport)
	if len(report.Trace) != 4 { // 2 requests × 2 events
		t.Errorf("expected 4 trace records, got %d", len(report.Trace))
	}
	for _, r := range report.Trace {
		if r.Detector != "composite" {
			t.Errorf("single-detector trace should only contain composite records, got %q", r.Detector)
		}
	}
}

// TestSaturationTracer_ZeroRequestsWritesEmptyTrace verifies the degenerate-input
// contract (R20) through the CLI tracer: zero completed requests writes a valid
// {"trace":[]} file (not {"trace":null}, not an error), for both the single and
// bank paths.
func TestSaturationTracer_ZeroRequestsWritesEmptyTrace(t *testing.T) {
	for _, sel := range []string{"composite", "all"} {
		t.Run(sel, func(t *testing.T) {
			resetSaturationGlobals()
			detectorName = sel
			saturationReport = filepath.Join(t.TempDir(), "empty.json")
			tracer, err := resolveSaturation()
			if err != nil {
				t.Fatalf("resolveSaturation(%q): %v", sel, err)
			}
			if err := tracer.trace(nil); err != nil { // zero requests
				t.Fatalf("trace(nil): %v", err)
			}
			data, err := os.ReadFile(saturationReport)
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			if got, want := string(data), "{\n  \"trace\": []\n}\n"; got != want {
				t.Errorf("detectors=%q: empty trace = %q, want %q", sel, got, want)
			}
		})
	}
}

// TestSaturationTracer_BankWritesAllDetectors verifies --detectors all writes a
// trace containing records for every detector in the roster.
func TestSaturationTracer_BankWritesAllDetectors(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "all"
	saturationReport = filepath.Join(t.TempDir(), "trace.json")
	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation: %v", err)
	}
	if err := tracer.trace(twoRequests()); err != nil {
		t.Fatalf("trace: %v", err)
	}
	report := readReport(t, saturationReport)
	// 2 requests × 2 events × 3 detectors = 12 records.
	if len(report.Trace) != 12 {
		t.Errorf("expected 12 trace records (2 req × 2 ev × 3 det), got %d", len(report.Trace))
	}
	seen := map[string]bool{}
	for _, r := range report.Trace {
		seen[r.Detector] = true
	}
	for _, name := range []string{"composite", "threshold", "backlog-drift"} {
		if !seen[name] {
			t.Errorf("bank trace missing records for %q", name)
		}
	}
}

// TestSaturationTracer_AllEqualsExplicitList verifies --detectors all produces a
// byte-identical trace file to the explicit full comma-list (INV-6): selection
// order and spelling never change how detectors see traffic.
func TestSaturationTracer_AllEqualsExplicitList(t *testing.T) {
	write := func(sel string) []byte {
		resetSaturationGlobals()
		detectorName = sel
		saturationReport = filepath.Join(t.TempDir(), "trace.json")
		tracer, err := resolveSaturation()
		if err != nil {
			t.Fatalf("resolveSaturation(%q): %v", sel, err)
		}
		if err := tracer.trace(twoRequests()); err != nil {
			t.Fatalf("trace(%q): %v", sel, err)
		}
		data, err := os.ReadFile(saturationReport)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		return data
	}
	all := write("all")
	explicit := write("threshold,backlog-drift,composite") // scrambled order
	if string(all) != string(explicit) {
		t.Errorf("--detectors all and the explicit full list produced different trace bytes")
	}
}

// TestSaturationTracer_SubsetMatchesRecordsUnderAll verifies a subset detector's
// records in the file are byte-for-byte the same as its records under all
// (INV-6 / INV-13) — selection filters WHICH detectors run, never HOW.
func TestSaturationTracer_SubsetMatchesRecordsUnderAll(t *testing.T) {
	writeAndFilter := func(sel, only string) []saturation.TraceRecord {
		resetSaturationGlobals()
		detectorName = sel
		saturationReport = filepath.Join(t.TempDir(), "trace.json")
		tracer, err := resolveSaturation()
		if err != nil {
			t.Fatalf("resolveSaturation(%q): %v", sel, err)
		}
		if err := tracer.trace(twoRequests()); err != nil {
			t.Fatalf("trace(%q): %v", sel, err)
		}
		report := readReport(t, saturationReport)
		out := make([]saturation.TraceRecord, 0)
		for _, r := range report.Trace {
			if only == "" || r.Detector == only {
				out = append(out, r)
			}
		}
		return out
	}
	underAll := writeAndFilter("all", "threshold")
	alone := writeAndFilter("threshold", "")
	if len(underAll) != len(alone) {
		t.Fatalf("record count differs: %d under all vs %d alone", len(underAll), len(alone))
	}
	for i := range underAll {
		// TraceRecord embeds a map (Result.Signals) so it is not comparable with
		// ==; compare the scalar fields directly and the Signals map via DeepEqual.
		a, b := underAll[i], alone[i]
		if a.Timestamp != b.Timestamp || a.Detector != b.Detector ||
			a.Result.Level != b.Result.Level || a.Result.Score != b.Result.Score ||
			a.Result.Confidence != b.Result.Confidence ||
			!reflect.DeepEqual(a.Result.Signals, b.Result.Signals) {
			t.Errorf("threshold record %d differs under all vs alone: %+v vs %+v", i, a, b)
		}
	}
}

// readReport reads and unmarshals a saturation report file.
func readReport(t *testing.T, path string) saturation.CombinedReport {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	var report saturation.CombinedReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return report
}
