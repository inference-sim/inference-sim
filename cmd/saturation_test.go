package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
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

// TestResolveSaturation_Off verifies that with no --detectors and no config/report,
// resolveSaturation returns (nil, nil, nil) — saturation is off.
func TestResolveSaturation_Off(t *testing.T) {
	resetSaturationGlobals()
	det, coll, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if det != nil || coll != nil {
		t.Errorf("expected (nil, nil) when off, got det=%v coll=%v", det, coll)
	}
}

// TestResolveSaturation_ConfigOrReportWithoutDetectors verifies the hard-error
// contract: --saturation-config or --saturation-report given without --detectors.
func TestResolveSaturation_ConfigOrReportWithoutDetectors(t *testing.T) {
	t.Run("config without detectors", func(t *testing.T) {
		resetSaturationGlobals()
		saturationConfigPath = "some.yaml"
		if _, _, err := resolveSaturation(); err == nil || !strings.Contains(err.Error(), "--saturation-config requires --detectors") {
			t.Errorf("expected 'requires --detectors' error, got: %v", err)
		}
	})
	t.Run("report without detectors", func(t *testing.T) {
		resetSaturationGlobals()
		saturationReport = "some.json"
		if _, _, err := resolveSaturation(); err == nil || !strings.Contains(err.Error(), "--saturation-report requires --detectors") {
			t.Errorf("expected 'requires --detectors' error, got: %v", err)
		}
	})
}

// TestResolveSaturation_BankRejected verifies "all" and comma-lists error,
// pointing at the bank (#1519) — this PR takes exactly one detector name.
func TestResolveSaturation_BankRejected(t *testing.T) {
	for _, name := range []string{"all", "composite,threshold", "threshold,backlog-drift"} {
		resetSaturationGlobals()
		detectorName = name
		saturationReport = filepath.Join(t.TempDir(), "x.json")
		_, _, err := resolveSaturation()
		if err == nil || !strings.Contains(err.Error(), "1519") {
			t.Errorf("detectors=%q: expected bank (#1519) error, got: %v", name, err)
		}
	}
}

// TestResolveSaturation_UnknownName verifies an unknown single name errors listing
// the valid names.
func TestResolveSaturation_UnknownName(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "bogus"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	_, _, err := resolveSaturation()
	if err == nil {
		t.Fatal("expected error for unknown detector name")
	}
	for _, name := range []string{"composite", "threshold", "backlog-drift"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error should list %q, got: %v", name, err)
		}
	}
}

// TestResolveSaturation_UnwritableReportPath verifies the report path is validated
// up front (fast-fail before the run).
func TestResolveSaturation_UnwritableReportPath(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite"
	saturationReport = filepath.Join(t.TempDir(), "nonexistent-dir", "x.json")
	if _, _, err := resolveSaturation(); err == nil {
		t.Error("expected error for unwritable report path")
	}
}

// TestResolveSaturation_ValidSingleDetector verifies the happy path returns a
// non-nil detector and collector.
func TestResolveSaturation_ValidSingleDetector(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite"
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	det, coll, err := resolveSaturation()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if det == nil || coll == nil {
		t.Errorf("expected non-nil detector and collector, got det=%v coll=%v", det, coll)
	}
	if det.Name() != "composite" {
		t.Errorf("expected composite, got %q", det.Name())
	}
}

// TestResolveSaturation_CompositeWithEmptyConfig verifies the integration path
// --detectors composite + an empty --saturation-config round-trips cleanly
// (composite has no block; an empty file = all defaults, not an error).
func TestResolveSaturation_CompositeWithEmptyConfig(t *testing.T) {
	resetSaturationGlobals()
	cfgPath := filepath.Join(t.TempDir(), "empty.yaml")
	if err := os.WriteFile(cfgPath, []byte(""), 0644); err != nil {
		t.Fatalf("write empty config: %v", err)
	}
	detectorName = "composite"
	saturationConfigPath = cfgPath
	saturationReport = filepath.Join(t.TempDir(), "x.json")
	det, coll, err := resolveSaturation()
	if err != nil {
		t.Fatalf("composite + empty config should succeed, got: %v", err)
	}
	if det == nil || coll == nil || det.Name() != "composite" {
		t.Errorf("expected composite detector + collector, got det=%v coll=%v", det, coll)
	}
}

// TestRunSaturationTrace_NoOpWhenOff verifies runSaturationTrace writes nothing
// when the detector is nil or no report path is set (INV-6 no-op).
func TestRunSaturationTrace_NoOpWhenOff(t *testing.T) {
	reqs := []sim.RequestMetrics{{ID: "request_0", ArrivedAt: 0, E2E: 100}}

	t.Run("nil detector", func(t *testing.T) {
		resetSaturationGlobals()
		saturationReport = filepath.Join(t.TempDir(), "x.json")
		if err := runSaturationTrace(nil, saturation.NewInMemoryCollector(), reqs); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, err := os.Stat(saturationReport); !os.IsNotExist(err) {
			t.Errorf("expected no file written with nil detector, stat err=%v", err)
		}
	})

	t.Run("empty report path", func(t *testing.T) {
		resetSaturationGlobals()
		det, _ := saturation.BuildDetector("composite", saturation.SaturationConfig{})
		if err := runSaturationTrace(det, saturation.NewInMemoryCollector(), reqs); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// saturationReport is "" — nothing to check on disk; the assertion is that
		// no error occurs and the call is inert.
	})
}

// TestRunSaturationTrace_WritesTrace verifies the happy path writes a {"trace":[...]}
// file with one record per event.
func TestRunSaturationTrace_WritesTrace(t *testing.T) {
	resetSaturationGlobals()
	detectorName = "composite"
	saturationReport = filepath.Join(t.TempDir(), "trace.json")
	det, coll, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation: %v", err)
	}
	reqs := []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 100},
		{ID: "request_1", ArrivedAt: 1, E2E: 200},
	}
	if err := runSaturationTrace(det, coll, reqs); err != nil {
		t.Fatalf("runSaturationTrace: %v", err)
	}
	data, err := os.ReadFile(saturationReport)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	var report saturation.CombinedReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(report.Trace) != 4 { // 2 requests × 2 events
		t.Errorf("expected 4 trace records, got %d", len(report.Trace))
	}
}
