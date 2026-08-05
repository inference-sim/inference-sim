// sim/saturation/config_test.go
package saturation

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTempConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "sat.yaml")
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

// TestLoadSaturationConfig_EmptyPath_Defaults verifies an empty path yields the
// zero config (all defaults) without touching disk.
func TestLoadSaturationConfig_EmptyPath_Defaults(t *testing.T) {
	cfg, err := LoadSaturationConfig("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Threshold != nil || cfg.BacklogDrift != nil {
		t.Errorf("expected all-nil config for empty path, got %+v", cfg)
	}
}

// TestLoadSaturationConfig_EmptyFile_Defaults verifies an empty file is valid
// (all defaults), not an error.
func TestLoadSaturationConfig_EmptyFile_Defaults(t *testing.T) {
	path := writeTempConfig(t, "")
	cfg, err := LoadSaturationConfig(path)
	if err != nil {
		t.Fatalf("empty file should be valid, got: %v", err)
	}
	if cfg.Threshold != nil || cfg.BacklogDrift != nil {
		t.Errorf("expected all-nil config for empty file, got %+v", cfg)
	}
}

// TestLoadSaturationConfig_UnknownKey_Errors verifies strict parsing: a
// composite: block (or any unknown key) fails, naming the offending field.
func TestLoadSaturationConfig_UnknownKey_Errors(t *testing.T) {
	for _, contents := range []string{
		"composite:\n  anything: 1\n",
		"threshold:\n  bogus_field: 1\n",
		"unknown_top: 5\n",
	} {
		if _, err := LoadSaturationConfig(writeTempConfig(t, contents)); err == nil {
			t.Errorf("expected error for unknown key in %q, got nil", contents)
		}
	}
}

// TestBuildDetector_ThresholdOverride verifies a partial threshold block
// overrides only the field it names.
func TestBuildDetector_ThresholdOverride(t *testing.T) {
	cfg, err := LoadSaturationConfig(writeTempConfig(t, "threshold:\n  threshold_ms: 250\n"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	det, err := BuildDetector("threshold", cfg)
	if err != nil {
		t.Fatalf("BuildDetector: %v", err)
	}
	// Feed one completion above the override threshold (250ms) → OVERLOADED.
	det.Observe(Event{Timestamp: 1_000_000, Type: Completion, RequestID: "r0", LatencyMs: 300})
	if got := det.Detect().Level; got != Overloaded {
		t.Errorf("expected OVERLOADED with threshold_ms=250 and 300ms latency, got %v", got)
	}
}

// TestBuildDetector_UnknownName_Errors verifies an unknown name returns an error
// listing the valid single names (no panic — R6).
func TestBuildDetector_UnknownName_Errors(t *testing.T) {
	_, err := BuildDetector("bogus", SaturationConfig{})
	if err == nil {
		t.Fatal("expected error for unknown detector name")
	}
	for _, name := range []string{"composite", "threshold", "backlog-drift"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error should list valid name %q, got: %v", name, err)
		}
	}
}

// TestBuildDetector_BadParam_ErrorsNamingField verifies out-of-range params are
// rejected with the field named.
func TestBuildDetector_BadParam_ErrorsNamingField(t *testing.T) {
	tests := []struct {
		contents string
		wantSub  string
	}{
		{"threshold:\n  threshold_ms: -1\n", "threshold.threshold_ms"},
		{"backlog_drift:\n  window_size_sec: 0\n", "backlog_drift.window_size_sec"},
		{"backlog_drift:\n  confidence_ci: 1.5\n", "backlog_drift.confidence_ci"},
		{"backlog_drift:\n  saturated_drain_ratio: 0.99\n  transient_drain_ratio: 0.90\n", "saturated_drain_ratio"},
	}
	for _, tc := range tests {
		cfg, err := LoadSaturationConfig(writeTempConfig(t, tc.contents))
		if err != nil {
			t.Fatalf("load %q: %v", tc.contents, err)
		}
		name := "threshold"
		if strings.Contains(tc.contents, "backlog_drift") {
			name = "backlog-drift"
		}
		_, err = BuildDetector(name, cfg)
		if err == nil || !strings.Contains(err.Error(), tc.wantSub) {
			t.Errorf("config %q: expected error naming %q, got: %v", tc.contents, tc.wantSub, err)
		}
	}
}

// TestBuildDetector_BlockOwnershipMismatch verifies a config block for a
// detector other than the selected one is rejected (not silently dropped, R1).
func TestBuildDetector_BlockOwnershipMismatch(t *testing.T) {
	tests := []struct {
		name     string
		contents string
		wantSub  string
	}{
		{"composite", "threshold:\n  threshold_ms: 3000\n", "threshold block is not valid for --detectors composite"},
		{"composite", "backlog_drift:\n  window_size_sec: 30\n", "backlog_drift block is not valid for --detectors composite"},
		{"threshold", "backlog_drift:\n  window_size_sec: 30\n", "backlog_drift block is not valid for --detectors threshold"},
		{"backlog-drift", "threshold:\n  threshold_ms: 3000\n", "threshold block is not valid for --detectors backlog-drift"},
	}
	for _, tc := range tests {
		cfg, err := LoadSaturationConfig(writeTempConfig(t, tc.contents))
		if err != nil {
			t.Fatalf("load %q: %v", tc.contents, err)
		}
		_, err = BuildDetector(tc.name, cfg)
		if err == nil || !strings.Contains(err.Error(), tc.wantSub) {
			t.Errorf("detector %q with %q: expected error containing %q, got: %v", tc.name, tc.contents, tc.wantSub, err)
		}
	}
}

// TestBuildDetector_BacklogDriftPartialOverride verifies a partial backlog_drift
// block keeps defaults for unnamed fields and succeeds.
func TestBuildDetector_BacklogDriftPartialOverride(t *testing.T) {
	cfg, err := LoadSaturationConfig(writeTempConfig(t, "backlog_drift:\n  window_size_sec: 1\n"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	det, err := BuildDetector("backlog-drift", cfg)
	if err != nil {
		t.Fatalf("BuildDetector: %v", err)
	}
	if det.Name() != "backlog-drift" {
		t.Errorf("expected backlog-drift, got %q", det.Name())
	}
}
