// sim/saturation/factory_test.go
package saturation

import (
	"strings"
	"testing"
)

func TestNewDetector_UnknownName_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			msg := r.(string)
			if !strings.Contains(msg, "unknown saturation detector") {
				t.Errorf("Expected panic message to contain 'unknown saturation detector', got: %s", msg)
			}
		} else {
			t.Error("Expected panic, but none occurred")
		}
	}()
	NewDetector("invalid", DetectorOpts{})
}

func TestNewDetector_ValidNames(t *testing.T) {
	tests := []string{"composite", "threshold", "backlog-drift", "none"}
	for _, name := range tests {
		det := NewDetector(name, DetectorOpts{ThresholdMs: 5000})
		if det == nil {
			t.Errorf("NewDetector(%q) returned nil", name)
		}
		if det.Name() != name {
			t.Errorf("Expected name %q, got %q", name, det.Name())
		}
	}
}

func TestValidDetectorNames_CoversAllFactoryNames(t *testing.T) {
	// BC-1: ValidDetectorNames() must include all names accepted by NewDetector
	valid := ValidDetectorNames()
	factoryNames := []string{"composite", "threshold", "backlog-drift", "none"}

	for _, name := range factoryNames {
		if !valid[name] {
			t.Errorf("ValidDetectorNames() missing factory-supported name %q", name)
		}
	}

	// BC-2: All names in ValidDetectorNames() must be accepted by NewDetector (no false positives)
	for name := range valid {
		det := NewDetector(name, DetectorOpts{ThresholdMs: 5000})
		if det == nil {
			t.Errorf("ValidDetectorNames() includes %q but NewDetector panics/returns nil", name)
		}
	}
}
