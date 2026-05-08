// sim/workload/saturation_test.go
package workload

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

func TestBacklogDriftConfig_Validation_ZeroWindow(t *testing.T) {
	// GIVEN window size <= 0
	// WHEN constructing config
	// THEN panics with descriptive error
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for zero window size")
		} else if !strings.Contains(fmt.Sprint(r), "WindowSize must be > 0") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(0, 5, 2.0, 0.95)
}

func TestBacklogDriftConfig_Validation_NegativeMinWindows(t *testing.T) {
	// GIVEN MinWindows <= 0
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for negative MinWindows")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 0, 2.0, 0.95)
}

func TestBacklogDriftConfig_Validation_NaNPeakRatio(t *testing.T) {
	// GIVEN PeakRatio is NaN
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for NaN PeakRatio")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, math.NaN(), 0.95)
}

func TestBacklogDriftConfig_Validation_CIOutOfRange(t *testing.T) {
	// GIVEN ConfidenceCI not in (0, 1)
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for CI=1.5")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 1.5)
}

func TestBacklogDriftConfig_Validation_ValidConfig(t *testing.T) {
	// GIVEN all parameters valid
	// WHEN constructing config
	// THEN succeeds without panic
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)
	if cfg.WindowSize != 60*time.Second {
		t.Errorf("WindowSize mismatch: got %v", cfg.WindowSize)
	}
}
