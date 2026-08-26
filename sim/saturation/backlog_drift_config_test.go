// sim/saturation/backlog_drift_config_test.go
package saturation

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
	_ = NewBacklogDriftConfig(0, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98)
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
	_ = NewBacklogDriftConfig(60*time.Second, 0, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98)
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
	_ = NewBacklogDriftConfig(60*time.Second, 5, math.NaN(), 0.2, 0.95, 2, 1, 0.95, 0.98)
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
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 1.5, 2, 1, 0.95, 0.98)
}

func TestBacklogDriftConfig_Validation_NegativePeakRatioBand(t *testing.T) {
	// GIVEN PeakRatioBand < 0
	// WHEN constructing config
	// THEN panics identifying PeakRatioBand
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for negative PeakRatioBand")
		} else if !strings.Contains(fmt.Sprint(r), "PeakRatioBand must be >= 0") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, -0.1, 0.95, 2, 1, 0.95, 0.98)
}

func TestBacklogDriftConfig_Validation_NegativeWarmupWindows(t *testing.T) {
	// GIVEN WarmupWindows < 0
	// WHEN constructing config
	// THEN panics identifying WarmupWindows
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for negative WarmupWindows")
		} else if !strings.Contains(fmt.Sprint(r), "WarmupWindows must be >= 0") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, -1, 1, 0.95, 0.98)
}

func TestBacklogDriftConfig_Validation_NegativeTailWindows(t *testing.T) {
	// GIVEN TailWindows < 0
	// WHEN constructing config
	// THEN panics identifying TailWindows
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for negative TailWindows")
		} else if !strings.Contains(fmt.Sprint(r), "TailWindows must be >= 0") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, -1, 0.95, 0.98)
}

func TestBacklogDriftConfig_Validation_SaturatedDrainRatioOutOfRange(t *testing.T) {
	// GIVEN SaturatedDrainRatio outside (0, 1]
	// WHEN constructing config
	// THEN panics identifying SaturatedDrainRatio
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for SaturatedDrainRatio=1.5")
		} else if !strings.Contains(fmt.Sprint(r), "SaturatedDrainRatio must be in (0, 1]") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 1.5, 0.98)
}

func TestBacklogDriftConfig_Validation_TransientDrainRatioOutOfRange(t *testing.T) {
	// GIVEN TransientDrainRatio outside (0, 1]
	// WHEN constructing config
	// THEN panics identifying TransientDrainRatio
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for TransientDrainRatio=0")
		} else if !strings.Contains(fmt.Sprint(r), "TransientDrainRatio must be in (0, 1]") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.0)
}

func TestBacklogDriftConfig_Validation_DrainRatioOverlap(t *testing.T) {
	// GIVEN SaturatedDrainRatio > TransientDrainRatio (each individually in range)
	// WHEN constructing config
	// THEN panics because the PERSISTENTLY_SATURATED and TRANSIENT_BACKLOG regions
	//      would overlap
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for overlapping drain-ratio regions")
		} else if !strings.Contains(fmt.Sprint(r), "regions would overlap") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.99, 0.95)
}

func TestBacklogDriftConfig_Validation_ValidConfig(t *testing.T) {
	// GIVEN all parameters valid
	// WHEN constructing config
	// THEN succeeds without panic
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98)
	if cfg.WindowSize != 60*time.Second {
		t.Errorf("WindowSize mismatch: got %v", cfg.WindowSize)
	}
}
