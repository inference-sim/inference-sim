package workload

import (
	"os"
	"path/filepath"
	"testing"
)

// TestServeGenMultiPeriod_RealData verifies the implementation handles actual ServeGen
// data including zero-rate rows with empty distribution fields.
//
// BC-REAL-1: Zero-rate rows are safely skipped (rate <= 0 check at loadServeGenChunk:566-569)
// BC-REAL-2: Empty pattern fields default to "poisson" (buildArrivalSpecFromRow:754-756)
// BC-REAL-3: All 162 chunks are scanned, only active chunks assigned to cohorts
func TestServeGenMultiPeriod_RealData(t *testing.T) {
	// GIVEN: Real ServeGen data with zero-rate rows
	// ServeGen/data/language/m-large/ contains 162 chunks
	// Many chunks have rows like: "600,0,0,,0,0" (zero rate, empty distribution)

	// Try multiple possible paths (worktree vs main repo, different nesting levels)
	possiblePaths := []string{
		filepath.Join("..", "..", "ServeGen", "data", "language", "m-large"),
		filepath.Join("..", "..", "..", "ServeGen", "data", "language", "m-large"),
		filepath.Join("..", "..", "..", "..", "ServeGen", "data", "language", "m-large"),
	}

	var realDataPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			realDataPath = path
			break
		}
	}

	if realDataPath == "" {
		t.Skip("Real ServeGen data not available (checked multiple paths)")
	}

	// WHEN: ConvertServeGen processes real data with default window/drain settings
	spec, err := ConvertServeGen(realDataPath, 600, 180)

	// THEN: Conversion succeeds without errors
	if err != nil {
		t.Fatalf("ConvertServeGen failed on real data: %v", err)
	}

	// BC-REAL-1: Output has 15 cohorts (3 periods × 5 SLO classes)
	if len(spec.Cohorts) != 15 {
		t.Errorf("BC-REAL-1: expected 15 cohorts, got %d", len(spec.Cohorts))
	}

	// BC-REAL-2: Total population < 162 (inactive chunks filtered out)
	totalPop := 0
	for _, cohort := range spec.Cohorts {
		totalPop += cohort.Population
	}
	if totalPop == 0 {
		t.Error("BC-REAL-2: no chunks assigned (expected some active chunks)")
	}
	if totalPop >= 162 {
		t.Errorf("BC-REAL-2: expected some inactive chunks filtered, got %d/%d assigned", totalPop, 162)
	}

	// BC-REAL-3: All cohorts have valid arrival specs (no empty pattern crashes)
	for _, cohort := range spec.Cohorts {
		if cohort.Arrival.Process == "" {
			t.Errorf("BC-REAL-3: cohort %s has empty arrival process", cohort.ID)
		}
		// Verify trace_rate is positive (absolute rate mode)
		if cohort.Spike == nil || cohort.Spike.TraceRate == nil || *cohort.Spike.TraceRate <= 0 {
			t.Errorf("BC-REAL-3: cohort %s has invalid trace_rate", cohort.ID)
		}
	}

	// BC-REAL-4: Absolute rate mode (aggregate_rate=0)
	if spec.AggregateRate != 0 {
		t.Errorf("BC-REAL-4: expected aggregate_rate=0, got %f", spec.AggregateRate)
	}
}
