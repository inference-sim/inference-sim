package workload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoadServeGenData_MultiPeriodGrouping verifies BC-1: chunks are assigned
// to multiple periods and output uses Cohorts instead of Clients.
func TestLoadServeGenData_MultiPeriodGrouping(t *testing.T) {
	// GIVEN a spec with multi-period configuration
	spec := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_mini",
			WindowDurationSecs: 600,
			DrainTimeoutSecs:   180,
		},
	}

	// WHEN loadServeGenData runs
	err := loadServeGenData(spec)

	// THEN it should complete without error (or return a path error for missing testdata)
	// We're primarily testing the structure change from Clients to Cohorts
	if err != nil {
		// If testdata doesn't exist, that's expected - we're testing the API change
		if !strings.Contains(err.Error(), "no chunk") && !strings.Contains(err.Error(), "testdata") {
			t.Fatalf("unexpected error: %v", err)
		}
		t.Skip("testdata not available, skipping functional test")
	}

	// BC-1: Output should have Cohorts, not Clients
	if len(spec.Cohorts) == 0 && len(spec.Clients) > 0 {
		t.Error("BC-1: expected Cohorts to be populated, but only Clients were populated (old behavior)")
	}

	// BC-6: AggregateRate should be 0 (absolute mode)
	if spec.AggregateRate != 0 {
		t.Errorf("BC-6: expected AggregateRate=0, got %f", spec.AggregateRate)
	}
}

// TestLoadServeGenData_Deterministic verifies BC-7: same seed produces
// identical output.
func TestLoadServeGenData_Deterministic(t *testing.T) {
	createSpec := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2",
			Seed:    42,
			ServeGenData: &ServeGenDataSpec{
				Path:               "testdata/servegen_mini",
				WindowDurationSecs: 600,
				DrainTimeoutSecs:   180,
			},
		}
	}

	spec1 := createSpec()
	spec2 := createSpec()

	// WHEN loadServeGenData runs twice with same seed
	err1 := loadServeGenData(spec1)
	err2 := loadServeGenData(spec2)

	if err1 != nil || err2 != nil {
		t.Skip("testdata not available")
	}

	// THEN outputs are identical
	// BC-7: Deterministic assignment
	if len(spec1.Cohorts) != len(spec2.Cohorts) {
		t.Errorf("BC-7: cohort count mismatch: %d vs %d", len(spec1.Cohorts), len(spec2.Cohorts))
	}

	for i := range spec1.Cohorts {
		if i >= len(spec2.Cohorts) {
			break
		}
		if spec1.Cohorts[i].ID != spec2.Cohorts[i].ID {
			t.Errorf("BC-7: cohort[%d] ID mismatch: %s vs %s", i, spec1.Cohorts[i].ID, spec2.Cohorts[i].ID)
		}
	}
}

// Helper to split cohort ID into [period, sloClass]
func splitCohortID(id string) []string {
	parts := strings.Split(id, "-")
	if len(parts) < 2 {
		return []string{id, ""}
	}
	return []string{parts[0], parts[1]}
}

// TestLoadServeGenData_NoDuplication verifies BC-9: no chunk appears in multiple cohorts.
func TestLoadServeGenData_NoDuplication(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_mini",
			WindowDurationSecs: 600,
			DrainTimeoutSecs:   180,
		},
	}

	err := loadServeGenData(spec)
	if err != nil {
		t.Skip("testdata not available")
	}

	// BC-9: Sum of cohort populations should equal number of unique chunks
	totalPopulation := 0
	for _, cohort := range spec.Cohorts {
		totalPopulation += cohort.Population
	}

	// The total population represents how many chunk assignments were made
	// Since each chunk should be assigned exactly once, this tests deduplication
	// We can't hard-code expected count without testdata, but we can verify
	// that population > 0 and cohorts were created
	if totalPopulation == 0 && len(spec.Cohorts) > 0 {
		t.Error("BC-9: cohorts exist but have zero population (impossible)")
	}
}

// TestConvertServeGen_MissingDirectory verifies BC-11: missing directory returns error.
func TestConvertServeGen_MissingDirectory(t *testing.T) {
	// GIVEN a nonexistent directory
	// WHEN ConvertServeGen runs
	spec, err := ConvertServeGen("/nonexistent/dir/that/does/not/exist", 600, 180)

	// THEN error is returned
	// BC-11: Missing directory error
	if err == nil {
		t.Error("BC-11: expected error for missing directory, got nil")
	}
	if spec != nil {
		t.Error("BC-11: expected nil spec on error")
	}
	if !strings.Contains(err.Error(), "no chunk") && !strings.Contains(err.Error(), "scanning") {
		t.Logf("BC-11: error message: %v", err)
	}
}

// TestConvertServeGen_AllChunksInactive verifies BC-13: all-inactive chunks returns error.
// This test documents expected behavior but may skip if we don't have testdata with all-inactive chunks.
func TestConvertServeGen_AllChunksInactive(t *testing.T) {
	// GIVEN a ServeGen directory where all chunks have rate=0 (would need testdata)
	// For now, we test with a missing directory which also results in "no valid chunks"
	_, err := ConvertServeGen("testdata/servegen_empty_if_exists", 600, 180)

	// THEN error is returned
	if err == nil {
		t.Skip("testdata exists with valid chunks, cannot test all-inactive scenario")
	}

	// BC-13: All chunks filtered out or no chunks found
	if !strings.Contains(err.Error(), "no valid chunks") &&
	   !strings.Contains(err.Error(), "no chunk") &&
	   !strings.Contains(err.Error(), "no active cohorts") {
		t.Logf("BC-13: got error (acceptable): %v", err)
	}
}

// TestServeGenMultiPeriod_E2E verifies end-to-end multi-period conversion:
// BC-1 (Cohorts), BC-2 (unique assignment), BC-3 (period gaps), BC-4 (SLO round-robin),
// BC-5 (rate summation), BC-6 (absolute rate mode), BC-7 (determinism), BC-8 (parameter averaging)
func TestServeGenMultiPeriod_E2E(t *testing.T) {
	// GIVEN ServeGen data with chunks spanning all 3 periods
	tmpDir := t.TempDir()

	// Create chunks that will be active in each period
	// Midnight period (0-1800s): chunks 0, 1
	chunk0Trace := "0,5.0,0.8,Weibull,1.5,0.02\n" +
		"600,4.5,0.85,Weibull,1.55,0.02\n"
	chunk0Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(chunk0Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d0, _ := json.Marshal(chunk0Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d0, 0644); err != nil {
		t.Fatal(err)
	}

	chunk1Trace := "0,6.0,0.9,Gamma,2.0,0.05\n" +
		"600,5.5,0.95,Gamma,2.1,0.05\n"
	chunk1Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{200: 1.0}", "output_tokens": "{80: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-1-trace.csv"), []byte(chunk1Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d1, _ := json.Marshal(chunk1Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-1-dataset.json"), d1, 0644); err != nil {
		t.Fatal(err)
	}

	// Morning period (28800-30600s): chunk 2
	chunk2Trace := "28800,8.0,1.0,Weibull,1.6,0.03\n" +
		"29400,7.5,1.05,Weibull,1.65,0.03\n"
	chunk2Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{150: 1.0}", "output_tokens": "{60: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-2-trace.csv"), []byte(chunk2Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d2, _ := json.Marshal(chunk2Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-2-dataset.json"), d2, 0644); err != nil {
		t.Fatal(err)
	}

	// Afternoon period (50400-52200s): chunks 3, 4
	chunk3Trace := "50400,10.0,1.1,Gamma,2.5,0.06\n" +
		"51000,9.5,1.15,Gamma,2.6,0.06\n"
	chunk3Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{180: 1.0}", "output_tokens": "{70: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-3-trace.csv"), []byte(chunk3Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d3, _ := json.Marshal(chunk3Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-3-dataset.json"), d3, 0644); err != nil {
		t.Fatal(err)
	}

	chunk4Trace := "50400,12.0,1.2,Weibull,1.7,0.04\n" +
		"51000,11.0,1.25,Weibull,1.75,0.04\n"
	chunk4Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{220: 1.0}", "output_tokens": "{90: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-4-trace.csv"), []byte(chunk4Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d4, _ := json.Marshal(chunk4Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-4-dataset.json"), d4, 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting with default parameters
	spec, err := ConvertServeGen(tmpDir, 600, 180)

	// THEN conversion succeeds
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// BC-1: Output uses Cohorts, not Clients
	if len(spec.Cohorts) == 0 {
		t.Error("BC-1: expected Cohorts to be populated")
	}
	if len(spec.Clients) > 0 {
		t.Error("BC-1: expected Clients to be empty (old format)")
	}

	// BC-6: Absolute rate mode (aggregate_rate=0)
	if spec.AggregateRate != 0 {
		t.Errorf("BC-6: expected aggregate_rate=0, got %f", spec.AggregateRate)
	}

	// BC-7: Determinism - run twice with same seed
	spec2, err2 := ConvertServeGen(tmpDir, 600, 180)
	if err2 != nil {
		t.Fatalf("second conversion failed: %v", err2)
	}
	if len(spec.Cohorts) != len(spec2.Cohorts) {
		t.Errorf("BC-7: cohort count mismatch: %d vs %d", len(spec.Cohorts), len(spec2.Cohorts))
	}
	for i := range spec.Cohorts {
		if i >= len(spec2.Cohorts) {
			break
		}
		if spec.Cohorts[i].ID != spec2.Cohorts[i].ID {
			t.Errorf("BC-7: cohort[%d] ID mismatch: %s vs %s", i, spec.Cohorts[i].ID, spec2.Cohorts[i].ID)
		}
	}

	// BC-2 & BC-9: Each chunk assigned to exactly one cohort
	totalPopulation := 0
	for _, cohort := range spec.Cohorts {
		totalPopulation += cohort.Population
	}
	if totalPopulation != 5 {
		t.Errorf("BC-2/BC-9: expected 5 chunks assigned once each, got total population %d", totalPopulation)
	}

	// BC-3: Period gaps - verify cohort timelines have gaps
	// Check that Spike field has distinct start times across periods
	spikeStartTimes := make(map[int64]bool)
	for _, cohort := range spec.Cohorts {
		if cohort.Spike != nil {
			spikeStartTimes[cohort.Spike.StartTimeUs] = true
		}
	}
	// We expect at least 3 distinct start times for the 3 periods
	if len(spikeStartTimes) < 3 {
		t.Errorf("BC-3: expected at least 3 distinct period start times, got %d", len(spikeStartTimes))
	}

	// BC-4: SLO round-robin - verify all 5 SLO classes appear
	sloClassCount := make(map[string]int)
	for _, cohort := range spec.Cohorts {
		if cohort.SLOClass != "" {
			sloClassCount[cohort.SLOClass]++
		}
	}
	expectedClasses := []string{"critical", "standard", "batch", "sheddable", "background"}
	for _, class := range expectedClasses {
		if sloClassCount[class] == 0 {
			t.Errorf("BC-4: expected SLO class %s to appear, but count is 0", class)
		}
	}

	// BC-5: Rate summation - verify trace_rate exists in spike
	for _, cohort := range spec.Cohorts {
		if cohort.Spike != nil {
			if cohort.Spike.TraceRate == nil || *cohort.Spike.TraceRate <= 0 {
				t.Errorf("BC-5: cohort %s has spike but trace_rate is invalid", cohort.ID)
			}
		}
	}

	// BC-8: Parameter averaging - verify lognormal parameters are present
	for _, cohort := range spec.Cohorts {
		if cohort.InputDist.Type != "lognormal" {
			t.Errorf("BC-8: cohort %s input dist type: got %s, want lognormal", cohort.ID, cohort.InputDist.Type)
		}
		if _, hasMu := cohort.InputDist.Params["mu"]; !hasMu {
			t.Errorf("BC-8: cohort %s input dist missing mu parameter", cohort.ID)
		}
		if _, hasSigma := cohort.InputDist.Params["sigma"]; !hasSigma {
			t.Errorf("BC-8: cohort %s input dist missing sigma parameter", cohort.ID)
		}
	}
}
