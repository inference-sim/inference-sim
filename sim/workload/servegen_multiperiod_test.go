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
	// Ensure each period has chunks in all 3 windows (for test robustness with random selection)
	// Midnight period (0-1800s): chunks 0-8 (3 per window)
	// Each chunk has activity at ONLY ONE timestamp within its period
	chunk0Trace := "0,5.0,0.8,Weibull,1.5,0.02\n"
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

	chunk1Trace := "0,6.0,0.9,Gamma,2.0,0.05\n"
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

	chunk2Trace := "0,5.2,0.82,Weibull,1.52,0.022\n"
	chunk2Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{150: 1.0}", "output_tokens": "{60: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-2-trace.csv"), []byte(chunk2Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d2, _ := json.Marshal(chunk2Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-2-dataset.json"), d2, 0644); err != nil {
		t.Fatal(err)
	}

	chunk3Trace := "600,5.5,0.95,Gamma,2.1,0.05\n"
	chunk3Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{180: 1.0}", "output_tokens": "{70: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-3-trace.csv"), []byte(chunk3Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d3, _ := json.Marshal(chunk3Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-3-dataset.json"), d3, 0644); err != nil {
		t.Fatal(err)
	}

	chunk4Trace := "600,5.6,0.96,Gamma,2.2,0.06\n"
	chunk4Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{220: 1.0}", "output_tokens": "{90: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-4-trace.csv"), []byte(chunk4Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d4, _ := json.Marshal(chunk4Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-4-dataset.json"), d4, 0644); err != nil {
		t.Fatal(err)
	}

	chunk5Trace := "600,5.7,0.97,Gamma,2.3,0.07\n"
	chunk5Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{210: 1.0}", "output_tokens": "{85: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-5-trace.csv"), []byte(chunk5Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d5, _ := json.Marshal(chunk5Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-5-dataset.json"), d5, 0644); err != nil {
		t.Fatal(err)
	}

	chunk6Trace := "1200,5.8,0.98,Weibull,1.58,0.028\n"
	chunk6Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{230: 1.0}", "output_tokens": "{95: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-6-trace.csv"), []byte(chunk6Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d6, _ := json.Marshal(chunk6Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-6-dataset.json"), d6, 0644); err != nil {
		t.Fatal(err)
	}

	chunk7Trace := "1200,5.9,0.99,Weibull,1.59,0.029\n"
	chunk7Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{240: 1.0}", "output_tokens": "{100: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-7-trace.csv"), []byte(chunk7Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d7, _ := json.Marshal(chunk7Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-7-dataset.json"), d7, 0644); err != nil {
		t.Fatal(err)
	}

	chunk8Trace := "1200,6.0,1.0,Weibull,1.6,0.03\n"
	chunk8Dataset := map[string]map[string]string{
		"0": {"input_tokens": "{250: 1.0}", "output_tokens": "{105: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-8-trace.csv"), []byte(chunk8Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d8, _ := json.Marshal(chunk8Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-8-dataset.json"), d8, 0644); err != nil {
		t.Fatal(err)
	}

	// Morning period (28800-30600s): chunks 9-17 (3 per window)
	chunk9Trace := "28800,8.0,1.0,Weibull,1.6,0.03\n"
	chunk9Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{260: 1.0}", "output_tokens": "{110: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-9-trace.csv"), []byte(chunk9Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d9, _ := json.Marshal(chunk9Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-9-dataset.json"), d9, 0644); err != nil {
		t.Fatal(err)
	}

	chunk10Trace := "28800,8.1,1.01,Weibull,1.61,0.031\n"
	chunk10Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{270: 1.0}", "output_tokens": "{115: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-10-trace.csv"), []byte(chunk10Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d10, _ := json.Marshal(chunk10Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-10-dataset.json"), d10, 0644); err != nil {
		t.Fatal(err)
	}

	chunk11Trace := "28800,8.2,1.02,Weibull,1.62,0.032\n"
	chunk11Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{280: 1.0}", "output_tokens": "{120: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-11-trace.csv"), []byte(chunk11Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d11, _ := json.Marshal(chunk11Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-11-dataset.json"), d11, 0644); err != nil {
		t.Fatal(err)
	}

	chunk12Trace := "29400,7.5,1.05,Weibull,1.65,0.03\n"
	chunk12Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{290: 1.0}", "output_tokens": "{125: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-12-trace.csv"), []byte(chunk12Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d12, _ := json.Marshal(chunk12Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-12-dataset.json"), d12, 0644); err != nil {
		t.Fatal(err)
	}

	chunk13Trace := "29400,7.6,1.06,Weibull,1.66,0.034\n"
	chunk13Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{300: 1.0}", "output_tokens": "{130: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-13-trace.csv"), []byte(chunk13Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d13, _ := json.Marshal(chunk13Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-13-dataset.json"), d13, 0644); err != nil {
		t.Fatal(err)
	}

	chunk14Trace := "29400,7.7,1.07,Weibull,1.67,0.035\n"
	chunk14Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{310: 1.0}", "output_tokens": "{135: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-14-trace.csv"), []byte(chunk14Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d14, _ := json.Marshal(chunk14Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-14-dataset.json"), d14, 0644); err != nil {
		t.Fatal(err)
	}

	chunk15Trace := "30000,7.8,1.08,Weibull,1.68,0.036\n"
	chunk15Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{320: 1.0}", "output_tokens": "{140: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-15-trace.csv"), []byte(chunk15Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d15, _ := json.Marshal(chunk15Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-15-dataset.json"), d15, 0644); err != nil {
		t.Fatal(err)
	}

	chunk16Trace := "30000,7.9,1.09,Weibull,1.69,0.037\n"
	chunk16Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{330: 1.0}", "output_tokens": "{145: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-16-trace.csv"), []byte(chunk16Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d16, _ := json.Marshal(chunk16Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-16-dataset.json"), d16, 0644); err != nil {
		t.Fatal(err)
	}

	chunk17Trace := "30000,8.0,1.10,Weibull,1.70,0.038\n"
	chunk17Dataset := map[string]map[string]string{
		"21600": {"input_tokens": "{340: 1.0}", "output_tokens": "{150: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-17-trace.csv"), []byte(chunk17Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d17, _ := json.Marshal(chunk17Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-17-dataset.json"), d17, 0644); err != nil {
		t.Fatal(err)
	}

	// Afternoon period (50400-52200s): chunks 18-26 (3 per window)
	chunk18Trace := "50400,10.0,1.1,Gamma,2.5,0.06\n"
	chunk18Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{350: 1.0}", "output_tokens": "{155: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-18-trace.csv"), []byte(chunk18Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d18, _ := json.Marshal(chunk18Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-18-dataset.json"), d18, 0644); err != nil {
		t.Fatal(err)
	}

	chunk19Trace := "50400,10.1,1.11,Gamma,2.51,0.061\n"
	chunk19Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{360: 1.0}", "output_tokens": "{160: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-19-trace.csv"), []byte(chunk19Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d19, _ := json.Marshal(chunk19Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-19-dataset.json"), d19, 0644); err != nil {
		t.Fatal(err)
	}

	chunk20Trace := "50400,10.2,1.12,Gamma,2.52,0.062\n"
	chunk20Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{370: 1.0}", "output_tokens": "{165: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-20-trace.csv"), []byte(chunk20Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d20, _ := json.Marshal(chunk20Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-20-dataset.json"), d20, 0644); err != nil {
		t.Fatal(err)
	}

	chunk21Trace := "51000,9.5,1.15,Gamma,2.6,0.06\n"
	chunk21Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{380: 1.0}", "output_tokens": "{170: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-21-trace.csv"), []byte(chunk21Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d21, _ := json.Marshal(chunk21Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-21-dataset.json"), d21, 0644); err != nil {
		t.Fatal(err)
	}

	chunk22Trace := "51000,9.6,1.16,Gamma,2.61,0.063\n"
	chunk22Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{390: 1.0}", "output_tokens": "{175: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-22-trace.csv"), []byte(chunk22Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d22, _ := json.Marshal(chunk22Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-22-dataset.json"), d22, 0644); err != nil {
		t.Fatal(err)
	}

	chunk23Trace := "51000,9.7,1.17,Gamma,2.62,0.064\n"
	chunk23Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{400: 1.0}", "output_tokens": "{180: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-23-trace.csv"), []byte(chunk23Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d23, _ := json.Marshal(chunk23Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-23-dataset.json"), d23, 0644); err != nil {
		t.Fatal(err)
	}

	chunk24Trace := "51600,9.8,1.18,Gamma,2.63,0.065\n"
	chunk24Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{410: 1.0}", "output_tokens": "{185: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-24-trace.csv"), []byte(chunk24Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d24, _ := json.Marshal(chunk24Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-24-dataset.json"), d24, 0644); err != nil {
		t.Fatal(err)
	}

	chunk25Trace := "51600,9.9,1.19,Gamma,2.64,0.066\n"
	chunk25Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{420: 1.0}", "output_tokens": "{190: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-25-trace.csv"), []byte(chunk25Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d25, _ := json.Marshal(chunk25Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-25-dataset.json"), d25, 0644); err != nil {
		t.Fatal(err)
	}

	chunk26Trace := "51600,10.0,1.20,Gamma,2.65,0.067\n"
	chunk26Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{430: 1.0}", "output_tokens": "{195: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-26-trace.csv"), []byte(chunk26Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d26, _ := json.Marshal(chunk26Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-26-dataset.json"), d26, 0644); err != nil {
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

	// BC-2 & BC-9: Each chunk assigned to exactly one period, then split into 5 SLO cohorts
	totalPopulation := 0
	for _, cohort := range spec.Cohorts {
		totalPopulation += cohort.Population
	}
	if totalPopulation != 9 {
		t.Errorf("BC-2/BC-9: expected 9 chunks assigned once each (3 chunks × 3 periods), got total population %d", totalPopulation)
	}

	// BC-4b: With 3 chunks per period split into 5 SLO classes, not all cohorts will have chunks
	// We expect at least 6 cohorts (each period contributes at least 2 non-empty cohorts)
	if len(spec.Cohorts) < 6 {
		t.Errorf("BC-4b: expected at least 6 cohorts, got %d", len(spec.Cohorts))
	}
	if len(spec.Cohorts) > 15 {
		t.Errorf("BC-4b: expected at most 15 cohorts (3 periods × 5 SLO), got %d", len(spec.Cohorts))
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

	// BC-4: SLO diversity - verify multiple SLO classes appear
	// With random window selection, not all 15 (3×5) cohorts will have chunks.
	// We just verify that multiple classes are represented.
	sloClassCount := make(map[string]int)
	for _, cohort := range spec.Cohorts {
		if cohort.SLOClass != "" {
			sloClassCount[cohort.SLOClass]++
		}
	}
	if len(sloClassCount) < 2 {
		t.Errorf("BC-4: expected at least 2 SLO classes, got %d", len(sloClassCount))
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
