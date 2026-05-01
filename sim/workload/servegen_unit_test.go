package workload

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestFindNearestDataset verifies nearest-preceding timestamp lookup.
// Review finding #2: Direct unit tests for findNearestDataset.
func TestFindNearestDataset(t *testing.T) {
	makeDataset := func(timestamps ...int) map[int]datasetWindow {
		m := make(map[int]datasetWindow)
		for _, ts := range timestamps {
			m[ts] = datasetWindow{
				inputPDF:  map[int]float64{100: 1.0},
				outputPDF: map[int]float64{50: 1.0},
			}
		}
		return m
	}

	tests := []struct {
		name      string
		query     int
		datasets  map[int]datasetWindow
		wantKey   int
		wantFound bool
	}{
		{
			name:      "exact match",
			query:     21600,
			datasets:  makeDataset(0, 21600, 43200),
			wantKey:   21600,
			wantFound: true,
		},
		{
			name:      "nearest preceding",
			query:     25000,
			datasets:  makeDataset(0, 21600, 43200),
			wantKey:   21600,
			wantFound: true,
		},
		{
			name:      "query before all datasets",
			query:     100,
			datasets:  makeDataset(21600, 43200),
			wantKey:   0,
			wantFound: false,
		},
		{
			name:      "query at zero with zero dataset",
			query:     0,
			datasets:  makeDataset(0, 21600),
			wantKey:   0,
			wantFound: true,
		},
		{
			name:      "empty map",
			query:     1000,
			datasets:  makeDataset(),
			wantKey:   0,
			wantFound: false,
		},
		{
			name:      "query after all datasets",
			query:     90000,
			datasets:  makeDataset(0, 21600, 43200),
			wantKey:   43200,
			wantFound: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, gotKey, gotFound := findNearestDataset(tt.query, tt.datasets)
			if gotFound != tt.wantFound {
				t.Errorf("findNearestDataset(%d): found=%v, want %v", tt.query, gotFound, tt.wantFound)
			}
			if gotFound && gotKey != tt.wantKey {
				t.Errorf("findNearestDataset(%d): key=%d, want %d", tt.query, gotKey, tt.wantKey)
			}
		})
	}
}

// TestLoadServeGenDatasetAllWindows verifies dataset JSON loading with span filtering.
// Review finding #2: Direct unit tests for loadServeGenDatasetAllWindows.
func TestLoadServeGenDatasetAllWindows(t *testing.T) {
	tmpDir := t.TempDir()

	// Create dataset JSON with timestamps at 0, 21600, 43200
	dataset := map[string]map[string]string{
		"0":     {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 1.0}"},
		"21600": {"input_tokens": "{150: 1.0}", "output_tokens": "{75: 1.0}"},
		"43200": {"input_tokens": "{300: 1.0}", "output_tokens": "{100: 1.0}"},
	}
	data, _ := json.Marshal(dataset)
	datasetPath := filepath.Join(tmpDir, "chunk-0-dataset.json")
	if err := os.WriteFile(datasetPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	t.Run("no span filtering", func(t *testing.T) {
		config := &ServeGenDataSpec{Path: tmpDir}
		result, err := loadServeGenDatasetAllWindows(datasetPath, config)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result) != 3 {
			t.Errorf("expected 3 dataset windows, got %d", len(result))
		}
	})

	t.Run("span end filtering", func(t *testing.T) {
		config := &ServeGenDataSpec{Path: tmpDir, SpanEnd: 30000}
		result, err := loadServeGenDatasetAllWindows(datasetPath, config)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Should include timestamps 0 and 21600 (both < 30000), exclude 43200
		if _, ok := result[43200]; ok {
			t.Error("expected timestamp 43200 to be filtered out by SpanEnd=30000")
		}
	})
}

// TestServeGenMultiPeriod_EmptyPeriod verifies BC-8: periods with no chunks are skipped.
// Review finding #3: Missing test for empty period handling.
func TestServeGenMultiPeriod_EmptyPeriod(t *testing.T) {
	tmpDir := t.TempDir()

	// Create chunks ONLY in midnight (0-1800s) and afternoon (50400-52200s).
	// Morning (28800-30600s) has no active chunks.
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

	chunk1Trace := "50400,10.0,1.1,Gamma,2.5,0.06\n"
	chunk1Dataset := map[string]map[string]string{
		"43200": {"input_tokens": "{180: 1.0}", "output_tokens": "{70: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-1-trace.csv"), []byte(chunk1Trace), 0644); err != nil {
		t.Fatal(err)
	}
	d1, _ := json.Marshal(chunk1Dataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-1-dataset.json"), d1, 0644); err != nil {
		t.Fatal(err)
	}

	spec, err := ConvertServeGen(tmpDir, 600, 180)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify no morning cohorts exist
	for _, cohort := range spec.Cohorts {
		if len(cohort.ID) > 7 && cohort.ID[:7] == "morning" {
			t.Errorf("BC-8: expected no morning cohort, but found %s", cohort.ID)
		}
	}

	// Verify midnight and afternoon cohorts DO exist
	hasMidnight := false
	hasAfternoon := false
	for _, cohort := range spec.Cohorts {
		if len(cohort.ID) >= 8 && cohort.ID[:8] == "midnight" {
			hasMidnight = true
		}
		if len(cohort.ID) >= 9 && cohort.ID[:9] == "afternoon" {
			hasAfternoon = true
		}
	}
	if !hasMidnight {
		t.Error("BC-8: expected midnight cohorts to be present")
	}
	if !hasAfternoon {
		t.Error("BC-8: expected afternoon cohorts to be present")
	}
}

// TestServeGenMultiPeriod_RateValue verifies BC-5: trace_rate is the correct sum
// of per-chunk rates for overlapping windows only.
// Review finding #4: BC-5 test must verify actual rate values, not just > 0.
func TestServeGenMultiPeriod_RateValue(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a chunk with two 10-min windows in midnight span.
	// With seed=42, midnight random window is [526, 1126).
	// Row [0, 600) overlaps [526, 1126): rate 3.0 included
	// Row [600, 1200) overlaps [526, 1126): rate 2.0 included
	// Row [1200, 1800) does NOT overlap [526, 1126): rate 99.0 excluded
	chunkTrace := "0,3.0,0.8,Weibull,1.5,0.02\n" +
		"600,2.0,0.85,Weibull,1.55,0.02\n" +
		"1200,99.0,0.9,Weibull,1.6,0.02\n"
	chunkDataset := map[string]map[string]string{
		"0": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}"},
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(chunkTrace), 0644); err != nil {
		t.Fatal(err)
	}
	d, _ := json.Marshal(chunkDataset)
	if err := os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644); err != nil {
		t.Fatal(err)
	}

	spec, err := ConvertServeGen(tmpDir, 600, 180)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// There should be exactly one cohort (midnight-critical with seed=42, sloIndex=0)
	if len(spec.Cohorts) == 0 {
		t.Fatal("expected at least one cohort")
	}

	// Find the midnight cohort
	var midnightCohort *CohortSpec
	for i := range spec.Cohorts {
		if spec.Cohorts[i].Spike != nil && spec.Cohorts[i].Spike.StartTimeUs == 0 {
			midnightCohort = &spec.Cohorts[i]
			break
		}
	}
	if midnightCohort == nil {
		t.Fatal("no midnight cohort found")
	}

	// BC-5: trace_rate should be 3.0 + 2.0 = 5.0 (NOT 3.0 + 2.0 + 99.0 = 104.0)
	expectedRate := 5.0
	if midnightCohort.Spike.TraceRate == nil {
		t.Fatal("BC-5: trace_rate is nil")
	}
	actualRate := *midnightCohort.Spike.TraceRate
	if math.Abs(actualRate-expectedRate) > 0.01 {
		t.Errorf("BC-5: trace_rate = %.2f, want %.2f (only windows overlapping period should be summed)", actualRate, expectedRate)
	}

	// BC-3: Token distributions should be lognormal (fitted from dataset), not gaussian fallback.
	// Dataset has {100: 1.0} for input and {50: 1.0} for output.
	// Lognormal fit of a single-point PDF: mu = ln(value), sigma = 0.
	if midnightCohort.InputDist.Type != "lognormal" {
		t.Errorf("BC-3: InputDist.Type = %q, want \"lognormal\" (should be fitted from dataset)", midnightCohort.InputDist.Type)
	}
	if midnightCohort.OutputDist.Type != "lognormal" {
		t.Errorf("BC-3: OutputDist.Type = %q, want \"lognormal\" (should be fitted from dataset)", midnightCohort.OutputDist.Type)
	}
	// mu = ln(100) ≈ 4.605 for input, ln(50) ≈ 3.912 for output
	if mu, ok := midnightCohort.InputDist.Params["mu"]; ok {
		expectedMu := math.Log(100)
		if math.Abs(mu-expectedMu) > 0.01 {
			t.Errorf("BC-3: InputDist mu = %.4f, want %.4f (ln(100))", mu, expectedMu)
		}
	}
	if mu, ok := midnightCohort.OutputDist.Params["mu"]; ok {
		expectedMu := math.Log(50)
		if math.Abs(mu-expectedMu) > 0.01 {
			t.Errorf("BC-3: OutputDist mu = %.4f, want %.4f (ln(50))", mu, expectedMu)
		}
	}

	// BC-4: Arrival params should be averaged from overlapping windows.
	// Two windows overlap: row[0] (cv=0.8, shape=1.5, scale=0.02) and row[1] (cv=0.85, shape=1.55, scale=0.02).
	// But only the first overlapping window per chunk is used (break after first match).
	// Single chunk → the first overlapping window's params are used directly.
	if midnightCohort.Arrival.Process != "weibull" {
		t.Errorf("BC-4: Arrival.Process = %q, want \"weibull\"", midnightCohort.Arrival.Process)
	}
	if midnightCohort.Arrival.CV != nil {
		expectedCV := 0.8
		if math.Abs(*midnightCohort.Arrival.CV-expectedCV) > 0.01 {
			t.Errorf("BC-4: Arrival.CV = %.4f, want %.4f", *midnightCohort.Arrival.CV, expectedCV)
		}
	} else {
		t.Error("BC-4: Arrival.CV is nil, expected non-nil")
	}
}
