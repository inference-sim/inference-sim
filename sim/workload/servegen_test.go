package workload

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseServeGenPDF_PythonDictString_ConvertsCorrectly(t *testing.T) {
	input := "{100: 0.5, 200: 0.3, 300: 0.2}"
	pdf, err := parseServeGenPDF(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 3 {
		t.Fatalf("expected 3 bins, got %d", len(pdf))
	}
	if pdf[100] != 0.5 || pdf[200] != 0.3 || pdf[300] != 0.2 {
		t.Errorf("unexpected PDF values: %v", pdf)
	}
}

func TestParseServeGenPDF_ScientificNotation(t *testing.T) {
	input := "{100: 3e-4, 200: 9.997e-1}"
	pdf, err := parseServeGenPDF(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 2 {
		t.Fatalf("got %d bins, want 2", len(pdf))
	}
	if pdf[100] < 0.0002 || pdf[100] > 0.0004 {
		t.Errorf("pdf[100] = %v, want ≈ 0.0003", pdf[100])
	}
}

func TestParseServeGenPDF_ExtraWhitespace(t *testing.T) {
	input := "{ 100 : 0.5 , 200 : 0.5 }"
	pdf, err := parseServeGenPDF(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 2 {
		t.Errorf("got %d bins, want 2", len(pdf))
	}
}

func TestParseServeGenPDF_TrailingComma(t *testing.T) {
	input := "{100: 0.5, 200: 0.5,}"
	pdf, err := parseServeGenPDF(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 2 {
		t.Errorf("got %d bins, want 2", len(pdf))
	}
}

func TestParseServeGenPDF_LargeDict(t *testing.T) {
	// 1000-bin PDF
	s := "{"
	for i := 0; i < 1000; i++ {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%d: 0.001", i)
	}
	s += "}"
	pdf, err := parseServeGenPDF(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 1000 {
		t.Errorf("got %d bins, want 1000", len(pdf))
	}
}

func TestParseServeGenPDF_EmptyDict_ReturnsError(t *testing.T) {
	_, err := parseServeGenPDF("{}")
	if err == nil {
		t.Fatal("expected error for empty dict")
	}
}

func TestParseServeGenTrace_AllShortRows_ReturnsEmptySlice(t *testing.T) {
	// GIVEN a CSV file where all rows have fewer than 4 fields
	dir := t.TempDir()
	csvContent := "short,row\nonly,two\n"
	path := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(path, []byte(csvContent), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN no error is returned but the result is empty (all rows skipped)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 0 {
		t.Errorf("got %d rows, want 0 (all rows should be skipped)", len(rows))
	}
}

// TestParseServeGenTrace_NonNumericFields_SkippedAndWarned verifies BC-2.
// Rows with non-numeric startTime, rate, or cv are counted in skippedRows.
func TestParseServeGenTrace_NonNumericFields_SkippedAndWarned(t *testing.T) {
	// GIVEN a CSV with 3 rows: 1 valid, 1 with non-numeric rate, 1 with non-numeric startTime
	dir := t.TempDir()
	csvContent := "0,1.5,2.0,Gamma\nBAD_TIME,1.0,2.0,Poisson\n100,NOT_A_NUMBER,2.0,Weibull\n"
	path := filepath.Join(dir, "trace.csv")
	require.NoError(t, os.WriteFile(path, []byte(csvContent), 0644))

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN no error is returned
	require.NoError(t, err)

	// AND only the valid row is included
	assert.Len(t, rows, 1, "only the valid row should be parsed")
	assert.InDelta(t, 1.5, rows[0].rate, 0.001)

	// AND a warning was logged about 2 skipped rows
	assert.Contains(t, buf.String(), "2 rows", "should warn about 2 skipped rows")
}

// TestLoadServeGenDataset_EmptyDictWindows_SkippedUntilValid tests that the loader
// skips time windows with empty PDF dictionaries (serialized as "{}") and finds the first
// window with actual traffic data, matching ServeGen Python library behavior.
// Keys are chosen for lexicographic sort order ("100" < "200" < "300") to ensure
// asymmetric partial-empty windows (window 200: input="{}", output=valid) are evaluated.
func TestLoadServeGenDataset_EmptyDictWindows_SkippedUntilValid(t *testing.T) {
	// GIVEN a dataset with empty dict windows followed by a valid window
	dir := t.TempDir()
	datasetJSON := `{
		"100": {"input_tokens": "{}", "output_tokens": "{}"},
		"200": {"input_tokens": "{}", "output_tokens": "{50: 1.0}"},
		"300": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}
	}`
	path := filepath.Join(dir, "dataset.json")
	require.NoError(t, os.WriteFile(path, []byte(datasetJSON), 0644))

	// WHEN loading the dataset
	inputPDF, outputPDF, err := loadServeGenDataset(path, &ServeGenDataSpec{})

	// THEN the function succeeds and uses the first valid window (timestamp 300)
	// Windows 100 (both empty) and 200 (input empty, output valid) are skipped
	require.NoError(t, err, "should skip empty dict windows and find valid window")
	assert.Len(t, inputPDF, 2, "input PDF should have 2 bins from window 300")
	assert.Len(t, outputPDF, 2, "output PDF should have 2 bins from window 300")
	assert.Equal(t, 0.5, inputPDF[100])
	assert.Equal(t, 0.5, inputPDF[200])
	assert.Equal(t, 0.7, outputPDF[50])
	assert.Equal(t, 0.3, outputPDF[100])
}

// TestLoadServeGenDataset_OutputEmptyDictWindow_Skipped tests the mirror asymmetric case:
// when the output field is "{}" but input is valid, the window is correctly skipped.
// This independently verifies the outputPDFStr != "{}" clause of the break condition.
func TestLoadServeGenDataset_OutputEmptyDictWindow_Skipped(t *testing.T) {
	// GIVEN a dataset with output="{}" followed by a valid window
	dir := t.TempDir()
	datasetJSON := `{
		"100": {"input_tokens": "{100: 1.0}", "output_tokens": "{}"},
		"200": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}
	}`
	path := filepath.Join(dir, "dataset.json")
	require.NoError(t, os.WriteFile(path, []byte(datasetJSON), 0644))

	// WHEN loading the dataset
	inputPDF, outputPDF, err := loadServeGenDataset(path, &ServeGenDataSpec{})

	// THEN the function succeeds and uses the first valid window (timestamp 200)
	// Window 100 (input valid, output empty) is skipped
	require.NoError(t, err, "should skip window with output={}")
	assert.Len(t, inputPDF, 2, "input PDF should have 2 bins from window 200")
	assert.Len(t, outputPDF, 2, "output PDF should have 2 bins from window 200")
	assert.Equal(t, 0.5, inputPDF[100])
	assert.Equal(t, 0.5, inputPDF[200])
	assert.Equal(t, 0.7, outputPDF[50])
	assert.Equal(t, 0.3, outputPDF[100])
}

// TestLoadServeGenDataset_AllEmptyDictWindows_ReturnsError tests that when ALL windows
// contain empty dicts ("{}"), the loader returns the correct "no valid PDF windows" error
// rather than falling through to the parser with misleading "empty PDF dictionary" error.
func TestLoadServeGenDataset_AllEmptyDictWindows_ReturnsError(t *testing.T) {
	// GIVEN a dataset where every window has empty dicts
	dir := t.TempDir()
	datasetJSON := `{
		"100": {"input_tokens": "{}", "output_tokens": "{}"},
		"200": {"input_tokens": "{}", "output_tokens": "{}"},
		"300": {"input_tokens": "{}", "output_tokens": "{}"}
	}`
	path := filepath.Join(dir, "dataset.json")
	require.NoError(t, os.WriteFile(path, []byte(datasetJSON), 0644))

	// WHEN loading the dataset
	_, _, err := loadServeGenDataset(path, &ServeGenDataSpec{})

	// THEN the function returns an error indicating no valid windows were found
	require.Error(t, err, "should fail when all windows are empty dicts")
	assert.Contains(t, err.Error(), "no valid PDF windows", "error should indicate no valid windows, not parser error")
}

// TestLoadServeGenDataset_NonNumericKey_SkippedWithWarning verifies BC-3.
// JSON keys that are not valid floats are skipped with a warning.
// When ALL keys are non-numeric, the function returns an error after warning.
func TestLoadServeGenDataset_NonNumericKey_SkippedWithWarning(t *testing.T) {
	// GIVEN a dataset JSON where the only key is non-numeric
	dir := t.TempDir()
	datasetJSON := `{
		"metadata": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}
	}`
	path := filepath.Join(dir, "dataset.json")
	require.NoError(t, os.WriteFile(path, []byte(datasetJSON), 0644))

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN loading the dataset
	_, _, err := loadServeGenDataset(path, &ServeGenDataSpec{})

	// THEN the function returns an error (no valid windows found)
	require.Error(t, err, "should fail when all keys are non-numeric")
	assert.Contains(t, err.Error(), "no valid PDF windows", "error should indicate no valid windows")

	// AND a warning was logged about the non-numeric key
	assert.Contains(t, buf.String(), "metadata", "should warn about non-numeric key 'metadata'")
}

func TestParseServeGenTrace_WithShapeScale(t *testing.T) {
	// GIVEN a trace CSV with 6 columns including shape/scale
	tmpfile, err := os.CreateTemp("", "trace-*.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Remove(tmpfile.Name()) }()

	// Write test data with high CV and fitted shape/scale
	content := "199200,22.46,173.81,Weibull,0.0575,0.000573\n"
	if _, err := tmpfile.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatal(err)
	}

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(tmpfile.Name())

	// THEN shape and scale are parsed correctly
	if err != nil {
		t.Fatalf("parseServeGenTrace failed: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	row := rows[0]
	if row.cv != 173.81 {
		t.Errorf("expected cv=173.81, got %f", row.cv)
	}
	if row.shapeParam != 0.0575 {
		t.Errorf("expected shapeParam=0.0575, got %f", row.shapeParam)
	}
	if row.scaleParam != 0.000573 {
		t.Errorf("expected scaleParam=0.000573, got %f", row.scaleParam)
	}
}

func TestParseServeGenTrace_FourColumnsBackwardCompat(t *testing.T) {
	// GIVEN a trace CSV with only 4 columns (no shape/scale)
	dir := t.TempDir()
	csvContent := "0,1.5,2.5,Gamma\n600,0.8,1.2,Weibull\n"
	path := filepath.Join(dir, "trace.csv")
	require.NoError(t, os.WriteFile(path, []byte(csvContent), 0644))

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN both rows are parsed successfully
	require.NoError(t, err)
	require.Len(t, rows, 2)

	// AND shapeParam and scaleParam default to zero for both rows
	assert.Equal(t, float64(0), rows[0].shapeParam, "4-column row should have shapeParam=0")
	assert.Equal(t, float64(0), rows[0].scaleParam, "4-column row should have scaleParam=0")
	assert.Equal(t, float64(0), rows[1].shapeParam, "4-column row should have shapeParam=0")
	assert.Equal(t, float64(0), rows[1].scaleParam, "4-column row should have scaleParam=0")

	// AND the core fields are parsed correctly
	assert.InDelta(t, 1.5, rows[0].rate, 0.001)
	assert.InDelta(t, 2.5, rows[0].cv, 0.001)
	assert.Equal(t, "Gamma", rows[0].pattern)
}

func TestParseServeGenTrace_BadShapeScale_FallsBackToZero(t *testing.T) {
	// GIVEN a trace CSV with 6 columns where columns 5-6 are non-numeric
	dir := t.TempDir()
	csvContent := "0,1.0,2.5,Gamma,BAD,BAD\n"
	path := filepath.Join(dir, "trace.csv")
	require.NoError(t, os.WriteFile(path, []byte(csvContent), 0644))

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN the row is still parsed (not skipped)
	require.NoError(t, err)
	require.Len(t, rows, 1, "row with bad shape/scale should not be skipped")

	// AND shapeParam and scaleParam fall back to zero
	assert.Equal(t, float64(0), rows[0].shapeParam, "bad shape should fall back to 0")
	assert.Equal(t, float64(0), rows[0].scaleParam, "bad scale should fall back to 0")

	// AND core fields are still parsed correctly
	assert.InDelta(t, 1.0, rows[0].rate, 0.001)
	assert.InDelta(t, 2.5, rows[0].cv, 0.001)
	assert.Equal(t, "Gamma", rows[0].pattern)
}

func TestLoadServeGenChunk_PopulatesShapeScale(t *testing.T) {
	// GIVEN a ServeGen chunk with high CV and fitted parameters
	traceDir, err := os.MkdirTemp("", "servegen-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(traceDir) }()

	// Write trace file
	tracePath := filepath.Join(traceDir, "chunk-0-trace.csv")
	traceContent := "0,22.46,173.81,Weibull,0.0575,0.000573\n"
	if err := os.WriteFile(tracePath, []byte(traceContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Write dataset file
	datasetPath := filepath.Join(traceDir, "chunk-0-dataset.json")
	datasetContent := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
	if err := os.WriteFile(datasetPath, []byte(datasetContent), 0644); err != nil {
		t.Fatal(err)
	}

	sgConfig := &ServeGenDataSpec{}

	// WHEN loading the chunk
	client, err := loadServeGenChunk("0", tracePath, datasetPath, sgConfig)

	// THEN per-window ArrivalSpec contains shape and scale
	if err != nil {
		t.Fatalf("loadServeGenChunk failed: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
	// With temporal preservation, parameters are per-window, not client-level.
	// Client-level Arrival is the fallback default (poisson).
	require.NotNil(t, client.Lifecycle)
	require.Len(t, client.Lifecycle.Windows, 1)
	w := client.Lifecycle.Windows[0]
	require.NotNil(t, w.Arrival)
	assert.Equal(t, "weibull", w.Arrival.Process)
	require.NotNil(t, w.Arrival.CV)
	assert.Equal(t, 173.81, *w.Arrival.CV)
	require.NotNil(t, w.Arrival.Shape)
	assert.Equal(t, 0.0575, *w.Arrival.Shape)
	// Scale is converted from seconds (0.000573) to microseconds (573.0)
	expectedScale := 0.000573 * 1e6
	require.NotNil(t, w.Arrival.Scale)
	assert.Equal(t, expectedScale, *w.Arrival.Scale)
}

func TestLoadServeGenChunk_FourColumnTrace_ShapeScaleRemainNil(t *testing.T) {
	// GIVEN a ServeGen chunk with only 4 columns (no shape/scale)
	dir := t.TempDir()

	// Write trace file with 4-column format (no shape/scale columns)
	tracePath := filepath.Join(dir, "chunk-0-trace.csv")
	traceContent := "0,5.0,3.2,Gamma\n"
	require.NoError(t, os.WriteFile(tracePath, []byte(traceContent), 0644))

	// Write dataset file
	datasetPath := filepath.Join(dir, "chunk-0-dataset.json")
	datasetContent := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
	require.NoError(t, os.WriteFile(datasetPath, []byte(datasetContent), 0644))

	sgConfig := &ServeGenDataSpec{}

	// WHEN loading the chunk
	client, err := loadServeGenChunk("0", tracePath, datasetPath, sgConfig)

	// THEN no error and client is non-nil
	require.NoError(t, err)
	require.NotNil(t, client)

	// With temporal preservation, parameters are per-window.
	require.NotNil(t, client.Lifecycle)
	require.Len(t, client.Lifecycle.Windows, 1)
	w := client.Lifecycle.Windows[0]
	require.NotNil(t, w.Arrival)

	// AND process and CV are set from the trace (on the window, not client-level)
	assert.Equal(t, "gamma", w.Arrival.Process)
	require.NotNil(t, w.Arrival.CV)
	assert.InDelta(t, 3.2, *w.Arrival.CV, 0.001)

	// AND Shape/Scale remain nil (not non-nil pointers to 0.0)
	// This preserves the pointer-nil idiom: nil means "derive from CV"
	assert.Nil(t, w.Arrival.Shape, "Shape must be nil for 4-column trace (no MLE params)")
	assert.Nil(t, w.Arrival.Scale, "Scale must be nil for 4-column trace (no MLE params)")
}

func TestLoadServeGenChunk_BadShapeScale_ShapeScaleRemainNil(t *testing.T) {
	// GIVEN a ServeGen chunk with 6 columns but non-numeric shape/scale
	dir := t.TempDir()

	// Write trace file with non-numeric shape/scale (falls back to 0)
	tracePath := filepath.Join(dir, "chunk-0-trace.csv")
	traceContent := "0,5.0,3.2,Weibull,BAD,BAD\n"
	require.NoError(t, os.WriteFile(tracePath, []byte(traceContent), 0644))

	// Write dataset file
	datasetPath := filepath.Join(dir, "chunk-0-dataset.json")
	datasetContent := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
	require.NoError(t, os.WriteFile(datasetPath, []byte(datasetContent), 0644))

	sgConfig := &ServeGenDataSpec{}

	// WHEN loading the chunk
	client, err := loadServeGenChunk("0", tracePath, datasetPath, sgConfig)

	// THEN no error and client is non-nil
	require.NoError(t, err)
	require.NotNil(t, client)

	// With temporal preservation, parameters are per-window.
	require.NotNil(t, client.Lifecycle)
	require.Len(t, client.Lifecycle.Windows, 1)
	w := client.Lifecycle.Windows[0]
	require.NotNil(t, w.Arrival)

	// AND process and CV are set (on the window)
	assert.Equal(t, "weibull", w.Arrival.Process)
	require.NotNil(t, w.Arrival.CV)

	// AND Shape/Scale remain nil (parse-failure fallback produced zeros)
	assert.Nil(t, w.Arrival.Shape, "Shape must be nil when parse falls back to 0")
	assert.Nil(t, w.Arrival.Scale, "Scale must be nil when parse falls back to 0")
}

// TestServeGenConversion_HighCVTrace verifies the end-to-end flow from a
// ServeGen trace CSV with extreme CV (173.81) through workload generation.
// This exercises the critical path: parse 6-column trace -> populate ArrivalSpec
// with MLE-fitted shape/scale -> validation passes despite high CV ->
// sampler uses explicit params -> IAT generation produces valid values.
//
// Behavioral contract:
//
//	GIVEN a ServeGen trace CSV with high-CV window (CV=173.81, shape=0.0575, scale=0.000573)
//	WHEN GenerateRequests processes the trace
//	THEN conversion succeeds, the chunk uses weibull with explicit shape/scale,
//	     validation passes, and sampled IATs are finite and positive.
func TestServeGenConversion_HighCVTrace(t *testing.T) {
	// GIVEN a ServeGen directory with high-CV trace and dataset
	dir := t.TempDir()

	// ServeGen 6-column trace: timestamp, rate, cv, pattern, shape, scale
	// CV=173.81 exceeds the normal Weibull CV validator bound of [0.01, 10.4]
	// but MLE-fitted shape=0.0575, scale=0.000573 are used directly.
	// Create chunks at all 3 windows in midnight period to ensure random selection finds chunks
	for i := 0; i < 3; i++ {
		timestamp := float64(i * 600) // 0, 600, 1200
		traceCSV := fmt.Sprintf("%f,22.46,173.81,Weibull,0.0575,0.000573\n", timestamp)
		require.NoError(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-trace.csv", i)), []byte(traceCSV), 0644))

		// Minimal dataset with empirical PDFs
		datasetJSON := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
		require.NoError(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-dataset.json", i)), []byte(datasetJSON), 0644))
	}

	// WHEN generating requests through the full pipeline
	spec := &WorkloadSpec{
		Version:      "2",
		Seed:         42,
		Category:     "language",
		AggregateRate: 22.46,
		ServeGenData: &ServeGenDataSpec{Path: dir},
	}
	requests, err := GenerateRequests(spec, 1e6, 0)

	// THEN conversion succeeds despite high CV
	require.NoError(t, err, "GenerateRequests should succeed with high-CV trace when shape/scale are provided")
	require.NotEmpty(t, requests, "Should generate requests from ServeGen data")

	// AND the loaded cohort has weibull with MLE-fitted parameters
	require.NotEmpty(t, spec.Cohorts, "ServeGen should populate cohorts (multi-period)")
	cohort := spec.Cohorts[0]
	assert.Equal(t, "weibull", cohort.Arrival.Process)
	require.NotNil(t, cohort.Arrival.Shape, "Shape should be populated from trace column 5")
	require.NotNil(t, cohort.Arrival.Scale, "Scale should be populated from trace column 6")
	assert.InDelta(t, 0.0575, *cohort.Arrival.Shape, 0.0001, "Shape should match trace value")
	// Scale converted from seconds (0.000573) to microseconds (573.0)
	assert.InDelta(t, 0.000573*1e6, *cohort.Arrival.Scale, 0.001, "Scale should be in microseconds")

	// AND the CV is preserved as informational metadata
	require.NotNil(t, cohort.Arrival.CV, "CV should be preserved from trace")
	assert.InDelta(t, 173.81, *cohort.Arrival.CV, 0.01, "CV should match trace value")

	// AND sampled IATs are finite and positive (behavioral verification)
	ratePerMicros := 22.46 / 1e6 // Use trace rate directly
	sampler := NewArrivalSampler(cohort.Arrival, ratePerMicros)
	require.NotNil(t, sampler, "Sampler should be created")

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		iat := sampler.SampleIAT(rng)
		assert.Greater(t, iat, int64(0), "IAT should be positive (iteration %d)", i)
	}
}

func TestNormalizeLifecycleTimestamps_EmptyClientList(t *testing.T) {
	// GIVEN an empty client list
	clients := []ClientSpec{}

	// WHEN normalization is called
	normalizeLifecycleTimestamps(&clients)

	// THEN it completes without panic and leaves list unchanged
	if len(clients) != 0 {
		t.Errorf("expected empty list to remain empty, got %d clients", len(clients))
	}
}

func TestNormalizeLifecycleTimestamps_Scenarios(t *testing.T) {
	tests := []struct {
		name    string
		clients []ClientSpec
		want    []ClientSpec
		checkFn func(t *testing.T, original, normalized []ClientSpec)
	}{
		{
			name: "single client with single window at non-zero time",
			clients: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 28800000000, EndUs: 29400000000}, // 8:00-8:10 AM
						},
					},
				},
			},
			want: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 600000000}, // 0:00-0:10 (10 min duration preserved)
						},
					},
				},
			},
			checkFn: func(t *testing.T, original, normalized []ClientSpec) {
				// BC-1: earliest window starts at 0
				if normalized[0].Lifecycle.Windows[0].StartUs != 0 {
					t.Errorf("BC-1 violation: expected StartUs=0, got %d", normalized[0].Lifecycle.Windows[0].StartUs)
				}
				// BC-6: duration preserved
				origDuration := original[0].Lifecycle.Windows[0].EndUs - original[0].Lifecycle.Windows[0].StartUs
				normDuration := normalized[0].Lifecycle.Windows[0].EndUs - normalized[0].Lifecycle.Windows[0].StartUs
				if origDuration != normDuration {
					t.Errorf("BC-6 violation: duration changed from %d to %d", origDuration, normDuration)
				}
			},
		},
		{
			name: "multiple clients with different start times",
			clients: []ClientSpec{
				{
					ID: "clientA",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 100000000, EndUs: 200000000}, // starts at 100s
						},
					},
				},
				{
					ID: "clientB",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 50000000, EndUs: 150000000}, // starts at 50s (earlier)
						},
					},
				},
			},
			want: []ClientSpec{
				{
					ID: "clientA",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 50000000, EndUs: 150000000}, // shifted by -50s
						},
					},
				},
				{
					ID: "clientB",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 100000000}, // shifted by -50s
						},
					},
				},
			},
			checkFn: func(t *testing.T, original, normalized []ClientSpec) {
				// BC-5: global minimum across all clients
				minStart := int64(math.MaxInt64)
				for _, c := range normalized {
					if c.Lifecycle != nil {
						for _, w := range c.Lifecycle.Windows {
							if w.StartUs < minStart {
								minStart = w.StartUs
							}
						}
					}
				}
				if minStart != 0 {
					t.Errorf("BC-5 violation: global minimum is %d, expected 0", minStart)
				}
			},
		},
		{
			name: "consecutive windows preserve relative timing",
			clients: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 1000, EndUs: 2000}, // window 1
							{StartUs: 3000, EndUs: 4000}, // window 2 (2000us gap)
						},
					},
				},
			},
			want: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 1000},
							{StartUs: 2000, EndUs: 3000}, // 2000us gap preserved
						},
					},
				},
			},
			checkFn: func(t *testing.T, original, normalized []ClientSpec) {
				// BC-2: relative timing preserved
				origGap := original[0].Lifecycle.Windows[1].StartUs - original[0].Lifecycle.Windows[0].EndUs
				normGap := normalized[0].Lifecycle.Windows[1].StartUs - normalized[0].Lifecycle.Windows[0].EndUs
				if origGap != normGap {
					t.Errorf("BC-2 violation: gap changed from %d to %d", origGap, normGap)
				}
			},
		},
		{
			name: "client with nil lifecycle is skipped safely",
			clients: []ClientSpec{
				{ID: "client1", Lifecycle: nil},
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 1000, EndUs: 2000},
						},
					},
				},
			},
			want: []ClientSpec{
				{ID: "client1", Lifecycle: nil}, // unchanged
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 1000}, // normalized
						},
					},
				},
			},
			checkFn: func(t *testing.T, original, normalized []ClientSpec) {
				// Verify nil lifecycle wasn't touched
				if normalized[0].Lifecycle != nil {
					t.Error("nil Lifecycle was unexpectedly modified")
				}
			},
		},
		{
			name: "client with empty windows slice is skipped safely",
			clients: []ClientSpec{
				{
					ID:        "client1",
					Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{}},
				},
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 5000, EndUs: 6000},
						},
					},
				},
			},
			want: []ClientSpec{
				{
					ID:        "client1",
					Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{}}, // unchanged
				},
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 1000},
						},
					},
				},
			},
			checkFn: nil, // structural check via want comparison is sufficient
		},
		{
			name: "idempotency: normalizing twice equals normalizing once",
			clients: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 28800000000, EndUs: 29400000000}, // 8:00-8:10 AM
							{StartUs: 29400000000, EndUs: 30000000000}, // 8:10-8:20 AM
						},
					},
				},
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 30000000000, EndUs: 30600000000}, // 8:20-8:30 AM
						},
					},
				},
			},
			want: []ClientSpec{
				{
					ID: "client1",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 0, EndUs: 600000000},
							{StartUs: 600000000, EndUs: 1200000000},
						},
					},
				},
				{
					ID: "client2",
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{StartUs: 1200000000, EndUs: 1800000000},
						},
					},
				},
			},
			checkFn: func(t *testing.T, _, normalized []ClientSpec) {
				// Snapshot the state after the first normalization
				snapshotWindows := make([][]ActiveWindow, len(normalized))
				for i, c := range normalized {
					if c.Lifecycle != nil {
						ws := make([]ActiveWindow, len(c.Lifecycle.Windows))
						copy(ws, c.Lifecycle.Windows)
						snapshotWindows[i] = ws
					}
				}

				// Apply normalization a second time
				normalizeLifecycleTimestamps(&normalized)

				// Verify all timestamps are identical after the second call
				for i, c := range normalized {
					if c.Lifecycle == nil {
						continue
					}
					for j, w := range c.Lifecycle.Windows {
						if w.StartUs != snapshotWindows[i][j].StartUs || w.EndUs != snapshotWindows[i][j].EndUs {
							t.Errorf("idempotency violation: client[%d] window[%d] changed on second normalize: "+
								"StartUs %d->%d, EndUs %d->%d",
								i, j,
								snapshotWindows[i][j].StartUs, w.StartUs,
								snapshotWindows[i][j].EndUs, w.EndUs)
						}
					}
				}
			},
		},
		{
			name: "all clients with nil lifecycle: no-op (exercises MaxInt64 guard)",
			clients: []ClientSpec{
				{ID: "client1", Lifecycle: nil},
				{ID: "client2", Lifecycle: nil},
				{ID: "client3", Lifecycle: nil},
			},
			want: []ClientSpec{
				{ID: "client1", Lifecycle: nil},
				{ID: "client2", Lifecycle: nil},
				{ID: "client3", Lifecycle: nil},
			},
			checkFn: func(t *testing.T, _, normalized []ClientSpec) {
				// All lifecycles must remain nil — the MaxInt64 guard
				// at line 744 must prevent any modification.
				for i, c := range normalized {
					if c.Lifecycle != nil {
						t.Errorf("client[%d] Lifecycle was unexpectedly modified from nil", i)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a deep copy of original for checkFn comparison
			// (shallow copy is insufficient: ClientSpec.Lifecycle is a pointer,
			// so mutations via normalizeLifecycleTimestamps would be visible
			// through both originalCopy and tt.clients, making BC-2/BC-6 checks vacuous)
			originalCopy := make([]ClientSpec, len(tt.clients))
			for i, c := range tt.clients {
				originalCopy[i] = c
				if c.Lifecycle != nil {
					lc := *c.Lifecycle
					lc.Windows = make([]ActiveWindow, len(c.Lifecycle.Windows))
					copy(lc.Windows, c.Lifecycle.Windows)
					originalCopy[i].Lifecycle = &lc
				}
			}

			// WHEN normalization is called
			normalizeLifecycleTimestamps(&tt.clients)

			// THEN structural expectations match
			if len(tt.clients) != len(tt.want) {
				t.Fatalf("client count mismatch: got %d, want %d", len(tt.clients), len(tt.want))
			}

			for i := range tt.clients {
				gotClient := tt.clients[i]
				wantClient := tt.want[i]

				if gotClient.ID != wantClient.ID {
					t.Errorf("client[%d] ID mismatch: got %q, want %q", i, gotClient.ID, wantClient.ID)
				}

				if (gotClient.Lifecycle == nil) != (wantClient.Lifecycle == nil) {
					t.Errorf("client[%d] Lifecycle nil mismatch", i)
					continue
				}

				if gotClient.Lifecycle != nil {
					if len(gotClient.Lifecycle.Windows) != len(wantClient.Lifecycle.Windows) {
						t.Errorf("client[%d] window count mismatch: got %d, want %d",
							i, len(gotClient.Lifecycle.Windows), len(wantClient.Lifecycle.Windows))
						continue
					}

					for j := range gotClient.Lifecycle.Windows {
						gotWindow := gotClient.Lifecycle.Windows[j]
						wantWindow := wantClient.Lifecycle.Windows[j]

						if gotWindow.StartUs != wantWindow.StartUs {
							t.Errorf("client[%d] window[%d] StartUs: got %d, want %d",
								i, j, gotWindow.StartUs, wantWindow.StartUs)
						}
						if gotWindow.EndUs != wantWindow.EndUs {
							t.Errorf("client[%d] window[%d] EndUs: got %d, want %d",
								i, j, gotWindow.EndUs, wantWindow.EndUs)
						}
					}
				}
			}

			// THEN behavioral contracts verified by custom check function
			if tt.checkFn != nil {
				tt.checkFn(t, originalCopy, tt.clients)
			}
		})
	}
}

// This test uses trace data starting at t=0, so normalization is a no-op.
// The test validates per-window shape/scale parameter population from trace columns 5-6,
// not normalization behavior (which is thoroughly covered by other tests).
func TestServeGenDataLoading_SyntheticDataset_ProducesClients(t *testing.T) {
	dir := t.TempDir()
	// Create chunks at all 3 windows in midnight period to ensure random selection finds chunks
	// Scale parameters in seconds: Gamma(shape=0.16, scale=6.25s), Weibull(shape=1.0, scale=2.0s)
	for i := 0; i < 3; i++ {
		timestamp := i * 600 // 0, 600, 1200
		// Alternate between Gamma and Weibull
		process := "Gamma"
		shape, scale := 0.16, 6.25
		if i%2 == 1 {
			process = "Weibull"
			shape, scale = 1.0, 2.0
		}
		traceCSV := fmt.Sprintf("%d,1.0,2.5,%s,%.2f,%.2f\n", timestamp, process, shape, scale)
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-trace.csv", i)), []byte(traceCSV), 0644); err != nil {
			t.Fatal(err)
		}
		// Create dataset.json
		datasetJSON := `{"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}}`
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-dataset.json", i)), []byte(datasetJSON), 0644); err != nil {
			t.Fatal(err)
		}
	}

	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		ServeGenData: &ServeGenDataSpec{Path: dir},
	}
	// Load to verify cohort creation with shape/scale parameters
	if err := loadServeGenData(spec); err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}
	if len(spec.Cohorts) == 0 {
		t.Fatal("expected at least one cohort")
	}
	// With 3 chunks split across 5 SLO classes, we expect 3-5 non-empty cohorts
	if len(spec.Cohorts) > 5 {
		t.Fatalf("expected at most 5 cohorts (3 chunks / 5 SLO classes), got %d", len(spec.Cohorts))
	}
	// Test the first cohort (which should have shape/scale from its selected window)
	cohort := spec.Cohorts[0]
	// Verify MLE-fitted shape/scale parameters were populated from 6-column trace
	require.NotNil(t, cohort.Arrival.Shape, "Shape should be populated from trace column 5")
	require.NotNil(t, cohort.Arrival.Scale, "Scale should be populated from trace column 6")
	// Shape/scale values depend on which window was selected and averaged across chunks
	// Just verify they're positive and in reasonable range
	assert.Greater(t, *cohort.Arrival.Shape, 0.0, "Shape should be positive")
	assert.Greater(t, *cohort.Arrival.Scale, 0.0, "Scale should be positive (in microseconds)")

	// Clear ServeGenData since cohorts are now populated
	spec.ServeGenData = nil
	// Use a 10-second horizon; the time-varying generator uses per-window
	// distributions from the lifecycle windows.
	requests, err := GenerateRequests(spec, 10e6, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from ServeGen data")
	}
	// Verify input token lengths come from fitted lognormal distribution
	// Lognormal tails can extend beyond the original empirical PDF range (100, 200)
	// Allow wider range to accommodate lognormal tail samples
	for _, req := range requests[:min(10, len(requests))] {
		l := len(req.InputTokens)
		if l < 1 || l > 1000 {
			t.Errorf("input length %d outside reasonable range [1, 1000]", l)
		}
	}
}

func TestConvertServeGen_MultiPeriodAbsoluteTimestamps(t *testing.T) {
	// This test verifies that multi-period ServeGen conversion uses absolute
	// timestamps (no normalization). Chunks at Hour 8 (morning) are assigned
	// to the morning period with absolute start times.

	dir := t.TempDir()

	// GIVEN ServeGen chunks with absolute clock timestamps in morning period (8:00 AM onwards).
	// Create chunks at all 3 windows to ensure random selection finds chunks.
	// Morning windows: 28800 (8:00), 29400 (8:10), 30000 (8:20)
	morningWindows := []int{28800, 29400, 30000}
	for i, timestamp := range morningWindows {
		traceCSV := fmt.Sprintf("%d,5.0,2.5,Gamma,0.16,6.25\n", timestamp)
		require.NoError(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-trace.csv", i)), []byte(traceCSV), 0644))

		// Dataset entry at timestamp 21600 (Hour 6) — nearest-preceding for Hour 8 windows.
		datasetJSON := `{"21600": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
		require.NoError(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("chunk-%d-dataset.json", i)), []byte(datasetJSON), 0644))
	}

	// WHEN ConvertServeGen processes these files
	spec, err := ConvertServeGen(dir, 600, 180)

	// THEN conversion succeeds
	require.NoError(t, err)
	require.NotNil(t, spec)
	require.NotEmpty(t, spec.Cohorts, "expected cohorts from chunk-0")

	// BC-10: No timestamp normalization for cohort-based specs
	// Cohorts use absolute timestamps within the day, with period start times
	// that preserve gaps (morning period starts at windowDur + drain after midnight)
	cohort := spec.Cohorts[0]
	require.NotNil(t, cohort.Spike, "cohort should have spike for the period")

	// The morning period starts at (600 + 180) * 1e6 = 780,000,000 µs
	// (This is the fixed period start, not the original trace timestamp)
	expectedMorningStart := int64((600 + 180) * 1e6)
	assert.Equal(t, expectedMorningStart, cohort.Spike.StartTimeUs,
		"morning period cohort should start at absolute period time (not normalized to 0)")

	// Duration should match the configured window duration (600s)
	assert.Equal(t, int64(600*1e6), cohort.Spike.DurationUs,
		"spike duration should be the configured window duration")
}

func TestParseServeGenFloatPDF_ValidRatios(t *testing.T) {
	// GIVEN a reason_ratio PDF with float keys in [0.0, 1.0]
	input := `{"0.0": 0.1, "0.5": 0.3, "1.0": 0.6}`

	// WHEN parsed
	pdf, err := parseServeGenFloatPDF(input)

	// THEN it succeeds and returns float map
	require.NoError(t, err)
	assert.InDelta(t, 0.1, pdf[0.0], 0.0001)
	assert.InDelta(t, 0.3, pdf[0.5], 0.0001)
	assert.InDelta(t, 0.6, pdf[1.0], 0.0001)
	assert.Equal(t, 3, len(pdf))
}

func TestParseServeGenFloatPDF_OutOfRangeKey(t *testing.T) {
	// GIVEN PDF with key outside [0.0, 1.0]
	input := `{"1.5": 0.2}`

	// WHEN parsed
	_, err := parseServeGenFloatPDF(input)

	// THEN it returns error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "outside valid range")
}

func TestParseServeGenFloatPDF_EmptyDict(t *testing.T) {
	// GIVEN empty PDF dict
	input := `{}`

	// WHEN parsed
	_, err := parseServeGenFloatPDF(input)

	// THEN it returns error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty PDF dictionary")
}

func TestParseServeGenFloatPDF_NegativeProbability(t *testing.T) {
	// GIVEN PDF with negative probability value
	input := `{"0.5": -0.3}`

	// WHEN parsed
	_, err := parseServeGenFloatPDF(input)

	// THEN it returns error
	require.Error(t, err)
	assert.Contains(t, err.Error(), "negative probability")
}

func TestDetectReasoningCategory_FromDataset(t *testing.T) {
	// GIVEN ServeGen data directory with reason_ratio field in dataset
	tmpDir := t.TempDir()

	// Create trace file with 144 windows (24 hours at 10-min intervals)
	// to ensure the randomly-selected window has data
	var traceContent string
	for i := 0; i < 144; i++ {
		traceContent += fmt.Sprintf("%d,1.5,0.8,Gamma,2.0,300\n", i*600)
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-trace.csv"), []byte(traceContent), 0644))

	// Create dataset with reason_ratio field
	datasetContent := `{"0": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}", "reason_ratio": "{0.5: 1.0}"}}`
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-dataset.json"), []byte(datasetContent), 0644))

	// WHEN ConvertServeGen is called
	spec, err := ConvertServeGen(tmpDir, 600, 180)

	// THEN category is set to "reasoning"
	require.NoError(t, err)
	assert.Equal(t, "reasoning", spec.Category)
}

func TestReasoningCohorts_BehavioralContract(t *testing.T) {
	// GIVEN ServeGen reasoning dataset with 5 chunks
	tmpDir := t.TempDir()

	// Create 5 chunks with varying rates to ensure round-robin distribution
	for i := 0; i < 5; i++ {
		chunkID := fmt.Sprintf("%03d", i)

		// Full-day trace (144 windows) with non-zero rate
		var traceContent string
		for w := 0; w < 144; w++ {
			rate := 1.0 + float64(i)*0.5 // Different rate per chunk
			traceContent += fmt.Sprintf("%d,%.1f,0.8,Gamma,2.0,300\n", w*600, rate)
		}
		require.NoError(t, os.WriteFile(
			filepath.Join(tmpDir, fmt.Sprintf("chunk-%s-trace.csv", chunkID)),
			[]byte(traceContent), 0644))

		// Dataset with reason_ratio (bimodal: low=24%, high=61%)
		datasetContent := `{"0": {
			"input_tokens": "{100: 0.5, 200: 0.5}",
			"output_tokens": "{50: 0.5, 100: 0.5}",
			"reason_ratio": "{0.24: 0.6, 0.61: 0.4}"
		}}`
		require.NoError(t, os.WriteFile(
			filepath.Join(tmpDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID)),
			[]byte(datasetContent), 0644))
	}

	// WHEN ConvertServeGen is called
	spec, err := ConvertServeGen(tmpDir, 600, 180)
	require.NoError(t, err)

	// THEN behavioral contract guarantees are verified

	// BC-1: AggregateRate is 0 (absolute mode)
	assert.Equal(t, 0.0, spec.AggregateRate, "should use absolute rate mode")

	// BC-2: All 5 SLO cohorts are present
	assert.Len(t, spec.Cohorts, 5, "should have 5 SLO cohorts")

	sloClassesFound := make(map[string]bool)
	for _, cohort := range spec.Cohorts {
		// BC-3: Cohort IDs follow reasoning-<sloClass> format
		assert.Regexp(t, `^reasoning-(critical|standard|batch|sheddable|background)$`,
			cohort.ID, "cohort ID should follow reasoning-<sloClass> format")

		sloClassesFound[cohort.SLOClass] = true

		// BC-4: Population is non-zero (chunks distributed round-robin)
		assert.Greater(t, cohort.Population, 0, "cohort %s should have non-zero population", cohort.ID)

		// BC-5: ClosedLoop is explicitly false (spike-based, not sessions)
		require.NotNil(t, cohort.ClosedLoop, "cohort %s should have ClosedLoop set", cohort.ID)
		assert.False(t, *cohort.ClosedLoop, "cohort %s should be open-loop (spike-based)", cohort.ID)

		// BC-6: Spike config with positive TraceRate
		require.NotNil(t, cohort.Spike, "cohort %s should have Spike config", cohort.ID)
		require.NotNil(t, cohort.Spike.TraceRate, "cohort %s should have TraceRate", cohort.ID)
		assert.Greater(t, *cohort.Spike.TraceRate, 0.0, "cohort %s should have positive rate", cohort.ID)

		// BC-7: Reasoning config with empirical distribution
		require.NotNil(t, cohort.Reasoning, "cohort %s should have Reasoning config", cohort.ID)
		assert.Equal(t, "empirical", cohort.Reasoning.ReasonRatioDist.Type,
			"cohort %s should use empirical distribution", cohort.ID)
		assert.NotEmpty(t, cohort.Reasoning.ReasonRatioDist.Params,
			"cohort %s should have non-empty reason_ratio distribution", cohort.ID)

		// BC-8: MultiTurn workaround (MaxRounds=1)
		require.NotNil(t, cohort.Reasoning.MultiTurn, "cohort %s should have MultiTurn", cohort.ID)
		assert.Equal(t, 1, cohort.Reasoning.MultiTurn.MaxRounds,
			"cohort %s should have MaxRounds=1", cohort.ID)
	}

	// Verify all 5 SLO classes are present
	expectedClasses := []string{"critical", "standard", "batch", "sheddable", "background"}
	for _, sloClass := range expectedClasses {
		assert.True(t, sloClassesFound[sloClass], "should have %s cohort", sloClass)
	}
}

func TestReasoningConversion_Determinism(t *testing.T) {
	// GIVEN ServeGen reasoning dataset
	tmpDir := t.TempDir()

	// Create minimal reasoning dataset (2 chunks for round-robin distribution)
	for i := 0; i < 2; i++ {
		chunkID := fmt.Sprintf("%03d", i)

		var traceContent string
		for w := 0; w < 144; w++ {
			traceContent += fmt.Sprintf("%d,%.1f,0.8,Gamma,2.0,300\n", w*600, 1.0+float64(i)*0.1)
		}
		require.NoError(t, os.WriteFile(
			filepath.Join(tmpDir, fmt.Sprintf("chunk-%s-trace.csv", chunkID)),
			[]byte(traceContent), 0644))

		datasetContent := `{"0": {
			"input_tokens": "{100: 0.4, 200: 0.6}",
			"output_tokens": "{50: 0.3, 100: 0.7}",
			"reason_ratio": "{0.3: 0.5, 0.7: 0.5}"
		}}`
		require.NoError(t, os.WriteFile(
			filepath.Join(tmpDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID)),
			[]byte(datasetContent), 0644))
	}

	// WHEN ConvertServeGen is called twice
	spec1, err1 := ConvertServeGen(tmpDir, 600, 180)
	require.NoError(t, err1)

	spec2, err2 := ConvertServeGen(tmpDir, 600, 180)
	require.NoError(t, err2)

	// THEN outputs are identical (INV-6: deterministic conversion)
	require.Equal(t, len(spec1.Cohorts), len(spec2.Cohorts), "cohort count should match")

	for i := range spec1.Cohorts {
		c1, c2 := spec1.Cohorts[i], spec2.Cohorts[i]

		assert.Equal(t, c1.ID, c2.ID, "cohort %d ID should match", i)
		assert.Equal(t, c1.Population, c2.Population, "cohort %d population should match", i)
		assert.Equal(t, c1.SLOClass, c2.SLOClass, "cohort %d SLO class should match", i)
		assert.Equal(t, c1.ClosedLoop, c2.ClosedLoop, "cohort %d ClosedLoop should match", i)

		// Spike config
		require.NotNil(t, c1.Spike)
		require.NotNil(t, c2.Spike)
		assert.Equal(t, c1.Spike.StartTimeUs, c2.Spike.StartTimeUs, "cohort %d spike start should match", i)
		assert.Equal(t, c1.Spike.DurationUs, c2.Spike.DurationUs, "cohort %d spike duration should match", i)
		if c1.Spike.TraceRate != nil && c2.Spike.TraceRate != nil {
			assert.InDelta(t, *c1.Spike.TraceRate, *c2.Spike.TraceRate, 1e-9, "cohort %d trace rate should match", i)
		}

		// Input/Output distributions
		assert.Equal(t, c1.InputDist.Type, c2.InputDist.Type, "cohort %d input dist type should match", i)
		assert.InDelta(t, c1.InputDist.Params["mu"], c2.InputDist.Params["mu"], 1e-9, "cohort %d input mu should match", i)
		assert.InDelta(t, c1.InputDist.Params["sigma"], c2.InputDist.Params["sigma"], 1e-9, "cohort %d input sigma should match", i)
		assert.Equal(t, c1.OutputDist.Type, c2.OutputDist.Type, "cohort %d output dist type should match", i)
		assert.InDelta(t, c1.OutputDist.Params["mu"], c2.OutputDist.Params["mu"], 1e-9, "cohort %d output mu should match", i)
		assert.InDelta(t, c1.OutputDist.Params["sigma"], c2.OutputDist.Params["sigma"], 1e-9, "cohort %d output sigma should match", i)

		// Reasoning distribution params
		require.NotNil(t, c1.Reasoning)
		require.NotNil(t, c2.Reasoning)
		assert.Equal(t, len(c1.Reasoning.ReasonRatioDist.Params), len(c2.Reasoning.ReasonRatioDist.Params),
			"cohort %d reason ratio param count should match", i)
		for key, val1 := range c1.Reasoning.ReasonRatioDist.Params {
			val2, ok := c2.Reasoning.ReasonRatioDist.Params[key]
			assert.True(t, ok, "cohort %d reason ratio key %s should exist in both", i, key)
			assert.InDelta(t, val1, val2, 1e-9, "cohort %d reason ratio[%s] should match", i, key)
		}
	}
}

func TestFindBestCoverageWindow_TieBreaking(t *testing.T) {
	// GIVEN trace files where multiple windows have same coverage
	tmpDir := t.TempDir()

	// Chunk 1: windows 0 and 5 both have rate > 0
	trace1 := "0,1.0,0.8,Gamma,2.0,300\n3000,1.0,0.8,Gamma,2.0,300\n"
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-trace.csv"), []byte(trace1), 0644))

	// Chunk 2: windows 0 and 5 both have rate > 0
	trace2 := "0,2.0,0.8,Gamma,2.0,300\n3000,2.0,0.8,Gamma,2.0,300\n"
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-002-trace.csv"), []byte(trace2), 0644))

	traceFiles, _ := filepath.Glob(filepath.Join(tmpDir, "chunk-*-trace.csv"))

	// WHEN findBestCoverageWindow is called
	bestWindow, err := findBestCoverageWindow(traceFiles)

	// THEN it picks the first window with max coverage (tie-breaking rule)
	require.NoError(t, err)
	assert.Equal(t, 0, bestWindow, "should pick first window when tied")
}

func TestFindBestCoverageWindow_AllRatesZero(t *testing.T) {
	// GIVEN trace files where all rates are 0
	tmpDir := t.TempDir()

	trace := "0,0.0,0.8,Gamma,2.0,300\n600,0.0,0.8,Gamma,2.0,300\n"
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-trace.csv"), []byte(trace), 0644))

	traceFiles, _ := filepath.Glob(filepath.Join(tmpDir, "chunk-*-trace.csv"))

	// WHEN findBestCoverageWindow is called
	_, err := findBestCoverageWindow(traceFiles)

	// THEN it returns error (zero coverage = total failure)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "all windows have zero coverage")
}

func TestFindBestCoverageWindow_SingleChunk(t *testing.T) {
	// GIVEN single trace file with one active window
	tmpDir := t.TempDir()

	// Window 7 has rate > 0, all others zero
	trace := ""
	for i := 0; i < 144; i++ {
		rate := 0.0
		if i == 7 {
			rate = 5.0
		}
		trace += fmt.Sprintf("%d,%.1f,0.8,Gamma,2.0,300\n", i*600, rate)
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-trace.csv"), []byte(trace), 0644))

	traceFiles, _ := filepath.Glob(filepath.Join(tmpDir, "chunk-*-trace.csv"))

	// WHEN findBestCoverageWindow is called
	bestWindow, err := findBestCoverageWindow(traceFiles)

	// THEN it picks the only active window
	require.NoError(t, err)
	assert.Equal(t, 7, bestWindow)
}

func TestDetectLanguageCategory_NoReasonRatio(t *testing.T) {
	// GIVEN ServeGen data without reason_ratio field (sufficient for success)
	tmpDir := t.TempDir()

	// Full-day trace to guarantee valid window selection
	var traceContent string
	for i := 0; i < 144; i++ {
		traceContent += fmt.Sprintf("%d,1.5,0.8,Gamma,2.0,300\n", i*600)
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-trace.csv"), []byte(traceContent), 0644))

	// Dataset WITHOUT reason_ratio (language workload)
	datasetContent := `{"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.5, 100: 0.5}"}}`
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-001-dataset.json"), []byte(datasetContent), 0644))

	// WHEN ConvertServeGen is called
	spec, err := ConvertServeGen(tmpDir, 600, 180)

	// THEN conversion succeeds and category is empty (language workload)
	require.NoError(t, err, "conversion should succeed with sufficient data")
	assert.Equal(t, "", spec.Category, "category should be empty (language, not reasoning)")
	assert.NotEqual(t, "reasoning", spec.Category, "reasoning should not be detected")
}

func TestAverageReasonRatioPMFs_PreservesBimodality(t *testing.T) {
	// GIVEN chunks with bimodal distributions (two peaks: low and high)
	chunks := []chunkData{
		// Chunk A: 80% at 10%, 20% at 90%
		{id: "A", reasonRatioPDF: map[float64]float64{0.1: 0.8, 0.9: 0.2}},
		// Chunk B: 70% at 10%, 30% at 90%
		{id: "B", reasonRatioPDF: map[float64]float64{0.1: 0.7, 0.9: 0.3}},
	}

	// WHEN averaged
	dist, err := averageReasonRatioPMFs(chunks)

	// THEN result is empirical distribution preserving bimodality
	require.NoError(t, err)
	assert.Equal(t, "empirical", dist.Type)
	assert.NotNil(t, dist.Params)

	// Should have two peaks at 10% and 90%
	assert.Contains(t, dist.Params, "10") // 10% key
	assert.Contains(t, dist.Params, "90") // 90% key

	// Average probabilities: (0.8+0.7)/2 = 0.75 and (0.2+0.3)/2 = 0.25
	assert.InDelta(t, 0.75, dist.Params["10"], 0.01)
	assert.InDelta(t, 0.25, dist.Params["90"], 0.01)
}

func TestAverageReasonRatioPMFs_EmptyChunks(t *testing.T) {
	// GIVEN empty chunks list
	chunks := []chunkData{}

	// WHEN averaged
	_, err := averageReasonRatioPMFs(chunks)

	// THEN it returns error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no chunks provided")
}
