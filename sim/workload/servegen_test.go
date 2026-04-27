package workload

import (
	"bytes"
	"fmt"
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
	traceCSV := "0.0,22.46,173.81,Weibull,0.0575,0.000573\n1.0,22.46,173.81,Weibull,0.0575,0.000573\n"
	require.NoError(t, os.WriteFile(filepath.Join(dir, "chunk-0-trace.csv"), []byte(traceCSV), 0644))

	// Minimal dataset with empirical PDFs
	datasetJSON := `{"0": {"input_tokens": "{256: 1.0}", "output_tokens": "{100: 1.0}"}}`
	require.NoError(t, os.WriteFile(filepath.Join(dir, "chunk-0-dataset.json"), []byte(datasetJSON), 0644))

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

	// AND the loaded client has per-window weibull with MLE-fitted parameters
	require.NotEmpty(t, spec.Clients, "ServeGen should populate clients")
	client := spec.Clients[0]
	// With temporal preservation, arrival params are per-window.
	require.NotNil(t, client.Lifecycle)
	require.Greater(t, len(client.Lifecycle.Windows), 0)
	w := client.Lifecycle.Windows[0]
	require.NotNil(t, w.Arrival)
	assert.Equal(t, "weibull", w.Arrival.Process)
	require.NotNil(t, w.Arrival.Shape, "Shape should be populated from trace column 5")
	require.NotNil(t, w.Arrival.Scale, "Scale should be populated from trace column 6")
	assert.InDelta(t, 0.0575, *w.Arrival.Shape, 0.0001, "Shape should match trace value")
	// Scale converted from seconds (0.000573) to microseconds (573.0)
	assert.InDelta(t, 0.000573*1e6, *w.Arrival.Scale, 0.001, "Scale should be in microseconds")

	// AND the CV is preserved as informational metadata
	require.NotNil(t, w.Arrival.CV, "CV should be preserved from trace")
	assert.InDelta(t, 173.81, *w.Arrival.CV, 0.01, "CV should match trace value")

	// AND sampled IATs are finite and positive (behavioral verification)
	ratePerMicros := 22.46 / 1e6 // Use trace rate directly
	sampler := NewArrivalSampler(*w.Arrival, ratePerMicros)
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

func TestServeGenDataLoading_SyntheticDataset_ProducesClients(t *testing.T) {
	dir := t.TempDir()
	// Create chunk-0-trace.csv
	// Scale parameters in seconds: Gamma(shape=0.16, scale=6.25s), Weibull(shape=1.0, scale=2.0s)
	traceCSV := "0,1.0,2.5,Gamma,0.16,6.25\n600,0.5,1.0,Weibull,1.0,2.0\n"
	if err := os.WriteFile(filepath.Join(dir, "chunk-0-trace.csv"), []byte(traceCSV), 0644); err != nil {
		t.Fatal(err)
	}
	// Create chunk-0-dataset.json - needs entries for both trace timestamps
	datasetJSON := `{"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}, "600": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}}`
	if err := os.WriteFile(filepath.Join(dir, "chunk-0-dataset.json"), []byte(datasetJSON), 0644); err != nil {
		t.Fatal(err)
	}

	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		ServeGenData: &ServeGenDataSpec{Path: dir},
	}
	// Load to verify client creation with per-window shape/scale parameters
	if err := loadServeGenData(spec); err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("expected 1 client, got %d", len(spec.Clients))
	}
	client := spec.Clients[0]
	// With temporal preservation, parameters are per-window.
	require.NotNil(t, client.Lifecycle, "Client should have lifecycle with windows")
	require.Greater(t, len(client.Lifecycle.Windows), 0, "Client should have at least one window")
	w := client.Lifecycle.Windows[0]
	require.NotNil(t, w.Arrival, "Window should have arrival spec")
	// Verify MLE-fitted shape/scale parameters were populated from 6-column trace
	require.NotNil(t, w.Arrival.Shape, "Shape should be populated from trace column 5")
	require.NotNil(t, w.Arrival.Scale, "Scale should be populated from trace column 6")
	// Scale converted from seconds (6.25) to microseconds (6.25 * 1e6)
	assert.Equal(t, 0.16, *w.Arrival.Shape, "Shape should match trace value")
	assert.Equal(t, 6.25e6, *w.Arrival.Scale, "Scale should be converted to microseconds")

	// Clear ServeGenData since clients are now populated
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
	// Verify input token lengths come from the empirical PDF (around 100 or 200)
	// Lognormal can have tail samples slightly outside the core range
	for _, req := range requests[:min(10, len(requests))] {
		l := len(req.InputTokens)
		if l < 50 || l > 350 {
			t.Errorf("input length %d outside expected range [50, 350]", l)
		}
	}
}
