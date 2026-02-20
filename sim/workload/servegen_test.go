package workload

import (
	"bytes"
	"fmt"
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

func TestServeGenDataLoading_SyntheticDataset_ProducesClients(t *testing.T) {
	dir := t.TempDir()
	// Create chunk-0-trace.csv
	traceCSV := "0,1.0,2.5,Gamma,0.16,6.25\n600,0.5,1.0,Weibull,1.0,2000000\n"
	if err := os.WriteFile(filepath.Join(dir, "chunk-0-trace.csv"), []byte(traceCSV), 0644); err != nil {
		t.Fatal(err)
	}
	// Create chunk-0-dataset.json
	datasetJSON := `{"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}}`
	if err := os.WriteFile(filepath.Join(dir, "chunk-0-dataset.json"), []byte(datasetJSON), 0644); err != nil {
		t.Fatal(err)
	}

	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		ServeGenData: &ServeGenDataSpec{Path: dir},
	}
	requests, err := GenerateRequests(spec, 1e6, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from ServeGen data")
	}
	// Verify input token lengths come from the empirical PDF (around 100 or 200)
	for _, req := range requests[:min(10, len(requests))] {
		l := len(req.InputTokens)
		if l < 50 || l > 300 {
			t.Errorf("input length %d outside expected range [50, 300]", l)
		}
	}
}
