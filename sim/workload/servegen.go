package workload

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// parseServeGenPDF parses a Python dict string like "{100: 0.5, 200: 0.3}"
// into a Go map[int]float64. Handles scientific notation, trailing commas,
// and extra whitespace.
func parseServeGenPDF(s string) (map[int]float64, error) {
	// Strip outer braces
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "{") || !strings.HasSuffix(s, "}") {
		return nil, fmt.Errorf("expected dict string starting with '{' and ending with '}', got: %.40s", s)
	}
	s = s[1 : len(s)-1]
	s = strings.TrimSpace(s)

	if s == "" {
		return nil, fmt.Errorf("empty PDF dictionary")
	}

	pdf := make(map[int]float64)

	// Split by comma, parse each "key: value" pair
	pairs := strings.Split(s, ",")
	for _, pair := range pairs {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue // trailing comma
		}
		parts := strings.SplitN(pair, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid key:value pair: %q", pair)
		}

		keyStr := strings.TrimSpace(parts[0])
		valStr := strings.TrimSpace(parts[1])

		// Parse key as int (may have .0 suffix from Python)
		keyFloat, err := strconv.ParseFloat(keyStr, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid key %q: %w", keyStr, err)
		}
		key := int(keyFloat)

		// Parse value as float64 (supports scientific notation)
		val, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid value %q for key %d: %w", valStr, key, err)
		}

		pdf[key] = val
	}

	if len(pdf) == 0 {
		return nil, fmt.Errorf("no valid entries in PDF dictionary")
	}
	return pdf, nil
}

// serveGenTraceRow represents one row from a ServeGen chunk-*-trace.csv.
// Format: start_time(s), rate(req/s), cv, pattern_type, param1, param2
type serveGenTraceRow struct {
	startTimeSec float64
	rate         float64
	cv           float64
	pattern      string // "Gamma", "Weibull", or empty
}

// loadServeGenData loads ServeGen data files and populates the spec's Clients list.
// Scans for chunk-*-trace.csv and chunk-*-dataset.json files.
func loadServeGenData(spec *WorkloadSpec) error {
	dataDir := spec.ServeGenData.Path

	// Find all chunk trace files
	traceFiles, err := filepath.Glob(filepath.Join(dataDir, "chunk-*-trace.csv"))
	if err != nil {
		return fmt.Errorf("scanning trace files: %w", err)
	}
	sort.Strings(traceFiles)

	if len(traceFiles) == 0 {
		return fmt.Errorf("no chunk-*-trace.csv files found in %s", dataDir)
	}

	for _, tracePath := range traceFiles {
		// Derive chunk ID from filename
		base := filepath.Base(tracePath)
		// "chunk-0-trace.csv" → "0"
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")

		// Load corresponding dataset
		datasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))

		client, err := loadServeGenChunk(chunkID, tracePath, datasetPath, spec.ServeGenData)
		if err != nil {
			return fmt.Errorf("loading chunk %s: %w", chunkID, err)
		}
		if client != nil {
			spec.Clients = append(spec.Clients, *client)
		}
	}

	if len(spec.Clients) == 0 {
		return fmt.Errorf("no valid chunks found in %s", dataDir)
	}
	return nil
}

// loadServeGenChunk loads a single chunk's trace + dataset into a ClientSpec.
func loadServeGenChunk(chunkID, tracePath, datasetPath string, sgConfig *ServeGenDataSpec) (*ClientSpec, error) {
	// Parse trace CSV for arrival pattern
	rows, err := parseServeGenTrace(tracePath)
	if err != nil {
		return nil, err
	}

	// Find the best (highest rate) window for arrival parameters
	var bestRow serveGenTraceRow
	for _, row := range rows {
		// Filter by time span if configured
		if sgConfig.SpanStart > 0 && row.startTimeSec < float64(sgConfig.SpanStart) {
			continue
		}
		if sgConfig.SpanEnd > 0 && row.startTimeSec >= float64(sgConfig.SpanEnd) {
			continue
		}
		if row.rate > bestRow.rate {
			bestRow = row
		}
	}
	if bestRow.rate <= 0 {
		return nil, nil // skip inactive chunks
	}

	// Load dataset JSON for empirical PDFs
	inputPDF, outputPDF, err := loadServeGenDataset(datasetPath, sgConfig)
	if err != nil {
		return nil, err
	}

	// Build ArrivalSpec from trace pattern
	arrivalSpec := ArrivalSpec{Process: "poisson"} // default
	if bestRow.pattern != "" {
		process := strings.ToLower(bestRow.pattern)
		if process == "gamma" || process == "weibull" {
			arrivalSpec.Process = process
			cv := bestRow.cv
			arrivalSpec.CV = &cv
		}
	}

	// Build ClientSpec
	client := &ClientSpec{
		ID:           fmt.Sprintf("servegen-chunk-%s", chunkID),
		TenantID:     fmt.Sprintf("chunk-%s", chunkID),
		RateFraction: bestRow.rate, // will be normalized later
		Arrival:      arrivalSpec,
		InputDist:    DistSpec{Type: "empirical"},
		OutputDist:   DistSpec{Type: "empirical"},
	}

	// Store PDFs — we need to convert map[int]float64 to EmpiricalPDFSampler later.
	// For now, store in Params as string-keyed map (matching DistSpec.Params type).
	client.InputDist.Params = intMapToStringMap(inputPDF)
	client.OutputDist.Params = intMapToStringMap(outputPDF)

	return client, nil
}

func intMapToStringMap(m map[int]float64) map[string]float64 {
	result := make(map[string]float64, len(m))
	for k, v := range m {
		result[strconv.Itoa(k)] = v
	}
	return result
}

func parseServeGenTrace(path string) ([]serveGenTraceRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening trace: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	var rows []serveGenTraceRow
	skippedRows := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading trace CSV: %w", err)
		}
		if len(record) < 4 {
			skippedRows++
			continue
		}
		startTime, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		rate, err := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		cv, err := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		pattern := strings.TrimSpace(record[3])

		rows = append(rows, serveGenTraceRow{
			startTimeSec: startTime,
			rate:         rate,
			cv:           cv,
			pattern:      pattern,
		})
	}
	if skippedRows > 0 {
		logrus.Warnf("parseServeGenTrace: %d rows in %s were skipped (short rows or parse errors)", skippedRows, path)
	}
	return rows, nil
}

func loadServeGenDataset(path string, sgConfig *ServeGenDataSpec) (map[int]float64, map[int]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("reading dataset: %w", err)
	}

	// Parse JSON: map of window_start_time → {input_tokens: "...", output_tokens: "..."}
	var dataset map[string]map[string]string
	if err := json.Unmarshal(data, &dataset); err != nil {
		return nil, nil, fmt.Errorf("parsing dataset JSON: %w", err)
	}

	// Find the first valid window (or the one matching span)
	var inputPDFStr, outputPDFStr string
	// Sort keys for determinism
	keys := make([]string, 0, len(dataset))
	for k := range dataset {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		window := dataset[k]
		startTime, _ := strconv.ParseFloat(k, 64)
		if sgConfig.SpanStart > 0 && startTime < float64(sgConfig.SpanStart) {
			continue
		}
		if sgConfig.SpanEnd > 0 && startTime >= float64(sgConfig.SpanEnd) {
			continue
		}
		inputPDFStr = window["input_tokens"]
		outputPDFStr = window["output_tokens"]
		if inputPDFStr != "" && outputPDFStr != "" {
			break
		}
	}

	if inputPDFStr == "" || outputPDFStr == "" {
		return nil, nil, fmt.Errorf("no valid PDF windows found in dataset")
	}

	inputPDF, err := parseServeGenPDF(inputPDFStr)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing input PDF: %w", err)
	}
	outputPDF, err := parseServeGenPDF(outputPDFStr)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing output PDF: %w", err)
	}

	return inputPDF, outputPDF, nil
}
