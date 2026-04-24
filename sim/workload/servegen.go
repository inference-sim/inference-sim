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
	shapeParam   float64
	scaleParam   float64
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

// datasetWindow holds parsed PDFs for a single timestamp window.
type datasetWindow struct {
	inputPDF  map[int]float64
	outputPDF map[int]float64
}

// loadServeGenDatasetAllWindows loads per-window token distributions from
// a ServeGen dataset JSON file. Returns map[timestamp] -> {inputPDF, outputPDF}.
// Skips empty windows (represented as "{}" in JSON) and applies span filtering.
// Non-numeric keys are skipped with a warning.
func loadServeGenDatasetAllWindows(path string, sgConfig *ServeGenDataSpec) (map[int]datasetWindow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading dataset: %w", err)
	}

	// Parse JSON: map[timestamp_str] -> {input_tokens: "...", output_tokens: "..."}
	var raw map[string]map[string]string
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parsing dataset JSON: %w", err)
	}

	result := make(map[int]datasetWindow)

	for tsStr, window := range raw {
		timestamp, parseErr := strconv.Atoi(tsStr)
		if parseErr != nil {
			// Try float parse for keys like "600.0"
			tsFloat, floatErr := strconv.ParseFloat(tsStr, 64)
			if floatErr != nil {
				logrus.Warnf("loadServeGenDatasetAllWindows: skipping non-numeric key %q", tsStr)
				continue
			}
			timestamp = int(tsFloat)
		}

		// Filter by span
		if sgConfig.SpanStart > 0 && int64(timestamp) < sgConfig.SpanStart {
			continue
		}
		if sgConfig.SpanEnd > 0 && int64(timestamp) >= sgConfig.SpanEnd {
			continue
		}

		// Parse PDFs
		inputPDFStr := window["input_tokens"]
		outputPDFStr := window["output_tokens"]

		// Skip empty windows
		if inputPDFStr == "" || inputPDFStr == "{}" ||
			outputPDFStr == "" || outputPDFStr == "{}" {
			logrus.Debugf("loadServeGenDatasetAllWindows: skipping empty window at t=%d", timestamp)
			continue
		}

		inputPDF, parseInputErr := parseServeGenPDF(inputPDFStr)
		if parseInputErr != nil {
			return nil, fmt.Errorf("parsing input PDF at timestamp %d: %w", timestamp, parseInputErr)
		}

		outputPDF, parseOutputErr := parseServeGenPDF(outputPDFStr)
		if parseOutputErr != nil {
			return nil, fmt.Errorf("parsing output PDF at timestamp %d: %w", timestamp, parseOutputErr)
		}

		result[timestamp] = datasetWindow{
			inputPDF:  inputPDF,
			outputPDF: outputPDF,
		}
	}

	return result, nil
}

// serveGenWindowDurationSec is the standard ServeGen window duration (10 minutes).
const serveGenWindowDurationSec = 600

// loadServeGenChunk loads a single chunk's trace + dataset into a ClientSpec
// with per-window temporal parameters. Each active trace row (rate > 0) that has
// a matching dataset entry becomes a lifecycle window with per-window arrival
// parameters, token distributions, and trace rate. Inactive windows (rate=0) and
// windows without matching dataset entries are skipped.
//
// The resulting ClientSpec uses per-window overrides on ActiveWindow, triggering
// the time-varying generator path in GenerateRequests. Client-level fields
// provide fallback defaults (used only if a window lacks overrides).
func loadServeGenChunk(chunkID, tracePath, datasetPath string, sgConfig *ServeGenDataSpec) (*ClientSpec, error) {
	// Parse all trace rows.
	rows, err := parseServeGenTrace(tracePath)
	if err != nil {
		return nil, err
	}

	// Load per-window distributions from dataset JSON.
	datasetByTimestamp, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
	if err != nil {
		return nil, err
	}

	// Build lifecycle windows from trace rows + matching dataset windows.
	var windows []ActiveWindow

	for _, row := range rows {
		// Filter by time span if configured.
		if sgConfig.SpanStart > 0 && row.startTimeSec < float64(sgConfig.SpanStart) {
			continue
		}
		if sgConfig.SpanEnd > 0 && row.startTimeSec >= float64(sgConfig.SpanEnd) {
			continue
		}

		// Skip inactive windows (rate = 0).
		if row.rate <= 0 {
			continue
		}

		// Get distributions for this timestamp.
		dataset, ok := datasetByTimestamp[int(row.startTimeSec)]
		if !ok {
			logrus.Debugf("loadServeGenChunk: no dataset for chunk %s at t=%.0f, skipping window", chunkID, row.startTimeSec)
			continue
		}

		// Build per-window arrival spec.
		arrivalSpec := buildArrivalSpecFromRow(row)

		// Build per-window trace rate (copy to avoid pointer aliasing across loop iterations).
		traceRate := row.rate

		// Build window with per-window parameters.
		window := ActiveWindow{
			StartUs:   int64(row.startTimeSec * 1e6),
			EndUs:     int64((row.startTimeSec + serveGenWindowDurationSec) * 1e6),
			TraceRate: &traceRate,
			Arrival:   &arrivalSpec,
			InputDist: &DistSpec{
				Type:   "empirical",
				Params: intMapToStringMap(dataset.inputPDF),
			},
			OutputDist: &DistSpec{
				Type:   "empirical",
				Params: intMapToStringMap(dataset.outputPDF),
			},
		}

		windows = append(windows, window)
	}

	if len(windows) == 0 {
		return nil, nil // Inactive chunk: no active windows found.
	}

	// Build ClientSpec with per-window lifecycle.
	// Client-level fields are fallback defaults; all windows have full overrides.
	client := &ClientSpec{
		ID:           fmt.Sprintf("servegen-chunk-%s", chunkID),
		TenantID:     fmt.Sprintf("chunk-%s", chunkID),
		RateFraction: 1.0, // Normalized by proportional allocation in the generator.
		SLOClass:     "standard",
		Streaming:    true,

		Lifecycle: &LifecycleSpec{
			Windows: windows,
		},

		// Client-level defaults (fallback only; each window has full overrides).
		Arrival:    ArrivalSpec{Process: "poisson"},
		InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 512, "stddev": 128}},
		OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "stddev": 32}},
	}

	return client, nil
}

// buildArrivalSpecFromRow constructs an ArrivalSpec from a ServeGen trace row.
// Maps the pattern field (Gamma, Weibull) to the corresponding process name,
// sets CV from the trace row, and populates MLE-fitted shape/scale when
// positive values are present (columns 5-6). Scale is converted from
// ServeGen seconds to BLIS microseconds.
func buildArrivalSpecFromRow(row serveGenTraceRow) ArrivalSpec {
	spec := ArrivalSpec{Process: "poisson"} // default

	if row.pattern == "" {
		return spec
	}

	process := strings.ToLower(row.pattern)
	if process != "gamma" && process != "weibull" {
		return spec
	}

	spec.Process = process
	if row.cv > 0 {
		cv := row.cv
		spec.CV = &cv
	}

	// Store MLE-fitted parameters from ServeGen trace columns 5-6.
	// Only set when both values are positive -- zero means the trace
	// had only 4 columns or the parse fell back to defaults. Nil
	// pointers signal "derive from CV" downstream.
	if row.shapeParam > 0 && row.scaleParam > 0 {
		shape := row.shapeParam
		// Convert scale from seconds (ServeGen units) to microseconds (BLIS units).
		scale := row.scaleParam * 1e6
		spec.Shape = &shape
		spec.Scale = &scale
	}

	return spec
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

		// Parse shape and scale parameters (columns 5-6)
		var shapeParam, scaleParam float64
		if len(record) >= 6 {
			shape, shapeErr := strconv.ParseFloat(strings.TrimSpace(record[4]), 64)
			scale, scaleErr := strconv.ParseFloat(strings.TrimSpace(record[5]), 64)
			if shapeErr != nil || scaleErr != nil {
				logrus.Debugf("parseServeGenTrace: row at t=%.0f has non-numeric shape/scale, falling back to 0", startTime)
			} else {
				shapeParam = shape
				scaleParam = scale
			}
		} else if len(record) == 5 {
			// Anomalous case: 5 columns means one of shape/scale is missing
			logrus.Warnf("parseServeGenTrace: row at t=%.0f has 5 columns (expected 4 or 6); shape/scale will be derived from CV", startTime)
		}

		rows = append(rows, serveGenTraceRow{
			startTimeSec: startTime,
			rate:         rate,
			cv:           cv,
			pattern:      pattern,
			shapeParam:   shapeParam,
			scaleParam:   scaleParam,
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
		startTime, parseErr := strconv.ParseFloat(k, 64)
		if parseErr != nil {
			logrus.Warnf("loadServeGenDataset: skipping non-numeric key %q: %v", k, parseErr)
			continue
		}
		if sgConfig.SpanStart > 0 && startTime < float64(sgConfig.SpanStart) {
			continue
		}
		if sgConfig.SpanEnd > 0 && startTime >= float64(sgConfig.SpanEnd) {
			continue
		}
		inputPDFStr = window["input_tokens"]
		outputPDFStr = window["output_tokens"]
		// Skip empty dicts (represented as "{}" string) and truly empty strings
		// Matches ServeGen Python library behavior (clientpool.py:166-168)
		if inputPDFStr != "" && inputPDFStr != "{}" &&
			outputPDFStr != "" && outputPDFStr != "{}" {
			break
		}
		// Log skipped windows for debugging (common in real ServeGen data warm-up periods)
		logrus.Debugf("loadServeGenDataset: skipping window %q: input=%q output=%q (empty dict or missing)", k, inputPDFStr, outputPDFStr)
	}

	if inputPDFStr == "" || inputPDFStr == "{}" ||
		outputPDFStr == "" || outputPDFStr == "{}" {
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
