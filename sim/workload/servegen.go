package workload

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
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
// If TimeWindow is specified, applies 30-minute temporal filtering and computes
// aggregate rate as peak within the window.
func loadServeGenData(spec *WorkloadSpec) error {
	dataDir := spec.ServeGenData.Path

	// Apply time window filtering if specified
	if spec.ServeGenData.TimeWindow != "" {
		start, end := getTimeWindowBounds(spec.ServeGenData.TimeWindow)
		if start == 0 && end == 0 {
			return fmt.Errorf("invalid time window %q (must be midnight, morning, or afternoon)", spec.ServeGenData.TimeWindow)
		}
		spec.ServeGenData.SpanStart = start
		spec.ServeGenData.SpanEnd = end
		logrus.Infof("loadServeGenData: applying time window %q (%ds - %ds)", spec.ServeGenData.TimeWindow, start, end)
	}

	// Find all chunk trace files
	traceFiles, err := filepath.Glob(filepath.Join(dataDir, "chunk-*-trace.csv"))
	if err != nil {
		return fmt.Errorf("scanning trace files: %w", err)
	}
	sort.Strings(traceFiles)

	if len(traceFiles) == 0 {
		return fmt.Errorf("no chunk-*-trace.csv files found in %s", dataDir)
	}

	// Track aggregate rate at each timestamp for peak calculation
	// Map[timestamp] -> sum of rates from all chunks at that timestamp
	aggregateRateAtTime := make(map[int64]float64)

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

			// Accumulate this chunk's rate at each timestamp for peak calculation
			for _, window := range client.Lifecycle.Windows {
				if window.TraceRate != nil {
					timestamp := window.StartUs / 1e6 // Convert to seconds
					aggregateRateAtTime[timestamp] += *window.TraceRate
				}
			}
		} else {
			logrus.Warnf("loadServeGenData: chunk %s produced no active windows (no matching dataset entries)", chunkID)
		}
	}

	if len(spec.Clients) == 0 {
		return fmt.Errorf("no valid chunks found in %s", dataDir)
	}

	// Set aggregate_rate to 0 to signal absolute rate mode (ServeGen temporal parity).
	// ServeGen workloads have time-varying aggregate load encoded in per-window trace_rate.
	// Setting aggregate_rate=0 tells the generator to use trace_rate as absolute rates
	// instead of proportional weights, preserving temporal variation.
	spec.AggregateRate = 0

	// Log the peak for informational purposes
	if spec.ServeGenData.TimeWindow != "" {
		var peakAggregate float64
		for _, rate := range aggregateRateAtTime {
			if rate > peakAggregate {
				peakAggregate = rate
			}
		}
		logrus.Infof("loadServeGenData: using absolute rate mode (aggregate_rate=0); peak rate was %.2f within %s window", peakAggregate, spec.ServeGenData.TimeWindow)
	}

	// Normalize lifecycle window timestamps to start from zero.
	// ServeGen traces contain absolute clock times (e.g., 8:00 AM = 28800s).
	// Without normalization, the generator waits for simulated time to reach
	// those absolute timestamps before dispatching requests.
	normalizeLifecycleTimestamps(&spec.Clients)

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

		// Filter dataset entries that are after the span end.
		// Keep entries before span start - they may be needed for nearest-preceding lookup.
		// Example: Hour 8 trace windows (28800-30600) need Hour 6 dataset (21600).
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

// getTimeWindowBounds converts a time window name to (start, end) bounds in seconds.
// Uses 30-minute windows aligned with ServeGen's diurnal analysis.
// Note: ServeGen datasets contain token distributions at 6-hour intervals (0, 21600, 43200, 64800...),
// while trace files have 10-minute granularity. Trace windows use nearest-preceding dataset entry.
// Returns (0, 0) if the window name is empty/invalid.
func getTimeWindowBounds(window string) (int64, int64) {
	switch window {
	case "midnight":
		// Hour 0:00-0:30 (0s - 1800s)
		// Trace windows at 0s, 600s, 1200s all use dataset entry at timestamp 0
		return 0, 1800
	case "morning":
		// Hour 8:00-8:30 (28800s - 30600s)
		// Trace windows at 28800s, 29400s, 30000s all use dataset entry at timestamp 21600 (Hour 6)
		return 28800, 30600
	case "afternoon":
		// Hour 14:00-14:30 (50400s - 52200s)
		// Trace windows at 50400s, 51000s, 51600s all use dataset entry at timestamp 43200 (Hour 12)
		return 50400, 52200
	default:
		return 0, 0
	}
}

// findNearestDataset finds the dataset entry for the given timestamp.
// If an exact match exists, returns it. Otherwise, returns the nearest-preceding
// dataset entry (largest timestamp <= queryTimestamp).
// Returns (datasetWindow{}, false) if no dataset entry exists at or before the timestamp.
//
// This handles ServeGen's dataset granularity mismatch: datasets have 6-hour intervals
// (0, 21600, 43200, 64800...) while traces have 10-minute intervals.
// Example: trace window at 28800s (Hour 8) uses dataset at 21600s (Hour 6).
func findNearestDataset(queryTimestamp int, datasetByTimestamp map[int]datasetWindow) (datasetWindow, int, bool) {
	// Try exact match first (fast path for timestamps like 0, 21600, 43200...)
	if dataset, ok := datasetByTimestamp[queryTimestamp]; ok {
		return dataset, queryTimestamp, true
	}

	// Find largest dataset timestamp <= queryTimestamp
	var bestTimestamp = -1
	for dsTimestamp := range datasetByTimestamp {
		if dsTimestamp <= queryTimestamp && dsTimestamp > bestTimestamp {
			bestTimestamp = dsTimestamp
		}
	}

	if bestTimestamp >= 0 {
		return datasetByTimestamp[bestTimestamp], bestTimestamp, true
	}

	// No dataset entry at or before this timestamp
	return datasetWindow{}, 0, false
}

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
	// Track unique datasets used across all windows to determine if we can
	// deduplicate distributions at client-level vs per-window.
	var windows []ActiveWindow
	datasetTimestampsUsed := make(map[int]bool)

	// First pass: collect windows and track which unique datasets are used
	type windowInfo struct {
		window     ActiveWindow
		datasetKey int
	}
	var windowInfos []windowInfo

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

		// Find which dataset this window will use (nearest-preceding).
		// ServeGen datasets have 6-hour granularity while traces have 10-minute granularity.
		// A trace window at Hour 8 uses the dataset from Hour 6, etc.
		_, datasetKey, ok := findNearestDataset(int(row.startTimeSec), datasetByTimestamp)
		if !ok {
			logrus.Debugf("loadServeGenChunk: no dataset for chunk %s at or before t=%.0f, skipping window", chunkID, row.startTimeSec)
			continue
		}

		// Track which unique dataset this window uses
		datasetTimestampsUsed[datasetKey] = true

		// Build per-window arrival spec.
		arrivalSpec := buildArrivalSpecFromRow(row)

		// Build per-window trace rate (copy to avoid pointer aliasing across loop iterations).
		traceRate := row.rate

		// Build window. Distributions will be set either at client-level or per-window
		// depending on whether all windows share the same dataset.
		window := ActiveWindow{
			StartUs:   int64(row.startTimeSec * 1e6),
			EndUs:     int64((row.startTimeSec + serveGenWindowDurationSec) * 1e6),
			TraceRate: &traceRate,
			Arrival:   &arrivalSpec,
			// InputDist/OutputDist set below after deduplication check
		}

		windowInfos = append(windowInfos, windowInfo{
			window:     window,
			datasetKey: datasetKey,
		})
	}

	if len(windowInfos) == 0 {
		return nil, nil // Inactive chunk: no active windows found.
	}

	// Determine if all windows use the same dataset.
	// If yes, fit distributions once at client-level and don't include in windows.
	// If no, fit per-window (rare for time-window extractions but possible for full-day).
	var clientInputDist, clientOutputDist DistSpec

	// Find the unique dataset timestamp(s) used
	uniqueDatasets := make([]int, 0, len(datasetTimestampsUsed))
	for ts := range datasetTimestampsUsed {
		uniqueDatasets = append(uniqueDatasets, ts)
	}

	if len(uniqueDatasets) == 1 {
		// All windows use the same dataset - fit once at client level
		datasetKey := uniqueDatasets[0]
		dataset := datasetByTimestamp[datasetKey]

		inputDist, err := fitLognormalFromPDF(dataset.inputPDF)
		if err != nil {
			return nil, fmt.Errorf("fitting input distribution: %w", err)
		}
		outputDist, err := fitLognormalFromPDF(dataset.outputPDF)
		if err != nil {
			return nil, fmt.Errorf("fitting output distribution: %w", err)
		}

		clientInputDist = inputDist
		clientOutputDist = outputDist

		// Build windows WITHOUT distribution overrides (use client-level)
		for _, info := range windowInfos {
			windows = append(windows, info.window)
		}
	} else {
		// Multiple datasets used - need per-window distributions
		// Fit each unique dataset once and cache it
		fittedDists := make(map[int]struct {
			input  DistSpec
			output DistSpec
		})

		for datasetKey := range datasetTimestampsUsed {
			dataset := datasetByTimestamp[datasetKey]

			inputDist, err := fitLognormalFromPDF(dataset.inputPDF)
			if err != nil {
				return nil, fmt.Errorf("fitting input distribution for dataset t=%d: %w", datasetKey, err)
			}
			outputDist, err := fitLognormalFromPDF(dataset.outputPDF)
			if err != nil {
				return nil, fmt.Errorf("fitting output distribution for dataset t=%d: %w", datasetKey, err)
			}

			fittedDists[datasetKey] = struct {
				input  DistSpec
				output DistSpec
			}{inputDist, outputDist}
		}

		// Build windows WITH distribution overrides from cache
		for _, info := range windowInfos {
			w := info.window
			dists := fittedDists[info.datasetKey]
			w.InputDist = &dists.input
			w.OutputDist = &dists.output
			windows = append(windows, w)
		}

		// Fallback client-level distributions (never used for sampling when per-window overrides exist)
		clientInputDist = DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 512, "std_dev": 128, "min": 1, "max": 32768}}
		clientOutputDist = DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "std_dev": 32, "min": 1, "max": 32768}}
	}

	// Build ClientSpec with per-window lifecycle.
	client := &ClientSpec{
		ID:           fmt.Sprintf("servegen-chunk-%s", chunkID),
		TenantID:     fmt.Sprintf("chunk-%s", chunkID),
		RateFraction: 1.0, // Normalized by proportional allocation in the generator.
		SLOClass:     "standard",
		Streaming:    true,

		Lifecycle: &LifecycleSpec{
			Windows: windows,
		},

		// Client-level distributions: either the single shared distribution
		// (if all windows use same dataset) or fallback defaults (if per-window).
		Arrival:    ArrivalSpec{Process: "poisson"},
		InputDist:  clientInputDist,
		OutputDist: clientOutputDist,
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

// fitLognormalFromPDF fits a lognormal distribution to an empirical PMF.
// Returns a DistSpec with type "lognormal" and parameters mu, sigma.
// Lognormal is appropriate for strictly positive, right-skewed distributions
// like token counts. Zero and negative values are filtered out automatically.
func fitLognormalFromPDF(pdf map[int]float64) (DistSpec, error) {
	if len(pdf) == 0 {
		return DistSpec{}, fmt.Errorf("empty PDF")
	}

	// Compute weighted mean and variance of log(tokens)
	// Filter out zero/negative values (lognormal requires positive domain)
	// Sort keys for deterministic float accumulation (R2)
	keys := make([]int, 0, len(pdf))
	for k := range pdf {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	var sumProb, sumLogX, sumLogXSq float64
	for _, value := range keys {
		prob := pdf[value]
		if value <= 0 {
			continue // skip zero/negative tokens
		}
		if prob < 0 {
			return DistSpec{}, fmt.Errorf("negative probability %f for value %d", prob, value)
		}
		if prob == 0 {
			continue // skip zero-probability values
		}
		logX := math.Log(float64(value))
		sumProb += prob
		sumLogX += prob * logX
		sumLogXSq += prob * logX * logX
	}

	// Normalize if needed (R11: division guard)
	if sumProb <= 0 {
		return DistSpec{}, fmt.Errorf("sum of probabilities is zero or negative")
	}
	mu := sumLogX / sumProb
	variance := (sumLogXSq / sumProb) - (mu * mu)

	// Variance must be non-negative (guard against numerical errors)
	if variance < 0 {
		if variance > -1e-10 {
			variance = 0 // numerical error
		} else {
			return DistSpec{}, fmt.Errorf("negative variance %f (numerical instability)", variance)
		}
	}
	sigma := math.Sqrt(variance)

	return DistSpec{
		Type: "lognormal",
		Params: map[string]float64{
			"mu":    mu,
			"sigma": sigma,
		},
	}, nil
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

// normalizeLifecycleTimestamps shifts all lifecycle window timestamps to start
// from zero while preserving relative timing. Finds the minimum StartUs across
// all clients' lifecycle windows and subtracts it from every StartUs and EndUs.
// No-op if clients list is empty or no clients have lifecycle windows.
func normalizeLifecycleTimestamps(clients *[]ClientSpec) {
	if len(*clients) == 0 {
		return
	}

	// Pass 1: Find global minimum StartUs across all clients and windows.
	minStartUs := int64(math.MaxInt64)
	for i := range *clients {
		client := &(*clients)[i]
		if client.Lifecycle == nil || len(client.Lifecycle.Windows) == 0 {
			continue
		}
		for _, window := range client.Lifecycle.Windows {
			if window.StartUs < minStartUs {
				minStartUs = window.StartUs
			}
		}
	}

	// If no windows found, nothing to normalize.
	if minStartUs == math.MaxInt64 {
		return
	}

	// Pass 2: Shift all timestamps by subtracting the minimum.
	for i := range *clients {
		client := &(*clients)[i]
		if client.Lifecycle == nil {
			continue
		}
		for j := range client.Lifecycle.Windows {
			client.Lifecycle.Windows[j].StartUs -= minStartUs
			client.Lifecycle.Windows[j].EndUs -= minStartUs
		}
	}
}
