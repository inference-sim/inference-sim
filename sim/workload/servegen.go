package workload

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
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

// parseServeGenFloatPDF parses a Python dict string like {"0.0": 0.1, "0.5": 0.3}
// into a Go map[float64]float64. Used for reason_ratio distributions.
// Validates that all keys are in [0.0, 1.0] range.
func parseServeGenFloatPDF(s string) (map[float64]float64, error) {
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

	pdf := make(map[float64]float64)

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

		// Strip quotes from key (JSON format)
		keyStr = strings.Trim(keyStr, "\"")

		// Parse key as float64
		key, err := strconv.ParseFloat(keyStr, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid key %q: %w", keyStr, err)
		}

		// Validate range [0.0, 1.0]
		if key < 0.0 || key > 1.0 {
			return nil, fmt.Errorf("reason_ratio key %f outside valid range [0.0, 1.0]", key)
		}

		// Parse value as float64 (probability)
		val, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid value %q for key %f: %w", valStr, key, err)
		}

		if val < 0 {
			return nil, fmt.Errorf("negative probability %f for key %f", val, key)
		}

		pdf[key] = val
	}

	if len(pdf) == 0 {
		return nil, fmt.Errorf("no valid entries in PDF dictionary")
	}
	return pdf, nil
}

// classifyReasoningChunks splits chunks into low-reasoning and high-reasoning
// groups based on mean reason_ratio threshold. Returns error if any chunk has
// empty reason_ratio PDF.
func classifyReasoningChunks(chunks []chunkData, threshold float64) (low, high []chunkData, err error) {
	for _, chunk := range chunks {
		if len(chunk.reasonRatioPDF) == 0 {
			return nil, nil, fmt.Errorf("chunk %s: empty reason_ratio PDF", chunk.id)
		}

		// Compute weighted mean ratio
		var sumRatio, sumProb float64
		for ratio, prob := range chunk.reasonRatioPDF {
			if prob > 0 {
				sumRatio += ratio * prob
				sumProb += prob
			}
		}

		if sumProb <= 0 {
			return nil, nil, fmt.Errorf("chunk %s: sum of probabilities is zero", chunk.id)
		}

		meanRatio := sumRatio / sumProb

		if meanRatio < threshold {
			low = append(low, chunk)
		} else {
			high = append(high, chunk)
		}
	}
	return low, high, nil
}

// chunkData represents a single ServeGen chunk during loading.
type chunkData struct {
	id             string
	client         *ClientSpec // Temporarily use ClientSpec for loading; will convert to cohort
	datasetPath    string      // For lognormal fitting in cohort loop
	reasonRatioPDF map[float64]float64 // For reasoning workloads
}

// serveGenTraceRow represents one row from a ServeGen chunk-*-trace.csv.
// Format: start_time(s), rate(req/s), cv, pattern_type, param1, param2
type serveGenTraceRow struct{
	startTimeSec float64
	rate         float64
	cv           float64
	pattern      string // "Gamma", "Weibull", or empty
	shapeParam   float64
	scaleParam   float64
}

// ServeGen Day 1 time period boundaries (in seconds since midnight)
const (
	midnightSpanStart  = 0     // 00:00
	midnightSpanEnd    = 1800  // 00:30
	morningSpanStart   = 28800 // 08:00
	morningSpanEnd     = 30600 // 08:30
	afternoonSpanStart = 50400 // 14:00
	afternoonSpanEnd   = 52200 // 14:30
)

// periodInfo holds metadata for one time period (midnight, morning, afternoon).
type periodInfo struct {
	name       string // "midnight", "morning", "afternoon"
	startUs    int64  // Start time in microseconds
	durationUs int64  // Duration in microseconds
	spanStart  int64  // ServeGen time range start (seconds)
	spanEnd    int64  // ServeGen time range end (seconds)
}

// loadServeGenData loads ServeGen data files and populates the spec's Cohorts list.
// Implements multi-period conversion: groups chunks into 3 time periods (midnight, morning, afternoon),
// assigns to 5 SLO classes via round-robin, aggregates into CohortSpec with averaged params and summed rates.
func loadServeGenData(spec *WorkloadSpec) error {
	dataDir := spec.ServeGenData.Path
	windowDurSec := spec.ServeGenData.WindowDurationSecs
	drainSec := spec.ServeGenData.DrainTimeoutSecs

	// Apply defaults if not set (for backwards compatibility with tests)
	if windowDurSec <= 0 {
		windowDurSec = 600 // 10 minutes default
	}
	if drainSec < 0 {
		drainSec = 180 // 3 minutes default
	}

	// BC-7: Deterministic RNG from spec.Seed
	rng := rand.New(rand.NewSource(spec.Seed))

	// Define three time periods (ServeGen Day 1 spans 0-86400s; each period is 30 minutes)
	// startUs values create gaps between periods for drain timeout
	periods := []periodInfo{
		{
			name:       "midnight",
			startUs:    0,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart:  midnightSpanStart,
			spanEnd:    midnightSpanEnd,
		},
		{
			name:       "morning",
			startUs:    int64(windowDurSec+drainSec) * 1e6,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart:  morningSpanStart,
			spanEnd:    morningSpanEnd,
		},
		{
			name:       "afternoon",
			startUs:    int64(2*(windowDurSec+drainSec)) * 1e6,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart:  afternoonSpanStart,
			spanEnd:    afternoonSpanEnd,
		},
	}

	// Window selection happens per-cohort (not per-period) below

	// Find all chunk trace files
	traceFiles, err := filepath.Glob(filepath.Join(dataDir, "chunk-*-trace.csv"))
	if err != nil {
		return fmt.Errorf("scanning trace files: %w", err)
	}
	sort.Strings(traceFiles)

	// Detect reasoning category by checking first chunk's dataset for reason_ratio field
	isReasoningWorkload := false
	if len(traceFiles) > 0 {
		firstTrace := traceFiles[0]
		base := filepath.Base(firstTrace)
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")
		firstDatasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))

		// Load first dataset window to check for reason_ratio
		data, readErr := os.ReadFile(firstDatasetPath)
		if readErr == nil {
			var raw map[string]map[string]string
			if jsonErr := json.Unmarshal(data, &raw); jsonErr == nil {
				// Check first window for reason_ratio field
				for _, window := range raw {
					if _, hasReasonRatio := window["reason_ratio"]; hasReasonRatio {
						isReasoningWorkload = true
						spec.Category = "reasoning"
						logrus.Infof("Detected reasoning workload (reason_ratio field found in chunk %s)", chunkID)
						break
					}
				}
			}
		}
	}

	// Route to appropriate conversion path
	if isReasoningWorkload {
		return buildReasoningCohorts(spec, traceFiles, rng)
	}

	if len(traceFiles) == 0 {
		return fmt.Errorf("no chunk-*-trace.csv files found in %s", dataDir)
	}

	// Load all chunks (no time filtering)
	var allChunks []chunkData

	for _, tracePath := range traceFiles {
		base := filepath.Base(tracePath)
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")

		datasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))

		// Load chunk without time filtering (pass empty ServeGenDataSpec with just path)
		client, err := loadServeGenChunk(chunkID, tracePath, datasetPath, &ServeGenDataSpec{Path: dataDir})
		if err != nil {
			return fmt.Errorf("loading chunk %s: %w", chunkID, err)
		}
		if client != nil {
			allChunks = append(allChunks, chunkData{id: chunkID, client: client, datasetPath: datasetPath})
		}
	}

	if len(allChunks) == 0 {
		return fmt.Errorf("no valid chunks found in %s", dataDir)
	}

	// BC-2: For each period, select one of 3 windows uniformly at random.
	// Then split chunks active in that window across 5 SLO cohorts.

	sloClasses := []string{"critical", "standard", "batch", "sheddable", "background"}

	type cohortKey struct {
		period   int
		sloClass string
	}

	// Pre-select one window per period (not per cohort)
	periodWindows := make(map[int]int64) // Maps period index -> selected window start (seconds)
	for periodIdx, period := range periods {
		windowIndex := rng.Intn(3) // Pick window 0, 1, or 2
		windowStart := period.spanStart + int64(windowIndex*600)
		periodWindows[periodIdx] = windowStart
	}

	// Group chunks by period based on activity at selected window across ALL days
	// Use modulo matching: timestamp % 86400 == selectedWindow to aggregate multi-day traces
	// FIXED: Allow chunks to be assigned to multiple periods (removed "first match wins" constraint)
	periodChunks := make(map[int][]chunkData)
	for _, chunk := range allChunks {
		if chunk.client.Lifecycle == nil || len(chunk.client.Lifecycle.Windows) == 0 {
			continue
		}

		// Check which period(s) this chunk is active in
		for periodIdx, selectedWindow := range periodWindows {
			hasActivity := false
			for _, window := range chunk.client.Lifecycle.Windows {
				windowStartSec := window.StartUs / 1e6
				// Match any timestamp at the same time-of-day across all days
				if int64(windowStartSec)%86400 == selectedWindow {
					hasActivity = true
					break
				}
			}
			if hasActivity {
				// Chunks can be assigned to multiple periods if active in multiple time windows
				periodChunks[periodIdx] = append(periodChunks[periodIdx], chunk)
			}
		}
	}

	// Split each period's chunks into 5 SLO cohorts using round-robin
	cohortGroups := make(map[cohortKey][]chunkData)

	// INV-6: Sort period indices for deterministic RNG consumption
	// Map iteration is non-deterministic in Go; shuffling periods in different
	// orders changes the RNG state consumed and breaks determinism.
	periodIndices := make([]int, 0, len(periodChunks))
	for idx := range periodChunks {
		periodIndices = append(periodIndices, idx)
	}
	sort.Ints(periodIndices)

	for _, periodIdx := range periodIndices {
		chunks := periodChunks[periodIdx]
		// Shuffle chunks for this period to randomize SLO assignment
		shuffled := make([]chunkData, len(chunks))
		copy(shuffled, chunks)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		// Round-robin assignment to 5 SLO classes
		for i, chunk := range shuffled {
			sloClass := sloClasses[i%5]
			key := cohortKey{period: periodIdx, sloClass: sloClass}
			cohortGroups[key] = append(cohortGroups[key], chunk)
		}
	}

	// BC-7: Sort cohort keys for deterministic ordering
	// Iterate by period index first, then by SLO class (alphabetical)
	type sortableKey struct {
		period   int
		sloClass string
	}
	var keys []sortableKey
	for key := range cohortGroups {
		keys = append(keys, sortableKey(key))
	}
	// Sort by period, then by SLO class
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].period != keys[j].period {
			return keys[i].period < keys[j].period
		}
		return keys[i].sloClass < keys[j].sloClass
	})

	// BC-8: Build cohorts (skip empty groups)
	for _, sortedKey := range keys {
		key := cohortKey(sortedKey)
		chunks := cohortGroups[key]
		if len(chunks) == 0 {
			continue
		}

		period := periods[key.period]
		cohortID := fmt.Sprintf("%s-%s", period.name, key.sloClass)
		selectedWindow := periodWindows[key.period] // This period's selected window

		// BC-3: Fit lognormal distributions from each chunk's dataset at the cohort's window.
		// Load the actual dataset file and find the nearest-preceding PDF entry,
		// then average the fitted mu/sigma across all chunks in this cohort.
		var sumMuInput, sumSigmaInput, sumMuOutput, sumSigmaOutput float64
		var distCount int
		var totalRate float64

		for _, chunk := range chunks {
			// Load this chunk's dataset to fit lognormal for this cohort's window
			datasets, dsErr := loadServeGenDatasetAllWindows(chunk.datasetPath, &ServeGenDataSpec{Path: dataDir})
			if dsErr != nil {
				logrus.Warnf("Skipping chunk %s for cohort %s: dataset load failed: %v", chunk.id, cohortID, dsErr)
				continue
			}

			// Aggregate across ALL days at the same time-of-day (consistent with rate/arrival param averaging)
			// First find the nearest 6-hour boundary within day 1 using the existing findNearestDataset logic
			_, nearestBoundary, found := findNearestDataset(int(selectedWindow), datasets)
			if !found {
				continue
			}

			// Now collect all dataset entries at this time-of-day across all days
			// using modulo matching (same pattern as rate/arrival aggregation)
			var chunkInputMuSum, chunkInputSigmaSum, chunkOutputMuSum, chunkOutputSigmaSum float64
			var chunkDistCount int

			// INV-6: Sort dataset timestamps for deterministic float accumulation
			// (same pattern as fitLognormalFromPDF:828-833). Map iteration is
			// non-deterministic; floating-point addition is not associative, so
			// accumulation order affects the final sum.
			dsSortedTimestamps := make([]int, 0, len(datasets))
			for ts := range datasets {
				dsSortedTimestamps = append(dsSortedTimestamps, ts)
			}
			sort.Ints(dsSortedTimestamps)

			for _, dsTimestamp := range dsSortedTimestamps {
				dataset := datasets[dsTimestamp]
				// Match datasets at the same time-of-day across all days
				if int64(dsTimestamp)%86400 == int64(nearestBoundary)%86400 {
					inputFit, fitErr := fitLognormalFromPDF(dataset.inputPDF)
					if fitErr == nil {
						chunkInputMuSum += inputFit.Params["mu"]
						chunkInputSigmaSum += inputFit.Params["sigma"]
					}
					outputFit, fitErr := fitLognormalFromPDF(dataset.outputPDF)
					if fitErr == nil {
						chunkOutputMuSum += outputFit.Params["mu"]
						chunkOutputSigmaSum += outputFit.Params["sigma"]
						chunkDistCount++
					}
				}
			}

			// Average this chunk's multi-day distributions
			if chunkDistCount > 0 {
				n := float64(chunkDistCount)
				sumMuInput += chunkInputMuSum / n
				sumSigmaInput += chunkInputSigmaSum / n
				sumMuOutput += chunkOutputMuSum / n
				sumSigmaOutput += chunkOutputSigmaSum / n
				distCount++
			}

			// BC-5: Average rates from all windows matching the selected time-of-day (across all days)
			// Mathematical consistency: averaging arrival params requires averaging rates too
			if chunk.client.Lifecycle != nil {
				var chunkRateSum float64
				var chunkRateCount int
				for _, window := range chunk.client.Lifecycle.Windows {
					windowStartSec := window.StartUs / 1e6

					// Include all windows at the same time-of-day across all days
					if int64(windowStartSec)%86400 == selectedWindow {
						if window.TraceRate != nil {
							chunkRateSum += *window.TraceRate
							chunkRateCount++
						}
					}
				}
				if chunkRateCount > 0 {
					totalRate += chunkRateSum / float64(chunkRateCount)
				}
			}
		}

		// BC-3: Build averaged distributions
		var inputDist, outputDist DistSpec
		if distCount > 0 {
			n := float64(distCount)
			inputDist = DistSpec{
				Type: "lognormal",
				Params: map[string]float64{
					"mu":    sumMuInput / n,
					"sigma": sumSigmaInput / n,
				},
			}
			outputDist = DistSpec{
				Type: "lognormal",
				Params: map[string]float64{
					"mu":    sumMuOutput / n,
					"sigma": sumSigmaOutput / n,
				},
			}
		} else {
			// Fallback (only if no datasets available at all)
			inputDist = DistSpec{
				Type: "gaussian",
				Params: map[string]float64{
					"mean":    512,
					"std_dev": 128,
					"min":     1,
					"max":     32768,
				},
			}
			outputDist = DistSpec{
				Type: "gaussian",
				Params: map[string]float64{
					"mean":    128,
					"std_dev": 32,
					"min":     1,
					"max":     32768,
				},
			}
		}

		// BC-4: Average arrival parameters from all windows at the selected time-of-day (across all days).
		// Majority-vote the process type; simple-average CV, shape, scale.
		var sumCV, sumShape, sumScale float64
		var arrivalCount int
		gammaCount, weibullCount := 0, 0

		for _, chunk := range chunks {
			if chunk.client.Lifecycle == nil {
				continue
			}
			for _, window := range chunk.client.Lifecycle.Windows {
				windowStartSec := window.StartUs / 1e6
				// Include all windows at the same time-of-day across all days
				if int64(windowStartSec)%86400 == selectedWindow {
					if window.Arrival != nil {
						if window.Arrival.CV != nil {
							sumCV += *window.Arrival.CV
						}
						if window.Arrival.Shape != nil {
							sumShape += *window.Arrival.Shape
						}
						if window.Arrival.Scale != nil {
							sumScale += *window.Arrival.Scale
						}
						arrivalCount++
						switch window.Arrival.Process {
						case "gamma":
							gammaCount++
						case "weibull":
							weibullCount++
						}
					}
				}
			}
		}

		var arrivalSpec ArrivalSpec
		if arrivalCount > 0 {
			n := float64(arrivalCount)
			if weibullCount > gammaCount {
				arrivalSpec.Process = "weibull"
			} else {
				arrivalSpec.Process = "gamma"
			}
			avgCV := sumCV / n
			avgShape := sumShape / n
			avgScale := sumScale / n
			arrivalSpec.CV = &avgCV
			arrivalSpec.Shape = &avgShape
			arrivalSpec.Scale = &avgScale
		}

		// Skip cohorts with zero or negative rate (happens when selected window has no active chunks)
		if totalRate <= 0 {
			logrus.Warnf("Skipping cohort %s: zero rate at selected window (window %d had no active chunks)", cohortID, selectedWindow)
			continue
		}

		cohort := CohortSpec{
			ID:           cohortID,
			Population:   len(chunks),
			SLOClass:     key.sloClass,
			Streaming:    true, // ServeGen traces are streaming
			RateFraction: 1.0,  // Unused in absolute rate mode
			Arrival:      arrivalSpec,
			InputDist:    inputDist,
			OutputDist:   outputDist,
			Spike: &SpikeSpec{
				StartTimeUs: period.startUs,
				DurationUs:  period.durationUs,
				TraceRate:   &totalRate,
			},
		}

		spec.Cohorts = append(spec.Cohorts, cohort)
	}

	if len(spec.Cohorts) == 0 {
		return fmt.Errorf("no active cohorts generated (all chunks filtered out)")
	}

	// BC-6: Set aggregate_rate to 0 (absolute mode)
	spec.AggregateRate = 0

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

// buildReasoningCohorts converts reasoning chunks into cohorts with single-window
// temporal structure (no morning/afternoon periods). Generates 10 cohorts:
// 2 reasoning levels (low, high) × 5 SLO classes.
func buildReasoningCohorts(spec *WorkloadSpec, traceFiles []string, rng *rand.Rand) error {
	if len(traceFiles) == 0 {
		return fmt.Errorf("no chunks provided for reasoning cohort building")
	}

	dataDir := spec.ServeGenData.Path
	windowDurSec := spec.ServeGenData.WindowDurationSecs
	if windowDurSec <= 0 {
		windowDurSec = 600
	}

	// BC-4: Select ONE random window from 144 (24-hour period at 10-min intervals)
	selectedWindowIndex := rng.Intn(144)
	selectedWindowStartSec := int64(selectedWindowIndex * 600) // 600s per window

	logrus.Infof("Selected reasoning window %d (time offset %d seconds)", selectedWindowIndex, selectedWindowStartSec)

	// Load all chunks and parse reason_ratio PDFs
	var chunks []chunkData
	for _, tracePath := range traceFiles {
		base := filepath.Base(tracePath)
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")
		datasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))

		// Read dataset JSON
		data, err := os.ReadFile(datasetPath)
		if err != nil {
			logrus.Warnf("Skipping chunk %s: dataset read failed: %v", chunkID, err)
			continue
		}

		var raw map[string]map[string]string
		if err := json.Unmarshal(data, &raw); err != nil {
			logrus.Warnf("Skipping chunk %s: JSON parse failed: %v", chunkID, err)
			continue
		}

		// Find reason_ratio at selected window (or first available)
		var reasonRatioPDF map[float64]float64
		for _, window := range raw {
			if ratioStr, ok := window["reason_ratio"]; ok && ratioStr != "" && ratioStr != "{}" {
				parsed, parseErr := parseServeGenFloatPDF(ratioStr)
				if parseErr == nil {
					reasonRatioPDF = parsed
					break
				}
			}
		}

		if len(reasonRatioPDF) == 0 {
			logrus.Warnf("Skipping chunk %s: no valid reason_ratio found", chunkID)
			continue
		}

		chunks = append(chunks, chunkData{
			id:             chunkID,
			reasonRatioPDF: reasonRatioPDF,
			datasetPath:    datasetPath,
		})
	}

	if len(chunks) == 0 {
		return fmt.Errorf("no valid reasoning chunks found")
	}

	// BC-3: Classify chunks by mean reasoning intensity
	threshold := 0.4
	lowChunks, highChunks, err := classifyReasoningChunks(chunks, threshold)
	if err != nil {
		return fmt.Errorf("classifying reasoning chunks: %w", err)
	}

	logrus.Infof("Classified %d low-reasoning chunks, %d high-reasoning chunks", len(lowChunks), len(highChunks))

	// BC-5: Build 10 cohorts (2 levels × 5 SLO classes)
	sloClasses := []string{"critical", "standard", "batch", "sheddable", "background"}
	reasoningLevels := []struct {
		name   string
		chunks []chunkData
	}{
		{"low", lowChunks},
		{"high", highChunks},
	}

	for _, level := range reasoningLevels {
		if len(level.chunks) == 0 {
			logrus.Warnf("No chunks in %s-reasoning group; some cohorts will be empty", level.name)
			continue
		}

		// Shuffle chunks for random SLO assignment
		shuffled := make([]chunkData, len(level.chunks))
		copy(shuffled, level.chunks)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		// Round-robin assign to 5 SLO classes
		groups := make(map[string][]chunkData)
		for i, chunk := range shuffled {
			sloClass := sloClasses[i%5]
			groups[sloClass] = append(groups[sloClass], chunk)
		}

		// Build cohorts
		for _, sloClass := range sloClasses {
			cohortChunks := groups[sloClass]
			if len(cohortChunks) == 0 {
				continue
			}

			cohortID := fmt.Sprintf("reasoning-%s-%s", sloClass, level.name)

			// BC-6: Fit input/output distributions from datasets
			var sumMuInput, sumSigmaInput, sumMuOutput, sumSigmaOutput float64
			var distCount int

			// BC-6: Fit reasoning ratio distributions from reason_ratio PDFs
			var sumMuRatio, sumSigmaRatio float64
			var ratioFitCount int

			for _, chunk := range cohortChunks {
				// Load dataset to fit input/output token distributions
				datasets, dsErr := loadServeGenDatasetAllWindows(chunk.datasetPath, &ServeGenDataSpec{Path: dataDir})
				if dsErr != nil {
					logrus.Warnf("Skipping chunk %s for cohort %s: dataset load failed: %v", chunk.id, cohortID, dsErr)
					continue
				}

				// Find nearest dataset boundary for the selected window
				_, nearestBoundary, found := findNearestDataset(int(selectedWindowStartSec), datasets)
				if !found {
					logrus.Warnf("Skipping chunk %s for cohort %s: no dataset at selected window", chunk.id, cohortID)
					continue
				}

				// Aggregate across all matching time-of-day windows (deterministic order)
				dsSortedTimestamps := make([]int, 0, len(datasets))
				for ts := range datasets {
					dsSortedTimestamps = append(dsSortedTimestamps, ts)
				}
				sort.Ints(dsSortedTimestamps)

				var chunkInputMuSum, chunkInputSigmaSum, chunkOutputMuSum, chunkOutputSigmaSum float64
				var chunkDistCount int

				for _, dsTimestamp := range dsSortedTimestamps {
					dataset := datasets[dsTimestamp]
					// Match time-of-day across all days
					if int64(dsTimestamp)%86400 == int64(nearestBoundary)%86400 {
						inputFit, fitErr := fitLognormalFromPDF(dataset.inputPDF)
						if fitErr == nil {
							chunkInputMuSum += inputFit.Params["mu"]
							chunkInputSigmaSum += inputFit.Params["sigma"]
						}
						outputFit, fitErr := fitLognormalFromPDF(dataset.outputPDF)
						if fitErr == nil {
							chunkOutputMuSum += outputFit.Params["mu"]
							chunkOutputSigmaSum += outputFit.Params["sigma"]
							chunkDistCount++
						}
					}
				}

				// Average this chunk's multi-day distributions
				if chunkDistCount > 0 {
					n := float64(chunkDistCount)
					sumMuInput += chunkInputMuSum / n
					sumSigmaInput += chunkInputSigmaSum / n
					sumMuOutput += chunkOutputMuSum / n
					sumSigmaOutput += chunkOutputSigmaSum / n
					distCount++
				}

				// Fit reasoning ratio distribution from float PDF
				// Convert float PDF to int PDF by scaling to [0, 100]
				intPDF := make(map[int]float64)
				for ratio, prob := range chunk.reasonRatioPDF {
					pct := int(ratio * 100)
					if pct < 0 {
						pct = 0
					}
					if pct > 100 {
						pct = 100
					}
					intPDF[pct] += prob
				}

				fit, fitErr := fitLognormalFromPDF(intPDF)
				if fitErr == nil {
					sumMuRatio += fit.Params["mu"]
					sumSigmaRatio += fit.Params["sigma"]
					ratioFitCount++
				}
			}

			if distCount == 0 {
				logrus.Warnf("Skipping cohort %s: no valid input/output distributions", cohortID)
				continue
			}
			if ratioFitCount == 0 {
				logrus.Warnf("Skipping cohort %s: no valid reasoning ratio fits", cohortID)
				continue
			}

			avgMuInput := sumMuInput / float64(distCount)
			avgSigmaInput := sumSigmaInput / float64(distCount)
			avgMuOutput := sumMuOutput / float64(distCount)
			avgSigmaOutput := sumSigmaOutput / float64(distCount)
			avgMuRatio := sumMuRatio / float64(ratioFitCount)
			avgSigmaRatio := sumSigmaRatio / float64(ratioFitCount)

			// BC-7: Inject MultiTurn workaround
			cohort := CohortSpec{
				ID:           cohortID,
				Population:   len(cohortChunks),
				SLOClass:     sloClass,
				Streaming:    true,
				RateFraction: 1.0, // Divided by population during ExpandCohorts
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{
					Type: "lognormal",
					Params: map[string]float64{
						"mu":    avgMuInput,
						"sigma": avgSigmaInput,
					},
				},
				OutputDist: DistSpec{
					Type: "lognormal",
					Params: map[string]float64{
						"mu":    avgMuOutput,
						"sigma": avgSigmaOutput,
					},
				},
				Spike: &SpikeSpec{
					StartTimeUs: 0,
					DurationUs:  int64(windowDurSec) * 1e6,
					TraceRate:   func() *float64 { r := 1.0; return &r }(), // Placeholder rate
				},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{
						Type: "lognormal",
						Params: map[string]float64{
							"mu":    avgMuRatio,
							"sigma": avgSigmaRatio,
						},
					},
					MultiTurn: &MultiTurnSpec{
						MaxRounds:     1,
						ThinkTimeUs:   0,
						ContextGrowth: "",
					},
				},
			}

			spec.Cohorts = append(spec.Cohorts, cohort)
		}
	}

	if len(spec.Cohorts) == 0 {
		return fmt.Errorf("no cohorts generated (all groups empty)")
	}

	// Set aggregate_rate to 0 (absolute mode)
	spec.AggregateRate = 0

	return nil
}
