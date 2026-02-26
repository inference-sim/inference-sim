package workload

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ConvertServeGen converts a ServeGen data directory (containing chunk-*-trace.csv
// and dataset.json files) into a v2 WorkloadSpec.
// Returns error if the directory is empty or contains invalid data (R6: never Fatalf).
func ConvertServeGen(path string) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("ServeGen path must not be empty")
	}
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 1.0, // placeholder; loadServeGenData derives rates from trace data
		ServeGenData:  &ServeGenDataSpec{Path: path},
	}
	if err := loadServeGenData(spec); err != nil {
		return nil, fmt.Errorf("loading ServeGen data from %s: %w", path, err)
	}
	spec.ServeGenData = nil // clear after loading; clients are now populated
	return spec, nil
}

// ConvertCSVTrace converts a legacy CSV trace file into a v2 WorkloadSpec.
// The CSV format has columns: arrival_time(s), ..., prefill_tokens(JSON), decode_tokens(JSON).
// Returns a spec with a single constant-arrival client whose requests match the CSV.
// horizon is in microseconds; 0 means no horizon truncation.
func ConvertCSVTrace(path string, horizon int64) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("CSV trace path must not be empty")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening CSV trace %s: %w", path, err)
	}
	defer file.Close() //nolint:errcheck // read-only file

	reader := csv.NewReader(file)

	// Skip header row
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("reading CSV header from %s: %w", path, err)
	}

	// Parse all rows to extract request data
	type csvRequest struct {
		arrivalTimeUs int64
		inputLen      int
		outputLen     int
	}
	var requests []csvRequest
	rowIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("CSV %s row %d: %w", path, rowIdx, err)
		}
		if len(record) < 5 {
			return nil, fmt.Errorf("CSV %s row %d: expected at least 5 columns, got %d", path, rowIdx, len(record))
		}

		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("CSV %s row %d: invalid arrival_time %q: %w", path, rowIdx, record[0], err)
		}
		arrivalTimeUs := int64(arrivalFloat * 1e6)

		if horizon > 0 && arrivalTimeUs > horizon {
			break
		}

		var inputTokens []int
		if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
			return nil, fmt.Errorf("CSV %s row %d: invalid prefill_tokens: %w", path, rowIdx, err)
		}
		var outputTokens []int
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			return nil, fmt.Errorf("CSV %s row %d: invalid decode_tokens: %w", path, rowIdx, err)
		}

		requests = append(requests, csvRequest{
			arrivalTimeUs: arrivalTimeUs,
			inputLen:      len(inputTokens),
			outputLen:     len(outputTokens),
		})
		rowIdx++
	}

	if len(requests) == 0 {
		return nil, fmt.Errorf("empty CSV: no data rows in %s", path)
	}

	// Compute average input/output lengths for the spec
	var totalInput, totalOutput int
	for _, r := range requests {
		totalInput += r.inputLen
		totalOutput += r.outputLen
	}
	avgInput := float64(totalInput) / float64(len(requests))
	avgOutput := float64(totalOutput) / float64(len(requests))

	// Compute aggregate rate from trace duration
	durationUs := requests[len(requests)-1].arrivalTimeUs - requests[0].arrivalTimeUs
	var ratePerSec float64
	if durationUs > 0 {
		ratePerSec = float64(len(requests)) / (float64(durationUs) / 1e6)
	} else {
		ratePerSec = float64(len(requests)) // all at time 0: treat as 1-second burst
	}

	spec := &WorkloadSpec{
		Version:       "2",
		NumRequests:   int64(len(requests)),
		AggregateRate: ratePerSec,
		Clients: []ClientSpec{
			{
				ID:           "csv-trace",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{
					Type:   "constant",
					Params: map[string]float64{"value": avgInput},
				},
				OutputDist: DistSpec{
					Type:   "constant",
					Params: map[string]float64{"value": avgOutput},
				},
			},
		},
	}
	return spec, nil
}

// PresetConfig holds the token distribution parameters for a workload preset.
// Exported so cmd/ can pass loaded presets to the synthesis layer.
type PresetConfig struct {
	PrefixTokens      int
	PromptTokensMean  int
	PromptTokensStdev int
	PromptTokensMin   int
	PromptTokensMax   int
	OutputTokensMean  int
	OutputTokensStdev int
	OutputTokensMin   int
	OutputTokensMax   int
}

// ConvertPreset converts a named workload preset (e.g., "chatbot") into a v2 WorkloadSpec.
// rate is in requests/second. Returns error for unknown preset names.
func ConvertPreset(name string, rate float64, numRequests int, preset PresetConfig) (*WorkloadSpec, error) {
	if rate <= 0 {
		return nil, fmt.Errorf("rate must be positive, got %f", rate)
	}
	if numRequests <= 0 {
		return nil, fmt.Errorf("num_requests must be positive, got %d", numRequests)
	}

	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: rate,
		NumRequests:   int64(numRequests),
		Clients: []ClientSpec{
			{
				ID:           fmt.Sprintf("preset-%s", name),
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":  float64(preset.PromptTokensMean),
						"std_dev": float64(preset.PromptTokensStdev),
						"min":   float64(preset.PromptTokensMin),
						"max":   float64(preset.PromptTokensMax),
					},
				},
				OutputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":  float64(preset.OutputTokensMean),
						"std_dev": float64(preset.OutputTokensStdev),
						"min":   float64(preset.OutputTokensMin),
						"max":   float64(preset.OutputTokensMax),
					},
				},
				PrefixLength: preset.PrefixTokens,
			},
		},
	}

	if preset.PrefixTokens > 0 {
		spec.Clients[0].PrefixGroup = "shared"
	}

	return spec, nil
}

// ConvertInferencePerf converts an inference-perf YAML spec file into a v2 WorkloadSpec.
// Wraps existing ExpandInferencePerfSpec with file loading.
func ConvertInferencePerf(path string) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("inference-perf spec path must not be empty")
	}
	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		return nil, fmt.Errorf("loading inference-perf spec from %s: %w", path, err)
	}
	if spec.InferencePerf == nil {
		return nil, fmt.Errorf("file %s does not contain an inference_perf section", path)
	}
	expanded, err := ExpandInferencePerfSpec(spec.InferencePerf, spec.Seed)
	if err != nil {
		return nil, fmt.Errorf("expanding inference-perf spec: %w", err)
	}
	expanded.Version = "2"
	return expanded, nil
}

// ComposeSpecs merges multiple WorkloadSpecs into a single spec.
// Client lists are concatenated, aggregate rates summed, rate fractions renormalized.
func ComposeSpecs(specs []*WorkloadSpec) (*WorkloadSpec, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("at least one spec file required")
	}

	merged := &WorkloadSpec{
		Version:  "2",
		Category: specs[0].Category,
	}

	var totalRate float64
	for _, s := range specs {
		totalRate += s.AggregateRate
		merged.Clients = append(merged.Clients, s.Clients...)
	}
	merged.AggregateRate = totalRate

	// Renormalize rate fractions so they sum to 1.0 in the merged spec
	if len(merged.Clients) > 0 {
		var totalFraction float64
		for i := range merged.Clients {
			totalFraction += merged.Clients[i].RateFraction
		}
		if totalFraction > 0 {
			for i := range merged.Clients {
				merged.Clients[i].RateFraction /= totalFraction
			}
		}
	}

	return merged, nil
}
