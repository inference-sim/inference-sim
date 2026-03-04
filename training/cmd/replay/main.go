// Package main implements a BLIS replay harness for comparing simulated latency
// against real vLLM traces. It reads per-request data extracted from
// per_request_lifecycle_metrics.json and runs it through the actual BLIS
// simulator with the crossmodel (or blackbox) latency backend.
//
// Usage:
//
//	go run training/cmd/replay/main.go --input training/replay_data/<experiment>.json
//	go run training/cmd/replay/main.go --input <file>.json --beta 116.1,1226.9,19.9,9445.2 --alpha 13732,0,860.6 --delta 13000
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/kv"     // register KV store factory
	_ "github.com/inference-sim/inference-sim/sim/latency" // register latency model factory
)

// ReplayInput is the JSON format produced by training/extract_replay.py.
type ReplayInput struct {
	Experiment  string            `json:"experiment"`
	ModelID     string            `json:"model_id"`
	ModelShort  string            `json:"model_short"`
	Profile     string            `json:"profile"`
	Split       string            `json:"split"`
	Config      ReplayConfig      `json:"config"`
	ModelConfig ReplayModelConfig `json:"model_config"`
	Requests    []ReplayRequest   `json:"requests"`
}

type ReplayConfig struct {
	TP               int   `json:"tensor_parallelism"`
	MaxNumSeqs       int   `json:"max_num_seqs"`
	MaxBatchedTokens int   `json:"max_num_batched_tokens"`
	KVBlocksTotal    int64 `json:"kv_blocks_total_gpu"`
	BlockSize        int   `json:"block_size"`
	NumPrefixGroups  int   `json:"num_prefix_groups"`
	UsersPerGroup    int   `json:"users_per_group"`
	SystemPromptLen  int   `json:"system_prompt_len"`
}

type ReplayModelConfig struct {
	NumLayers        int `json:"num_layers"`
	HiddenDim        int `json:"hidden_dim"`
	NumHeads         int `json:"num_heads"`
	NumKVHeads       int `json:"num_kv_heads"`
	IntermediateDim  int `json:"intermediate_dim"`
	NumLocalExperts  int `json:"num_local_experts"`
	NumExpertsPerTok int `json:"num_experts_per_tok"`
	VocabSize        int `json:"vocab_size"`
}

type ReplayRequest struct {
	Index       int    `json:"index"`
	ArrivalUs   int64  `json:"arrival_time_us"`
	InputTokens int    `json:"input_tokens"`
	OutputTokens int   `json:"output_tokens"`
	PrefixGroup string `json:"prefix_group"`
	HasError    bool   `json:"has_error"`
}

// ReplayOutput is the JSON output format for comparison.
type ReplayOutput struct {
	Experiment  string                `json:"experiment"`
	ModelShort  string                `json:"model_short"`
	Profile     string                `json:"profile"`
	Split       string                `json:"split"`
	Backend     string                `json:"backend"`
	Coefficients ReplayCoefficients   `json:"coefficients"`
	Summary     ReplaySummary         `json:"summary"`
	StepMetrics *ReplayStepMetrics    `json:"step_metrics,omitempty"`
	Requests    []ReplayRequestResult `json:"requests"`
}

type ReplayCoefficients struct {
	Beta  []float64 `json:"beta"`
	Alpha []float64 `json:"alpha"`
	Delta []float64 `json:"delta,omitempty"`
}

type ReplaySummary struct {
	Completed         int     `json:"completed"`
	StillQueued       int     `json:"still_queued"`
	StillRunning      int     `json:"still_running"`
	DroppedUnservable int     `json:"dropped_unservable"`
	PreemptionCount   int64   `json:"preemption_count"`
	TTFTMeanMs        float64 `json:"ttft_mean_ms"`
	TTFTP50Ms         float64 `json:"ttft_p50_ms"`
	TTFTP90Ms         float64 `json:"ttft_p90_ms"`
	TTFTP99Ms         float64 `json:"ttft_p99_ms"`
	E2EMeanMs         float64 `json:"e2e_mean_ms"`
	E2EP50Ms          float64 `json:"e2e_p50_ms"`
	E2EP90Ms          float64 `json:"e2e_p90_ms"`
	E2EP99Ms          float64 `json:"e2e_p99_ms"`
	ResponsesPerSec   float64 `json:"responses_per_sec"`
	TokensPerSec      float64 `json:"tokens_per_sec"`
}

type ReplayStepMetrics struct {
	NumSteps      int     `json:"num_steps"`
	Durations     []int64 `json:"durations"`
	PrefillTokens []int64 `json:"prefill_tokens"`
	DecodeTokens  []int64 `json:"decode_tokens"`
}

type ReplayRequestResult struct {
	Index        int     `json:"index"`
	TTFTMs       float64 `json:"ttft_ms"`
	E2EMs        float64 `json:"e2e_ms"`
	SchedDelayMs float64 `json:"scheduling_delay_ms"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
}

// Iter 3 defaults (from defaults.yaml crossmodel_defaults)
var (
	defaultBeta  = []float64{116.110, 1226.868, 19.943, 9445.157}
	defaultAlpha = []float64{13732.0, 0.0, 860.6}
)

// parseFloatSlice parses a comma-separated string of floats.
func parseFloatSlice(s string) ([]float64, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	result := make([]float64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float at position %d: %q: %w", i, p, err)
		}
		result[i] = v
	}
	return result, nil
}

func main() {
	inputPath := flag.String("input", "", "Path to extracted replay JSON")
	backend := flag.String("backend", "crossmodel", "Latency backend: crossmodel or blackbox")
	seed := flag.Int64("seed", 42, "Random seed for token ID generation")
	betaStr := flag.String("beta", "", "Comma-separated beta coefficients (default: Iter 3 crossmodel)")
	alphaStr := flag.String("alpha", "", "Comma-separated alpha coefficients (default: Iter 3 crossmodel)")
	deltaStr := flag.String("delta", "", "Comma-separated inter-step overhead coefficients (default: none)")
	emitStepMetrics := flag.Bool("emit-step-metrics", false, "Include per-step arrays in JSON output")
	flag.Parse()

	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required")
		os.Exit(1)
	}

	// Parse coefficient overrides
	betaCoeffs, err := parseFloatSlice(*betaStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error parsing --beta: %v\n", err)
		os.Exit(1)
	}
	if betaCoeffs == nil {
		betaCoeffs = make([]float64, len(defaultBeta))
		copy(betaCoeffs, defaultBeta)
	}

	alphaCoeffs, err := parseFloatSlice(*alphaStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error parsing --alpha: %v\n", err)
		os.Exit(1)
	}
	if alphaCoeffs == nil {
		alphaCoeffs = make([]float64, len(defaultAlpha))
		copy(alphaCoeffs, defaultAlpha)
	}

	var deltaCoeffs []float64
	deltaCoeffs, err = parseFloatSlice(*deltaStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error parsing --delta: %v\n", err)
		os.Exit(1)
	}

	// Read input
	data, err := os.ReadFile(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading %s: %v\n", *inputPath, err)
		os.Exit(1)
	}
	var input ReplayInput
	if err := json.Unmarshal(data, &input); err != nil {
		fmt.Fprintf(os.Stderr, "error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	// Validate backend
	if *backend != "crossmodel" && *backend != "blackbox" {
		fmt.Fprintf(os.Stderr, "error: --backend must be 'crossmodel' (4 beta) or 'blackbox' (3 beta)\n")
		os.Exit(1)
	}

	// Create sim.Request objects with exact arrival times and token counts
	rng := rand.New(rand.NewSource(*seed))

	// Generate shared prefix tokens per group (for prefix cache)
	prefixTokens := make(map[string][]int)
	for g := 0; g < input.Config.NumPrefixGroups; g++ {
		group := fmt.Sprintf("prompt-%d", g)
		tokens := make([]int, input.Config.SystemPromptLen)
		for j := range tokens {
			tokens[j] = rng.Intn(32000) + 1
		}
		prefixTokens[group] = tokens
	}

	// Build sim.Request objects for all requests (including errored)
	var requests []*sim.Request
	indexMap := make(map[string]int) // request ID -> original index

	for _, r := range input.Requests {
		// Generate synthetic token IDs of the right length
		// Prepend shared prefix for prefix cache
		var inputToks []int
		if prefix, ok := prefixTokens[r.PrefixGroup]; ok {
			inputToks = make([]int, 0, len(prefix)+r.InputTokens)
			inputToks = append(inputToks, prefix...)
			// Add remaining non-prefix tokens
			remaining := max(0, r.InputTokens-len(prefix))
			for j := 0; j < remaining; j++ {
				inputToks = append(inputToks, rng.Intn(32000)+1)
			}
		} else {
			inputToks = make([]int, r.InputTokens)
			for j := range inputToks {
				inputToks[j] = rng.Intn(32000) + 1
			}
		}

		outputToks := make([]int, r.OutputTokens)
		for j := range outputToks {
			outputToks[j] = rng.Intn(32000) + 1
		}

		reqID := fmt.Sprintf("req-%d", r.Index)
		req := &sim.Request{
			ID:           reqID,
			ArrivalTime:  r.ArrivalUs,
			InputTokens:  inputToks,
			OutputTokens: outputToks,
			State:        sim.StateQueued,
		}
		requests = append(requests, req)
		indexMap[reqID] = r.Index
	}

	// Determine horizon: last arrival + generous buffer for completion
	var maxArrival int64
	for _, req := range requests {
		if req.ArrivalTime > maxArrival {
			maxArrival = req.ArrivalTime
		}
	}
	horizon := maxArrival + 600_000_000 // +600s buffer

	// Configure BLIS
	mc := input.ModelConfig
	config := cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: horizon,
			Seed:    *seed,
			KVCacheConfig: sim.NewKVCacheConfig(
				input.Config.KVBlocksTotal,
				int64(input.Config.BlockSize),
				0,   // no CPU tier
				0.0, // no offload
				0.0, // no transfer bandwidth
				0,   // no transfer latency
			),
			BatchConfig: sim.NewBatchConfig(
				int64(input.Config.MaxNumSeqs),
				int64(input.Config.MaxBatchedTokens),
				0, // no long-prefill threshold (matches vLLM default)
			),
			LatencyCoeffs: sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs, deltaCoeffs),
			ModelHardwareConfig: sim.NewModelHardwareConfig(
				sim.ModelConfig{
					NumLayers:       mc.NumLayers,
					HiddenDim:       mc.HiddenDim,
					NumHeads:        mc.NumHeads,
					NumKVHeads:      mc.NumKVHeads,
					IntermediateDim: mc.IntermediateDim,
					NumLocalExperts: mc.NumLocalExperts,
					VocabSize:       mc.VocabSize,
				},
				sim.HardwareCalib{}, // not needed for crossmodel
				input.ModelID,
				"H100",
				input.Config.TP,
				*backend,
			),
			PolicyConfig: sim.NewPolicyConfig("constant", "fcfs"),
		},
		NumInstances:    1,
		AdmissionPolicy: "always-admit",
		RoutingPolicy:   "round-robin",
	}

	// Run simulation
	fmt.Fprintf(os.Stderr, "Replaying %s (%s/%s): %d requests, KV=%d blocks, TP=%d, backend=%s\n",
		input.Experiment, input.ModelShort, input.Profile,
		len(requests), input.Config.KVBlocksTotal, input.Config.TP, *backend)
	fmt.Fprintf(os.Stderr, "  beta=%v alpha=%v delta=%v seed=%d\n",
		betaCoeffs, alphaCoeffs, deltaCoeffs, *seed)

	cs := cluster.NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Simulation failed: %v\n", err)
		os.Exit(1)
	}

	// Collect results
	metrics := cs.AggregatedMetrics()

	// Build per-request results
	// Note: metrics.RequestTTFTs and RequestE2Es are in ticks (microseconds).
	// Convert to milliseconds for output (÷1000).
	var reqResults []ReplayRequestResult
	for id, rm := range metrics.Requests {
		idx, ok := indexMap[id]
		if !ok {
			continue
		}
		reqResults = append(reqResults, ReplayRequestResult{
			Index:        idx,
			TTFTMs:       metrics.RequestTTFTs[id] / 1e3,
			E2EMs:        metrics.RequestE2Es[id] / 1e3,
			SchedDelayMs: float64(metrics.RequestSchedulingDelays[id]) / 1e3,
			InputTokens:  rm.NumPrefillTokens,
			OutputTokens: rm.NumDecodeTokens,
		})
	}

	// Compute percentiles from per-request data (in ticks → convert to ms)
	ttfts := make([]float64, 0, len(metrics.RequestTTFTs))
	for _, v := range metrics.RequestTTFTs {
		ttfts = append(ttfts, v/1e3) // ticks → ms
	}
	e2es := make([]float64, 0, len(metrics.RequestE2Es))
	for _, v := range metrics.RequestE2Es {
		e2es = append(e2es, v/1e3) // ticks → ms
	}

	output := ReplayOutput{
		Experiment: input.Experiment,
		ModelShort: input.ModelShort,
		Profile:    input.Profile,
		Split:      input.Split,
		Backend:    *backend,
		Coefficients: ReplayCoefficients{
			Beta:  betaCoeffs,
			Alpha: alphaCoeffs,
			Delta: deltaCoeffs,
		},
		Summary: ReplaySummary{
			Completed:         metrics.CompletedRequests,
			StillQueued:       metrics.StillQueued,
			StillRunning:      metrics.StillRunning,
			DroppedUnservable: metrics.DroppedUnservable,
			PreemptionCount:   metrics.PreemptionCount,
			TTFTMeanMs:        mean(ttfts),
			TTFTP50Ms:         percentile(ttfts, 50),
			TTFTP90Ms:         percentile(ttfts, 90),
			TTFTP99Ms:         percentile(ttfts, 99),
			E2EMeanMs:         mean(e2es),
			E2EP50Ms:          percentile(e2es, 50),
			E2EP90Ms:          percentile(e2es, 90),
			E2EP99Ms:          percentile(e2es, 99),
		},
	}

	// Compute throughput
	if metrics.SimEndedTime > 0 {
		durationSec := float64(metrics.SimEndedTime) / 1e6
		output.Summary.ResponsesPerSec = float64(metrics.CompletedRequests) / durationSec
		output.Summary.TokensPerSec = float64(metrics.TotalOutputTokens) / durationSec
	}

	// Optionally include per-step metrics
	if *emitStepMetrics {
		output.StepMetrics = &ReplayStepMetrics{
			NumSteps:      len(metrics.StepDurations),
			Durations:     metrics.StepDurations,
			PrefillTokens: metrics.StepPrefillTokens,
			DecodeTokens:  metrics.StepDecodeTokens,
		}
	}

	output.Requests = reqResults

	// Write JSON to stdout
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "error encoding output: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Completed: %d requests, dropped=%d, preemptions=%d\n",
		metrics.CompletedRequests, metrics.DroppedUnservable, metrics.PreemptionCount)
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func percentile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sortFloat64s(sorted)
	idx := int(p / 100.0 * float64(len(sorted)-1))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func sortFloat64s(a []float64) {
	// Simple sort for percentile computation
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}
