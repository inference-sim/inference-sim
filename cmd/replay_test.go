package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// setupTrainedPhysicsTestFixtures creates temp model config and hardware config
// files for replay integration tests that need a working latency backend.
// Returns the model config folder and hardware config file path.
func setupTrainedPhysicsTestFixtures(t *testing.T) (mcFolder, hwPath string) {
	t.Helper()
	dir := t.TempDir()

	// Minimal HF config.json (Llama-like, 2-layer for fast simulation)
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir model config: %v", err)
	}
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_key_value_heads": 4,
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}

	// Minimal hardware config
	hwFile := filepath.Join(dir, "hw.json")
	hwJSON := `{
  "H100": {
    "MemoryGiB": 80.0,
    "TFlopsPeak": 1.0,
    "BwPeakTBs": 0.001
  }
}`
	if err := os.WriteFile(hwFile, []byte(hwJSON), 0644); err != nil {
		t.Fatalf("write hw config: %v", err)
	}

	return mcDir, hwFile
}

// setupTrainedPhysicsTestFixturesWithDefaults extends setupTrainedPhysicsTestFixtures
// by also creating a defaults.yaml with trained_physics_coefficients.
// Returns model config folder, hardware config path, and defaults file path.
func setupTrainedPhysicsTestFixturesWithDefaults(t *testing.T) (mcFolder, hwPath, defaultsPath string) {
	t.Helper()
	mcFolder, hwPath = setupTrainedPhysicsTestFixtures(t)

	// Create minimal defaults.yaml with trained_physics_coefficients
	defaultsPath = filepath.Join(filepath.Dir(hwPath), "defaults.yaml")
	defaultsYAML := `trained_physics_coefficients:
  alpha_coeffs: [100.0, 1.0, 100.0]
  beta_coeffs: [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
`
	if err := os.WriteFile(defaultsPath, []byte(defaultsYAML), 0644); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	return mcFolder, hwPath, defaultsPath
}

// TestReplayCmd_SimConfigFlags_Registered verifies BC-4:
// all sim config flags registered on replayCmd.
func TestReplayCmd_SimConfigFlags_Registered(t *testing.T) {
	flags := []string{
		// registerSimConfigFlags: general
		"seed", "horizon", "log", "defaults-filepath",
		"model-config-folder", "hardware-config",

		// registerSimConfigFlags: vLLM server configs
		"total-kv-blocks", "max-num-running-reqs", "max-num-scheduled-tokens",
		"beta-coeffs", "alpha-coeffs", "block-size-in-tokens",
		"long-prefill-token-threshold",

		// registerSimConfigFlags: BLIS model configs
		"model", "hardware", "tp", "vllm-version",
		"latency-model", "max-model-len",

		// registerSimConfigFlags: cluster config
		"num-instances",

		// registerSimConfigFlags: online routing pipeline
		"admission-policy", "admission-latency", "routing-latency",
		"token-bucket-capacity", "token-bucket-refill-rate",

		// registerSimConfigFlags: routing policy
		"routing-policy", "routing-scorers",

		// registerSimConfigFlags: priority, scheduler, and preemption
		"scheduler", "preemption-policy",

		// registerSimConfigFlags: policy bundle
		"policy-config",

		// registerSimConfigFlags: fitness evaluation
		"fitness-weights",

		// registerSimConfigFlags: decision trace
		"trace-level", "counterfactual-k", "summarize-trace",

		// registerSimConfigFlags: tiered KV cache
		"kv-cpu-blocks", "kv-offload-threshold",
		"kv-transfer-bandwidth", "kv-transfer-base-latency",
		"snapshot-refresh-interval",

		// registerSimConfigFlags: cache signal delay
		"cache-signal-delay",

		// registerSimConfigFlags: flow control
		"flow-control", "saturation-detector", "dispatch-order",
		"max-gateway-queue-depth", "queue-depth-threshold",
		"kv-cache-util-threshold", "max-concurrency",
		"per-band-capacity", "usage-limit-threshold",

		// replay-specific: results
		"results-path",

		// replay-specific flags
		"trace-header", "trace-data",
	}
	for _, name := range flags {
		f := replayCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("replayCmd missing flag --%s", name)
		}
	}
}

func TestSimResult_JSONRoundTrip(t *testing.T) {
	// GIVEN a workload.SimResult with known values
	// workload.SimResult is in sim/workload/calibrate.go — JSON tags added by Task 2.
	sr := workload.SimResult{
		RequestID:    42,
		TTFT:         12345.0,
		E2E:          98765.0,
		InputTokens:  256,
		OutputTokens: 128,
	}

	// WHEN marshaled to JSON and back
	data, err := json.Marshal(sr)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var got workload.SimResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	// THEN all fields round-trip correctly (BC-2)
	if got.RequestID != 42 {
		t.Errorf("RequestID: got %d, want 42", got.RequestID)
	}
	if got.TTFT != 12345.0 {
		t.Errorf("TTFT: got %f, want 12345.0", got.TTFT)
	}
	if got.E2E != 98765.0 {
		t.Errorf("E2E: got %f, want 98765.0", got.E2E)
	}
	if got.InputTokens != 256 {
		t.Errorf("InputTokens: got %d, want 256", got.InputTokens)
	}
	if got.OutputTokens != 128 {
		t.Errorf("OutputTokens: got %d, want 128", got.OutputTokens)
	}

	// THEN JSON keys match the calibrate contract
	if !strings.Contains(string(data), `"request_id":42`) {
		t.Errorf("JSON must contain integer request_id, got: %s", data)
	}
	if !strings.Contains(string(data), `"ttft_us"`) {
		t.Errorf("JSON must contain ttft_us key, got: %s", data)
	}
	if !strings.Contains(string(data), `"e2e_us"`) {
		t.Errorf("JSON must contain e2e_us key, got: %s", data)
	}
}

func TestExtractSimResults_SortsAndConverts(t *testing.T) {
	// GIVEN a Metrics struct with 3 completed requests
	m := sim.NewMetrics()
	// Populate as simulator does (RequestTTFTs in ticks = microseconds)
	m.RequestTTFTs["request_2"] = 2000.0
	m.RequestTTFTs["request_0"] = 1000.0
	m.RequestTTFTs["request_1"] = 1500.0
	m.RequestE2Es["request_2"] = 20000.0
	m.RequestE2Es["request_0"] = 10000.0
	m.RequestE2Es["request_1"] = 15000.0
	m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
	m.Requests["request_1"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}
	m.Requests["request_2"] = sim.RequestMetrics{NumPrefillTokens: 300, NumDecodeTokens: 70}

	// WHEN extractSimResults is called
	results := extractSimResults(m) // returns []workload.SimResult

	// THEN 3 results are returned in ascending request_id order (BC-5: determinism, R2)
	if len(results) != 3 {
		t.Fatalf("want 3 results, got %d", len(results))
	}
	if results[0].RequestID != 0 || results[1].RequestID != 1 || results[2].RequestID != 2 {
		t.Errorf("results not sorted by request_id: %v", results)
	}

	// THEN TTFT and E2E are in microseconds (BC-2, BC-6)
	if results[0].TTFT != 1000.0 {
		t.Errorf("results[0].TTFT: got %f, want 1000.0 (microseconds)", results[0].TTFT)
	}
	if results[0].E2E != 10000.0 {
		t.Errorf("results[0].E2E: got %f, want 10000.0 (microseconds)", results[0].E2E)
	}
	if results[0].InputTokens != 100 || results[0].OutputTokens != 50 {
		t.Errorf("token counts wrong for results[0]: %+v", results[0])
	}
}

func TestExtractSimResults_SkipsNonNumericIDs(t *testing.T) {
	// GIVEN metrics with a non-numeric ID (session follow-up)
	m := sim.NewMetrics()
	m.RequestTTFTs["request_0"] = 1000.0
	m.RequestTTFTs["session_follow_abc"] = 2000.0
	m.RequestE2Es["request_0"] = 5000.0
	m.RequestE2Es["session_follow_abc"] = 8000.0
	m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
	m.Requests["session_follow_abc"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}

	// WHEN extractSimResults is called
	results := extractSimResults(m)

	// THEN only the numeric-ID request is included (BC-7)
	if len(results) != 1 {
		t.Fatalf("want 1 result (non-numeric ID skipped), got %d", len(results))
	}
	if results[0].RequestID != 0 {
		t.Errorf("wrong RequestID: got %d, want 0", results[0].RequestID)
	}
}

func TestExtractSimResults_ExcludesPartialTTFT(t *testing.T) {
	// GIVEN a request with TTFT but no E2E (timed out during decode)
	m := sim.NewMetrics()
	m.RequestTTFTs["request_0"] = 1000.0
	// No entry in RequestE2Es for request_0
	m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 0}

	// WHEN extractSimResults is called
	results := extractSimResults(m)

	// THEN the incomplete request is excluded (no E2E = timeout after prefill)
	if len(results) != 0 {
		t.Errorf("want 0 results (no E2E = incomplete), got %d", len(results))
	}
}

func TestExtractSimResults_EmptyMetrics_ReturnsEmptySlice(t *testing.T) {
	// GIVEN empty metrics (all requests timed out before prefill)
	m := sim.NewMetrics()

	// WHEN extractSimResults is called
	results := extractSimResults(m)

	// THEN an initialized empty slice is returned (not nil)
	// A nil slice marshals to JSON "null"; an empty slice marshals to "[]"
	if results == nil {
		t.Error("want initialized empty slice (not nil) so JSON marshal produces [] not null")
	}
	data, err := json.Marshal(results)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	if string(data) != "[]" {
		t.Errorf("want JSON [], got %s", data)
	}
}

func TestExtractSimResults_MixedRequests_OnlyCompletedIncluded(t *testing.T) {
	// GIVEN metrics with completed, timed-out, and non-numeric IDs mixed
	m := sim.NewMetrics()
	// Completed request
	m.RequestTTFTs["request_1"] = 1500.0
	m.RequestE2Es["request_1"] = 15000.0
	m.Requests["request_1"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}
	// Timed out after prefill (TTFT but no E2E)
	m.RequestTTFTs["request_2"] = 2000.0
	m.Requests["request_2"] = sim.RequestMetrics{NumPrefillTokens: 300, NumDecodeTokens: 0}
	// Session follow-up (non-numeric ID)
	m.RequestTTFTs["session_followup_abc"] = 3000.0
	m.RequestE2Es["session_followup_abc"] = 30000.0
	m.Requests["session_followup_abc"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}

	// WHEN extractSimResults is called
	results := extractSimResults(m)

	// THEN only the fully-completed numeric-ID request is included
	if len(results) != 1 {
		t.Fatalf("want 1 result (only completed numeric request), got %d: %v", len(results), results)
	}
	if results[0].RequestID != 1 {
		t.Errorf("want RequestID=1, got %d", results[0].RequestID)
	}
}

func TestExtractSimResults_DeterminismInvariant(t *testing.T) {
	// GIVEN the same metrics populated in two different key-insertion orders
	makeMetrics := func() *sim.Metrics {
		m := sim.NewMetrics()
		for _, id := range []string{"request_2", "request_0", "request_1"} {
			m.RequestTTFTs[id] = float64(len(id)) * 1000
			m.RequestE2Es[id] = float64(len(id)) * 5000
			m.Requests[id] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
		}
		return m
	}

	// WHEN extractSimResults is called twice
	r1 := extractSimResults(makeMetrics())
	r2 := extractSimResults(makeMetrics())

	// THEN the output is identical (INV-6: determinism)
	if len(r1) != len(r2) {
		t.Fatalf("different lengths: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].RequestID != r2[i].RequestID {
			t.Errorf("index %d: RequestID %d vs %d — output is non-deterministic", i, r1[i].RequestID, r2[i].RequestID)
		}
	}
	// Verify order is ascending (the invariant being tested)
	for i := 1; i < len(r1); i++ {
		if r1[i].RequestID <= r1[i-1].RequestID {
			t.Errorf("results not sorted: index %d (%d) <= index %d (%d)", i, r1[i].RequestID, i-1, r1[i-1].RequestID)
		}
	}
}

// TestReplayCmd_TraceOutputFlag_Registered verifies BC-6:
// --trace-output is registered with empty default (flag is optional).
func TestReplayCmd_TraceOutputFlag_Registered(t *testing.T) {
	f := replayCmd.Flags().Lookup("trace-output")
	if f == nil {
		t.Fatal("replayCmd missing --trace-output flag")
	}
	if f.DefValue != "" {
		t.Errorf("--trace-output default must be empty (optional flag), got %q", f.DefValue)
	}
}

func TestReplayCmd_TraceHeaderFlag_Registered(t *testing.T) {
	// GIVEN the replay command
	// WHEN checking for --trace-header flag
	f := replayCmd.Flags().Lookup("trace-header")
	// THEN it must exist with empty default (BC-6: missing = fail fast)
	if f == nil {
		t.Error("replayCmd missing --trace-header flag")
	}
	if f != nil && f.DefValue != "" {
		t.Errorf("--trace-header default must be empty (required), got %q", f.DefValue)
	}
}

func TestReplayCmd_TraceDataFlag_Registered(t *testing.T) {
	f := replayCmd.Flags().Lookup("trace-data")
	if f == nil {
		t.Error("replayCmd missing --trace-data flag")
	}
	if f != nil && f.DefValue != "" {
		t.Errorf("--trace-data default must be empty (required), got %q", f.DefValue)
	}
}

func TestComputeReplayHorizon_TwiceMaxArrival(t *testing.T) {
	// BC-3: horizon = max(arrivals) * 2
	requests := []*sim.Request{
		{ArrivalTime: 1000},
		{ArrivalTime: 5000},
		{ArrivalTime: 3000},
	}
	horizon := computeReplayHorizon(requests)
	if horizon != 10000 {
		t.Errorf("want horizon 10000 (5000*2), got %d", horizon)
	}
}

func TestComputeReplayHorizon_EmptyRequests_ReturnsMaxInt64(t *testing.T) {
	// Edge case: no requests → MaxInt64 fallback
	horizon := computeReplayHorizon([]*sim.Request{})
	if horizon != math.MaxInt64 {
		t.Errorf("want math.MaxInt64 for empty requests, got %d", horizon)
	}
}

func TestComputeReplayHorizon_AllArrivalsAtZero_ReturnsFixedBuffer(t *testing.T) {
	// Edge case: all requests at t=0 (common for synthetic traces)
	// Must NOT return math.MaxInt64 (would hang simulation)
	requests := []*sim.Request{{ArrivalTime: 0}, {ArrivalTime: 0}}
	horizon := computeReplayHorizon(requests)
	if horizon <= 0 || horizon == math.MaxInt64 {
		t.Errorf("want a finite positive buffer for all-zero arrivals, got %d", horizon)
	}
}

func TestComputeReplayHorizon_LargeArrival_NoOverflow(t *testing.T) {
	// Overflow guard: maxArrival > MaxInt64/2 must not wrap to negative
	requests := []*sim.Request{{ArrivalTime: math.MaxInt64/2 + 1}}
	horizon := computeReplayHorizon(requests)
	if horizon <= 0 {
		t.Errorf("want positive horizon for large arrival (no overflow), got %d", horizon)
	}
	if horizon != math.MaxInt64 {
		t.Errorf("want MaxInt64 as overflow fallback, got %d", horizon)
	}
}

// TestReplayCmd_TraceOutput_FilesCreated verifies BC-1 and BC-2:
// --trace-output creates <prefix>.yaml with mode:"replayed" and <prefix>.csv.
func TestReplayCmd_TraceOutput_FilesCreated(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	outputPrefix := filepath.Join(dir, "out")

	// Write header YAML
	headerContent := `trace_version: 2
time_unit: microseconds
mode: generated
warm_up_requests: 0
`
	if err := os.WriteFile(headerPath, []byte(headerContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Write data CSV: 2 requests
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	// Save and restore all package-level flag vars (same pattern as EndToEnd test)
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder
	origHwConfigPath := hwConfigPath
	origGPU := gpu
	origTP := tensorParallelism
	origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode
	origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
		flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder
		hwConfigPath = origHwConfigPath
		gpu = origGPU
		tensorParallelism = origTP
		defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	// Set package-level vars
	model = "test-model"
	latencyModelBackend = "trained-physics"
	// Note: betaCoeffs and alphaCoeffs NOT set → auto-loads from defaults.yaml trained_physics_coefficients
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 1
	seed = 42
	resultsPath = ""
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	simulationHorizon = math.MaxInt64
	replayTraceOutput = outputPrefix
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = "../defaults.yaml" // Load trained-physics coefficients (relative to cmd/ test dir)

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&replayTraceOutput, "trace-output", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		// Note: --beta-coeffs and --alpha-coeffs omitted → auto-loads from defaults.yaml
		"--total-kv-blocks", "1000",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--trace-header", headerPath,
		"--trace-data", dataPath,
		"--trace-output", outputPrefix,
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	// Run replay
	replayCmd.Run(testCmd, nil)

	// BC-1: both output files must exist
	yamlPath := outputPrefix + ".yaml"
	csvPath := outputPrefix + ".csv"
	if _, err := os.Stat(yamlPath); err != nil {
		t.Fatalf("BC-1: output YAML not created: %v", err)
	}
	if _, err := os.Stat(csvPath); err != nil {
		t.Fatalf("BC-1: output CSV not created: %v", err)
	}

	// BC-1: files must round-trip through LoadTraceV2
	loaded, err := workload.LoadTraceV2(yamlPath, csvPath)
	if err != nil {
		t.Fatalf("BC-1: LoadTraceV2 failed on output files: %v", err)
	}

	// BC-2: header mode must be "replayed"
	if loaded.Header.Mode != "replayed" {
		t.Errorf("BC-2: header.Mode = %q, want \"replayed\"", loaded.Header.Mode)
	}

	// BC-1: record count matches input
	if len(loaded.Records) != 2 {
		t.Errorf("BC-1: want 2 records, got %d", len(loaded.Records))
	}

	// BC-3: for all requests, send_time_us = arrival_time_us (universal)
	for i, rec := range loaded.Records {
		if rec.SendTimeUs != rec.ArrivalTimeUs {
			t.Errorf("BC-3: record[%d] send_time_us=%d != arrival_time_us=%d", i, rec.SendTimeUs, rec.ArrivalTimeUs)
		}
	}

	// BC-3: completed requests have simulation-computed timing (non-zero chunk times)
	for i, rec := range loaded.Records {
		if rec.Status == "ok" {
			if rec.FirstChunkTimeUs <= 0 {
				t.Errorf("BC-3: record[%d] status=ok but first_chunk_time_us=%d (want >0)", i, rec.FirstChunkTimeUs)
			}
			if rec.LastChunkTimeUs < rec.FirstChunkTimeUs {
				t.Errorf("BC-3: record[%d] last_chunk_time_us=%d < first_chunk_time_us=%d", i, rec.LastChunkTimeUs, rec.FirstChunkTimeUs)
			}
		}
	}
}

func TestReplayCmd_EndToEnd_TrainedPhysicsMode(t *testing.T) {
	// NOTE: This test mutates package-level flag vars shared with runCmd.
	// Do NOT use t.Parallel() — concurrent execution would create data races.

	// GIVEN a minimal TraceV2 header + data in a temp directory
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	resultsFilePath := filepath.Join(dir, "results.json")

	// Write header YAML
	header := `trace_version: 2
time_unit: microseconds
mode: generated
warm_up_requests: 0
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	// Write data CSV: 3 requests with arrival times spread over 200ms
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,,\n" +
		"2,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,200000,200000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder, hwCfgPath := setupTrainedPhysicsTestFixtures(t)

	// Save and restore package-level flag vars (this test mutates them)
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder
	origHwConfigPath := hwConfigPath
	origGPU := gpu
	origTP := tensorParallelism
	origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode
	origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
		flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder
		hwConfigPath = origHwConfigPath
		gpu = origGPU
		tensorParallelism = origTP
		defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	// Library-level BC-1 verification: trace loads correctly and requests are correct
	trace, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2 failed: %v", err)
	}
	if len(trace.Records) != 3 {
		t.Errorf("want 3 records, got %d", len(trace.Records))
	}

	reqs, err := workload.LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests failed: %v", err)
	}
	if len(reqs) != 3 {
		t.Fatalf("want 3 requests, got %d (BC-1)", len(reqs))
	}

	// Verify token counts preserved (BC-1)
	for _, req := range reqs {
		if len(req.InputTokens) != 10 {
			t.Errorf("want 10 input tokens, got %d", len(req.InputTokens))
		}
		if len(req.OutputTokens) != 5 {
			t.Errorf("want 5 output tokens, got %d", len(req.OutputTokens))
		}
	}

	// Verify horizon computation (BC-3): max arrival = 200000, horizon = 400000
	horizon := computeReplayHorizon(reqs)
	if horizon != 400000 {
		t.Errorf("want horizon 400000 (200000*2), got %d (BC-3)", horizon)
	}

	// Full simulation via replayCmd.Run (BC-2: verifies SimResult JSON output)
	model = "test-model"
	latencyModelBackend = "trained-physics"
	// Note: betaCoeffs and alphaCoeffs NOT set → auto-loads from defaults.yaml
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 1
	seed = 42
	resultsPath = resultsFilePath
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	simulationHorizon = math.MaxInt64
	modelConfigFolder = mcFolder
	hwConfigPath = hwCfgPath
	gpu = "H100"
	tensorParallelism = 1

	// Create a cobra command with Changed() tracking for the flags the Run closure checks.
	// This is required so cmd.Flags().Changed("latency-model") etc. return correct values.
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		// Note: --beta-coeffs and --alpha-coeffs omitted → auto-loads from defaults.yaml
		"--total-kv-blocks", "1000",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwCfgPath,
		"--trace-header", headerPath,
		"--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	// Run the replay command
	replayCmd.Run(testCmd, nil)

	// Verify SimResult file was written (BC-2)
	data, err := os.ReadFile(resultsFilePath)
	if err != nil {
		t.Fatalf("results file not written: %v", err)
	}
	var simResults []workload.SimResult
	if err := json.Unmarshal(data, &simResults); err != nil {
		t.Fatalf("failed to parse SimResult JSON: %v\ncontent: %s", err, data)
	}

	// All 3 requests should have completed (BC-1: fidelity)
	if len(simResults) != 3 {
		t.Errorf("want 3 SimResult entries (one per trace record), got %d", len(simResults))
	}

	// Verify integer request IDs 0, 1, 2 in sorted order (BC-2)
	for i, sr := range simResults {
		if sr.RequestID != i {
			t.Errorf("simResults[%d].RequestID = %d, want %d", i, sr.RequestID, i)
		}
		if sr.TTFT <= 0 {
			t.Errorf("simResults[%d].TTFT must be > 0, got %f", i, sr.TTFT)
		}
		if sr.E2E <= 0 {
			t.Errorf("simResults[%d].E2E must be > 0, got %f", i, sr.E2E)
		}
		if sr.InputTokens != 10 {
			t.Errorf("simResults[%d].InputTokens = %d, want 10", i, sr.InputTokens)
		}
		if sr.OutputTokens != 5 {
			t.Errorf("simResults[%d].OutputTokens = %d, want 5", i, sr.OutputTokens)
		}
	}

	// TTFT must be in microseconds (not ms) and positive.
	// With trained-physics (β₅=100 µs/layer, L=2), TTFT ≈ 200+ µs.
	if len(simResults) > 0 && simResults[0].TTFT <= 0 {
		t.Errorf("TTFT %f must be positive (microseconds)", simResults[0].TTFT)
	}
}

// TestReplayCmd_TraceOutput_NoOp verifies BC-4:
// omitting --trace-output produces no .yaml/.csv files.
func TestReplayCmd_TraceOutput_NoOp(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")

	headerContent := "trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"
	if err := os.WriteFile(headerPath, []byte(headerContent), 0644); err != nil {
		t.Fatal(err)
	}
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder3, hwPath3 := setupTrainedPhysicsTestFixtures(t)

	// Save/restore package-level vars
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder
	origHwConfigPath := hwConfigPath
	origGPU := gpu
	origTP := tensorParallelism
	origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode
	origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
		flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder
		hwConfigPath = origHwConfigPath
		gpu = origGPU
		tensorParallelism = origTP
		defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	model = "test-model"
	latencyModelBackend = "trained-physics"
	// Note: betaCoeffs and alphaCoeffs NOT set → auto-loads from defaults.yaml
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 1
	seed = 42
	resultsPath = ""
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	simulationHorizon = math.MaxInt64
	replayTraceOutput = "" // BC-4: no --trace-output flag set
	modelConfigFolder = mcFolder3
	hwConfigPath = hwPath3
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = "../defaults.yaml" // Load trained-physics coefficients (relative to cmd/ test dir)

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		// Note: --beta-coeffs and --alpha-coeffs omitted → auto-loads from defaults.yaml
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder3, "--hardware-config", hwPath3,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	replayCmd.Run(testCmd, nil)

	// BC-4: no output files written — check a prefix that was NOT requested
	prefix := filepath.Join(dir, "out")
	if _, err := os.Stat(prefix + ".yaml"); !os.IsNotExist(err) {
		t.Error("BC-4: unexpected .yaml file written when --trace-output was absent")
	}
	if _, err := os.Stat(prefix + ".csv"); !os.IsNotExist(err) {
		t.Error("BC-4: unexpected .csv file written when --trace-output was absent")
	}
}

// TestReplayCmd_TraceOutput_Determinism verifies BC-5 (INV-6):
// same seed + same trace produces byte-identical output files.
func TestReplayCmd_TraceOutput_Determinism(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")

	headerContent := "trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"
	if err := os.WriteFile(headerPath, []byte(headerContent), 0644); err != nil {
		t.Fatal(err)
	}
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,0,,0,false,20,8,20,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder4, hwPath4 := setupTrainedPhysicsTestFixtures(t)

	// runOnce runs the replay and returns the content of the output files
	runOnce := func(prefix string) (yamlBytes, csvBytes []byte) {
		t.Helper()

		origModel := model
		origBackend := latencyModelBackend
		origBeta := betaCoeffs
		origAlpha := alphaCoeffs
		origTotalKV := totalKVBlocks
		origBlockSize := blockSizeTokens
		origMaxRunning := maxRunningReqs
		origMaxSched := maxScheduledTokens
		origInstances := numInstances
		origSeed := seed
		origResults := resultsPath
		origThreshold := longPrefillTokenThreshold
		origKVCPU := kvCPUBlocks
		origOffload := kvOffloadThreshold
		origBandwidth := kvTransferBandwidth
		origBaseLatency := kvTransferBaseLatency
		origSnapRefresh := snapshotRefreshInterval
		origAdmission := admissionPolicy
		origRouting := routingPolicy
		origScheduler := scheduler
		origPolicyConfig := policyConfigPath
		origMaxModelLen := maxModelLen
		origTraceLevel := traceLevel
		origCounterfactualK := counterfactualK
		origTraceHeader := traceHeaderPath
		origTraceData := traceDataPath
		origSimHorizon := simulationHorizon
		origTraceOutput := replayTraceOutput
		origCacheSignalDelay := cacheSignalDelay
		origFlowControlEnabled := flowControlEnabled
		origFlowControlDetector := flowControlDetector
		origFlowControlDispatchOrder := flowControlDispatchOrder
		origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
		origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
		origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
		origFlowControlMaxConcurrency := flowControlMaxConcurrency
		origModelConfigFolder := modelConfigFolder
		origHwConfigPath := hwConfigPath
		origGPU := gpu
		origTP := tensorParallelism
		origSessionMode := replaySessionMode
		origThinkTimeMs := replayThinkTimeMs
		origThinkTimeDist := replayThinkTimeDist
		origDefaultsFilePathInner := defaultsFilePath
		defer func() {
			model = origModel
			latencyModelBackend = origBackend
			betaCoeffs = origBeta
			alphaCoeffs = origAlpha
			totalKVBlocks = origTotalKV
			blockSizeTokens = origBlockSize
			maxRunningReqs = origMaxRunning
			maxScheduledTokens = origMaxSched
			numInstances = origInstances
			seed = origSeed
			resultsPath = origResults
			longPrefillTokenThreshold = origThreshold
			kvCPUBlocks = origKVCPU
			kvOffloadThreshold = origOffload
			kvTransferBandwidth = origBandwidth
			kvTransferBaseLatency = origBaseLatency
			snapshotRefreshInterval = origSnapRefresh
			admissionPolicy = origAdmission
			routingPolicy = origRouting
			scheduler = origScheduler
			policyConfigPath = origPolicyConfig
			maxModelLen = origMaxModelLen
			traceLevel = origTraceLevel
			counterfactualK = origCounterfactualK
			traceHeaderPath = origTraceHeader
			traceDataPath = origTraceData
			simulationHorizon = origSimHorizon
			replayTraceOutput = origTraceOutput
			cacheSignalDelay = origCacheSignalDelay
			flowControlEnabled = origFlowControlEnabled
			flowControlDetector = origFlowControlDetector
			flowControlDispatchOrder = origFlowControlDispatchOrder
			flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
			flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
			flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
			flowControlMaxConcurrency = origFlowControlMaxConcurrency
			modelConfigFolder = origModelConfigFolder
			hwConfigPath = origHwConfigPath
			gpu = origGPU
			tensorParallelism = origTP
			replaySessionMode = origSessionMode
			replayThinkTimeMs = origThinkTimeMs
			replayThinkTimeDist = origThinkTimeDist
			defaultsFilePath = origDefaultsFilePathInner
		}()

		model = "test-model"
		latencyModelBackend = "trained-physics"
		// Note: betaCoeffs and alphaCoeffs NOT set → auto-loads from defaults.yaml
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxRunningReqs = 64
		maxScheduledTokens = 2048
		numInstances = 1
		seed = 42
		resultsPath = ""
		longPrefillTokenThreshold = 0
		kvCPUBlocks = 0
		kvOffloadThreshold = 0.9
		kvTransferBandwidth = 100.0
		kvTransferBaseLatency = 0
		snapshotRefreshInterval = 0
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		policyConfigPath = ""
		maxModelLen = 0
		traceLevel = "none"
		counterfactualK = 0
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		simulationHorizon = math.MaxInt64
		replayTraceOutput = prefix
		modelConfigFolder = mcFolder4
		hwConfigPath = hwPath4
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml" // Load trained-physics coefficients (relative to cmd/ test dir)

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		testCmd.Flags().StringVar(&replayTraceOutput, "trace-output", "", "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			// Note: --beta-coeffs and --alpha-coeffs omitted → auto-loads from defaults.yaml
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder4, "--hardware-config", hwPath4,
			"--trace-header", headerPath,
			"--trace-data", dataPath, "--trace-output", prefix,
			"--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			t.Fatalf("ParseFlags failed: %v", err)
		}
		replayCmd.Run(testCmd, nil)

		y, err := os.ReadFile(prefix + ".yaml")
		if err != nil {
			t.Fatalf("output YAML not found: %v", err)
		}
		c, err := os.ReadFile(prefix + ".csv")
		if err != nil {
			t.Fatalf("output CSV not found: %v", err)
		}
		return y, c
	}

	prefix1 := filepath.Join(dir, "run1")
	prefix2 := filepath.Join(dir, "run2")

	yaml1, csv1 := runOnce(prefix1)
	yaml2, csv2 := runOnce(prefix2)

	// BC-5 / INV-6: byte-identical output
	if string(yaml1) != string(yaml2) {
		t.Error("BC-5: YAML output is non-deterministic across runs with same seed")
	}
	if string(csv1) != string(csv2) {
		t.Error("BC-5: CSV output is non-deterministic across runs with same seed")
	}
}

// TestReplayCmd_AnomalyBlock_TimedOutRequests verifies BC-1:
// when replay produces TimedOutRequests > 0, the anomaly block includes "Timed Out Requests: N".
func TestReplayCmd_AnomalyBlock_TimedOutRequests(t *testing.T) {
	// GIVEN a trace with deadline_us=1 — request deadline expires at t=1µs, before any execution step
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")

	headerContent := "trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"
	if err := os.WriteFile(headerPath, []byte(headerContent), 0644); err != nil {
		t.Fatal(err)
	}
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,1,0,0,0,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	// Save and restore package-level vars (same pattern as TestReplayCmd_EndToEnd_TrainedPhysicsMode)
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder
	origHwConfigPath := hwConfigPath
	origGPU := gpu
	origTP := tensorParallelism
	origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode
	origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
		flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder
		hwConfigPath = origHwConfigPath
		gpu = origGPU
		tensorParallelism = origTP
		defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	model = "test-model"
	latencyModelBackend = "trained-physics"
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 1
	seed = 42
	resultsPath = ""
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	simulationHorizon = math.MaxInt64
	replayTraceOutput = ""
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = "../defaults.yaml"
	cacheSignalDelay = 0
	flowControlEnabled = false
	flowControlDetector = "utilization"
	flowControlDispatchOrder = "fifo"
	flowControlMaxQueueDepth = 0
	flowControlQueueDepthThreshold = 5.0
	flowControlKVCacheUtilThreshold = 0.8
	flowControlMaxConcurrency = 0

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		"--total-kv-blocks", "1000",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--trace-header", headerPath,
		"--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	// Capture stdout (same pattern as root_test.go)
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	replayCmd.Run(testCmd, nil)
	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	out := buf.String()

	// THEN: anomaly block fires and "Timed Out Requests: 1" appears (BC-1)
	if !strings.Contains(out, "=== Anomaly Counters ===") {
		t.Errorf("BC-1: expected anomaly block header in output:\n%s", out)
	}
	if !strings.Contains(out, "Timed Out Requests: 1") {
		t.Errorf("BC-1: expected 'Timed Out Requests: 1' in anomaly block, got:\n%s", out)
	}
}

// TestReplayCmd_AutoscalerBundleFatal verifies BC-2:
// policy bundle with autoscaler config causes fatal exit in replay.
func TestReplayCmd_AutoscalerBundleFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		// Running as subprocess: set up and trigger the fatal path.
		dir := t.TempDir()
		bundleYAML := "autoscaler:\n  interval_us: 500000\n"
		bundlePath := filepath.Join(dir, "bundle.yaml")
		if err := os.WriteFile(bundlePath, []byte(bundleYAML), 0644); err != nil {
			os.Exit(2)
		}
		headerPath := filepath.Join(dir, "trace.yaml")
		dataPath := filepath.Join(dir, "trace.csv")
		_ = os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
		_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)

		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxRunningReqs = 64
		maxScheduledTokens = 2048
		numInstances = 1
		seed = 42
		longPrefillTokenThreshold = 0
		kvCPUBlocks = 0
		kvOffloadThreshold = 0.9
		kvTransferBandwidth = 100.0
		kvTransferBaseLatency = 0
		snapshotRefreshInterval = 0
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		policyConfigPath = bundlePath
		maxModelLen = 0
		traceLevel = "none"
		counterfactualK = 0
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed"
		resultsPath = ""
		replayTraceOutput = ""

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--policy-config", bundlePath, "--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2) // distinct from logrus.Fatalf exit code (1)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)                  // reached only if no fatal = parent test failure
	}

	// Parent: re-run this test as subprocess and expect exit code 1 (logrus.Fatalf) (BC-2).
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_AutoscalerBundleFatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("BC-2: expected non-zero exit when autoscaler bundle is present, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("BC-2: unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("BC-2: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "autoscaler") {
		t.Errorf("BC-2: fatal message should mention 'autoscaler', got:\n%s", out)
	}
}

// TestReplayCmd_NodePoolsBundleFatal verifies BC-4:
// policy bundle with node_pools causes fatal exit in replay.
func TestReplayCmd_NodePoolsBundleFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		dir := t.TempDir()
		bundleYAML := "node_pools:\n  - name: pool-a\n    gpu_type: H100\n    gpus_per_node: 8\n    gpu_memory_gib: 80\n    initial_nodes: 1\n    min_nodes: 1\n    max_nodes: 4\n    cost_per_hour: 32.0\n"
		bundlePath := filepath.Join(dir, "bundle.yaml")
		if err := os.WriteFile(bundlePath, []byte(bundleYAML), 0644); err != nil {
			os.Exit(2)
		}
		headerPath := filepath.Join(dir, "trace.yaml")
		dataPath := filepath.Join(dir, "trace.csv")
		_ = os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
		_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)

		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxRunningReqs = 64
		maxScheduledTokens = 2048
		numInstances = 1
		seed = 42
		longPrefillTokenThreshold = 0
		kvCPUBlocks = 0
		kvOffloadThreshold = 0.9
		kvTransferBandwidth = 100.0
		kvTransferBaseLatency = 0
		snapshotRefreshInterval = 0
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		policyConfigPath = bundlePath
		maxModelLen = 0
		traceLevel = "none"
		counterfactualK = 0
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed"
		resultsPath = ""
		replayTraceOutput = ""

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--policy-config", bundlePath, "--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2) // distinct from logrus.Fatalf exit code (1)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_NodePoolsBundleFatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("BC-4: expected non-zero exit when node_pools bundle is present, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("BC-4: unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("BC-4: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "node_pools") {
		t.Errorf("BC-4: fatal message should mention 'node_pools', got:\n%s", out)
	}
}

// TestReplayCmd_PD_BasicSmoke verifies BC-1 (pre-parity smoke):
// replay with PD flags set does not panic and completes simulation.
// Requires Track A (Task 3) wiring to exercise the disaggregated path.
func TestReplayCmd_PD_BasicSmoke(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644); err != nil {
		t.Fatal(err)
	}
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n" +
		"1,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	// Save/restore PD-related package-level vars.
	origPrefillInstances := prefillInstances
	origDecodeInstances := decodeInstances
	origSharedInstances := prefillDecodeInstances
	origPDDecider := pdDecider
	origPDTransferBandwidth := pdTransferBandwidth
	origPDTransferBaseLatency := pdTransferBaseLatency
	origPDTransferContention := pdTransferContention
	origPDPrefixThreshold := pdPrefixThreshold
	origPrefillScorers := prefillRoutingScorers
	origDecodeScorers := decodeRoutingScorers
	defer func() {
		prefillInstances = origPrefillInstances
		decodeInstances = origDecodeInstances
		prefillDecodeInstances = origSharedInstances
		pdDecider = origPDDecider
		pdTransferBandwidth = origPDTransferBandwidth
		pdTransferBaseLatency = origPDTransferBaseLatency
		pdTransferContention = origPDTransferContention
		pdPrefixThreshold = origPDPrefixThreshold
		prefillRoutingScorers = origPrefillScorers
		decodeRoutingScorers = origDecodeScorers
	}()

	// Save/restore standard vars.
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder
	origHwConfigPath := hwConfigPath
	origGPU := gpu
	origTP := tensorParallelism
	origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode
	origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
		flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder
		hwConfigPath = origHwConfigPath
		gpu = origGPU
		tensorParallelism = origTP
		defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	// WHEN: replay with PD config: 1 prefill + 1 decode out of 2 total instances.
	model = "test-model"
	latencyModelBackend = "trained-physics"
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 2
	seed = 42
	resultsPath = ""
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = headerPath
	traceDataPath = dataPath
	simulationHorizon = math.MaxInt64
	replayTraceOutput = ""
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = "../defaults.yaml"
	replaySessionMode = "fixed"
	replayThinkTimeMs = 0
	replayThinkTimeDist = ""
	cacheSignalDelay = 0
	flowControlEnabled = false
	flowControlDetector = "utilization"
	flowControlDispatchOrder = "fifo"
	flowControlMaxQueueDepth = 0
	flowControlQueueDepthThreshold = 5.0
	flowControlKVCacheUtilThreshold = 0.8
	flowControlMaxConcurrency = 0

	// PD config: 1 prefill + 1 decode.
	prefillInstances = 1
	decodeInstances = 1
	prefillDecodeInstances = 0
	pdDecider = "always"
	pdTransferBandwidth = 25.0
	pdTransferBaseLatency = 0.05
	pdTransferContention = false
	pdPrefixThreshold = 0
	prefillRoutingScorers = ""
	decodeRoutingScorers = ""

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--num-instances", "2",
		"--prefill-instances", "1", "--decode-instances", "1",
		"--pd-decider", "always", "--pd-transfer-bandwidth", "25.0",
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}

	// THEN: simulation completes without panic (BC-1 smoke test).
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("BC-1: replay with PD config panicked: %v", r)
		}
	}()
	replayCmd.Run(testCmd, nil)
}

// TestINV13_RunReplayParity_PD verifies INV-13 for PD disaggregation:
// running the same requests through a PD cluster directly vs. through
// trace-export-then-replay produces identical per-request TTFT and E2E.
func TestINV13_RunReplayParity_PD(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeMinimalPDRequests(t)

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	dir := t.TempDir()

	// Write defaults.yaml with trained-physics coefficients.
	defaultsContent := `trained_physics_coefficients:
  alpha_coeffs: [100.0, 1.0, 100.0]
  beta_coeffs: [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
`
	defaultsPath := filepath.Join(filepath.Dir(hwPath), "defaults.yaml")
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0644); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	// Build SimConfig from model config files.
	hfPath := filepath.Join(mcFolder, "config.json")
	hfConfig, err := latency.ParseHFConfig(hfPath)
	if err != nil {
		t.Fatalf("ParseHFConfig: %v", err)
	}
	mc, err := latency.GetModelConfigFromHF(hfConfig)
	if err != nil {
		t.Fatalf("GetModelConfigFromHF: %v", err)
	}
	hwCfg, err := latency.GetHWConfig(hwPath, "H100")
	if err != nil {
		t.Fatalf("GetHWConfig: %v", err)
	}

	betaCfg := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alphaCfg := []float64{100.0, 1.0, 100.0}

	// INV-13 SYNC POINT: cfg must match the DeploymentConfig built by replayCmd.Run
	// for the same flags. Keep in sync with cmd/replay.go (see cmd/root.go:1500).
	cfg := cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                fixedSeed,
			KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0.9, 100.0, 0),
			BatchConfig:         sim.NewBatchConfig(64, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betaCfg, alphaCfg),
			ModelHardwareConfig: sim.NewModelHardwareConfig(*mc, hwCfg, "test-model", "H100", 1, "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
		},
		NumInstances:            2,
		AdmissionPolicy:         "always-admit",
		RoutingPolicy:           "round-robin",
		PrefillInstances:        1,
		DecodeInstances:         1,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
	}

	// WHEN: direct run.
	cs1 := cluster.NewClusterSimulator(cfg, requests, nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	runTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	runE2Es := cs1.AggregatedMetrics().RequestE2Es

	if len(runTTFTs) == 0 {
		t.Fatal("INV-13: direct run produced no completed requests — cannot verify parity")
	}

	// WHEN: export to trace → reload → replay with same config.
	traceRecords := workload.RequestsToTraceRecords(requests)
	traceHdr := &workload.TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	traceHeaderFile := filepath.Join(dir, "trace.yaml")
	traceDataFile := filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(traceHdr, traceRecords, traceHeaderFile, traceDataFile); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	traceData, err := workload.LoadTraceV2(traceHeaderFile, traceDataFile)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	replayReqs, err := workload.LoadTraceV2Requests(traceData, fixedSeed)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}

	cs2 := cluster.NewClusterSimulator(cfg, replayReqs, nil)
	if err := cs2.Run(); err != nil {
		t.Fatalf("replay run failed: %v", err)
	}
	replayTTFTs := cs2.AggregatedMetrics().RequestTTFTs
	replayE2Es := cs2.AggregatedMetrics().RequestE2Es

	// THEN: per-request metrics must be identical (INV-13, BC-1).
	if len(runTTFTs) != len(replayTTFTs) {
		t.Errorf("INV-13: TTFT map size mismatch: run=%d replay=%d", len(runTTFTs), len(replayTTFTs))
	}
	for id, ttft := range runTTFTs {
		if got, ok := replayTTFTs[id]; !ok {
			t.Errorf("INV-13: request %s present in run but missing from replay TTFTs", id)
		} else if got != ttft {
			t.Errorf("INV-13: request %s TTFT mismatch: run=%f replay=%f", id, ttft, got)
		}
	}
	for id, e2e := range runE2Es {
		if got, ok := replayE2Es[id]; !ok {
			t.Errorf("INV-13: request %s present in run but missing from replay E2Es", id)
		} else if got != e2e {
			t.Errorf("INV-13: request %s E2E mismatch: run=%f replay=%f", id, e2e, got)
		}
	}
}

// makeMinimalPDRequests creates a small set of deterministic requests for PD parity testing.
func makeMinimalPDRequests(t *testing.T) []*sim.Request {
	t.Helper()
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		inputToks := make([]int, 10)
		for j := range inputToks {
			inputToks[j] = 100 + i*10 + j
		}
		outputToks := make([]int, 5)
		for j := range outputToks {
			outputToks[j] = 200 + j
		}
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * 100_000,
			InputTokens:  inputToks,
			OutputTokens: outputToks,
			MaxOutputLen: 100,
		}
	}
	return reqs
}

// TestINV13_RunReplayParity_PD_CLI verifies INV-13 end-to-end through replayCmd.Run:
// export trace from a PD cluster, replay via the CLI path, and confirm per-request
// TTFT/E2E match the direct library run. This catches bugs in the replayCmd CLI
// wiring that TestINV13_RunReplayParity_PD (library-level) would miss.
func TestINV13_RunReplayParity_PD_CLI(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeMinimalPDRequests(t)

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	dir := t.TempDir()

	defaultsContent := `trained_physics_coefficients:
  alpha_coeffs: [100.0, 1.0, 100.0]
  beta_coeffs: [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
`
	defaultsPath := filepath.Join(filepath.Dir(hwPath), "defaults.yaml")
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0644); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	hfPath := filepath.Join(mcFolder, "config.json")
	hfConfig, err := latency.ParseHFConfig(hfPath)
	if err != nil {
		t.Fatalf("ParseHFConfig: %v", err)
	}
	mc, err := latency.GetModelConfigFromHF(hfConfig)
	if err != nil {
		t.Fatalf("GetModelConfigFromHF: %v", err)
	}
	hwCfg, err := latency.GetHWConfig(hwPath, "H100")
	if err != nil {
		t.Fatalf("GetHWConfig: %v", err)
	}

	betaCfg := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alphaCfg := []float64{100.0, 1.0, 100.0}

	// WHEN: direct library run (reference values).
	cfg := cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                fixedSeed,
			KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0.9, 100.0, 0),
			BatchConfig:         sim.NewBatchConfig(64, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betaCfg, alphaCfg),
			ModelHardwareConfig: sim.NewModelHardwareConfig(*mc, hwCfg, "test-model", "H100", 1, "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
		},
		NumInstances:            2,
		AdmissionPolicy:         "always-admit",
		RoutingPolicy:           "round-robin",
		PrefillInstances:        1,
		DecodeInstances:         1,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
	}
	cs1 := cluster.NewClusterSimulator(cfg, requests, nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	wantTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	wantE2Es := cs1.AggregatedMetrics().RequestE2Es
	if len(wantTTFTs) == 0 {
		t.Fatal("INV-13 CLI: direct run produced no completed requests")
	}

	// WHEN: export trace → replay through replayCmd.Run → read SimResult JSON.
	traceRecords := workload.RequestsToTraceRecords(requests)
	traceHdr := &workload.TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	traceHeaderFile := filepath.Join(dir, "trace.yaml")
	traceDataFile := filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(traceHdr, traceRecords, traceHeaderFile, traceDataFile); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}

	resultsFile := filepath.Join(dir, "results.json")

	// Save/restore all package-level vars including PD vars.
	origPrefillInstances := prefillInstances
	origDecodeInstances := decodeInstances
	origSharedInstances := prefillDecodeInstances
	origPDDecider := pdDecider
	origPDTransferBandwidth := pdTransferBandwidth
	origPDTransferBaseLatency := pdTransferBaseLatency
	origPDTransferContention := pdTransferContention
	origPDPrefixThreshold := pdPrefixThreshold
	origPrefillScorers := prefillRoutingScorers
	origDecodeScorers := decodeRoutingScorers
	defer func() {
		prefillInstances = origPrefillInstances
		decodeInstances = origDecodeInstances
		prefillDecodeInstances = origSharedInstances
		pdDecider = origPDDecider
		pdTransferBandwidth = origPDTransferBandwidth
		pdTransferBaseLatency = origPDTransferBaseLatency
		pdTransferContention = origPDTransferContention
		pdPrefixThreshold = origPDPrefixThreshold
		prefillRoutingScorers = origPrefillScorers
		decodeRoutingScorers = origDecodeScorers
	}()
	origModel := model; origBackend := latencyModelBackend; origBeta := betaCoeffs
	origAlpha := alphaCoeffs; origTotalKV := totalKVBlocks; origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs; origMaxSched := maxScheduledTokens
	origInstances := numInstances; origSeedV := seed; origResults := resultsPath
	origThreshold := longPrefillTokenThreshold; origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold; origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency; origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy; origRouting := routingPolicy
	origScheduler := scheduler; origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen; origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK; origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath; origSimHorizon := simulationHorizon
	origTraceOutput := replayTraceOutput; origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled; origFlowControlDetector := flowControlDetector
	origFlowControlDispatchOrder := flowControlDispatchOrder
	origFlowControlMaxQueueDepth := flowControlMaxQueueDepth
	origFlowControlQueueDepthThreshold := flowControlQueueDepthThreshold
	origFlowControlKVCacheUtilThreshold := flowControlKVCacheUtilThreshold
	origFlowControlMaxConcurrency := flowControlMaxConcurrency
	origModelConfigFolder := modelConfigFolder; origHwConfigPath := hwConfigPath
	origGPU := gpu; origTP := tensorParallelism; origDefaultsFilePath := defaultsFilePath
	origSessionMode := replaySessionMode; origThinkTimeMs := replayThinkTimeMs
	origThinkTimeDist := replayThinkTimeDist
	defer func() {
		model = origModel; latencyModelBackend = origBackend; betaCoeffs = origBeta
		alphaCoeffs = origAlpha; totalKVBlocks = origTotalKV; blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning; maxScheduledTokens = origMaxSched
		numInstances = origInstances; seed = origSeedV; resultsPath = origResults
		longPrefillTokenThreshold = origThreshold; kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload; kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency; snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission; routingPolicy = origRouting
		scheduler = origScheduler; policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen; traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK; traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData; simulationHorizon = origSimHorizon
		replayTraceOutput = origTraceOutput; cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled; flowControlDetector = origFlowControlDetector
		flowControlDispatchOrder = origFlowControlDispatchOrder
		flowControlMaxQueueDepth = origFlowControlMaxQueueDepth
		flowControlQueueDepthThreshold = origFlowControlQueueDepthThreshold
		flowControlKVCacheUtilThreshold = origFlowControlKVCacheUtilThreshold
		flowControlMaxConcurrency = origFlowControlMaxConcurrency
		modelConfigFolder = origModelConfigFolder; hwConfigPath = origHwConfigPath
		gpu = origGPU; tensorParallelism = origTP; defaultsFilePath = origDefaultsFilePath
		replaySessionMode = origSessionMode; replayThinkTimeMs = origThinkTimeMs
		replayThinkTimeDist = origThinkTimeDist
	}()

	model = "test-model"; latencyModelBackend = "trained-physics"
	totalKVBlocks = 1000; blockSizeTokens = 16; maxRunningReqs = 64
	maxScheduledTokens = 2048; numInstances = 2; seed = fixedSeed
	resultsPath = resultsFile; longPrefillTokenThreshold = 0; kvCPUBlocks = 0
	kvOffloadThreshold = 0.9; kvTransferBandwidth = 100.0; kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0; admissionPolicy = "always-admit"
	routingPolicy = "round-robin"; scheduler = "fcfs"; policyConfigPath = ""
	maxModelLen = 0; traceLevel = "none"; counterfactualK = 0
	traceHeaderPath = traceHeaderFile; traceDataPath = traceDataFile
	simulationHorizon = 10_000_000; replayTraceOutput = ""
	modelConfigFolder = mcFolder; hwConfigPath = hwPath
	gpu = "H100"; tensorParallelism = 1; defaultsFilePath = defaultsPath
	replaySessionMode = "fixed"; replayThinkTimeMs = 0; replayThinkTimeDist = ""
	cacheSignalDelay = 0; flowControlEnabled = false
	prefillInstances = 1; decodeInstances = 1; prefillDecodeInstances = 0
	pdDecider = "always"; pdTransferBandwidth = 25.0; pdTransferBaseLatency = 0.05
	pdTransferContention = false; pdPrefixThreshold = 0
	prefillRoutingScorers = ""; decodeRoutingScorers = ""

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&resultsPath, "results-path", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", traceHeaderFile, "--trace-data", traceDataFile,
		"--results-path", resultsFile,
		"--num-instances", "2",
		"--prefill-instances", "1", "--decode-instances", "1",
		"--pd-decider", "always", "--pd-transfer-bandwidth", "25.0",
		"--pd-transfer-base-latency", "0.05",
		"--horizon", "10000000",
		"--defaults-filepath", defaultsPath,
	}); err != nil {
		t.Fatalf("ParseFlags failed: %v", err)
	}
	replayCmd.Run(testCmd, nil)

	// Read per-request SimResult JSON written by replayCmd.Run.
	data, err := os.ReadFile(resultsFile)
	if err != nil {
		t.Fatalf("results file not written: %v", err)
	}
	var simResults []workload.SimResult
	if err := json.Unmarshal(data, &simResults); err != nil {
		t.Fatalf("parse SimResult JSON: %v", err)
	}

	// THEN: per-request TTFT and E2E must match the direct library run (INV-13).
	for _, sr := range simResults {
		reqID := fmt.Sprintf("request_%d", sr.RequestID)
		wantTTFT, ok := wantTTFTs[reqID]
		if !ok {
			t.Errorf("INV-13 CLI: request %s missing from library run TTFTs", reqID)
			continue
		}
		if sr.TTFT != wantTTFT {
			t.Errorf("INV-13 CLI: request %s TTFT: CLI=%f library=%f", reqID, sr.TTFT, wantTTFT)
		}
		wantE2E, ok := wantE2Es[reqID]
		if !ok {
			t.Errorf("INV-13 CLI: request %s missing from library run E2Es", reqID)
			continue
		}
		if sr.E2E != wantE2E {
			t.Errorf("INV-13 CLI: request %s E2E: CLI=%f library=%f", reqID, sr.E2E, wantE2E)
		}
	}
	if len(simResults) != len(wantTTFTs) {
		t.Errorf("INV-13 CLI: completed request count mismatch: CLI=%d library=%d", len(simResults), len(wantTTFTs))
	}
}

// TestReplayCmd_AutoscalerFlagFatal verifies BC-3:
// passing --model-autoscaler-interval-us directly to blis replay causes fatal exit.
func TestReplayCmd_AutoscalerFlagFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "trace.yaml")
		dataPath := filepath.Join(dir, "trace.csv")
		_ = os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
		_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)

		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxRunningReqs = 64
		maxScheduledTokens = 2048
		numInstances = 1
		seed = 42
		longPrefillTokenThreshold = 0
		kvCPUBlocks = 0
		kvOffloadThreshold = 0.9
		kvTransferBandwidth = 100.0
		kvTransferBaseLatency = 0
		snapshotRefreshInterval = 0
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		policyConfigPath = ""
		maxModelLen = 0
		traceLevel = "none"
		counterfactualK = 0
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed"
		resultsPath = ""
		replayTraceOutput = ""

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--model-autoscaler-interval-us", "500000",
			"--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	// Parent: expect exit code 1 from --model-autoscaler-interval-us (BC-3).
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_AutoscalerFlagFatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("BC-3: expected non-zero exit when --model-autoscaler-interval-us is set, got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("BC-3: unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("BC-3: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "model-autoscaler-interval-us") {
		t.Errorf("BC-3: fatal message should mention 'model-autoscaler-interval-us', got:\n%s", out)
	}
}

// TestReplayCmd_PDTopologyFatal verifies BC-5:
// an invalid PD pool topology causes a fatal exit in replay.
func TestReplayCmd_PDTopologyFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "trace.yaml")
		dataPath := filepath.Join(dir, "trace.csv")
		_ = os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
		_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)

		mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
		model = "test-model"
		latencyModelBackend = "trained-physics"
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxRunningReqs = 64
		maxScheduledTokens = 2048
		numInstances = 2 // 4 prefill + 0 decode > 2 total: invalid topology
		seed = 42
		longPrefillTokenThreshold = 0
		kvCPUBlocks = 0
		kvOffloadThreshold = 0.9
		kvTransferBandwidth = 25.0
		kvTransferBaseLatency = 0
		snapshotRefreshInterval = 0
		admissionPolicy = "always-admit"
		routingPolicy = "round-robin"
		scheduler = "fcfs"
		policyConfigPath = ""
		maxModelLen = 0
		traceLevel = "none"
		counterfactualK = 0
		traceHeaderPath = headerPath
		traceDataPath = dataPath
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		gpu = "H100"
		tensorParallelism = 1
		defaultsFilePath = "../defaults.yaml"
		replaySessionMode = "fixed"
		resultsPath = ""
		replayTraceOutput = ""
		prefillInstances = 4
		decodeInstances = 0
		prefillDecodeInstances = 0
		pdDecider = "always"
		pdTransferBandwidth = 25.0
		pdTransferBaseLatency = 0.05

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
		testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--trace-header", headerPath, "--trace-data", dataPath,
			"--num-instances", "2",
			"--prefill-instances", "4", "--decode-instances", "0",
			"--pd-decider", "always", "--pd-transfer-bandwidth", "25.0",
			"--defaults-filepath", "../defaults.yaml",
		}); err != nil {
			fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
			os.Exit(2)
		}
		replayCmd.Run(testCmd, nil) // must Fatalf before here
		os.Exit(0)
	}

	// Parent: expect exit code 1 from ValidatePoolTopology (BC-5).
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_PDTopologyFatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("BC-5: expected non-zero exit when PD topology is invalid (4 prefill > 2 total), got exit 0")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("BC-5: unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("BC-5: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "topology") {
		t.Errorf("BC-5: fatal message should mention 'topology', got:\n%s", out)
	}
}

func TestExtractSimResults_PropagatesSLOClassModelITL(t *testing.T) {
	// GIVEN a Metrics struct with one completed request that has SLOClass, Model, and ITL set
	m := sim.NewMetrics()
	m.RequestTTFTs["request_0"] = 1000.0
	m.RequestE2Es["request_0"] = 5000.0
	m.Requests["request_0"] = sim.RequestMetrics{
		NumPrefillTokens: 100,
		NumDecodeTokens:  50,
		SLOClass:         "standard",
		Model:            "qwen3-14b",
		ITL:              5.0, // milliseconds
	}

	// WHEN extractSimResults is called
	results := extractSimResults(m)

	// THEN SLOClass, Model, and ITLMeanUs are populated correctly (BC-1)
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	r := results[0]
	if r.SLOClass != "standard" {
		t.Errorf("SLOClass: got %q, want %q", r.SLOClass, "standard")
	}
	if r.Model != "qwen3-14b" {
		t.Errorf("Model: got %q, want %q", r.Model, "qwen3-14b")
	}
	// ITL is 5.0ms in RequestMetrics → ITLMeanUs = 5000.0µs
	if r.ITLMeanUs != 5000.0 {
		t.Errorf("ITLMeanUs: got %f, want 5000.0 (5ms * 1000)", r.ITLMeanUs)
	}
}

func TestSimResult_NewFields_JSONOmitWhenEmpty(t *testing.T) {
	// BC-2: omitempty means empty SLOClass/Model and zero ITLMeanUs are omitted from JSON
	sr := workload.SimResult{
		RequestID:    1,
		TTFT:         100.0,
		E2E:          200.0,
		InputTokens:  10,
		OutputTokens: 5,
		// SLOClass, Model, ITLMeanUs intentionally zero/empty
	}
	data, err := json.Marshal(sr)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	s := string(data)
	if strings.Contains(s, "slo_class") {
		t.Errorf("omitempty: slo_class should be absent, got: %s", s)
	}
	if strings.Contains(s, `"model"`) {
		t.Errorf("omitempty: model should be absent, got: %s", s)
	}
	if strings.Contains(s, "itl_mean_us") {
		t.Errorf("omitempty: itl_mean_us should be absent, got: %s", s)
	}

	// GIVEN non-empty fields: all three must round-trip correctly (BC-1)
	sr2 := workload.SimResult{
		RequestID:    2,
		TTFT:         100.0,
		E2E:          200.0,
		InputTokens:  10,
		OutputTokens: 5,
		SLOClass:     "standard",
		Model:        "qwen3-14b",
		ITLMeanUs:    5000.0,
	}
	data2, err := json.Marshal(sr2)
	if err != nil {
		t.Fatalf("json.Marshal (non-empty): %v", err)
	}
	var got workload.SimResult
	if err := json.Unmarshal(data2, &got); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if got.SLOClass != "standard" {
		t.Errorf("SLOClass round-trip: got %q, want %q", got.SLOClass, "standard")
	}
	if got.Model != "qwen3-14b" {
		t.Errorf("Model round-trip: got %q, want %q", got.Model, "qwen3-14b")
	}
	if got.ITLMeanUs != 5000.0 {
		t.Errorf("ITLMeanUs round-trip: got %f, want 5000.0", got.ITLMeanUs)
	}
}
