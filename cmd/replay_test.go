package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
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

		// registerSimConfigFlags: priority and scheduler
		"priority-policy", "scheduler",

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
	origPriority := priorityPolicy
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
		priorityPolicy = origPriority
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
	priorityPolicy = "constant"
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
	origPriority := priorityPolicy
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
		priorityPolicy = origPriority
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
	priorityPolicy = "constant"
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
	origPriority := priorityPolicy
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
		priorityPolicy = origPriority
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
	priorityPolicy = "constant"
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
		origPriority := priorityPolicy
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
			priorityPolicy = origPriority
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
		priorityPolicy = "constant"
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
