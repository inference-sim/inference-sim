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

func TestReplayCmd_EndToEnd_BlackboxMode(t *testing.T) {
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
	latencyModelBackend = "blackbox"
	betaCoeffs = []float64{10000.0, 1.0, 1.0} // ~10ms prefill base
	alphaCoeffs = []float64{0.0, 0.0, 0.0}
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

	// Create a cobra command with Changed() tracking for the flags the Run closure checks.
	// This is required so cmd.Flags().Changed("latency-model") etc. return correct values.
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model",
		"--latency-model", "blackbox",
		"--beta-coeffs", "10000.0,1.0,1.0",
		"--alpha-coeffs", "0.0,0.0,0.0",
		"--total-kv-blocks", "1000",
		"--trace-header", headerPath,
		"--trace-data", dataPath,
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

	// TTFT must be in microseconds (not ms): 10ms prefill base = ~10,000 µs
	if len(simResults) > 0 && simResults[0].TTFT < 1000 {
		t.Errorf("TTFT %f looks like it's in ms (expected ~10000 µs for 10ms prefill base)", simResults[0].TTFT)
	}
}
