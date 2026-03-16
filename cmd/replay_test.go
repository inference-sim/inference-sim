package cmd

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

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

		// registerSimConfigFlags: results
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
