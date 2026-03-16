package cmd

import (
	"encoding/json"
	"strings"
	"testing"

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
