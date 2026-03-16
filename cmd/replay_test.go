package cmd

import "testing"

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
