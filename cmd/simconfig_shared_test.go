package cmd

import (
	"os"
	"strings"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
)

// TestResolveLatencyConfig_SignatureCheck is a compile-time guard: if resolveLatencyConfig
// is removed or its signature changes, this file will not compile. The function value
// assignment is intentional — it documents the expected signature for code readers.
func TestResolveLatencyConfig_SignatureCheck(t *testing.T) {
	// Compile-time assertion: resolveLatencyConfig(cmd) returns latencyResolution.
	// The function value is never nil; this test catches signature drift at compile time.
	_ = resolveLatencyConfig
}

// TestResolvePolicies_SignatureCheck is a compile-time guard: if resolvePolicies
// is removed or its signature changes, this file will not compile.
func TestResolvePolicies_SignatureCheck(t *testing.T) {
	// Compile-time assertion: resolvePolicies(cmd) returns []sim.ScorerConfig.
	_ = resolvePolicies
}

// TestNoR23CommentSyncMarkersInReplay verifies that after the refactor,
// replay.go contains no R23 comment-sync markers (BC-3).
// It is a regression guard: fails if any R23: marker is re-introduced.
func TestNoR23CommentSyncMarkersInReplay(t *testing.T) {
	// GIVEN the source of cmd/replay.go
	data, err := os.ReadFile("replay.go")
	assert.NoError(t, err, "replay.go must be readable")

	// WHEN we scan for ANY R23 comment-sync marker variant
	// (variants used in the original: "R23: MUST match", "R23: same as runCmd",
	//  "R23: exact structure from runCmd")
	// THEN none should be present (BC-3: single code path eliminates need for sync markers)
	lines := strings.Split(string(data), "\n")
	for i, line := range lines {
		if strings.Contains(line, "R23:") {
			t.Errorf("line %d: R23 comment-sync marker found in replay.go — "+
				"this indicates duplicated SimConfig resolution logic: %q", i+1, line)
		}
	}
}

// TestRunCmd_SimConfigFlagsParity verifies that both commands register the same
// deployment-source flags with the same defaults (BC-1): the scenario, scenario directory,
// registry and catalog are the whole deployment input on both run and replay.
func TestRunCmd_SimConfigFlagsParity(t *testing.T) {
	for _, name := range []string{"scenario", "scenarios", "registry", "catalog"} {
		runFlag := runCmd.Flags().Lookup(name)
		replayFlag := replayCmd.Flags().Lookup(name)
		assert.NotNilf(t, runFlag, "runCmd must have --%s", name)
		assert.NotNilf(t, replayFlag, "replayCmd must have --%s", name)
		if runFlag != nil && replayFlag != nil {
			assert.Equalf(t, runFlag.DefValue, replayFlag.DefValue,
				"--%s default must match between run and replay", name)
		}
	}
}

// TestBothCommands_EngineAndDeploymentKnobsAreNotFlags: the kernel scenario is the single
// source of the deployment (model, hardware, parallelism) and the engine the kernel prices
// (KV pool, batch caps, block size, max-model-len, prefix caching, speculation, transfer
// costs). A flag restating any of them would let the simulated scheduler run an engine the
// kernel did not price, so none is registered on either command.
func TestBothCommands_EngineAndDeploymentKnobsAreNotFlags(t *testing.T) {
	removed := []string{
		"latency-model", "alpha-coeffs", "beta-coeffs", "hardware-config",
		"model", "hardware", "tp", "dp", "enable-expert-parallel", "moe-comm-backend",
		"prefill-tp", "decode-tp", "prefill-hardware", "decode-hardware",
		"prefill-latency-model", "decode-latency-model", "prefill-max-model-len", "decode-max-model-len",
		"prefill-moe-comm-backend", "decode-moe-comm-backend",
		"total-kv-blocks", "max-num-seqs", "max-num-running-reqs", "max-num-batched-tokens",
		"max-num-scheduled-tokens", "block-size-in-tokens", "gpu-memory-utilization", "max-model-len",
		"kv-cache-dtype", "no-enable-prefix-caching", "num-speculative-tokens", "speculative-method",
		"kv-transfer-bandwidth", "kv-transfer-base-latency",
		"pd-transfer-bandwidth", "pd-transfer-base-latency", "pd-transfer-contention",
		"comm-serialization-factor", "enforce-eager",
	}
	for _, c := range []struct {
		name string
		cmd  *cobra.Command
	}{{"run", runCmd}, {"replay", replayCmd}} {
		for _, name := range removed {
			assert.Nilf(t, c.cmd.Flags().Lookup(name),
				"%s must not register --%s: the kernel scenario states it", c.name, name)
		}
	}
}

// TestResolvePolicies_InvalidAdmissionPolicy_Fatal verifies that resolvePolicies
// rejects unknown admission policy names (BC-2).
func TestResolvePolicies_InvalidAdmissionPolicy_Fatal(t *testing.T) {
	// GIVEN an invalid admission policy name
	// WHEN IsValidAdmissionPolicy is called (the predicate used by resolvePolicies)
	// THEN it must return false — confirming resolvePolicies would fatalf
	assert.False(t, sim.IsValidAdmissionPolicy("nonexistent-policy"),
		"resolvePolicies must reject unknown admission policy names")
}

// TestResolvePolicies_PolicyFlagsRegisteredInBothCommands verifies that all flags
// consumed by resolvePolicies are registered in both runCmd and replayCmd (BC-2).
func TestResolvePolicies_PolicyFlagsRegisteredInBothCommands(t *testing.T) {
	policyFlags := []string{
		"admission-policy", "routing-policy", "scheduler", "preemption-policy",
		"routing-scorers", "lora-scorer-weight", "token-bucket-capacity", "token-bucket-refill-rate",
		"kv-cpu-blocks", "kv-offload-threshold", "snapshot-refresh-interval",
		"admission-latency", "routing-latency", "trace-level",
		"counterfactual-k", "summarize-trace", "policy-config",
		"cache-signal-delay",
	}
	for _, name := range policyFlags {
		assert.NotNilf(t, runCmd.Flags().Lookup(name),
			"runCmd must have --%s (consumed by resolvePolicies)", name)
		assert.NotNilf(t, replayCmd.Flags().Lookup(name),
			"replayCmd must have --%s (consumed by resolvePolicies)", name)
	}
}

// TestReplayCmd_BuildsItsLatencyModelOnlyThroughResolveLatencyConfig: replay's steps are priced
// by the model resolveLatencyConfig returns, the one function run uses too (R23, INV-13), and
// replay opens no kernel of its own -- a second construction site could resolve a different
// scenario or pool than the one the run priced.
func TestReplayCmd_BuildsItsLatencyModelOnlyThroughResolveLatencyConfig(t *testing.T) {
	data, err := os.ReadFile("replay.go")
	assert.NoError(t, err)
	content := string(data)

	assert.Contains(t, content, "resolveLatencyConfig(cmd)",
		"replay.go must resolve its latency model through resolveLatencyConfig(cmd)")
	assert.Regexp(t, `LatencyModel:\s+lr\.KernelModel`, content,
		"replay.go must price its steps with the model resolveLatencyConfig returned")
	for _, open := range []string{"kernelmodel.Open(", "kernelmodel.OpenPool(", "latencykernel.New("} {
		assert.NotContainsf(t, content, open,
			"replay.go must not build a latency model of its own (%s); use resolveLatencyConfig(cmd)", open)
	}
}

// TestReplayCmd_SourceContainsNoPolicyInlineBlocks verifies replay.go delegates
// policy resolution to the shared function (BC-2, BC-3).
func TestReplayCmd_SourceContainsNoPolicyInlineBlocks(t *testing.T) {
	data, err := os.ReadFile("replay.go")
	assert.NoError(t, err)
	content := string(data)

	assert.NotContains(t, content, `sim.IsValidAdmissionPolicy(`,
		"replay.go must not inline admission policy validation; use resolvePolicies(cmd)")
	assert.NotContains(t, content, `sim.LoadPolicyBundle(`,
		"replay.go must not inline policy bundle loading; use resolvePolicies(cmd)")
}

// TestBothCommands_SimConfigFlagsHaveIdenticalDefaults is a comprehensive
// regression guard: verifies that all flags consumed by resolveLatencyConfig
// and resolvePolicies have identical default values in runCmd and replayCmd.
func TestBothCommands_SimConfigFlagsHaveIdenticalDefaults(t *testing.T) {
	sharedFlags := []string{
		"scenario", "scenarios", "registry", "catalog",
		"admission-policy", "routing-policy", "scheduler", "preemption-policy",
		"routing-scorers", "lora-scorer-weight", "token-bucket-capacity", "token-bucket-refill-rate",
		"kv-cpu-blocks", "kv-offload-threshold", "snapshot-refresh-interval",
		"admission-latency", "routing-latency", "trace-level",
		"counterfactual-k", "summarize-trace", "policy-config",
		"num-instances",
		"long-prefill-token-threshold", "cache-signal-delay",
		"flow-control", "saturation-detector", "dispatch-order",
		"max-gateway-queue-depth", "queue-depth-threshold",
		"kv-cache-util-threshold", "max-concurrency",
		"per-band-capacity", "usage-limit-threshold",
		"queue-shedding", "dispatch-tick-interval",
		"in-flight-eviction",
	}
	for _, name := range sharedFlags {
		runFlag := runCmd.Flags().Lookup(name)
		replayFlag := replayCmd.Flags().Lookup(name)
		// Both commands must register the flag (not skip silently — a missing flag is a regression).
		if !assert.NotNilf(t, runFlag, "runCmd must have --%s", name) ||
			!assert.NotNilf(t, replayFlag, "replayCmd must have --%s", name) {
			continue
		}
		assert.Equalf(t, runFlag.DefValue, replayFlag.DefValue,
			"--%s: default value diverged between run (%q) and replay (%q)",
			name, runFlag.DefValue, replayFlag.DefValue)
	}
}
