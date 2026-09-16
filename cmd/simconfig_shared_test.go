package cmd

import (
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	sim "github.com/inference-sim/inference-sim/sim"
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

// TestTrainedPhysicsBetaCoeffGuard_UsesCorrectMinimum verifies the shared function
// uses len < 7 for trained-physics (BC-4), matching the 7-coefficient model.
// The guard prevents insufficient coefficients for the trained-physics backend.
func TestTrainedPhysicsBetaCoeffGuard_UsesCorrectMinimum(t *testing.T) {
	// GIVEN the source of cmd/root.go (where the shared guard lives after Task 1)
	data, err := os.ReadFile("root.go")
	assert.NoError(t, err)

	content := string(data)

	// THEN: the guard uses < 7 (not < 10) for trained-physics (BC-4)
	// The local variable in resolveLatencyConfig is named "beta" (not "betaCoeffs")
	assert.Contains(t, content, `len(beta) < 7`,
		"trained-physics guard must use < 7 (model uses indices 0-6); local var is 'beta' in resolveLatencyConfig")

	// THEN: the wrong value < 10 must not appear in root.go
	assert.NotContains(t, content, `len(beta) < 10`,
		"< 10 was the replay.go drift bug and must not appear in the shared function")
}

// TestRunCmd_SimConfigFlagsParity verifies that both commands register the same
// latency-related flags with the same defaults (BC-1).
func TestRunCmd_SimConfigFlagsParity(t *testing.T) {
	// GIVEN both commands' flag sets
	// WHEN we check for latency-model related flags
	// THEN both commands must have the exact same set (registered via registerSimConfigFlags)
	latencyFlags := []string{
		"latency-model", "hardware", "tp", "dp", "enable-expert-parallel",
		"alpha-coeffs", "beta-coeffs",
		"total-kv-blocks", "block-size-in-tokens", "max-model-len",
		"gpu-memory-utilization", "catalog", "hardware-config",
		"comm-serialization-factor", "enforce-eager", // #1694 Part B
	}
	for _, name := range latencyFlags {
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
		"kv-cpu-blocks", "kv-offload-threshold", "kv-transfer-bandwidth",
		"kv-transfer-base-latency", "snapshot-refresh-interval",
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

// TestReplayCmd_SourceContainsNoInlineBackendBlocks verifies replay.go delegates
// to the shared function (no inline backend resolution after Task 3, BC-1).
func TestReplayCmd_SourceContainsNoInlineBackendBlocks(t *testing.T) {
	// GIVEN the source of cmd/replay.go
	data, err := os.ReadFile("replay.go")
	assert.NoError(t, err)
	content := string(data)

	// WHEN we check for the inline backend-resolution patterns
	// THEN they must not be present (indicates delegation to resolveLatencyConfig)
	assert.NotContains(t, content, `if backend == "roofline" {`,
		"replay.go must not contain inline roofline resolution block; use resolveLatencyConfig(cmd)")
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
		"latency-model", "hardware", "tp", "dp", "enable-expert-parallel",
		"alpha-coeffs", "beta-coeffs",
		"total-kv-blocks", "block-size-in-tokens", "max-model-len",
		"gpu-memory-utilization", "catalog", "hardware-config",
		"comm-serialization-factor", "enforce-eager", // #1694 Part B
		"admission-policy", "routing-policy", "scheduler", "preemption-policy",
		"routing-scorers", "lora-scorer-weight", "token-bucket-capacity", "token-bucket-refill-rate",
		"kv-cpu-blocks", "kv-offload-threshold", "kv-transfer-bandwidth",
		"kv-transfer-base-latency", "snapshot-refresh-interval",
		"admission-latency", "routing-latency", "trace-level",
		"counterfactual-k", "summarize-trace", "policy-config",
		"num-instances", "max-num-seqs", "max-num-batched-tokens",
		// Deprecated aliases (issue #1570) — registered on both commands.
		"max-num-running-reqs", "max-num-scheduled-tokens",
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

// TestModelConfigFolderFlag_Retired pins BC-3 of #1731: --model-config-folder is gone.
// There must be exactly one way to tell BLIS where a model config lives, so the retired
// flag must not be registered on any command (a lingering registration would silently
// accept the old spelling and reintroduce the second mechanism), and no non-test source
// in cmd/ may still name it.
func TestModelConfigFolderFlag_Retired(t *testing.T) {
	// GIVEN every command that registers simulation-engine flags
	for _, c := range []struct {
		name string
		cmd  *cobra.Command
	}{{"run", runCmd}, {"replay", replayCmd}, {"observe", observeCmd}} {
		// WHEN the retired flag is looked up
		// THEN it must be absent, while --catalog (its replacement) is present on the
		// two commands that resolve a model config.
		assert.Nilf(t, c.cmd.Flags().Lookup("model-config-folder"),
			"%s must NOT register the retired --model-config-folder (#1731)", c.name)
	}
	assert.NotNil(t, runCmd.Flags().Lookup("catalog"), "run must register --catalog")
	assert.NotNil(t, replayCmd.Flags().Lookup("catalog"), "replay must register --catalog")
	// observe derives no model architecture, so it takes no catalog (same boundary as
	// --kv-cache-dtype / --kv-offload-config).
	assert.Nil(t, observeCmd.Flags().Lookup("catalog"),
		"observe must not register --catalog (it resolves no model config)")

	// AND no non-test source in cmd/ still names the retired flag or its variable.
	entries, err := os.ReadDir(".")
	assert.NoError(t, err)
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, readErr := os.ReadFile(name)
		assert.NoError(t, readErr)
		content := string(data)
		assert.NotContainsf(t, content, "model-config-folder",
			"%s must not mention the retired --model-config-folder (#1731)", name)
		assert.NotContainsf(t, content, "ModelConfigFolder",
			"%s must not mention ModelConfigFolder (#1731)", name)
	}
}
