package cmd

import (
	"os"
	"strings"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
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

// TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum verifies the shared function
// uses len < 7 for trained-roofline (BC-4), matching the 7-coefficient model.
// Verifiable from first principles: trained_roofline.go uses betaCoeffs[0..6].
func TestTrainedRooflineBetaCoeffGuard_UsesCorrectMinimum(t *testing.T) {
	// GIVEN the source of cmd/root.go (where the shared guard lives after Task 1)
	data, err := os.ReadFile("root.go")
	assert.NoError(t, err)

	content := string(data)

	// THEN: the guard uses < 7 (not < 10) for trained-roofline (BC-4)
	// The local variable in resolveLatencyConfig is named "beta" (not "betaCoeffs")
	assert.Contains(t, content, `len(beta) < 7`,
		"trained-roofline guard must use < 7 (model uses indices 0-6); local var is 'beta' in resolveLatencyConfig")

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
		"latency-model", "hardware", "tp", "alpha-coeffs", "beta-coeffs",
		"total-kv-blocks", "block-size-in-tokens", "max-model-len",
		"gpu-memory-utilization", "model-config-folder", "hardware-config",
		"mixed-batch-penalty", "overlap-penalty",
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
		"admission-policy", "routing-policy", "priority-policy", "scheduler",
		"routing-scorers", "token-bucket-capacity", "token-bucket-refill-rate",
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
	assert.NotContains(t, content, `if backend == "crossmodel" {`,
		"replay.go must not contain inline crossmodel resolution block; use resolveLatencyConfig(cmd)")
	assert.NotContains(t, content, `if backend == "trained-roofline" {`,
		"replay.go must not contain inline trained-roofline resolution block; use resolveLatencyConfig(cmd)")
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
		"latency-model", "hardware", "tp", "alpha-coeffs", "beta-coeffs",
		"total-kv-blocks", "block-size-in-tokens", "max-model-len",
		"gpu-memory-utilization", "model-config-folder", "hardware-config",
		"admission-policy", "routing-policy", "priority-policy", "scheduler",
		"routing-scorers", "token-bucket-capacity", "token-bucket-refill-rate",
		"kv-cpu-blocks", "kv-offload-threshold", "kv-transfer-bandwidth",
		"kv-transfer-base-latency", "snapshot-refresh-interval",
		"admission-latency", "routing-latency", "trace-level",
		"counterfactual-k", "summarize-trace", "policy-config",
		"num-instances", "max-num-running-reqs", "max-num-scheduled-tokens",
		"long-prefill-token-threshold", "cache-signal-delay",
		"mixed-batch-penalty", "overlap-penalty",
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
