package cmd

// CLI-surface tests for --kv-cache-dtype (issue #1565): an independent KV-cache
// storage precision (vLLM CacheConfig.cache_dtype parity), decoupled from the
// compute/activation dtype and from weight quantization.
//
// The KV-capacity math itself (fp8 halves per-token bytes → ~2x KV blocks; auto is
// byte-identical to the compute dtype) is proven directly on KVBytesPerToken /
// CalculateKVBlocks in sim/latency/kv_capacity_test.go. These tests pin the CLI
// wiring: the flag exists on both run and replay (INV-13 flag-surface parity), and it
// flows through the shared resolveLatencyConfig into ModelConfig.KVBytesPerParam — the
// exact resolution both `blis run` and `blis replay` share, so re-supplying it
// identically on replay is byte-identical (INV-13), and "auto" is a no-op (INV-6).

import (
	"testing"

	"github.com/spf13/cobra"
)

// TestKVCacheDtypeFlag_RegisteredOnRunAndReplay pins INV-13 at the flag surface:
// --kv-cache-dtype must exist on BOTH run and replay so a trace can be replayed with
// identical flags. (It is intentionally NOT on observe — observe is a black-box
// dispatcher that derives no KV capacity, matching --kv-offload-config's treatment.)
func TestKVCacheDtypeFlag_RegisteredOnRunAndReplay(t *testing.T) {
	if runCmd.Flags().Lookup("kv-cache-dtype") == nil {
		t.Error("runCmd missing --kv-cache-dtype flag")
	}
	if replayCmd.Flags().Lookup("kv-cache-dtype") == nil {
		t.Error("replayCmd missing --kv-cache-dtype flag")
	}
	// Default must be "auto" (the INV-6 no-op).
	if f := runCmd.Flags().Lookup("kv-cache-dtype"); f != nil && f.DefValue != "auto" {
		t.Errorf("--kv-cache-dtype default = %q, want \"auto\"", f.DefValue)
	}
}

// TestResolveLatencyConfig_KVCacheDtype_SetsModelConfigField verifies the flag reaches
// ModelConfig.KVBytesPerParam through the real, shared resolveLatencyConfig path (the
// same path run and replay use). --total-kv-blocks is passed explicitly so the auto-calc
// (and thus fixture completeness) is irrelevant — this isolates the flag→field wiring.
func TestResolveLatencyConfig_KVCacheDtype_SetsModelConfigField(t *testing.T) {
	orig := captureCmdLevelVars()
	origKV := kvCacheDtype
	defer func() {
		orig.restore()
		kvCacheDtype = origKV
	}()

	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	resolve := func(dtype string) float64 {
		model = "test-model"
		latencyModelBackend = "trained-physics"
		gpu = "H100"
		tensorParallelism = 1
		dataParallelism = 1
		totalKVBlocks = 1000
		blockSizeTokens = 16
		maxModelLen = 0
		gpuMemoryUtilization = 0.9
		modelConfigFolder = mcFolder
		hwConfigPath = hwPath
		defaultsFilePath = defaultsPath

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		if err := testCmd.ParseFlags([]string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcFolder, "--hardware-config", hwPath,
			"--total-kv-blocks", "1000", "--defaults-filepath", defaultsPath,
			"--kv-cache-dtype", dtype,
		}); err != nil {
			t.Fatalf("ParseFlags: %v", err)
		}
		return resolveLatencyConfig(testCmd).ModelConfig.KVBytesPerParam
	}

	// "auto" leaves KVBytesPerParam unset (falls back to compute dtype) — INV-6 no-op.
	if got := resolve("auto"); got != 0 {
		t.Errorf("--kv-cache-dtype auto → KVBytesPerParam = %v, want 0", got)
	}
	// fp8 → 1 byte/element (the ~2x KV capacity case).
	if got := resolve("fp8"); got != 1.0 {
		t.Errorf("--kv-cache-dtype fp8 → KVBytesPerParam = %v, want 1.0", got)
	}
	// An explicit bf16 KV dtype pins KV to 2 bytes/element regardless of compute dtype.
	if got := resolve("bf16"); got != 2.0 {
		t.Errorf("--kv-cache-dtype bf16 → KVBytesPerParam = %v, want 2.0", got)
	}
}
