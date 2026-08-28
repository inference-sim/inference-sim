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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
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

// TestResolveLatencyConfig_KVCacheDtype_GarbageRejected verifies R1: an
// unrecognized --kv-cache-dtype fails loudly (logrus.Fatalf → exit 1) at the CLI
// boundary rather than silently falling back to a default precision. Uses the
// shared BLIS_TEST_SUBPROCESS fatal-path harness (runFatalSubprocess), mirroring the
// sibling DP/EP flag-guard tests in dp_ep_cli_test.go.
func TestResolveLatencyConfig_KVCacheDtype_GarbageRejected(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		if os.Getenv("BLIS_DPEP_SCENARIO") == "kv-dtype-garbage" {
			kvDtypeResolve(t, "garbage-not-a-dtype")
		}
		return
	}
	runFatalSubprocess(t, "TestResolveLatencyConfig_KVCacheDtype_GarbageRejected", "kv-dtype-garbage", "is not recognized")
}

// kvDtypeResolve drives resolveLatencyConfig with the given --kv-cache-dtype against
// the dense trained-physics fixture; it is the body run inside the fatal-path
// subprocess. Mirrors dpEPResolve — everything but the dtype is valid, so the only
// fatal path exercised is the --kv-cache-dtype guard.
func kvDtypeResolve(t *testing.T, dtype string) {
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t) // dense fixture

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
	defaultsFilePath = "../defaults.yaml"
	kvCacheDtype = dtype

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--total-kv-blocks", "1000", "--defaults-filepath", "../defaults.yaml",
		"--kv-cache-dtype", dtype,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2) // distinct from logrus.Fatalf exit code (1)
	}
	resolveLatencyConfig(testCmd) // must Fatalf on the unrecognized dtype
	os.Exit(0)                    // reached only if no fatal (test failure)
}

// TestINV13_RunReplayParity_FP8KV pins INV-13 end-to-end for --kv-cache-dtype fp8 at
// the sim/cluster level: with fp8 KV precision (KVBytesPerParam=1) flowing through the
// REAL CalculateKVBlocks auto-calc — which it must, doubling the auto/bf16 block count
// — a run and a replay of the same requests under the same resolved config produce
// identical per-request TTFT/E2E. The CLI flag→ModelConfig wiring is covered by
// TestResolveLatencyConfig_KVCacheDtype_SetsModelConfigField; the capacity math by
// sim/latency TestCalculateKVBlocks_FP8_RoughlyDoublesCapacity. Mirrors the run→replay
// dance in offload_parity_test.go (assertOffloadRunReplayParity).
//
// NOTE: mutates no package-level CLI vars (drives cluster.NewClusterSimulator directly).
func TestINV13_RunReplayParity_FP8KV(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeSharedPrefixRequests(4) // shared helper from offload_parity_test.go

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

	hfConfig, err := latency.ParseHFConfig(filepath.Join(mcFolder, "config.json"))
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
	// The trained-physics fixture omits vocab_size; CalculateKVBlocks requires it.
	mc.VocabSize = 128256
	params := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)

	// Auto (compute dtype) vs fp8 KV — auto-computed block counts through the real
	// capacity function. fp8 (1 byte) must roughly double the bf16 (2 byte) count,
	// proving --kv-cache-dtype is load-bearing end-to-end (not just a stored field).
	mcAuto := *mc // KVBytesPerParam == 0 → follows compute dtype
	blocksAuto, err := latency.CalculateKVBlocks(mcAuto, hwCfg, 1, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks(auto): %v", err)
	}
	mcFP8 := *mc
	mcFP8.KVBytesPerParam = 1.0 // --kv-cache-dtype fp8
	blocksFP8, err := latency.CalculateKVBlocks(mcFP8, hwCfg, 1, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks(fp8): %v", err)
	}
	if ratio := float64(blocksFP8) / float64(blocksAuto); ratio < 1.9 || ratio > 2.1 {
		t.Fatalf("fp8 KV should ~double the auto block count: fp8=%d auto=%d (ratio %.3f)", blocksFP8, blocksAuto, ratio)
	}

	betaCfg := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alphaCfg := []float64{100.0, 1.0, 100.0}
	cfg := cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                fixedSeed,
			KVCacheConfig:       sim.NewKVCacheConfig(blocksFP8, 16, 0, 0.9, 100.0, 0),
			BatchConfig:         sim.NewBatchConfig(64, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betaCfg, alphaCfg),
			ModelHardwareConfig: sim.NewModelHardwareConfig(mcFP8, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
		},
		NumInstances:    1,
		AdmissionPolicy: "always-admit",
		RoutingPolicy:   "round-robin",
	}

	// Direct run with the fp8 config.
	cs1 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct fp8 run failed: %v", err)
	}
	runTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	runE2Es := cs1.AggregatedMetrics().RequestE2Es
	if len(runTTFTs) == 0 {
		t.Fatal("INV-13: fp8 run produced no completed requests")
	}

	// Export requests -> reload -> replay with the same fp8 config.
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

	cs2 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(replayReqs), nil)
	if err := cs2.Run(); err != nil {
		t.Fatalf("replay fp8 run failed: %v", err)
	}
	replayTTFTs := cs2.AggregatedMetrics().RequestTTFTs
	replayE2Es := cs2.AggregatedMetrics().RequestE2Es

	if len(runTTFTs) != len(replayTTFTs) {
		t.Fatalf("INV-13: TTFT map size mismatch: run=%d replay=%d", len(runTTFTs), len(replayTTFTs))
	}
	for id, ttft := range runTTFTs {
		if got, ok := replayTTFTs[id]; !ok || got != ttft {
			t.Errorf("INV-13: request %s TTFT mismatch: run=%f replay=%f ok=%v", id, ttft, got, ok)
		}
	}
	for id, e2e := range runE2Es {
		if got, ok := replayE2Es[id]; !ok || got != e2e {
			t.Errorf("INV-13: request %s E2E mismatch: run=%f replay=%f ok=%v", id, e2e, got, ok)
		}
	}
}
