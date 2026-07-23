package cmd

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/spf13/cobra"
)

// TestLoRAFlags_RegisteredOnRunAndReplay verifies the --lora-* config flags are
// registered on BOTH run and replay via the shared registerSimConfigFlags (INV-13
// parity: any config supported by run must be supported identically by replay).
func TestLoRAFlags_RegisteredOnRunAndReplay(t *testing.T) {
	flags := []string{
		"lora-config",
		"lora-adapter-capacity",
		"lora-load-base-latency-us",
		"lora-load-bandwidth-bytes-us",
		"lora-footprint-bytes-per-rank",
		"eviction-policy",
	}
	for _, name := range []string{"run", "replay"} {
		c := &cobra.Command{}
		registerSimConfigFlags(c)
		for _, fl := range flags {
			if c.Flags().Lookup(fl) == nil {
				t.Errorf("%s: --%s must be registered (registerSimConfigFlags parity)", name, fl)
			}
		}
	}
}

// TestResolveLoRAConfig_UnsetFlagsUseDefaults verifies R18: when the --lora-* scalar
// flags are NOT set, the resolved LoRAConfig takes its coefficient values from
// defaults.yaml (the flag's zero default does NOT clobber the file default).
func TestResolveLoRAConfig_UnsetFlagsUseDefaults(t *testing.T) {
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml"}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = ""

	cfg := resolveLoRAConfig(c)

	// defaults.yaml declares load_base_latency_us: 1500, load_bandwidth_bytes_us: 2e6,
	// footprint_bytes_per_rank: 2e6. Unset flags must NOT override these.
	if cfg.LoadBaseLatencyUs == nil || *cfg.LoadBaseLatencyUs != 1500.0 {
		t.Errorf("load_base_latency_us: want defaults.yaml value 1500, got %v", cfg.LoadBaseLatencyUs)
	}
	if cfg.LoadBandwidthBytesUs == nil || *cfg.LoadBandwidthBytesUs != 2.0e6 {
		t.Errorf("load_bandwidth_bytes_us: want defaults.yaml value 2e6, got %v", cfg.LoadBandwidthBytesUs)
	}
	// No adapters declared => inert (HasAdapters false).
	if cfg.HasAdapters() {
		t.Errorf("no --lora-config => no adapters => inert, got %d adapters", len(cfg.Adapters))
	}
}

// TestResolveLoRAConfig_FlagOverridesDefaults verifies that a set --lora-* flag wins
// over the defaults.yaml value (flags compose with, and override, file defaults).
func TestResolveLoRAConfig_FlagOverridesDefaults(t *testing.T) {
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{
		"--defaults-filepath", "../defaults.yaml",
		"--lora-load-base-latency-us", "9999",
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = ""

	cfg := resolveLoRAConfig(c)
	if cfg.LoadBaseLatencyUs == nil || *cfg.LoadBaseLatencyUs != 9999 {
		t.Errorf("set flag must override defaults.yaml, want 9999, got %v", cfg.LoadBaseLatencyUs)
	}
}

// TestResolveLoRAConfig_ConfigFileAdapters verifies a --lora-config file's adapter
// registry is loaded onto the resolved LoRAConfig.
func TestResolveLoRAConfig_ConfigFileAdapters(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "lora.yaml")
	content := "lora:\n  adapter_capacity: 4\n  adapters:\n    - id: adapter_0\n      rank: 8\n    - id: adapter_1\n      rank: 16\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml", "--lora-config", path}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = path

	cfg := resolveLoRAConfig(c)
	if !cfg.HasAdapters() || len(cfg.Adapters) != 2 {
		t.Fatalf("want 2 adapters from --lora-config, got %d", len(cfg.Adapters))
	}
	if cfg.AdapterCapacity == nil || *cfg.AdapterCapacity != 4 {
		t.Errorf("want adapter_capacity 4 from file, got %v", cfg.AdapterCapacity)
	}
	loraConfigPath = "" // reset shared state
}

// TestResolveLoRAConfig_EvictionPolicy pins the --eviction-policy resolution and
// precedence (B-4): an unset flag leaves EvictionPolicy "" (empty => lru, the no-op
// default, byte-identical to B-3); a --lora-config value survives an unset flag; and
// a set flag overrides the file value (flags win over file, matching the other
// --lora-* scalars, R18).
func TestResolveLoRAConfig_EvictionPolicy(t *testing.T) {
	defaultsFilePath = "../defaults.yaml"

	// (1) Unset flag, no file => "" (lru default).
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml"}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	loraConfigPath = ""
	if got := resolveLoRAConfig(c).EvictionPolicy; got != "" {
		t.Errorf("unset flag: EvictionPolicy = %q, want \"\" (empty => lru default)", got)
	}

	// (2) Set flag => flag value.
	c = &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml", "--eviction-policy", "rank-aware"}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	loraConfigPath = ""
	if got := resolveLoRAConfig(c).EvictionPolicy; got != "rank-aware" {
		t.Errorf("set flag: EvictionPolicy = %q, want \"rank-aware\"", got)
	}

	// (3) File value survives an unset flag.
	dir := t.TempDir()
	path := filepath.Join(dir, "lora.yaml")
	content := "lora:\n  adapter_capacity: 4\n  eviction_policy: rank-aware\n  adapters:\n    - id: adapter_0\n      rank: 8\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	c = &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml", "--lora-config", path}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	loraConfigPath = path
	if got := resolveLoRAConfig(c).EvictionPolicy; got != "rank-aware" {
		t.Errorf("file value, unset flag: EvictionPolicy = %q, want \"rank-aware\" (file survives)", got)
	}

	// (4) Set flag overrides the file value.
	c = &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml", "--lora-config", path, "--eviction-policy", "lru"}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	loraConfigPath = path
	if got := resolveLoRAConfig(c).EvictionPolicy; got != "lru" {
		t.Errorf("flag over file: EvictionPolicy = %q, want \"lru\" (flag overrides file)", got)
	}
	loraConfigPath = "" // reset shared state
}

// evictionFatalSubprocess drives resolveLoRAConfig with an unknown --eviction-policy
// in a subprocess; the CLI boundary must Fatalf (Principle V) rather than defer the
// error to NewSimulator.
func evictionFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	args := []string{"--defaults-filepath", "../defaults.yaml", "--eviction-policy", "bogus"}
	if err := c.ParseFlags(args); err != nil {
		os.Exit(2)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = ""
	resolveLoRAConfig(c) // must Fatalf before returning
	os.Exit(0)
}

// TestResolveLoRAConfig_UnknownEvictionPolicyRejected verifies an unknown
// --eviction-policy name aborts at the CLI boundary (exit 1) with a message listing
// the valid names — never a silent no-op or a deferred NewSimulator error.
func TestResolveLoRAConfig_UnknownEvictionPolicyRejected(t *testing.T) {
	evictionFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestResolveLoRAConfig_UnknownEvictionPolicyRejected", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit 1, got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "eviction-policy") {
		t.Errorf("fatal message must mention eviction-policy; output:\n%s", out)
	}
	if !strings.Contains(string(out), "rank-aware") {
		t.Errorf("fatal message must list valid names (rank-aware); output:\n%s", out)
	}
}

// loraFatalSubprocess drives resolveLoRAConfig in a subprocess for the
// zero-capacity-with-adapters rejection (contracts/cli-flags.md edge case).
func loraFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "lora.yaml")
	// adapters present + adapter_capacity 0 => unservable => must Fatalf.
	content := "lora:\n  adapter_capacity: 0\n  adapters:\n    - id: adapter_0\n      rank: 8\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		os.Exit(2)
	}

	c := &cobra.Command{}
	registerSimConfigFlags(c)
	args := []string{"--defaults-filepath", "../defaults.yaml", "--lora-config", path}
	if err := c.ParseFlags(args); err != nil {
		os.Exit(2)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = path
	resolveLoRAConfig(c) // must Fatalf before returning
	os.Exit(0)
}

// TestLoRAConfig_ZeroCapacityWithAdaptersRejected verifies the edge case:
// --lora-config declaring adapters with adapter_capacity 0 aborts with logrus.Fatalf
// so a misconfiguration never silently no-ops.
func TestLoRAConfig_ZeroCapacityWithAdaptersRejected(t *testing.T) {
	loraFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestLoRAConfig_ZeroCapacityWithAdaptersRejected", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit 1, got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "adapter_capacity") {
		t.Errorf("fatal message must mention adapter_capacity; output:\n%s", out)
	}
}

// TestAdapterReservedBytesFor covers the config→bytes resolution seam that production
// actually uses (both runCmd and replayCmd call it to set loraReservedBytesForKV). The
// resolveLatencyConfig tests set that package var directly, so without this a
// regression that made adapterReservedBytesFor return 0 — silently disabling the
// reservation for every real run/replay — would pass the suite green.
func TestAdapterReservedBytesFor(t *testing.T) {
	// Inert config (no adapters / no capacity): BuildAdapterCost returns nil ⇒ 0,
	// leaving KV byte-identical (INV-6).
	if got := adapterReservedBytesFor(sim.LoRAConfig{}); got != 0 {
		t.Errorf("inert config: adapterReservedBytesFor = %d, want 0", got)
	}

	// Populated config: equals the cost model's capacity × maxRank × footprint.
	fp := func(v float64) *float64 { return &v }
	capacity := 4
	cfg := sim.LoRAConfig{
		AdapterCapacity:       &capacity,
		LoadBaseLatencyUs:     fp(1000.0),
		LoadBandwidthBytesUs:  fp(2.0e6),
		FootprintBytesPerRank: fp(2.0e6),
		StepOverheadTiers:     map[int]sim.StepOverheadTier{8: {K6: fp(0.02), K7: fp(1.0)}},
		Adapters: []sim.AdapterSpec{
			{ID: "a8", Rank: 8},
			{ID: "a16", Rank: 16},
			{ID: "a32", Rank: 32}, // max rank sizes the per-slot footprint
		},
	}
	if got, want := adapterReservedBytesFor(cfg), int64(4*32*2.0e6); got != want {
		t.Errorf("adapterReservedBytesFor = %d, want %d (capacity × maxRank × footprint_per_rank)", got, want)
	}
}

// TestResolveLatencyConfig_AppliesAdapterHBMReservation is the cmd-level end-to-end
// check that the static LoRA HBM reservation threads into the auto-derived KV
// capacity (PR5). resolveLatencyConfig is the shared auto-calc chokepoint for BOTH
// run and replay, so exercising it pins the main-path INV-13 parity too: the same
// reservation reduces the same block count regardless of command. It resolves an
// identical fixture WITHOUT --total-kv-blocks (so the auto-capacity path runs) at
// reservation 0 vs a non-zero reservation and asserts the block count shrinks.
func TestResolveLatencyConfig_AppliesAdapterHBMReservation(t *testing.T) {
	dir := t.TempDir()
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// Dense Llama-like fixture: the auto-capacity path needs vocab_size + realistic
	// dims so the derived block count is comfortably positive on an 80 GiB GPU.
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "hidden_act": "silu",
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	hwPath := filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		t.Fatalf("write hw: %v", err)
	}

	// Full isolation: this test mutates several cmd-level package vars, so save and
	// restore ALL of them to avoid leaking state into other cmd tests.
	// captureCmdLevelVars covers most (model, backend, gpu, tp, totalKVBlocks,
	// modelConfigFolder, hwConfigPath, defaultsFilePath, blockSizeTokens,
	// maxModelLen, ...); save the few it does not, including the new
	// loraReservedBytesForKV.
	orig := captureCmdLevelVars()
	origDP, origEP, origMoE := dataParallelism, enableExpertParallel, moeCommBackend
	origUtil, origReserved := gpuMemoryUtilization, loraReservedBytesForKV
	defer func() {
		orig.restore()
		dataParallelism, enableExpertParallel, moeCommBackend = origDP, origEP, origMoE
		gpuMemoryUtilization, loraReservedBytesForKV = origUtil, origReserved
	}()

	resolve := func(reserved int64) int64 {
		// Reset the package-level vars resolveLatencyConfig reads.
		model = "test-model"
		latencyModelBackend = "trained-physics"
		gpu = "H100"
		tensorParallelism = 1
		dataParallelism = 1
		enableExpertParallel = false
		moeCommBackend = ""
		totalKVBlocks = 0 // auto-derive
		blockSizeTokens = 16
		maxModelLen = 0
		gpuMemoryUtilization = 0.9
		modelConfigFolder = mcDir
		hwConfigPath = hwPath
		defaultsFilePath = "../defaults.yaml"
		loraReservedBytesForKV = reserved

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		// No --total-kv-blocks, so Changed("total-kv-blocks") is false and the
		// auto-capacity path (which passes WithAdapterReservedBytes) runs.
		args := []string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "1",
			"--model-config-folder", mcDir, "--hardware-config", hwPath,
			"--defaults-filepath", "../defaults.yaml",
		}
		if err := testCmd.ParseFlags(args); err != nil {
			t.Fatalf("ParseFlags: %v", err)
		}
		resolveLatencyConfig(testCmd)
		return totalKVBlocks
	}

	base := resolve(0)
	withRes := resolve(8 << 30)   // 8 GiB reservation
	withRes2 := resolve(16 << 30) // 16 GiB reservation (double)
	if base <= 0 {
		t.Fatalf("baseline auto-derived blocks must be positive, got %d", base)
	}
	if !(withRes < base) {
		t.Errorf("adapter HBM reservation not applied via resolveLatencyConfig: base=%d withReservation=%d (expected fewer)", base, withRes)
	}
	// Magnitude law, not just direction: block loss must be LINEAR in the reserved
	// bytes (the reservation is subtracted as GiB from a fixed overhead, then blocks =
	// floor(remaining / per_block)), so doubling the reservation must remove ~twice
	// the blocks. This ratio catches a wiring bug that threads the value through with a
	// constant/fixed offset or otherwise non-linearly — it would still shrink the count
	// and pass a direction-only check, but breaks the 2× ratio. It intentionally does
	// NOT probe dp/tp scaling: at dp=tp=1 a dp/tp-scale error multiplies by 1 and a
	// uniform coefficient error preserves the ratio, so both are invisible here; the
	// per-DP-rank scaling law is pinned separately by the library test
	// TestCalculateKVBlocks_AdapterReservationPerDPRankScaling. Slack absorbs floor()
	// truncation at each of the two subtractions.
	lost1, lost2 := base-withRes, base-withRes2
	if lost1 <= 0 {
		t.Fatalf("8 GiB reservation removed no blocks: base=%d withReservation=%d", base, withRes)
	}
	if !(lost2 > lost1) {
		t.Errorf("16 GiB reservation must remove more blocks than 8 GiB: lost(8GiB)=%d lost(16GiB)=%d", lost1, lost2)
	}
	if diff := lost2 - 2*lost1; diff < -2 || diff > 2 {
		t.Errorf("block loss must scale linearly with the reservation: lost(8GiB)=%d, lost(16GiB)=%d, want lost(16GiB) ≈ 2×lost(8GiB) (±2 blocks truncation slack)", lost1, lost2)
	}
}

// TestResolveLatencyConfig_ExplicitTotalKVBlocksBypassesReservation pins the scope
// boundary documented in the --lora-config flag help: when --total-kv-blocks is set
// explicitly, the auto-calc path does NOT run, so the static LoRA HBM reservation is
// NOT subtracted — the explicit value is used as-is. Guards against a future refactor
// that accidentally applies the reservation to an explicit block count.
func TestResolveLatencyConfig_ExplicitTotalKVBlocksBypassesReservation(t *testing.T) {
	dir := t.TempDir()
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "hidden_act": "silu",
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	hwPath := filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		t.Fatalf("write hw: %v", err)
	}

	orig := captureCmdLevelVars()
	origDP, origEP, origMoE := dataParallelism, enableExpertParallel, moeCommBackend
	origUtil, origReserved := gpuMemoryUtilization, loraReservedBytesForKV
	defer func() {
		orig.restore()
		dataParallelism, enableExpertParallel, moeCommBackend = origDP, origEP, origMoE
		gpuMemoryUtilization, loraReservedBytesForKV = origUtil, origReserved
	}()

	model = "test-model"
	latencyModelBackend = "trained-physics"
	gpu = "H100"
	tensorParallelism = 1
	dataParallelism = 1
	enableExpertParallel = false
	moeCommBackend = ""
	blockSizeTokens = 16
	maxModelLen = 0
	gpuMemoryUtilization = 0.9
	modelConfigFolder = mcDir
	hwConfigPath = hwPath
	defaultsFilePath = "../defaults.yaml"
	loraReservedBytesForKV = 8 << 30 // a reservation IS configured...

	const explicit = int64(5000)
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	// ...but --total-kv-blocks is set explicitly, so Changed("total-kv-blocks") is
	// true and the auto-calc path (which would subtract the reservation) is skipped.
	args := []string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcDir, "--hardware-config", hwPath,
		"--defaults-filepath", "../defaults.yaml",
		"--total-kv-blocks", "5000",
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	resolveLatencyConfig(testCmd)

	if totalKVBlocks != explicit {
		t.Errorf("explicit --total-kv-blocks must be used as-is (reservation NOT applied), got %d, want %d", totalKVBlocks, explicit)
	}
}
