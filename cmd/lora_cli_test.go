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
	"pgregory.net/rapid"
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

// TestAdoptKernelDeployment_AdapterHBMReservationShrinksThePool is the cmd-level end-to-end
// law for the static LoRA HBM reservation (PR5): a run adopts the kernel's KV pool, then sets
// the configured adapter reservation aside. The pool is the kernel's plain answer with no
// reservation, and is non-increasing in the reserved bytes -- never larger than without one.
func TestAdoptKernelDeployment_AdapterHBMReservationShrinksThePool(t *testing.T) {
	catalogDir := setupKernelTestFixtures(t)
	orig := captureCmdLevelVars()
	origReserved := loraReservedBytesForKV
	t.Cleanup(func() { orig.restore(); loraReservedBytesForKV = origReserved })
	catalogPath = catalogDir

	adoptKernelDeployment(&cobra.Command{})
	adopted := totalKVBlocks
	poolWith := func(reserved int64) int64 {
		totalKVBlocks, loraReservedBytesForKV = adopted, reserved
		applyKernelLoRAReservation()
		return totalKVBlocks
	}
	plain := poolWith(0)
	if plain != adopted {
		t.Errorf("no reservation moved the adopted pool from %d to %d blocks", adopted, plain)
	}
	if plain <= 0 {
		t.Fatalf("non-vacuity: the kernel sized a %d-block pool", plain)
	}
	if got := poolWith(8 << 30); got >= plain {
		t.Errorf("an 8 GiB reservation left the pool at %d blocks, not below the plain %d", got, plain)
	}
	rapid.Check(t, func(rt *rapid.T) {
		a := rapid.Int64Range(0, 8<<30).Draw(rt, "reserved")
		b := rapid.Int64Range(a, 8<<30).Draw(rt, "more")
		if pa, pb := poolWith(a), poolWith(b); pb > pa {
			rt.Fatalf("reserving %d bytes left %d blocks, more than %d bytes' %d", b, pb, a, pa)
		}
	})
}
