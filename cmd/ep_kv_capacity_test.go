package cmd

import (
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// --- EP-aware KV capacity wiring (#1656) ---

// TestEPSizeForKVCapacity_ReadsLogicalTopology pins the helper every CLI auto-calc site
// funnels through: it must report the LOGICAL, user-requested EP group (TP·DP when
// --enable-expert-parallel is on for a MoE model), and 1 whenever expert parallelism
// cannot apply. A per-pool TP is honoured, so a prefill pool at a different TP gets its
// own group size.
func TestEPSizeForKVCapacity_ReadsLogicalTopology(t *testing.T) {
	origDP, origEP := dataParallelism, enableExpertParallel
	t.Cleanup(func() { dataParallelism, enableExpertParallel = origDP, origEP })

	tests := []struct {
		name  string
		isMoE bool
		tp    int
		dp    int
		ep    bool
		want  int
	}{
		{"ep_off", true, 8, 2, false, 1},
		{"dense_ep_on", false, 8, 2, true, 1},
		{"moe_ep_on_dp1", true, 8, 1, true, 8},
		{"moe_ep_on_dp2", true, 8, 2, true, 16},
		{"moe_ep_on_pool_tp", true, 4, 2, true, 8}, // a per-pool TP yields a per-pool group
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dataParallelism, enableExpertParallel = tc.dp, tc.ep
			if got := epSizeForKVCapacity(tc.isMoE, tc.tp); got != tc.want {
				t.Errorf("epSizeForKVCapacity(isMoE=%v, tp=%d) with dp=%d ep=%v = %d, want %d",
					tc.isMoE, tc.tp, tc.dp, tc.ep, got, tc.want)
			}
		})
	}
}

// TestEveryKVCapacityCallSiteIsEPAware is the BC-7 / R23 parity guard: a MoE+EP
// deployment must not be sized correctly by one code path and over-counted by another.
// It walks the whole repository (excluding tests and the function's own definition file)
// and requires every production CalculateKVBlocks call to pass an expert-parallel group
// size DERIVED from the topology — one of the two legitimate sources, the cmd helper
// epSizeForKVCapacity or the config accessor EffectiveEP. Requiring a derived source (not
// merely the option's presence) is what makes the guard bite: a site passing a literal
// WithExpertParallelSize(1) would be EP-blind, which is precisely the #1656 over-count it
// exists to prevent. A NEW auto-calc site added later fails this test instead.
//
// It is a shape check, deliberately cheap; the behavioral companion is
// TestResolveLatencyConfig_EPRaisesAutoKVCapacity, which drives the shared resolver and
// asserts the capacity actually changes.
func TestEveryKVCapacityCallSiteIsEPAware(t *testing.T) {
	const (
		callMarker   = "CalculateKVBlocks("
		optionMarker = "WithExpertParallelSize("
		// The definition and doc-comment references live in the implementing file.
		definitionFile = "kv_capacity.go"
	)
	// The only legitimate sources of the group size: the cmd helper (logical topology) or
	// the config-bound accessor (per-instance topology).
	derivedSources := []string{"epSizeForKVCapacity(", "EffectiveEP()"}

	found := 0
	err := filepath.WalkDir("..", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".worktrees", "site", "docs":
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		if filepath.Base(path) == definitionFile {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		content := string(data)
		for idx := 0; ; {
			hit := strings.Index(content[idx:], callMarker)
			if hit < 0 {
				break
			}
			start := idx + hit
			idx = start + len(callMarker)
			// A call's options sit on the immediately following lines; 600 bytes covers the
			// longest of them without reaching a neighbouring call site. The window also
			// reaches back 400 bytes, because a site may hoist the resolved group size into
			// a local just above the call (so the logged value is provably the one charged).
			end := start + 600
			if end > len(content) {
				end = len(content)
			}
			from := start - 400
			if from < 0 {
				from = 0
			}
			found++
			window := content[from:end]
			if !strings.Contains(window, optionMarker) {
				t.Errorf("%s: CalculateKVBlocks call at byte %d does not pass %s — a MoE+EP "+
					"deployment sized through this path would over-count routed-expert weights (#1656)",
					path, start, optionMarker)
				continue
			}
			derived := false
			for _, src := range derivedSources {
				if strings.Contains(window, src) {
					derived = true
					break
				}
			}
			if !derived {
				t.Errorf("%s: CalculateKVBlocks call at byte %d passes %s but not a topology-derived "+
					"group size (one of %v) — a literal value would be EP-blind (#1656)",
					path, start, optionMarker, derivedSources)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if found == 0 {
		t.Fatal("found no production CalculateKVBlocks call sites; the guard would be vacuous")
	}
}

// writeBigMoEFixture writes a MoE config.json whose 64 routed experts (~672 GiB at
// float16) do NOT fit on 8×80 GiB when charged to each DP rank's TP group, but DO fit
// when sharded across a 16-GPU expert-parallel group — the #1656 condition, sized so the
// two answers differ.
func writeBigMoEFixture(t *testing.T) (mcDir, hwPath string) {
	t.Helper()
	dir := t.TempDir()
	mcDir = filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	configJSON := `{
  "architectures": ["MixtralForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_key_value_heads": 8,
  "num_local_experts": 64,
  "num_experts_per_tok": 2,
  "vocab_size": 32000,
  "hidden_act": "silu",
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	hwPath = filepath.Join(dir, "hw.json")
	if err := os.WriteFile(hwPath, []byte(`{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`), 0644); err != nil {
		t.Fatalf("write hw: %v", err)
	}
	return mcDir, hwPath
}

// TestRunCmd_EPMoE_CapacityErrorNoLongerMasksPlacementGuard is BC-9, the user-visible
// effect of #1656 today. A large MoE at --tp 8 --dp 2 --enable-expert-parallel is a real
// EP deployment that fits on its 16 GPUs, so KV auto-calculation must now succeed and the
// run must fail on the honest reason — EP-on DP placement is not modelled yet (#1548) —
// rather than on a weight over-count that told the operator to buy more GPUs.
func TestRunCmd_EPMoE_CapacityErrorNoLongerMasksPlacementGuard(t *testing.T) {
	if os.Getenv("BLIS_RUN_EP_KV") == "1" {
		mcDir, hwPath := writeBigMoEFixture(t)
		rootCmd.SetArgs([]string{
			"run",
			"--model", "test-moe",
			"--model-config-folder", mcDir,
			"--hardware", "H100",
			"--hardware-config", hwPath,
			"--latency-model", "trained-physics",
			"--tp", "8",
			"--dp", "2",
			"--enable-expert-parallel",
			"--num-instances", "1",
			"--rate", "1",
			"--num-requests", "4",
			"--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
		})
		_ = rootCmd.Execute()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_EPMoE_CapacityErrorNoLongerMasksPlacementGuard$")
	cmd.Env = append(os.Environ(), "BLIS_RUN_EP_KV=1")
	out, err := cmd.CombinedOutput()

	// The run still ends at the #1548 EP-placement guard (exit 1) — that is the honest
	// unsupported-feature signal, and it is what must be visible.
	if err == nil {
		t.Fatalf("expected non-zero exit (the #1548 EP-placement guard), got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %v; output:\n%s", err, out)
	}
	if strings.Contains(string(out), "KV capacity auto-calculation failed") {
		t.Errorf("BC-9: KV auto-calculation must succeed for an EP-sharded MoE — the weight "+
			"over-count must no longer mask the placement guard; output:\n%s", out)
	}
	if !strings.Contains(string(out), "1548") {
		t.Errorf("BC-9: expected the #1548 EP-placement guard to be the failure reason; output:\n%s", out)
	}
}

// TestResolveLatencyConfig_EPRaisesAutoKVCapacity is the behavioral companion to the
// source-level guard above: it drives the SHARED auto-calc resolver that both `blis run`
// and `blis replay` call (resolveLatencyConfig — INV-13 parity by construction) and shows
// the wiring actually changes the answer. With expert parallelism on, an MoE model's
// routed experts are charged to the whole TP·DP group, so strictly more memory is left for
// KV. This survives a refactor of how the option reaches CalculateKVBlocks.
func TestResolveLatencyConfig_EPRaisesAutoKVCapacity(t *testing.T) {
	mcDir, hwPath := writeCompleteMoEFixture(t)

	origModel, origBackend, origGPU := model, latencyModelBackend, gpu
	origTP, origDP, origEP, origComm := tensorParallelism, dataParallelism, enableExpertParallel, moeCommBackend
	origBlocks, origBlockSize, origMML := totalKVBlocks, blockSizeTokens, maxModelLen
	origUtil, origMCFolder, origHW, origDefaults := gpuMemoryUtilization, modelConfigFolder, hwConfigPath, defaultsFilePath
	t.Cleanup(func() {
		model, latencyModelBackend, gpu = origModel, origBackend, origGPU
		tensorParallelism, dataParallelism, enableExpertParallel, moeCommBackend = origTP, origDP, origEP, origComm
		totalKVBlocks, blockSizeTokens, maxModelLen = origBlocks, origBlockSize, origMML
		gpuMemoryUtilization, modelConfigFolder, hwConfigPath, defaultsFilePath = origUtil, origMCFolder, origHW, origDefaults
	})

	resolveAutoKV := func(epOn bool) int64 {
		args := []string{
			"--model", "test-model", "--latency-model", "trained-physics",
			"--hardware", "H100", "--tp", "2", "--dp", "2",
			"--model-config-folder", mcDir, "--hardware-config", hwPath,
			"--defaults-filepath", "../defaults.yaml",
		}
		if epOn {
			args = append(args, "--enable-expert-parallel")
		}
		model, latencyModelBackend, gpu = "test-model", "trained-physics", "H100"
		tensorParallelism, dataParallelism, enableExpertParallel, moeCommBackend = 2, 2, epOn, ""
		totalKVBlocks, blockSizeTokens, maxModelLen = 0, 16, 0
		gpuMemoryUtilization, modelConfigFolder, hwConfigPath = 0.9, mcDir, hwPath
		defaultsFilePath = "../defaults.yaml"

		testCmd := &cobra.Command{}
		registerSimConfigFlags(testCmd)
		if err := testCmd.ParseFlags(args); err != nil {
			t.Fatalf("ParseFlags(epOn=%v): %v", epOn, err)
		}
		resolveLatencyConfig(testCmd)
		return totalKVBlocks
	}

	epOff := resolveAutoKV(false)
	epOn := resolveAutoKV(true)
	if epOff <= 0 || epOn <= 0 {
		t.Fatalf("both topologies must size: epOff=%d epOn=%d", epOff, epOn)
	}
	if epOn <= epOff {
		t.Errorf("expert parallelism must free routed-expert memory for KV: EP on gave %d blocks, "+
			"EP off gave %d (the option is not reaching the auto-calc)", epOn, epOff)
	}
}
