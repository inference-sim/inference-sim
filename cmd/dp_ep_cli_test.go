package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// writeMoEConfigFixture writes a minimal MoE config.json (num_local_experts > 1)
// and a hardware config into a temp dir, returning the model-config folder and
// hardware-config file path. Used by the DP/EP CLI validation tests.
func writeMoEConfigFixture(dir string) (mcFolder, hwPath string, err error) {
	mcDir := filepath.Join(dir, "config")
	if err = os.MkdirAll(mcDir, 0755); err != nil {
		return "", "", err
	}
	configJSON := `{
  "architectures": ["MixtralForCausalLM"],
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_key_value_heads": 4,
  "num_local_experts": 8,
  "num_experts_per_tok": 2,
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err = os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0644); err != nil {
		return "", "", err
	}
	hwPath = filepath.Join(dir, "hw.json")
	hwJSON := `{"H100": {"MemoryGiB": 80.0, "TFlopsPeak": 1.0, "BwPeakTBs": 0.001}}`
	if err = os.WriteFile(hwPath, []byte(hwJSON), 0644); err != nil {
		return "", "", err
	}
	return mcDir, hwPath, nil
}

// dpEPResolve drives resolveLatencyConfig with the given backend/dp/ep against a
// freshly written model fixture. It is the body run inside the fatal-path
// subprocess. moe selects the MoE fixture; otherwise the dense fixture is used.
func dpEPResolve(t *testing.T, backend string, dp int, ep, moe bool) {
	dir := t.TempDir()
	var mcFolder, hwPath string
	var err error
	if moe {
		mcFolder, hwPath, err = writeMoEConfigFixture(dir)
	} else {
		mcFolder, hwPath = setupTrainedPhysicsTestFixtures(t) // dense fixture
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "fixture setup failed: %v\n", err)
		os.Exit(2)
	}

	// Reset the package-level vars the resolver reads.
	model = "test-model"
	latencyModelBackend = backend
	gpu = "H100"
	tensorParallelism = 1
	dataParallelism = dp
	enableExpertParallel = ep
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxModelLen = 0
	gpuMemoryUtilization = 0.9
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	defaultsFilePath = "../defaults.yaml"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	args := []string{
		"--model", "test-model",
		"--latency-model", backend,
		"--hardware", "H100",
		"--tp", "1",
		"--dp", fmt.Sprintf("%d", dp),
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--total-kv-blocks", "1000",
		"--defaults-filepath", "../defaults.yaml",
	}
	if ep {
		args = append(args, "--enable-expert-parallel")
	}
	if err := testCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2) // distinct from logrus.Fatalf exit code (1)
	}
	resolveLatencyConfig(testCmd) // must Fatalf before returning for the rejection cases
	os.Exit(0)                    // reached only if no fatal
}

// runFatalSubprocess re-runs the named test with BLIS_TEST_SUBPROCESS=1 and the
// scenario selector, asserting exit code 1 (logrus.Fatalf) and that the fatal
// message contains wantMsg.
func runFatalSubprocess(t *testing.T, testName, scenario, wantMsg string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run="+testName, "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1", "BLIS_DPEP_SCENARIO="+scenario)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit for scenario %q, got exit 0; output:\n%s", scenario, out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type for scenario %q: %v", scenario, err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("scenario %q: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s",
			scenario, exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), wantMsg) {
		t.Errorf("scenario %q: fatal message should contain %q, got:\n%s", scenario, wantMsg, out)
	}
}

// TestResolveLatencyConfig_DPEP_RooflineRejected verifies INV BC-ROOFLINE:
// --dp > 1 (or --enable-expert-parallel) under a non-trained-physics backend
// fatals, because roofline step time is DP-blind while KV capacity is DP-aware.
func TestResolveLatencyConfig_DPEP_RooflineRejected(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		switch os.Getenv("BLIS_DPEP_SCENARIO") {
		case "roofline-dp":
			dpEPResolve(t, "roofline", 2, false, true) // MoE so only the backend gate trips
		case "roofline-ep":
			dpEPResolve(t, "roofline", 1, true, true)
		}
		return
	}
	runFatalSubprocess(t, "TestResolveLatencyConfig_DPEP_RooflineRejected", "roofline-dp", "trained-physics")
	runFatalSubprocess(t, "TestResolveLatencyConfig_DPEP_RooflineRejected", "roofline-ep", "trained-physics")
}

// TestResolveLatencyConfig_DPEP_DenseDPRejected verifies that --dp > 1 on a
// dense model fatals (dense DP is the router-replica mechanism, not a divisor).
func TestResolveLatencyConfig_DPEP_DenseDPRejected(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		if os.Getenv("BLIS_DPEP_SCENARIO") == "dense-dp" {
			dpEPResolve(t, "trained-physics", 2, false, false) // dense fixture
		}
		return
	}
	runFatalSubprocess(t, "TestResolveLatencyConfig_DPEP_DenseDPRejected", "dense-dp", "only supported for MoE")
}

// TestResolveLatencyConfig_DPEP_DefaultsAccepted verifies the regression guard:
// --dp 1 (default), EP off must NOT fatal — runs in-process and returns.
func TestResolveLatencyConfig_DPEP_DefaultsAccepted(t *testing.T) {
	dir := t.TempDir()
	mcFolder, hwPath, err := writeMoEConfigFixture(dir)
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}

	model = "test-model"
	latencyModelBackend = "trained-physics"
	gpu = "H100"
	tensorParallelism = 1
	dataParallelism = 1
	enableExpertParallel = false
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxModelLen = 0
	gpuMemoryUtilization = 0.9
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	defaultsFilePath = "../defaults.yaml"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--total-kv-blocks", "1000", "--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	lr := resolveLatencyConfig(testCmd) // must NOT fatal
	if lr.Backend != "trained-physics" {
		t.Errorf("backend: got %q, want trained-physics", lr.Backend)
	}
}

// TestResolveLatencyConfig_DenseEP_NoFatal verifies that --enable-expert-parallel
// on a DENSE model at DP=1 is accepted (a warned no-op), NOT a fatal. The issue
// only mandates rejecting dense DP>1; dense EP is harmless (EffectiveEP stays 1)
// and rejecting it would over-constrain. Runs in-process: if it fataled, the
// process would exit and the test would fail to reach the assertion.
func TestResolveLatencyConfig_DenseEP_NoFatal(t *testing.T) {
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t) // dense fixture

	model = "test-model"
	latencyModelBackend = "trained-physics"
	gpu = "H100"
	tensorParallelism = 2
	dataParallelism = 1 // DP=1 so the dense-DP gate does not trip
	enableExpertParallel = true
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxModelLen = 0
	gpuMemoryUtilization = 0.9
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	defaultsFilePath = "../defaults.yaml"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--hardware", "H100", "--tp", "2", "--enable-expert-parallel",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--total-kv-blocks", "1000", "--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	lr := resolveLatencyConfig(testCmd) // must NOT fatal (warns instead)
	if lr.Backend != "trained-physics" {
		t.Errorf("backend: got %q, want trained-physics", lr.Backend)
	}
}

// TestRunReplay_DPEPFlags_BothRegistered verifies INV-13 parity at the flag
// surface: --dp and --enable-expert-parallel exist on both run and replay.
func TestRunReplay_DPEPFlags_BothRegistered(t *testing.T) {
	for _, name := range []string{"dp", "enable-expert-parallel"} {
		if runCmd.Flags().Lookup(name) == nil {
			t.Errorf("runCmd missing --%s flag", name)
		}
		if replayCmd.Flags().Lookup(name) == nil {
			t.Errorf("replayCmd missing --%s flag", name)
		}
	}
}

// TestRunReplay_DPEPFlags_ThreadedIntoConstructor is a source-level wiring guard.
// The DP/EP flag values are threaded directly into NewModelHardwareConfig at the
// run (root.go) and replay (replay.go) call sites — they are NOT carried in
// latencyResolution, so no behavioral test exercises that threading (the only
// tests that set --dp>1 fatal before the config is built, by design). A
// regression that passed a literal "1, false" or swapped the two args would pass
// every other test in this PR. This guard reads both sources and asserts the
// constructor is called with the flag variables, in the correct positions
// (tensorParallelism, dataParallelism, enableExpertParallel) — defending INV-13
// parity at the one place it is established (identical wiring in run and replay).
func TestRunReplay_DPEPFlags_ThreadedIntoConstructor(t *testing.T) {
	// The exact positional substring both call sites must contain.
	const wantWiring = "tensorParallelism, dataParallelism, enableExpertParallel,"
	for _, src := range []string{"root.go", "replay.go"} {
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read %s: %v", src, err)
		}
		content := string(data)
		if !strings.Contains(content, "sim.NewModelHardwareConfig(") {
			t.Fatalf("%s: expected a NewModelHardwareConfig call site", src)
		}
		if !strings.Contains(content, wantWiring) {
			t.Errorf("%s: NewModelHardwareConfig must thread the flag vars in order %q "+
				"(guards against literal/swapped DP/EP args; INV-13 parity)", src, wantWiring)
		}
	}
}
