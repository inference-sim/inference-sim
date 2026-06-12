package cmd

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// TestMoECommBackendFlag_RegisteredOnRunAndReplay verifies the --moe-comm-backend
// flag is registered on BOTH run and replay (INV-13 parity: any config supported by
// run must be supported identically by replay). Both register via the shared
// registerSimConfigFlags, so a missing flag on either side is a parity break.
func TestMoECommBackendFlag_RegisteredOnRunAndReplay(t *testing.T) {
	for _, name := range []string{"run", "replay"} {
		c := &cobra.Command{}
		registerSimConfigFlags(c)
		f := c.Flags().Lookup("moe-comm-backend")
		if f == nil {
			t.Fatalf("%s: --moe-comm-backend must be registered (registerSimConfigFlags)", name)
		}
		if f.DefValue != "" {
			t.Errorf("%s: --moe-comm-backend default should be \"\" (defer to model factory), got %q", name, f.DefValue)
		}
	}
}

// moeCommBackendFatalSubprocess drives resolveLatencyConfig in a subprocess for the
// --moe-comm-backend rejection scenarios, mirroring the DP/EP CLI test harness.
func moeCommBackendFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	dir := t.TempDir()
	mcFolder, hwPath, err := writeMoEConfigFixture(dir)
	if err != nil {
		os.Exit(2)
	}

	scenario := os.Getenv("BLIS_MOECOMM_SCENARIO")
	var scenarioBackend, scenarioCommBackend string
	switch scenario {
	case "unknown-backend":
		scenarioBackend, scenarioCommBackend = "trained-physics", "not-a-real-backend"
	case "roofline-backend":
		scenarioBackend, scenarioCommBackend = "roofline", "deepep_low_latency"
	}

	model = "test-model"
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
	registerSimConfigFlags(testCmd) // resets bound package vars to flag defaults
	args := []string{
		"--model", "test-model", "--latency-model", scenarioBackend, "--hardware", "H100",
		"--tp", "1", "--moe-comm-backend", scenarioCommBackend,
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--total-kv-blocks", "1000", "--defaults-filepath", "../defaults.yaml",
	}
	if err := testCmd.ParseFlags(args); err != nil {
		os.Exit(2)
	}
	resolveLatencyConfig(testCmd) // must Fatalf before returning
	os.Exit(0)
}

func runMoECommFatal(t *testing.T, scenario, wantMsg string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestMoECommBackend_Rejections", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1", "BLIS_MOECOMM_SCENARIO="+scenario)
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("scenario %q: expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", scenario, err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("scenario %q: expected exit 1, got %d; output:\n%s", scenario, exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), wantMsg) {
		t.Errorf("scenario %q: fatal message must contain %q; output:\n%s", scenario, wantMsg, out)
	}
}

// TestMoECommBackend_Rejections verifies the two --moe-comm-backend validation
// fatals: an unknown backend name, and a (recognized) backend on a non-trained-physics
// model. Both must abort with logrus.Fatalf so a misconfiguration never silently no-ops.
func TestMoECommBackend_Rejections(t *testing.T) {
	moeCommBackendFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}
	runMoECommFatal(t, "unknown-backend", "not-a-real-backend")
	runMoECommFatal(t, "roofline-backend", "requires --latency-model trained-physics")
}
