package cmd

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"testing"
)

// baselineNoopGolden is the pre-feature no-op stdout golden captured in T002
// (./blis run --model qwen/qwen3-14b --seed 42). Path is relative to the cmd/ test cwd.
const baselineNoopGolden = "../specs/007-lora-control-plane/testdata/baseline_noop.json"

// TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline is the load-bearing INV-6 / SC-001
// regression: an adapter-blind run (no --lora-config, no --lora-* flags) MUST produce
// stdout byte-identical to the pre-feature baseline. This proves the LoRA subsystem is
// inert when unconfigured.
//
// The run is driven in a re-exec subprocess so the real cobra command tree executes
// (and os.Exit(0) suppresses the test framework's own stdout, leaving only the metrics
// JSON for a clean byte comparison). The qwen3-14b model config and hardware config are
// git-tracked under model_configs/ and hardware_config.json, so the run is offline-safe.
func TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline(t *testing.T) {
	if os.Getenv("BLIS_NOOP_SUBPROCESS") == "1" {
		rootCmd.SetArgs([]string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
		})
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	golden, err := os.ReadFile(baselineNoopGolden)
	if err != nil {
		t.Fatalf("read golden baseline: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline")
	cmd.Env = append(os.Environ(), "BLIS_NOOP_SUBPROCESS=1")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = io.Discard // logrus diagnostics go to stderr (INV-6: not part of deterministic output)
	if err := cmd.Run(); err != nil {
		t.Fatalf("subprocess run failed: %v\nstdout:\n%s", err, stdout.String())
	}

	if stdout.String() != string(golden) {
		t.Errorf("INV-6 VIOLATION: adapter-blind run stdout differs from pre-feature baseline.\n"+
			"--- got ---\n%s\n--- want (golden) ---\n%s", stdout.String(), string(golden))
	}
}
