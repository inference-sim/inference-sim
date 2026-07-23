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

// TestNoOpByteIdentity_MultiInstanceEvictionPolicyInert is the B-4 T020 / D-4-3
// cluster-scope INV-6 proof: across a 2-instance cluster with no LoRA configured,
// leaving --eviction-policy unset, selecting the default lru, and selecting rank-aware
// all produce byte-identical stdout. This closes the cluster-scope concern that the
// selector might introduce divergence — rank-aware is inert unless the LoRA subsystem
// is active, and --eviction-policy is one cluster-global name (not a per-pool
// override). Runs are compared to one another rather than to a stored golden, since no
// multi-instance baseline is tracked; the single-instance baseline golden above pins
// the absolute output.
func TestNoOpByteIdentity_MultiInstanceEvictionPolicyInert(t *testing.T) {
	if os.Getenv("BLIS_NOOP_SUBPROCESS") == "1" {
		args := []string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--num-instances", "2",
			"--defaults-filepath", "../defaults.yaml",
		}
		if pol := os.Getenv("BLIS_NOOP_EVICTION_POLICY"); pol != "" {
			args = append(args, "--eviction-policy", pol)
		}
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	run := func(policy string) string {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestNoOpByteIdentity_MultiInstanceEvictionPolicyInert")
		cmd.Env = append(os.Environ(), "BLIS_NOOP_SUBPROCESS=1", "BLIS_NOOP_EVICTION_POLICY="+policy)
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = io.Discard // logrus diagnostics go to stderr (INV-6: not part of deterministic output)
		if err := cmd.Run(); err != nil {
			t.Fatalf("subprocess run (policy=%q) failed: %v\nstdout:\n%s", policy, err, stdout.String())
		}
		return stdout.String()
	}

	unset := run("")     // flag not passed → EvictionPolicy "" → lru
	lru := run("lru")    // explicit default
	rank := run("rank-aware")

	if unset != lru {
		t.Errorf("INV-6 VIOLATION (cluster): unset --eviction-policy differs from explicit lru.\n"+
			"--- unset ---\n%s\n--- lru ---\n%s", unset, lru)
	}
	if unset != rank {
		t.Errorf("INV-6 VIOLATION (cluster): selecting rank-aware changed stdout on a LoRA-inert run.\n"+
			"--- unset ---\n%s\n--- rank-aware ---\n%s", unset, rank)
	}
}
