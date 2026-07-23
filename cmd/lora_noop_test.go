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

// TestCreationSeamInert_LoRAActiveOnDemandByteIdentity is the B-5 T021 / C-8 case (b)
// proof that the creation seam is inert when the LoRA subsystem is ACTIVE (not merely
// when absent). A --lora-config run declares adapters + capacity (so the registry,
// resident set, and creationPolicy are all built), but no request targets an adapter
// and no placement is declared, so on-demand's Initial seeds nothing and OnResidentMiss
// always admits. Three assertions, together closing "inert but genuinely active":
//   - determinism (INV-6): two identical LoRA-active runs are byte-identical;
//   - inert-while-active (INV-L1): the LoRA-active on-demand run is byte-identical to
//     the committed adapter-blind baseline — enabling the subsystem and adding the
//     creation seam changes nothing observable. With a negligible reservation, no
//     adapter-targeting request, and on-demand seeding nothing, byte-identity IS the
//     correct behavior (the tiny HBM reservation removes no blocks a light workload
//     can observe);
//   - liveness: a SEPARATE over-reservation --lora-config (≈190 GiB) MUST make the run
//     fatal on the KV auto-calc path. That workload-independent fatal proves
//     --lora-config genuinely threads into the simulator — so the byte-identity above
//     reflects an active-but-inert seam, not a silently disabled subsystem.
func TestCreationSeamInert_LoRAActiveOnDemandByteIdentity(t *testing.T) {
	if os.Getenv("BLIS_NOOP_SUBPROCESS") == "1" {
		loraCfg := os.Getenv("BLIS_LORA_CONFIG")
		if loraCfg == "" {
			loraCfg = "testdata/lora_ondemand.yaml"
		}
		rootCmd.SetArgs([]string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
			"--lora-config", loraCfg,
		})
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	// run executes the subprocess with the given --lora-config fixture and returns
	// its stdout plus whether it exited successfully (a logrus.Fatalf → os.Exit(1)
	// surfaces as a non-nil cmd.Run error).
	run := func(loraConfig string) (string, bool) {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestCreationSeamInert_LoRAActiveOnDemandByteIdentity")
		cmd.Env = append(os.Environ(), "BLIS_NOOP_SUBPROCESS=1", "BLIS_LORA_CONFIG="+loraConfig)
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = io.Discard // logrus diagnostics go to stderr (INV-6: not part of deterministic output)
		err := cmd.Run()
		return stdout.String(), err == nil
	}

	first, ok1 := run("testdata/lora_ondemand.yaml")
	second, ok2 := run("testdata/lora_ondemand.yaml")
	if !ok1 || !ok2 {
		t.Fatalf("LoRA-active on-demand run exited non-zero (ok1=%v ok2=%v); expected success", ok1, ok2)
	}
	if first != second {
		t.Errorf("INV-6 VIOLATION: LoRA-active on-demand run is non-deterministic across invocations.\n"+
			"--- first ---\n%s\n--- second ---\n%s", first, second)
	}

	golden, err := os.ReadFile(baselineNoopGolden)
	if err != nil {
		t.Fatalf("read golden baseline: %v", err)
	}
	if first != string(golden) {
		t.Errorf("INV-L1 VIOLATION: LoRA-active on-demand run differs from the adapter-blind baseline; "+
			"the creation seam + on-demand default must be byte-identical to no-LoRA.\n"+
			"--- got ---\n%s\n--- want (golden) ---\n%s", first, string(golden))
	}

	// Liveness: an over-reservation config must fatal on the KV auto-calc path,
	// proving --lora-config is genuinely threaded into the simulator (not ignored).
	if _, ok := run("testdata/lora_overreserved.yaml"); ok {
		t.Error("over-reservation --lora-config run exited 0; expected a fatal on the KV auto-calc " +
			"path — cannot prove --lora-config genuinely activates the LoRA subsystem")
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
