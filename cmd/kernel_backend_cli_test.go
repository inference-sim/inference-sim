package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Behavioral contract for `--latency-model blis-latency-kernel` on `blis run` and
// `blis replay`.
//
// The kernel already satisfied sim.LatencyModel and was reachable only from the
// standalone scoring binaries (cmd/metricscore and friends), which construct it
// directly. `blis run` accepted roofline and trained-physics alone, so the CLI with
// every serving knob on it -- P/D roles, --dp, speculative tokens, workload specs --
// could not be driven against the kernel at all. These tests pin the behaviour that
// closes that gap, at the CLI boundary rather than on the wiring.
//
// Each leg re-execs this test binary as a real `blis` invocation so the cobra tree
// executes and a logrus.Fatalf surfaces as a non-zero exit, following
// cmd/catalog_cli_test.go.
//
//	C1 — a kernel run completes and reports metrics.
//	C3 — --dp > 1 and --enable-expert-parallel are ACCEPTED on the kernel backend.
//	C5 — a missing scenario is refused naming what could not be resolved, never
//	     silently served by another backend.

const kernelLegEnv = "BLIS_KERNEL_LEG"

// kernelRepos locates the three repositories a kernel scenario resolves against. They
// are siblings of this one in a developer checkout; a leg skips rather than fails where
// any is absent, so the suite stays green on a machine that has only this repo.
func kernelRepos(t *testing.T) (scenarios, catalog, registry string) {
	t.Helper()
	root := filepath.Join("..", "..", "..", "Documents", "Projects")
	if env := os.Getenv("BLIS_PROJECTS"); env != "" {
		root = env
	}
	scenarios = filepath.Join(root, "blis-latency-kernel", "testdata", "aisimulate")
	catalog = filepath.Join(root, "blis-catalog")
	registry = filepath.Join(root, "blis-registry")
	for _, p := range []string{scenarios, catalog, registry} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("kernel scenario repositories not present (%s): %v", p, err)
		}
	}
	return scenarios, catalog, registry
}

func runKernelLeg(t *testing.T, leg string, extra ...string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_KernelBackend$")
	env := append(os.Environ(), kernelLegEnv+"="+leg)
	if len(extra) > 0 {
		env = append(env, "BLIS_KERNEL_EXTRA="+strings.Join(extra, "\x1f"))
	}
	cmd.Env = env
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// TestRunCmd_KernelBackend drives the real CLI on the kernel backend.
func TestRunCmd_KernelBackend(t *testing.T) {
	if leg := os.Getenv(kernelLegEnv); leg != "" {
		args := []string{
			"run", "--latency-model", "blis-latency-kernel",
			"--scenario", os.Getenv("BLIS_KERNEL_SCENARIO"),
			"--scenarios", os.Getenv("BLIS_KERNEL_SCENARIOS"),
			"--catalog", os.Getenv("BLIS_KERNEL_CATALOG"),
			"--registry", os.Getenv("BLIS_KERNEL_REGISTRY"),
			"--seed", "42", "--num-requests", "8", "--concurrency", "2",
			"--defaults-filepath", "../defaults.yaml",
		}
		if e := os.Getenv("BLIS_KERNEL_EXTRA"); e != "" {
			args = append(args, strings.Split(e, "\x1f")...)
		}
		rootCmd.SetArgs(args)
		if execErr := rootCmd.Execute(); execErr != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	scenarios, catalog, registry := kernelRepos(t)
	t.Setenv("BLIS_KERNEL_SCENARIOS", scenarios)
	t.Setenv("BLIS_KERNEL_CATALOG", catalog)
	t.Setenv("BLIS_KERNEL_REGISTRY", registry)

	tests := []struct {
		name      string
		scenario  string
		extra     []string
		wantFatal bool
		wantMsg   string
	}{
		{
			// C1: the run completes on the kernel and prints metrics.
			name:     "a kernel run reports metrics",
			scenario: "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
		},
		{
			// The deployment has one source. Re-supplying a fact the scenario states is
			// refused naming both, rather than reconciled by precedence logic.
			name:      "a duplicated deployment flag is refused",
			scenario:  "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
			extra:     []string{"--tp", "2"},
			wantFatal: true,
			wantMsg:   "--tp is not accepted",
		},
		{
			// C3: the kernel models DP/EP step time, so a scenario that states expert
			// parallelism runs -- the gate that names trained-physics as the only
			// option must admit this backend.
			name:     "a scenario with expert parallelism runs",
			scenario: "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
		},
		{
			// C5: an unresolvable scenario is refused naming it, not served by a
			// fallback backend.
			name:      "a missing scenario is refused",
			scenario:  "no-such-scenario.yaml",
			wantFatal: true,
			wantMsg:   "no-such-scenario",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("BLIS_KERNEL_SCENARIO", tt.scenario)
			stdout, stderr, err := runKernelLeg(t, "go", tt.extra...)

			if tt.wantFatal {
				if err == nil {
					t.Fatalf("expected refusal for scenario %q\nstdout:\n%s", tt.scenario, stdout)
				}
				if tt.wantMsg != "" && !strings.Contains(stderr, tt.wantMsg) {
					t.Errorf("refusal should name %q, got stderr:\n%s", tt.wantMsg, stderr)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected the run to succeed, got %v\nstderr:\n%s", err, stderr)
			}
			// Non-vacuity: a silently-empty run would otherwise satisfy "no error".
			if !strings.Contains(stdout, "completed_requests") {
				t.Errorf("expected simulation metrics on stdout, got:\n%s", stdout)
			}
		})
	}
}
