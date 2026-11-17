package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeLoRAPlacementRun writes the scenario (an engine that serves adapters: max_loras 2 and a CPU
// cache of 3), a --lora-config file declaring adapters a, b, c and d, and a policy bundle with the
// given lora_placement block, and returns the CLI arguments that run them.
func loraPlacementArgs(t *testing.T, placement string) []string {
	t.Helper()
	dir := writeColocatedVariant(t, "      max_model_len: 32768\n", "      max_model_len: 32768\n      max_loras: 2\n      max_cpu_loras: 3\n")
	tmp := t.TempDir()
	lora := filepath.Join(tmp, "lora.yaml")
	if err := os.WriteFile(lora, []byte("lora:\n  adapters:\n    - {id: a, rank: 8}\n    - {id: b, rank: 8}\n    - {id: c, rank: 8}\n    - {id: d, rank: 8}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	bundle := filepath.Join(tmp, "bundle.yaml")
	if err := os.WriteFile(bundle, []byte(placement), 0o644); err != nil {
		t.Fatal(err)
	}
	return []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-requests", "6", "--rate", "2",
		"--num-instances", "2", "--lora-config", lora, "--policy-config", bundle}
}

// The LoRA placement loop (#1493, #1495) through the CLI, on the engine that models vLLM's
// dynamic adapter load: a placement the cluster can honour runs; one it cannot is refused before
// the run, naming what is wrong.
func TestRunCmd_LoRAPlacement(t *testing.T) {
	for _, tc := range []struct {
		name, bundle, want string // want "" is a clean run
	}{
		{"an honourable placement", "lora_placement:\n  creation: pre-placement\n  periodic_interval_us: 500000\n  assignment:\n    0: [a, b]\n    1: [c]\n", ""},
		{"a bundle", "lora_placement:\n  bundle: static-placement\n  assignment:\n    0: [a]\n", ""},
		{"more adapters than the cache holds", "lora_placement:\n  creation: pre-placement\n  assignment:\n    0: [a, b, c, d]\n", "holds 3"},
		{"an undeclared adapter", "lora_placement:\n  creation: pre-placement\n  assignment:\n    0: [zzz]\n", "declares no such adapter"},
		{"an instance the run lacks", "lora_placement:\n  creation: pre-placement\n  assignment:\n    7: [a]\n", "instance 7"},
		{"an unknown bundle", "lora_placement:\n  bundle: nope\n", "not a bundle"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, stderr, err := runKernelCLI(t, loraPlacementArgs(t, tc.bundle)...)
			switch {
			case tc.want == "" && err != nil:
				t.Errorf("run failed: %v\n%s", err, lastLines(stderr, 4))
			case tc.want != "" && (err == nil || !strings.Contains(stderr, tc.want)):
				t.Errorf("want a refusal containing %q, got err=%v\n%s", tc.want, err, lastLines(stderr, 4))
			}
		})
	}
}
