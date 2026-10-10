package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// colocatedFixture is the vendored colocated kernel scenario the variants below edit.
const colocatedFixture = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"

// writeColocatedVariant writes colocatedFixture, edited by each old->new pair, into a fresh
// scenario directory as variant.yaml, so each case is driven by one stated difference.
func writeColocatedVariant(t *testing.T, edits ...string) string {
	t.Helper()
	scenarios, _, _ := kernelRepos(t)
	body := readFile(t, filepath.Join(scenarios, colocatedFixture))
	for i := 0; i+1 < len(edits); i += 2 {
		if !strings.Contains(body, edits[i]) {
			t.Fatalf("fixture has no %q to edit", edits[i])
		}
		body = strings.Replace(body, edits[i], edits[i+1], 1)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// writeNodePoolBundle writes a --policy-config bundle with one node pool of gpuType GPUs,
// stating gpuMemGiB.
func writeNodePoolBundle(t *testing.T, gpuType, gpuMemGiB string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "bundle.yaml")
	body := "node_pools:\n  - name: pool-a\n    gpu_type: " + gpuType + "\n    gpus_per_node: 8\n" +
		"    gpu_memory_gib: " + gpuMemGiB + "\n    initial_nodes: 1\n    min_nodes: 1\n    max_nodes: 4\n" +
		"    cost_per_hour: 32.0\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// What a colocated kernel run refuses, each named, rather than simulating an engine or device
// the scenario does not describe -- and, for every refusal, the nearest accepted input runs, so
// the refusal is the stated difference and not a broken fixture.
func TestRunCmd_KernelBackend_ScenarioRefusals(t *testing.T) {
	run := func(dir string, extra ...string) []string {
		return append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--num-requests", "4", "--rate", "2"}, extra...)
	}
	plain := writeColocatedVariant(t)
	pool := "  - role: colocated\n"
	scenario := readFile(t, filepath.Join(plain, "variant.yaml"))
	secondPool := scenario[strings.Index(scenario, pool):]
	for _, tt := range []struct {
		name, want string
		args       []string
	}{
		{"two colocated pools", "states 2 colocated pools",
			run(writeColocatedVariant(t,
				"  nodes: 1\n  gpus_per_node: 8\n", "  nodes: 2\n  gpus_per_node: 8\n  fabric: ib-400g\n",
				"      gpu_memory_utilization: 0.9\n", "      gpu_memory_utilization: 0.9\n"+secondPool))},
		// vLLM refuses to start an engine whose window one request cannot fit in its KV.
		{"a window larger than the KV budget", "vLLM refuses to start such an engine",
			run(writeColocatedVariant(t, "max_model_len: 32768", "max_model_len: 4000000"))},
		{"an unsimulated scheduling policy", `engine.scheduling_policy \"lifo\"`,
			run(writeColocatedVariant(t, "      gpu_memory_utilization: 0.9\n",
				"      gpu_memory_utilization: 0.9\n      scheduling_policy: lifo\n"))},
		{"a node pool of another GPU", "node pools must match it",
			run(plain, "--policy-config", writeNodePoolBundle(t, "H100", "141"))},
		{"a node pool stating other GPU memory", "states gpu_memory_gib 80",
			run(plain, "--policy-config", writeNodePoolBundle(t, "h200", "80"))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, stderr, err := runKernelCLI(t, tt.args...)
			if err == nil || !strings.Contains(stderr, tt.want) {
				t.Errorf("want a refusal naming %q, got err=%v\n%s", tt.want, err, lastLines(stderr, 3))
			}
		})
	}
	// The accepted neighbours: the plain fixture, and a node pool of the scenario's chip at
	// the catalog chip's memory.
	for _, args := range [][]string{run(plain), run(plain, "--policy-config", writeNodePoolBundle(t, "h200", "141"))} {
		if out, stderr, err := runKernelCLI(t, args...); err != nil || !strings.Contains(out, `"completed_requests": 4`) {
			t.Errorf("%v was not served: %v\n%s", args, err, lastLines(stderr, 3))
		}
	}
}

// The scenario's engine.scheduling_policy sets the instance scheduler: vLLM's priority is BLIS's
// priority-fcfs, an unstated policy leaves the default, and an explicit --scheduler wins.
func TestAdoptKernelDeployment_TakesTheScenariosSchedulingPolicy(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	setupKernelTestFixtures(t)
	savedScheduler, savedModel, savedGPU := scheduler, model, gpu
	saved := []any{totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen,
		tensorParallelism}
	t.Cleanup(func() {
		scheduler, model, gpu = savedScheduler, savedModel, savedGPU
		totalKVBlocks, blockSizeTokens, maxNumSeqs = saved[0].(int64), saved[1].(int64), saved[2].(int64)
		maxNumBatchedTokens, maxModelLen, tensorParallelism = saved[3].(int64), saved[4].(int64), saved[5].(int)
	})
	priority := writeColocatedVariant(t, "      gpu_memory_utilization: 0.9\n",
		"      gpu_memory_utilization: 0.9\n      scheduling_policy: priority\n")
	plain := writeColocatedVariant(t)
	for _, tt := range []struct {
		name, dir string
		extra     []string
		want      string
	}{
		{"priority becomes priority-fcfs", priority, nil, "priority-fcfs"},
		{"an unstated policy keeps the default", plain, nil, "fcfs"},
		{"an explicit --scheduler wins", priority, []string{"--scheduler", "sjf"}, "sjf"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cmd := &cobra.Command{}
			registerSimConfigFlags(cmd)
			if err := cmd.ParseFlags(append([]string{"--scenario", "variant.yaml", "--scenarios", tt.dir,
				"--registry", registry, "--catalog", catalog}, tt.extra...)); err != nil {
				t.Fatal(err)
			}
			adoptKernelDeployment(cmd)
			if scheduler != tt.want {
				t.Errorf("the run schedules with %q, want %q", scheduler, tt.want)
			}
		})
	}
	// An explicit --scheduler that departs from the scenario is kept, and says so.
	args := []string{"run", "--scenarios", priority, "--scenario", "variant.yaml", "--num-requests", "4", "--rate", "2"}
	_, stderr, err := runKernelCLI(t, append(args, "--scheduler", "fcfs")...)
	if err != nil || !strings.Contains(stderr, "overrides scenario") {
		t.Errorf("--scheduler fcfs over a priority scenario: want a warning naming the override, got %v\n%s",
			err, lastLines(stderr, 3))
	}
	if _, stderr, _ := runKernelCLI(t, append(args, "--scheduler", "priority-fcfs")...); strings.Contains(stderr, "overrides scenario") {
		t.Errorf("--scheduler restating the scenario's policy warned of an override:\n%s", lastLines(stderr, 3))
	}
}
