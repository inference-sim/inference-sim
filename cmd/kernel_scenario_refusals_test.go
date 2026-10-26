package cmd

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
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
				"      gpu_memory_utilization: 0.92\n", "      gpu_memory_utilization: 0.92\n"+secondPool))},
		// vLLM refuses to start an engine whose window one request cannot fit in its KV, with
		// its start-up message (vllm/v1/core/kv_cache_utils.py:917-935 at v0.31.0).
		{"a window larger than the KV budget", "To serve at least one request with the model's max seq len (4000000)",
			run(writeColocatedVariant(t, "max_model_len: 32768", "max_model_len: 4000000"))},
		{"an unsimulated scheduling policy", `engine.scheduling_policy \"lifo\"`,
			run(writeColocatedVariant(t, "      gpu_memory_utilization: 0.92\n",
				"      gpu_memory_utilization: 0.92\n      scheduling_policy: lifo\n"))},
		// A shared-only or encode+shared topology states no dedicated prefill or decode
		// instance yet is disaggregated; it is refused, not run into a missing handoff pricer.
		{"shared prefill-decode instances only", "--prefill-decode-instances and --encode-instances are not supported",
			run(plain, "--num-instances", "2", "--prefill-decode-instances", "1")},
		{"encode with shared instances", "--prefill-decode-instances and --encode-instances are not supported",
			run(plain, "--num-instances", "2", "--prefill-decode-instances", "1", "--encode-instances", "1")},
		// Encode alone is refused earlier, by the topology check (no decode-capable pool).
		{"encode instances only", "requires a decode-capable pool",
			run(plain, "--num-instances", "2", "--encode-instances", "1")},
		{"a node pool of another GPU", "node pools must match it",
			run(plain, "--policy-config", writeNodePoolBundle(t, "H100", "141"))},
		{"a node pool stating other GPU memory", "states gpu_memory_gib 80",
			run(plain, "--policy-config", writeNodePoolBundle(t, "h200", "80"))},
		// vLLM's own floor on the batch caps (ge=1, vllm/config/scheduler.py:48,62@5df63a72) is the
		// engine's to apply: blis-vllm's kernel model refuses a deployment that states none
		// (compat/kernelmodel Model.Engine), when the build site sizes the pool, and cmd keeps no
		// copy of the check.
		{"max_num_seqs 0", "deployment states no max_num_seqs",
			run(writeColocatedVariant(t, "      max_num_seqs: 256\n", "      max_num_seqs: 0\n"))},
		{"max_num_batched_tokens 0", "deployment states no max_num_batched_tokens",
			run(writeColocatedVariant(t, "      max_num_batched_tokens: 8192\n", "      max_num_batched_tokens: 0\n"))},
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
	for _, args := range [][]string{run(plain), run(plain, "--policy-config", writeNodePoolBundle(t, "h200", "139.80072021"))} {
		if out, stderr, err := runKernelCLI(t, args...); err != nil || !strings.Contains(out, `"completed_requests": 4`) {
			t.Errorf("%v was not served: %v\n%s", args, err, lastLines(stderr, 3))
		}
	}
}

// A DP engine vLLM refuses to start is refused before anything runs, exit status 1
// and vLLM's start-up message as the fatal line, whether the window is too long for one rank's
// KV or no memory is left for KV at all; the same DEP8 deployment at a utilization whose rank
// holds one max_model_len request starts and serves.
func TestRunCmd_KernelBackend_RefusesAnEngineVLLMWouldNotStart(t *testing.T) {
	src := readFile(t, filepath.Join("..", "testdata", "scenarios", "deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml"))
	variant := func(gmu string) string {
		dir := t.TempDir()
		body := strings.Replace(src, "gpu_memory_utilization: 0.85", "gpu_memory_utilization: "+gmu, 1)
		if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	run := func(dir string) []string {
		return []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
			"--num-requests", "2", "--rate", "1", "--speculative-acceptance-length", "3.5"}
	}
	for _, tt := range []struct{ gmu, want string }{
		{"0.74", "To serve at least one request with the model's max seq len (262144)"},
		{"0.70", "No available memory for the cache blocks"},
	} {
		_, stderr, err := runKernelCLI(t, run(variant(tt.gmu))...)
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 1 || !strings.Contains(stderr, "level=fatal") ||
			!strings.Contains(stderr, tt.want) {
			t.Errorf("gmu %s: want exit 1 with a fatal %q, got err=%v\n%s", tt.gmu, tt.want, err, lastLines(stderr, 4))
		}
	}
	if out, stderr, err := runKernelCLI(t, run(variant("0.85"))...); err != nil || !strings.Contains(out, `"completed_requests"`) {
		t.Errorf("gmu 0.85 on the vendored pins holds one window and must start: %v\n%s", err, lastLines(stderr, 3))
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
	priority := writeColocatedVariant(t, "      gpu_memory_utilization: 0.92\n",
		"      gpu_memory_utilization: 0.92\n      scheduling_policy: priority\n")
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
	if err != nil || !strings.Contains(stderr, "overriding scenario") {
		t.Errorf("--scheduler fcfs over a priority scenario: want a warning naming the override, got %v\n%s",
			err, lastLines(stderr, 3))
	}
	if _, stderr, _ := runKernelCLI(t, append(args, "--scheduler", "priority-fcfs")...); strings.Contains(stderr, "overriding scenario") {
		t.Errorf("--scheduler restating the scenario's policy warned of an override:\n%s", lastLines(stderr, 3))
	}
}

// A run that departs from its scenario says so in the results file, not only in a log line:
// an explicit --scheduler, or a policy bundle's scheduler, that differs from the scenario's
// engine.scheduling_policy is recorded under "overrides" in the --metrics-path file, with the
// two values and what chose the run's. A run that follows its scenario records nothing, and the
// block never reaches stdout (INV-6).
func TestRunCmd_ASchedulerOverrideIsRecordedInTheResultsFile(t *testing.T) {
	priority := writeColocatedVariant(t, "      gpu_memory_utilization: 0.92\n",
		"      gpu_memory_utilization: 0.92\n      scheduling_policy: priority\n")
	bundle := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(bundle, []byte("scheduler: sjf\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	type override struct{ Setting, Scenario, Run, Source string }
	overridesOf := func(t *testing.T, extra ...string) ([]override, string) {
		t.Helper()
		path := filepath.Join(t.TempDir(), "metrics.json")
		args := append([]string{"run", "--scenarios", priority, "--scenario", "variant.yaml",
			"--num-requests", "4", "--rate", "2", "--metrics-path", path}, extra...)
		stdout, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("%v: %v\n%s", extra, err, lastLines(stderr, 3))
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var file struct {
			Overrides []override `json:"overrides"`
		}
		if err := json.Unmarshal(raw, &file); err != nil {
			t.Fatal(err)
		}
		return file.Overrides, stdout
	}

	for _, tt := range []struct {
		name  string
		extra []string
		want  []override
	}{
		{"the scenario's own scheduler records nothing", nil, nil},
		{"an explicit --scheduler equal to the scenario's records nothing", []string{"--scheduler", "priority-fcfs"}, nil},
		{"an explicit --scheduler that differs is recorded", []string{"--scheduler", "sjf"},
			[]override{{"scheduler", "priority-fcfs", "sjf", "--scheduler"}}},
		{"a policy bundle's scheduler that differs is recorded", []string{"--policy-config", bundle},
			[]override{{"scheduler", "priority-fcfs", "sjf", "--policy-config"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, stdout := overridesOf(t, tt.extra...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("results file overrides %+v, want %+v", got, tt.want)
			}
			if strings.Contains(stdout, `"overrides"`) {
				t.Error("the overrides block reached stdout; it is file-only (INV-6)")
			}
		})
	}
}

// A scenario stating an engine release the compatibility engine does not model is refused
// before anything runs, naming the release (#2021 B6: one build models one release, and a
// version is a check, not a selector). A v0.32.0 deployment is never run as a 0.29 or 0.31
// one.
func TestRunCmd_RefusesAnEngineReleaseItDoesNotModel(t *testing.T) {
	dir := writeColocatedVariant(t, `engine_version: "0.29.0"`, `engine_version: "0.33.0"`)
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err == nil {
		t.Fatal("a 0.33.0 scenario ran: this build models no such release")
	}
	if !strings.Contains(stderr, `engine_version \"0.33.0\"`) {
		t.Errorf("refusal must name the stated release; stderr:\n%s", lastLines(stderr, 3))
	}
}

// A 0.32.0 scenario runs blis-vllm's translated v0.32.0 engine through blis-schemas' engine.Engine
// contract (GOAL.md): every request completes, and the latency kernel's 0.31.0 tuning, which
// prices the step, is disclosed on stderr.
func TestRunCmd_A032ScenarioRunsThroughTheEngineContract(t *testing.T) {
	dir := writeColocatedVariant(t, `engine_version: "0.29.0"`, `engine_version: "0.32.0"`)
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err != nil {
		t.Fatalf("a 0.32.0 scenario must run through the contract: %v\nstderr:\n%s", err, lastLines(stderr, 5))
	}
	if !strings.Contains(stdout, `"completed_requests": 4`) {
		t.Errorf("want 4 completed requests; stdout:\n%s", stdout)
	}
	if !strings.Contains(stderr, "engine assumption: engine_version = 0.31.0") {
		t.Errorf("the kernel's 0.31.0 tuning must be disclosed; stderr:\n%s", lastLines(stderr, 6))
	}
}

// A MultiConnector P/D run states its assumption about the connector's children however opening
// the pools ends: refused for counts that do not add up, the warning still reaches stderr, ahead
// of the refusal, as it did before the pools moved to blis-vllm.
func TestRunCmd_MultiConnectorWarningSurvivesARefusal(t *testing.T) {
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios,
		"--scenario", "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml",
		"--num-instances", "3", "--prefill-instances", "1", "--decode-instances", "1",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err == nil {
		t.Fatal("counts that do not add up must be refused")
	}
	warn := strings.Index(stderr, "pd_transfer.connector MultiConnector is simulated as NixlConnector")
	refusal := strings.Index(stderr, "the counts must add up")
	if warn < 0 || refusal < 0 || warn > refusal {
		t.Errorf("want the MultiConnector warning, then the refusal; stderr:\n%s", lastLines(stderr, 4))
	}
}

// A 0.32.0 deployment the v0.32.0 engine cannot build yet (here a DeepSeek-V4.1 checkpoint, whose
// sliding-window and indexer KV groups are hybrid) is refused at the build site: exit 1 with the
// engine's message naming the missing piece, not a panic out of the cluster's construction.
func TestRunCmd_A032ScenarioTheEngineCannotBuildIsRefusedByName(t *testing.T) {
	body := readFile(t, filepath.Join("..", "testdata", "scenarios", "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"))
	if !strings.Contains(body, `engine_version: "0.31.0"`) {
		t.Fatal("fixture states no engine_version 0.31.0 to edit")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "v032.yaml"),
		[]byte(strings.Replace(body, `engine_version: "0.31.0"`, `engine_version: "0.32.0"`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "v032.yaml",
		"--catalog", "../testdata/catalog", "--speculative-acceptance-rate", "0.7",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err == nil {
		t.Fatal("a 0.32.0 deployment needing hybrid KV groups ran")
	}
	if strings.Contains(stderr, "goroutine ") || strings.Contains(stderr, "panic:") {
		t.Errorf("the refusal must be a clean CLI error, not a panic; stderr:\n%s", lastLines(stderr, 8))
	}
	if !strings.Contains(stderr, "hybrid KV groups") {
		t.Errorf("the refusal must name the missing piece; stderr:\n%s", lastLines(stderr, 4))
	}
}
