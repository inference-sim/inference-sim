package cmd

import (
	"encoding/json"
	"errors"
	"math"
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

// INV-13 for a contract-built release: a 0.32.0 run's exported trace, replayed with the same
// flags, reproduces the run byte for byte. blis replay must build the same engine through the
// engine contract, not fall back to the compatibility engine.
func TestReplayCmd_A032ScenarioReproducesTheRun(t *testing.T) {
	dir := writeColocatedVariant(t, `engine_version: "0.29.0"`, `engine_version: "0.32.0"`)
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--seed", "7", "--horizon", "600000000"}
	runOut, runErr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "24", "--rate", "15",
		"--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\nstderr:\n%s", err, lastLines(runErr, 5))
	}
	repOut, repErr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\nstderr:\n%s", err, lastLines(repErr, 5))
	}
	if !strings.Contains(repErr, "engine assumption: engine_version = 0.31.0") {
		t.Errorf("a 0.32.0 replay must run the engine built through the contract, which discloses the "+
			"kernel's 0.31.0 tuning; stderr:\n%s", lastLines(repErr, 6))
	}
	if runOut != repOut {
		t.Errorf("a 0.32.0 replay does not reproduce its run (INV-13):\nrun:\n%s\nreplay:\n%s",
			lastLines(runOut, 30), lastLines(repOut, 30))
	}
}

// A 0.32.0 run keeps the host's API-server front end (INV-R1): requests that arrive together
// queue for the one render thread (vLLM's default, --renderer-num-workers 1) before they reach
// the engine, as on the 0.31.0 compatibility engine. The queue adds the same TTFT on both, and
// --renderer-num-workers 0 turns it off on both.
func TestRunCmd_A032ScenarioRendersOnTheHostFrontEnd(t *testing.T) {
	ttft := func(version string, extra ...string) float64 {
		dir := writeColocatedVariant(t, `engine_version: "0.29.0"`, `engine_version: "`+version+`"`)
		args := append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--concurrency", "128", "--num-requests", "128", "--prompt-tokens", "1024", "--output-tokens", "8",
			"--seed", "7"}, extra...)
		stdout, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("%s %v: %v\nstderr:\n%s", version, extra, err, lastLines(stderr, 5))
		}
		for _, raw := range extractJSONObjects(stdout) {
			var obj map[string]any
			if json.Unmarshal([]byte(raw), &obj) == nil && obj["instance_id"] == "cluster" {
				return obj["ttft_mean_ms"].(float64)
			}
		}
		t.Fatalf("no cluster metrics in stdout:\n%s", stdout)
		return 0
	}
	off := []string{"--renderer-num-workers", "0"}
	queued31, free31 := ttft("0.31.0"), ttft("0.31.0", off...)
	queued32, free32 := ttft("0.32.0"), ttft("0.32.0", off...)
	if queued31-free31 <= 0.05*free31 {
		t.Fatalf("0.31.0: the render queue adds %.1f ms of mean TTFT to %.1f ms; the workload no longer "+
			"exercises it", queued31-free31, free31)
	}
	if q, c := queued32-free32, queued31-free31; q < 0.5*c {
		t.Errorf("the render queue adds %.1f ms of mean TTFT on 0.32.0 against %.1f ms on 0.31.0; the front "+
			"end is the host's, the same for both engines", q, c)
	}
	for _, p := range [][2]float64{{queued32, queued31}, {free32, free31}} {
		if math.Abs(p[0]-p[1]) > 0.05*p[1] {
			t.Errorf("mean TTFT %.1f ms on 0.32.0 against %.1f ms on 0.31.0 (render queue %v), want within 5%%",
				p[0], p[1], p == [2]float64{queued32, queued31})
		}
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

// A 0.32.0 deployment the v0.32.0 engine cannot build (here a DeepSeek-V4.1 checkpoint stating
// max_model_len auto, a window the engine takes only as stated) is refused at the build site:
// exit 1 with the engine's message naming the missing piece, not a panic out of the cluster's
// construction.
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
		t.Fatal("a 0.32.0 deployment stating max_model_len auto ran")
	}
	if strings.Contains(stderr, "goroutine ") || strings.Contains(stderr, "panic:") {
		t.Errorf("the refusal must be a clean CLI error, not a panic; stderr:\n%s", lastLines(stderr, 8))
	}
	if !strings.Contains(stderr, "engine_version 0.32.0: vllm: max_model_len must be stated") {
		t.Errorf("the refusal must be the engine's, naming the missing piece; stderr:\n%s", lastLines(stderr, 4))
	}
}

// dp032Variant writes the vendored dp 2 MoE scenario (dpScenario: minimax-m2.5 on B200, TP2 x
// DP2 with expert parallelism) at engine_version 0.32.0, with extra engine lines, into a fresh
// scenario directory as v032-dp2.yaml.
func dp032Variant(t *testing.T, extra string) string {
	t.Helper()
	scenarios, _, _ := kernelRepos(t)
	body := readFile(t, filepath.Join(scenarios, dpScenario))
	const anchor = "      gpu_memory_utilization: 0.92\n"
	if !strings.Contains(body, `engine_version: "0.29.0"`) || !strings.Contains(body, anchor) {
		t.Fatalf("fixture %s has no engine_version 0.29.0 or %q to edit", dpScenario, anchor)
	}
	body = strings.Replace(body, `engine_version: "0.29.0"`, `engine_version: "0.32.0"`, 1)
	body = strings.Replace(body, anchor, anchor+extra, 1)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "v032-dp2.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// A 0.32.0 data-parallel deployment reaches the engine as one instance: the host places no rank
// replicas (the engine is the whole DP group, its balancer, waves and lockstep its own), and the
// build site no longer refuses dp > 1. With async scheduling off, which vLLM's DP group then
// runs with a queue of one batch per rank, the deployment runs to completion.
func TestRunCmd_A032DataParallelDeploymentIsOneEngineGroup(t *testing.T) {
	dir := dp032Variant(t, "      async_scheduling: false\n")
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "v032-dp2.yaml",
		"--rate", "4", "--num-requests", "8", "--seed", "1")
	if err != nil {
		t.Fatalf("the 0.32.0 dp 2 deployment did not run: %v; stderr:\n%s", err, lastLines(stderr, 6))
	}
	if strings.Contains(stderr, "DP-as-placement") {
		t.Errorf("a 0.32.0 DP group must not be expanded into rank replicas; stderr:\n%s", lastLines(stderr, 6))
	}
	if !strings.Contains(stdout, `"completed_requests": 8`) {
		t.Errorf("want all 8 requests completed; stdout:\n%s", lastLines(stdout, 30))
	}
}

// A 0.32.0 data-parallel group is vLLM's internal balancer: one endpoint whose engine routes to its
// ranks (core/vllm.DPGroup). Under the external balancer each rank is its own endpoint and the
// router picks the rank (vllm/entrypoints/cli/serve.py:107-116@5df63a72), which the contract path
// does not build: such a run is refused, naming the mode, rather than run as internal.
func TestRunCmd_A032DataParallelGroupRefusesAnExternalBalancer(t *testing.T) {
	dir := dp032Variant(t, "")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "v032-dp2.yaml",
		"--dp-load-balancing", "external", "--rate", "4", "--num-requests", "4", "--seed", "1")
	if err == nil {
		t.Fatal("a 0.32.0 dp 2 deployment under the external DP balancer ran")
	}
	if !strings.Contains(stderr, "--dp-load-balancing external") {
		t.Errorf("the refusal must name the balancing mode; stderr:\n%s", lastLines(stderr, 4))
	}
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "v032-dp2.yaml",
		"--dp-load-balancing", "internal", "--rate", "4", "--num-requests", "4", "--seed", "1"); err != nil {
		t.Errorf("the internal balancer must still run: %v\nstderr:\n%s", err, lastLines(stderr, 4))
	}
}

// A 0.32.0 data-parallel group's endpoint renders on vLLM's default API servers for the internal
// balancer: one per rank (api_server_count = data_parallel_size, vllm/entrypoints/cli/serve.py:
// 123-130@5df63a72), as the compatibility path's DP replica does (ReplicaAPIServers). With one
// server instead, requests that arrive together queue for its render thread twice as long, so
// the group's mean TTFT would be well above the stated-two-servers run, which it must match.
func TestRunCmd_A032DataParallelGroupRendersOnOneAPIServerPerRank(t *testing.T) {
	dir := dp032Variant(t, "")
	ttft := func(extra ...string) float64 {
		args := append([]string{"run", "--scenarios", dir, "--scenario", "v032-dp2.yaml",
			"--concurrency", "128", "--num-requests", "128", "--prompt-tokens", "1024", "--output-tokens", "8",
			"--seed", "7"}, extra...)
		stdout, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("%v: %v\nstderr:\n%s", extra, err, lastLines(stderr, 5))
		}
		for _, raw := range extractJSONObjects(stdout) {
			var obj map[string]any
			if json.Unmarshal([]byte(raw), &obj) == nil && obj["instance_id"] == "cluster" {
				return obj["ttft_mean_ms"].(float64)
			}
		}
		t.Fatalf("no cluster metrics in stdout:\n%s", stdout)
		return 0
	}
	byDefault, stated := ttft(), ttft("--api-server-count", "2")
	if byDefault != stated {
		t.Errorf("mean TTFT %.3f ms with the default API servers against %.3f ms with 2 stated; a dp 2 "+
			"group's default is one API server per rank", byDefault, stated)
	}
	if one := ttft("--api-server-count", "1"); one <= stated {
		t.Fatalf("one API server gives %.3f ms against %.3f ms with 2; the workload no longer exercises "+
			"render contention", one, stated)
	}
}
