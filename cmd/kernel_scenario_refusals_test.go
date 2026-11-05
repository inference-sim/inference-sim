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

	"github.com/inference-sim/inference-sim/internal/substratedata"
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
				"  nodes: 1\n  # gpus_per_node: 8 assumed: the corpus states none; every chip in it is an 8-GPU HGX board.\n  gpus_per_node: 8\n", "  nodes: 2\n  gpus_per_node: 8\n  fabric: ib-400g\n",
				"      max_model_len: 32768\n", "      max_model_len: 32768\n"+secondPool))},
		// vLLM refuses to start an engine whose window one request cannot fit in its KV, with
		// its start-up message (vllm/v1/core/kv_cache_utils.py:917-935 at v0.31.0).
		{"a window larger than the KV budget", "To serve at least one request with the model's max seq len (4000000)",
			run(writeColocatedVariant(t, "max_model_len: 32768", "max_model_len: 4000000"))},
		{"an unsimulated scheduling policy", `engine.scheduling_policy \"lifo\"`,
			run(writeColocatedVariant(t, "      max_model_len: 32768\n",
				"      max_model_len: 32768\n      scheduling_policy: lifo\n"))},
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
		// copy of the check. The fixture states neither, so each case states the zero.
		{"max_num_seqs 0", "max_num_seqs",
			run(writeColocatedVariant(t, "      max_model_len: 32768\n", "      max_num_seqs: 0\n      max_model_len: 32768\n"))},
		{"max_num_batched_tokens 0", "max_num_batched_tokens",
			run(writeColocatedVariant(t, "      max_model_len: 32768\n", "      max_num_batched_tokens: 0\n      max_model_len: 32768\n"))},
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
		if out, stderr, err := runKernelCLI(t, args...); err != nil || !completedIs(out, 4) {
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
	if out, stderr, err := runKernelCLI(t, run(variant("0.85"))...); err != nil || !completedAny(out) {
		t.Errorf("gmu 0.85 on the vendored pins holds one window and must start: %v\n%s", err, lastLines(stderr, 3))
	}
}

// The scenario's engine.scheduling_policy sets the instance scheduler: vLLM's priority is BLIS's
// priority-fcfs, an unstated policy leaves the default, and an explicit --scheduler wins. The
// adoption alone is checked here; a run of a priority scenario is refused (the host does not
// carry priority across the engine contract, TestRunCmd_A032ScenarioStatingPrioritySchedulingIsRefused).
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
	priority := writeColocatedVariant(t, "      max_model_len: 32768\n",
		"      max_model_len: 32768\n      scheduling_policy: priority\n")
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
	fcfs := writeColocatedVariant(t, "      max_model_len: 32768\n",
		"      max_model_len: 32768\n      scheduling_policy: fcfs\n")
	args := []string{"run", "--scenarios", fcfs, "--scenario", "variant.yaml", "--num-requests", "4", "--rate", "2"}
	_, stderr, err := runKernelCLI(t, append(args, "--scheduler", "sjf")...)
	if err != nil || !strings.Contains(stderr, "overriding scenario") {
		t.Errorf("--scheduler sjf over an fcfs scenario: want a warning naming the override, got %v\n%s",
			err, lastLines(stderr, 3))
	}
	if _, stderr, _ := runKernelCLI(t, append(args, "--scheduler", "fcfs")...); strings.Contains(stderr, "overriding scenario") {
		t.Errorf("--scheduler restating the scenario's policy warned of an override:\n%s", lastLines(stderr, 3))
	}
}

// A run that departs from its scenario says so in the results file, not only in a log line:
// an explicit --scheduler, or a policy bundle's scheduler, that differs from the scenario's
// engine.scheduling_policy is recorded under "overrides" in the --metrics-path file, with the
// two values and what chose the run's. A run that follows its scenario records nothing, and the
// block never reaches stdout (INV-6).
func TestRunCmd_ASchedulerOverrideIsRecordedInTheResultsFile(t *testing.T) {
	fcfs := writeColocatedVariant(t, "      max_model_len: 32768\n",
		"      max_model_len: 32768\n      scheduling_policy: fcfs\n")
	bundle := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(bundle, []byte("scheduler: sjf\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	type override struct{ Setting, Scenario, Run, Source string }
	overridesOf := func(t *testing.T, extra ...string) ([]override, string) {
		t.Helper()
		path := filepath.Join(t.TempDir(), "metrics.json")
		args := append([]string{"run", "--scenarios", fcfs, "--scenario", "variant.yaml",
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
		{"an explicit --scheduler equal to the scenario's records nothing", []string{"--scheduler", "fcfs"}, nil},
		{"an explicit --scheduler that differs is recorded", []string{"--scheduler", "sjf"},
			[]override{{"scheduler", "fcfs", "sjf", "--scheduler"}}},
		{"a policy bundle's scheduler that differs is recorded", []string{"--policy-config", bundle},
			[]override{{"scheduler", "fcfs", "sjf", "--policy-config"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, stdout := overridesOf(t, tt.extra...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("results file overrides %+v, want %+v", got, tt.want)
			}
			// The Result records the same policy overrides: an input the run simulated differently
			// is part of what the document says it predicts (blis-schemas#41 section A). Its other
			// rows are the command-line flags the harness passed (#1871 B1).
			var inDoc []override
			for _, o := range parseResult(t, stdout).Overrides {
				if strings.HasPrefix(o.Setting, "scheduler") || strings.HasPrefix(o.Setting, "preemption_policy") {
					inDoc = append(inDoc, override{o.Setting, o.Input, o.Run, o.Source})
				}
			}
			if !reflect.DeepEqual(inDoc, tt.want) {
				t.Errorf("Result overrides %+v, want %+v", inDoc, tt.want)
			}
		})
	}
}

// LIF-6, per engine kind: a scenario naming an engine release other than the one this build
// runs is refused before anything runs. The message names the release asked for, the one this
// build runs, and the newest inference-sim release that runs the one asked for, or says that
// none does. The closest accepted scenario, the same file naming vllm 0.32.0, runs (SIM-9).
// The accepted scenario names its engine with engine: {kind, version}; each case replaces that
// block with its own statement (engine_select.go).
func TestRunCmd_RefusesAnEngineReleaseItDoesNotRun(t *testing.T) {
	cases := []struct {
		name, edit string
		want       []string
	}{
		{"vllm 0.29.0, last run by v0.15.0", `engine_version: "0.29.0"`,
			[]string{"names engine vllm 0.29.0", "this build runs vllm 0.32.0", "newest inference-sim release that runs it is v0.15.0"}},
		{"vllm 0.31.0, last run by v0.15.0", `engine_version: "0.31.0"`,
			[]string{"names engine vllm 0.31.0", "this build runs vllm 0.32.0", "newest inference-sim release that runs it is v0.15.0"}},
		{"vllm 0.33.0, run by no release", `engine: {kind: vllm, version: "0.33.0"}`,
			[]string{"names engine vllm 0.33.0", "this build runs vllm 0.32.0", "no inference-sim release runs it"}},
		{"a kind this build does not register", `engine: {kind: sglang, version: "0.5.23"}`,
			[]string{`names engine kind \"sglang\"`, "this build runs the kinds vllm"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := writeColocatedVariant(t, "engine:\n  kind: vllm\n  version: \"0.32.0\"\n", c.edit+"\n")
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
				"--rate", "4", "--num-requests", "4", "--seed", "1")
			if err == nil {
				t.Fatalf("a scenario stating %s ran", c.edit)
			}
			for _, w := range c.want {
				if !strings.Contains(stderr, w) {
					t.Errorf("refusal must contain %q; stderr:\n%s", w, lastLines(stderr, 3))
				}
			}
		})
	}
	t.Run("the closest accepted scenario runs", func(t *testing.T) {
		dir := writeColocatedVariant(t)
		stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--rate", "4", "--num-requests", "4", "--seed", "1")
		if err != nil || !completedIs(stdout, 4) {
			t.Fatalf("vllm 0.32.0 must run: %v\n%s", err, lastLines(stderr, 3))
		}
	})
}

// A 0.32.0 scenario runs blis-vllm's translated v0.32.0 engine through blis-schemas' engine.Engine
// contract: every request completes, and the latency kernel's 0.31.0 tuning, which prices the
// step, is disclosed on stderr.
func TestRunCmd_A032ScenarioRunsThroughTheEngineContract(t *testing.T) {
	dir := writeColocatedVariant(t)
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err != nil {
		t.Fatalf("a 0.32.0 scenario must run through the contract: %v\nstderr:\n%s", err, lastLines(stderr, 5))
	}
	if !completedIs(stdout, 4) {
		t.Errorf("want 4 completed requests; stdout:\n%s", stdout)
	}
	if !strings.Contains(stderr, "engine assumption: engine_version = 0.31.0") {
		t.Errorf("the kernel's 0.31.0 tuning must be disclosed; stderr:\n%s", lastLines(stderr, 6))
	}
}

// The Result records the engine the scenario selected, by kind and upstream release (SIM-1).
func TestRunCmd_TheResultRecordsTheSelectedEngine(t *testing.T) {
	{
		dir := writeColocatedVariant(t)
		stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--rate", "4", "--num-requests", "4", "--seed", "1")
		if err != nil {
			t.Fatalf("%v\n%s", err, lastLines(stderr, 3))
		}
		if e := parseResult(t, stdout).Engine; e.Kind != "vllm" || e.Version != "0.32.0" {
			t.Errorf("engine = %+v, want vllm 0.32.0", e)
		}
	}
}

// INV-13 for a contract-built release: a 0.32.0 run's exported trace, replayed with the same
// flags, reproduces the run byte for byte. blis replay must build the same engine through the
// engine contract.
func TestReplayCmd_A032ScenarioReproducesTheRun(t *testing.T) {
	dir := writeColocatedVariant(t)
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
	if inv13View(t, runOut) != inv13View(t, repOut) {
		t.Errorf("a 0.32.0 replay does not reproduce its run (INV-13):\nrun:\n%s\nreplay:\n%s",
			lastLines(runOut, 30), lastLines(repOut, 30))
	}
}

// A 0.32.0 run reports its prefix cache's hit rate: each request's first admission's lookup,
// from the engine (engine.Scheduled.PrefixLookup). On a workload whose requests share prefixes,
// the engine hits.
func TestRunCmd_A032ScenarioReportsItsPrefixCacheHitRate(t *testing.T) {
	dir := writeColocatedVariant(t)
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--num-requests", "60")
	if err != nil {
		t.Fatalf("%v\nstderr:\n%s", err, lastLines(stderr, 5))
	}
	if resultNum(t, stdout, "engine.cluster.cache_hit_rate_whole_run") == 0 {
		t.Fatal("the 0.32.0 engine's prefix cache reported no hits on a workload of shared prefixes")
	}
}

// --server-series-interval-s writes a 0.32.0 run's server_series block: the engine bins its own
// vllm:* series, and it counts prefix-cache hits on a workload of shared prefixes.
func TestRunCmd_A032ScenarioWritesItsServerSeries(t *testing.T) {
	dir := writeColocatedVariant(t)
	path := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--num-requests", "60",
		"--server-series-interval-s", "1", "--metrics-path", path)
	if err != nil {
		t.Fatalf("%v\nstderr:\n%s", err, lastLines(stderr, 5))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		ServerSeries *struct {
			Engines []struct {
				Bins         int       `json:"bins"`
				PromptTokens []float64 `json:"prompt_tokens"`
				Hits         []float64 `json:"prefix_cache_hits"`
			} `json:"engines"`
		} `json:"server_series"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if out.ServerSeries == nil || len(out.ServerSeries.Engines) != 1 {
		t.Fatalf("want one engine's server_series, got %+v", out.ServerSeries)
	}
	e := out.ServerSeries.Engines[0]
	var prompt, hits float64
	for i := range e.PromptTokens {
		prompt += e.PromptTokens[i]
		hits += e.Hits[i]
	}
	if e.Bins == 0 || prompt == 0 || hits == 0 {
		t.Fatalf("server_series has %d bins, %v prompt tokens and %v prefix-cache hits on a workload of shared prefixes",
			e.Bins, prompt, hits)
	}
}

// A 0.32.0 run keeps the host's API-server front end (INV-R1): requests that arrive together
// queue for the one render thread (vLLM's default, --renderer-num-workers 1) before they reach
// the engine, and --renderer-num-workers 0 turns the queue off.
func TestRunCmd_A032ScenarioRendersOnTheHostFrontEnd(t *testing.T) {
	dir := writeColocatedVariant(t)
	ttft := func(extra ...string) float64 {
		args := append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--concurrency", "128", "--num-requests", "128", "--prompt-tokens", "1024", "--output-tokens", "8",
			"--seed", "7"}, extra...)
		stdout, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("%v: %v\nstderr:\n%s", extra, err, lastLines(stderr, 5))
		}
		return metric(t, stdout, "ttft_mean_ms")
	}
	queued, free := ttft(), ttft("--renderer-num-workers", "0")
	if queued-free <= 0.05*free {
		t.Errorf("the render queue adds %.1f ms of mean TTFT to %.1f ms; the host front end must queue "+
			"requests that arrive together", queued-free, free)
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
	if !strings.Contains(body, "      max_model_len: 1048576\n") {
		t.Fatal("fixture states no max_model_len 1048576 to edit")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "auto.yaml"),
		[]byte(strings.Replace(body, "      max_model_len: 1048576\n", "      max_model_len: auto\n", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "auto.yaml",
		"--catalog", substratedata.MustCatalog(), "--speculative-acceptance-rate", "0.7",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err == nil {
		t.Fatal("a 0.32.0 deployment stating max_model_len auto ran")
	}
	if strings.Contains(stderr, "goroutine ") || strings.Contains(stderr, "panic:") {
		t.Errorf("the refusal must be a clean CLI error, not a panic; stderr:\n%s", lastLines(stderr, 8))
	}
	if !strings.Contains(stderr, "engine vllm 0.32.0: vllm: max_model_len must be stated") {
		t.Errorf("the refusal must be the engine's, naming the missing piece; stderr:\n%s", lastLines(stderr, 4))
	}
}

// The host does not carry request priority across the engine contract yet (it sends every
// request to the v0.32.0 engine at priority 0), so a priority policy would silently run as FCFS. A
// scenario stating engine.scheduling_policy priority is refused at the build site, naming the
// policy; the same scenario stating fcfs runs.
func TestRunCmd_A032ScenarioStatingPrioritySchedulingIsRefused(t *testing.T) {
	priority := func(policy string) []string {
		dir := writeColocatedVariant(t,
			"      max_model_len: 32768\n", "      max_model_len: 32768\n      scheduling_policy: "+policy+"\n")
		return []string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--rate", "4", "--num-requests", "4", "--seed", "1"}
	}
	_, stderr, err := runKernelCLI(t, priority("priority")...)
	if err == nil || !strings.Contains(stderr, "engine.scheduling_policy priority would run as FCFS") {
		t.Errorf("a scenario stating priority scheduling: want a refusal naming the policy, got err=%v\n%s",
			err, lastLines(stderr, 3))
	}
	if stdout, stderr, err := runKernelCLI(t, priority("fcfs")...); err != nil || !completedIs(stdout, 4) {
		t.Errorf("the same scenario stating fcfs must run: %v\n%s", err, lastLines(stderr, 3))
	}
}

// An SLO priority override (--policy-config priority.slo_priorities) re-ranks the SLO classes a
// priority policy orders by, which reaches the v0.32.0 engine as nothing: a run stating one is
// refused at the build site, naming the setting. The same run without the bundle runs.
func TestRunCmd_A032ScenarioRefusesAnSLOPriorityOverride(t *testing.T) {
	bundle := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(bundle, []byte("priority:\n  slo_priorities:\n    critical: 9\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	dir := writeColocatedVariant(t)
	run := []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--rate", "4", "--num-requests", "4", "--seed", "1"}
	_, stderr, err := runKernelCLI(t, append(run, "--policy-config", bundle)...)
	if err == nil || !strings.Contains(stderr, "priority.slo_priorities override would be ignored") {
		t.Errorf("a run with an SLO priority override: want a refusal naming the override, got err=%v\n%s",
			err, lastLines(stderr, 3))
	}
	if stdout, stderr, err := runKernelCLI(t, run...); err != nil || !completedIs(stdout, 4) {
		t.Errorf("the same run without the override must run: %v\n%s", err, lastLines(stderr, 3))
	}
}

// The v0.32.0 engine schedules by its scenario's engine block alone: a priority scheduler or
// preemption policy the run asks for on the command line reaches it as nothing, so a run asking
// for one is refused, naming it. The FCFS values of the same flags run.
func TestRunCmd_A032ScenarioRefusesAPrioritySchedulerFlag(t *testing.T) {
	dir := writeColocatedVariant(t)
	run := []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--rate", "4", "--num-requests", "4", "--seed", "1"}
	for _, c := range []struct{ refused, accepted []string }{
		{[]string{"--scheduler", "priority-fcfs"}, []string{"--scheduler", "fcfs"}},
		{[]string{"--preemption-policy", "priority"}, []string{"--preemption-policy", "fcfs"}},
	} {
		_, stderr, err := runKernelCLI(t, append(run, c.refused...)...)
		if err == nil || !strings.Contains(stderr, "does not carry request priority across it yet") ||
			!strings.Contains(stderr, "so scheduler") || !strings.Contains(stderr, c.refused[1]) {
			t.Errorf("a run with %v: want a refusal naming it, got err=%v\n%s", c.refused, err, lastLines(stderr, 3))
		}
		if stdout, stderr, err := runKernelCLI(t, append(run, c.accepted...)...); err != nil || !completedIs(stdout, 4) {
			t.Errorf("%v must run: %v\n%s", c.accepted, err, lastLines(stderr, 3))
		}
	}
}

// dp032Variant writes the dp 2 MoE scenario (dpScenario: minimax-m2.5 on B200, TP2 x DP2 with
// expert parallelism), with extra engine lines, into a fresh scenario directory as v032-dp2.yaml.
func dp032Variant(t *testing.T, extra string) string {
	t.Helper()
	scenarios, _, _ := kernelRepos(t)
	body := readFile(t, filepath.Join(scenarios, dpScenario))
	const anchor = "      max_model_len: 32768\n"
	if !strings.Contains(body, anchor) {
		t.Fatalf("fixture %s has no %q to edit", dpScenario, anchor)
	}
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
	if !completedIs(stdout, 8) {
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
// 123-130@5df63a72), as sim.ReplicaAPIServers computes. With one
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
		return metric(t, stdout, "ttft_mean_ms")
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

// A scenario that names its engine with the deprecated engine_version is accepted until M4
// (release plan §3.3) with a warning naming engine: {kind, version}; the fixture states
// engine: {kind, version}, so this variant states the old field.
func TestRunCmd_EngineVersionDrawsADeprecationWarning(t *testing.T) {
	dir := writeColocatedVariant(t, "engine:\n  kind: vllm\n  version: \"0.32.0\"\n", "engine_version: \"0.32.0\"\n")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--rate", "4", "--num-requests", "4", "--seed", "1")
	if err != nil {
		t.Fatalf("a scenario stating engine_version must still run: %v\n%s", err, lastLines(stderr, 5))
	}
	if !strings.Contains(stderr, `state engine: {kind: vllm, version: \"0.32.0\"} instead`) {
		t.Errorf("engine_version must draw a deprecation warning naming engine: {kind, version}; stderr:\n%s",
			lastLines(stderr, 6))
	}
}
