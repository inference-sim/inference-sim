package cmd

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// contractCompatCases are the byte gate's cases the contract path carries (WS/tools/bytegate.cases):
// scenario directory, scenario and flags as the gate runs them. The two KV-pressure scenarios and
// the tight-decode P/D scenario are vendored under testdata/contract-compat/scenarios.
var contractCompatCases = []struct {
	name, dir, scenario, flags string
}{
	{"dsv41-tp8-dspark5", "examplescenarios", "deepseek-v4.1-flash-h100-tp8-dspark5.yaml", "--rate 4 --num-requests 60 --seed 7"},
	{"gptoss-tp2", "examplescenarios", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml", "--rate 8 --num-requests 80 --seed 3"},
	{"gptoss-tp2-2node", "examplescenarios", "gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml", "--rate 8 --num-requests 80 --seed 3"},
	{"dsv4pro-mtp2", "examplescenarios", "deepseek-v4-pro-b300-fp4-vllm-tp4-mtp2.yaml", "--rate 2 --num-requests 40 --seed 9"},
	{"td-dsv41-1m", "testdata/scenarios", "deepseek-v4.1-flash-h200-tp8-dspark5-1m.yaml", "--rate 2 --num-requests 30 --seed 4 --speculative-acceptance-length 3.2"},
	{"td-glm5-mtp3", "testdata/scenarios", "glm-5-h200-tp8-mtp3.yaml", "--rate 3 --num-requests 40 --seed 6 --speculative-acceptance-length 2.4"},
	{"td-llama-4node", "testdata/scenarios", "llama-3.1-70b-instruct-h200-tp4-4node.yaml", "--rate 6 --num-requests 80 --seed 8"},
	{"llama-preempt", "testdata/scenarios", "llama-3.1-70b-instruct-h200-tp4-4node.yaml", "--rate 60 --num-requests 300 --seed 21"},
	{"kvpressure", "testdata/contract-compat/scenarios", "llama-70b-kvpressure.yaml", "--rate 40 --num-requests 300 --seed 31 " + kvPressureLengths},
	{"kvpressure-chunk", "testdata/contract-compat/scenarios", "llama-70b-kvpressure-chunk.yaml", "--rate 40 --num-requests 300 --seed 32 " + kvPressureLengths},
	{"kvpressure-priority", "testdata/contract-compat/scenarios", "llama-70b-kvpressure.yaml", "--rate 40 --num-requests 300 --seed 33 " + kvPressureLengths + " --scheduler priority-fcfs --preemption-policy priority"},
	{"gptoss-4inst", "examplescenarios", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml", "--rate 20 --num-requests 200 --seed 13 --num-instances 4"},
	// P/D: the prefill engine's KV hold and the decode engine's handoffs through the contract
	// (Request.RemoteDecode and RemotePrefill, Result.Handoffs), with a CPU tier on both pools,
	// three prefill pools, and a decode engine that preempts a remote-prefilled request.
	{"td-dsv41-1p1d", "testdata/scenarios", "deepseek-v4.1-flash-h100-1p1d-dspark5.yaml", "--rate 2 --num-requests 40 --seed 1 --num-instances 2 --prefill-instances 1 --decode-instances 1 --speculative-acceptance-length 3.2"},
	{"td-dsv41-1p1d-cpu", "testdata/scenarios", "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml", "--rate 2 --num-requests 40 --seed 1 --num-instances 2 --prefill-instances 1 --decode-instances 1 --speculative-acceptance-length 3.2"},
	{"glm5-3p1d", "examplescenarios", "glm-5-h200-3p1d-ib.yaml", "--rate 2 --num-requests 40 --seed 11 --num-instances 4 --prefill-instances 3 --decode-instances 1"},
	{"pd-preempt-park", "testdata/contract-compat/scenarios", "glm-5-h200-3p1d-tightdecode.yaml", "--rate 1000 --num-requests 5 --seed 1 --num-instances 4 --prefill-instances 3 --decode-instances 1 " +
		"--prompt-tokens 4287 --prompt-tokens-stdev 0 --prompt-tokens-max 4287 --output-tokens 200 --output-tokens-stdev 0 --output-tokens-max 200 --timeout -1"},
	// Data parallelism: a replica's ranks step as one compatibility lockstep group behind the
	// contract (blis-vllm compat/contract.DPGroup), each rank its own instance: under vLLM's
	// external DP load balancer (dep8), and as the ranks of P/D pools (dp2).
	{"td-dsv41-dep8", "testdata/scenarios", "deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml", "--rate 2 --num-requests 40 --seed 2 --speculative-acceptance-length 3.2"},
	{"td-minimax-dp2", "testdata/scenarios", "minimax-m2.5-gb200-1p1d-dp2.yaml", "--rate 2 --num-requests 40 --seed 10 --num-instances 2 --prefill-instances 1 --decode-instances 1"},
}

const kvPressureLengths = "--prompt-tokens 6000 --prompt-tokens-stdev 1500 --prompt-tokens-max 12000 " +
	"--output-tokens 1500 --output-tokens-stdev 400 --output-tokens-max 3000"

// Phase 2c: a 0.29.0 or 0.31.0 run built through blis-vllm's compat/contract (the compatibility
// engine behind engine.Engine) and driven by sim.EngineDriver prints exactly what the
// compatibility engine's own host loop prints, for each of the byte gate's colocated cases:
// stdout byte for byte, both paths run by this binary on this machine. The two paths are compared
// with each other rather than with a stored file, because a stored reference is one machine's
// floating point: an x86-64 build rounds some percentiles in the last bit differently from the
// arm64 one the byte gate's reference was made on, on both paths alike. The byte gate (main
// against this branch, on one machine) pins the outputs themselves. The switched run must not be
// the old loop in disguise, so it must log none of the old loop's step events, and the old loop's
// must.
func TestContractCompatIsByteIdentical(t *testing.T) {
	for _, c := range contractCompatCases {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"run", "--scenarios", filepath.Join("..", c.dir), "--scenario", c.scenario,
				"--log", "debug"}, strings.Fields(c.flags)...)
			t.Setenv(ContractCompatEnv, "0")
			want, wantErr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the compatibility host loop: %v\nstderr:\n%s", err, lastLines(wantErr, 5))
			}
			if !strings.Contains(wantErr, "<< StepEvent") {
				t.Fatal("with the opt-out the compatibility host loop logged no step events; the comparison is vacuous")
			}
			t.Setenv(ContractCompatEnv, "")
			got, gotErr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
			}
			if strings.Contains(gotErr, "<< StepEvent") {
				t.Fatal("by default the compatibility engine's own host loop ran; the comparison is vacuous")
			}
			if got != want {
				t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
					lastLines(got, 30), lastLines(want, 30))
			}
		})
	}
}

// The router-driven paths the byte gate does not exercise, each run both ways: the precise-prefix
// producer (with speculative indexing, and without it, when routing reads only the engines' KV
// events), and in-flight eviction of a sheddable request.
func TestContractCompatRouterPathsAreByteIdentical(t *testing.T) {
	precise := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: precise.yaml\n", "",
		map[string]string{"precise.yaml": "testdata/epp/precise.yaml"})
	events := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: precise-events.yaml\n", "",
		map[string]string{"precise-events.yaml": "testdata/epp/precise-events.yaml"})
	evict := colocatedRouterVariant(t, evictionRouter("eviction.yaml"), "      app: vllm\n", evictionFiles("eviction.yaml"))
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"precise-prefix", []string{"run", "--scenarios", precise, "--scenario", "variant.yaml", "--num-instances", "4",
			"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--kv-event-delay", "3000"}, ""},
		{"precise-prefix-from-kv-events", []string{"run", "--scenarios", events, "--scenario", "variant.yaml", "--num-instances", "4",
			"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--kv-event-delay", "3000"}, ""},
		{"in-flight-eviction", []string{"run", "--scenarios", evict, "--scenario", "variant.yaml", "--num-instances", "2",
			"--workload-spec", "testdata/eviction-workload.yaml", "--seed", "1", noRenderQueue[0], noRenderQueue[1]},
			"Gateway Evicted (in-flight): "},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv(ContractCompatEnv, "0")
			want, wantErr, err := runKernelCLI(t, c.args...)
			if err != nil {
				t.Fatalf("the compatibility host loop: %v\nstderr:\n%s", err, lastLines(wantErr, 5))
			}
			if c.want != "" && !strings.Contains(want, c.want) {
				t.Fatalf("the run does not exercise the path (no %q)", c.want)
			}
			t.Setenv(ContractCompatEnv, "")
			got, gotErr, err := runKernelCLI(t, c.args...)
			if err != nil {
				t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
			}
			if got != want {
				t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
					lastLines(got, 40), lastLines(want, 40))
			}
		})
	}
}

// A KV offload tier, run both ways: prefix groups whose shared prefixes outgrow the GPU prefix
// cache, so a recurring prefix is found again on the CPU tier (--kv-cpu-blocks). The CPU tier's
// part of the admissions' prefix-cache hits reaches the report (cpu_tier_hit_rate), as do the
// offload store's thrash counts (KV Thrashing Rate on stdout).
func TestContractCompatOffloadIsByteIdentical(t *testing.T) {
	args := []string{"run", "--scenarios", filepath.Join("..", "testdata", "contract-compat", "scenarios"),
		"--scenario", "llama-70b-kvpressure.yaml", "--workload-spec", writePrefixPressureSpec(t, 200, 400),
		"--seed", "3", "--kv-cpu-blocks", "200000", "--log", "debug"}
	run := func(env string) (stdout, metrics, stderr string) {
		t.Helper()
		t.Setenv(ContractCompatEnv, env)
		path := filepath.Join(t.TempDir(), "metrics.json")
		stdout, stderr, err := runKernelCLI(t, append(args, "--metrics-path", path)...)
		if err != nil {
			t.Fatalf("ContractCompatEnv=%q: %v\nstderr:\n%s", env, err, lastLines(stderr, 5))
		}
		return stdout, readFile(t, path), stderr
	}
	want, wantMetrics, wantErr := run("0")
	if !strings.Contains(wantErr, "<< StepEvent") {
		t.Fatal("with the opt-out the compatibility host loop logged no step events; the comparison is vacuous")
	}
	got, gotMetrics, gotErr := run("")
	if strings.Contains(gotErr, "<< StepEvent") {
		t.Fatal("by default the compatibility engine's own host loop ran; the comparison is vacuous")
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(wantMetrics), &m); err != nil {
		t.Fatal(err)
	}
	if cpu, _ := m["cpu_tier_hit_rate"].(float64); cpu <= 0 {
		t.Fatalf("no CPU-tier hit (cpu_tier_hit_rate %v); the comparison is vacuous", m["cpu_tier_hit_rate"])
	}
	if strings.Contains(want, "KV Thrashing Rate: 0.0000") {
		t.Fatal("the offload store recorded no thrashing; the comparison is vacuous")
	}
	if got != want {
		t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
			lastLines(got, 30), lastLines(want, 30))
	}
	var gm map[string]any
	if err := json.Unmarshal([]byte(gotMetrics), &gm); err != nil {
		t.Fatal(err)
	}
	if gm["cpu_tier_hit_rate"] != m["cpu_tier_hit_rate"] {
		t.Errorf("cpu_tier_hit_rate: contract %v, host loop %v", gm["cpu_tier_hit_rate"], m["cpu_tier_hit_rate"])
	}
}

// The --metrics-path file, run both ways with the server time series on: every byte-gate case the
// contract carries writes the same file, byte for byte, as the compatibility host loop: its
// server_series block (the engine's own series, through the contract), the in-flight request
// mean and maximum, and each request's completion index (both of which follow the instant a
// step's results reach the cluster: its start, as on the host loop).
func TestContractCompatMetricsFileIsByteIdentical(t *testing.T) {
	for _, c := range contractCompatCases {
		t.Run(c.name, func(t *testing.T) {
			run := func(env string) (stdout, metrics string) {
				t.Helper()
				t.Setenv(ContractCompatEnv, env)
				path := filepath.Join(t.TempDir(), "metrics.json")
				args := append([]string{"run", "--scenarios", filepath.Join("..", c.dir), "--scenario", c.scenario,
					"--server-series-interval-s", "1", "--metrics-path", path}, strings.Fields(c.flags)...)
				stdout, stderr, err := runKernelCLI(t, args...)
				if err != nil {
					t.Fatalf("ContractCompatEnv=%q: %v\nstderr:\n%s", env, err, lastLines(stderr, 5))
				}
				return stdout, readFile(t, path)
			}
			want, wantMetrics := run("0")
			got, gotMetrics := run("")
			var m struct {
				InFlightMean float64 `json:"in_flight_requests_mean"`
				Series       *struct {
					Engines []struct {
						Bins int `json:"bins"`
					} `json:"engines"`
				} `json:"server_series"`
			}
			if err := json.Unmarshal([]byte(gotMetrics), &m); err != nil {
				t.Fatal(err)
			}
			if m.Series == nil || len(m.Series.Engines) == 0 || m.Series.Engines[0].Bins == 0 {
				t.Fatal("the contract path wrote no server_series; the comparison is vacuous")
			}
			if m.InFlightMean <= 0 {
				t.Fatal("no request was in flight; the comparison is vacuous")
			}
			if got != want {
				t.Errorf("the contract path's stdout differs from the compatibility host loop's")
			}
			if gotMetrics != wantMetrics {
				t.Errorf("the contract path's metrics file differs from the compatibility host loop's:\n%s",
					firstDifference(gotMetrics, wantMetrics))
			}
		})
	}
}

// firstDifference shows where two files first differ, with a little context.
func firstDifference(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	from := max(i-200, 0)
	return fmt.Sprintf("contract: ...%s\nhost loop: ...%s", a[from:min(i+200, len(a))], b[from:min(i+200, len(b))])
}

// LoRA adapters, run both ways: two adapters over two resident slots and KV pressure, so the
// run cold-loads adapters through the engine's load gate and preempts. The cold loads reach the
// report (each adapter's load_count in --metrics-path), as the run's output does.
func TestContractCompatLoRAIsByteIdentical(t *testing.T) {
	dir, lora, spec := writeLoRAPressureFixtures(t)
	args := []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--lora-config", lora,
		"--workload-spec", spec, "--seed", "5", "--horizon", "6000000000", "--log", "debug"}
	run := func(env string) (stdout, metrics, stderr string) {
		t.Helper()
		t.Setenv(ContractCompatEnv, env)
		path := filepath.Join(t.TempDir(), "metrics.json")
		stdout, stderr, err := runKernelCLI(t, append(args, "--metrics-path", path)...)
		if err != nil {
			t.Fatalf("ContractCompatEnv=%q: %v\nstderr:\n%s", env, err, lastLines(stderr, 5))
		}
		return stdout, readFile(t, path), stderr
	}
	want, wantMetrics, wantErr := run("0")
	if !strings.Contains(wantErr, "<< StepEvent") {
		t.Fatal("with the opt-out the compatibility host loop logged no step events; the comparison is vacuous")
	}
	if !strings.Contains(wantErr, "<< AdapterLoadCompletion") {
		t.Fatal("the run loaded no adapter; the comparison is vacuous")
	}
	got, gotMetrics, gotErr := run("")
	if strings.Contains(gotErr, "<< StepEvent") {
		t.Fatal("by default the compatibility engine's own host loop ran; the comparison is vacuous")
	}
	if got != want {
		t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
			lastLines(got, 30), lastLines(want, 30))
	}
	loads := func(metrics string) map[string]float64 {
		t.Helper()
		var m struct {
			Adapters map[string]struct {
				LoadCount float64 `json:"load_count"`
			} `json:"adapters"`
		}
		if err := json.Unmarshal([]byte(metrics), &m); err != nil {
			t.Fatal(err)
		}
		out := map[string]float64{}
		for id, a := range m.Adapters {
			out[id] = a.LoadCount
		}
		return out
	}
	wantLoads, gotLoads := loads(wantMetrics), loads(gotMetrics)
	if wantLoads["a0"] <= 0 || wantLoads["a1"] <= 0 {
		t.Fatalf("the host loop's report counts no cold load of each adapter (%v); the comparison is vacuous", wantLoads)
	}
	if !maps.Equal(gotLoads, wantLoads) {
		t.Errorf("adapter loads: contract %v, host loop %v", gotLoads, wantLoads)
	}
}

// lora-affinity routing over four instances, run both ways: the router reads the adapters each
// engine publishes (vllm:lora_requests_info: its slots, and the adapters with a running or
// waiting request), so its picks, and the cold loads they cause, follow the engines' LoRA view.
// Six adapters over two slots each make the instances load and evict adapters.
func TestContractCompatLoRAAffinityRoutingIsByteIdentical(t *testing.T) {
	dir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: lora.yaml\n", "", nil)
	if err := os.WriteFile(filepath.Join(dir, "lora.yaml"), []byte("apiVersion: llm-d.ai/v1\n"+
		"kind: EndpointPickerConfig\nplugins:\n- type: lora-affinity-scorer\n- type: max-score-picker\n"+
		"- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n"+
		"  - pluginRef: lora-affinity-scorer\n  - pluginRef: max-score-picker\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	adapters := []string{"a0", "a1", "a2", "a3", "a4", "a5"}
	var lora, spec strings.Builder
	lora.WriteString("lora:\n  adapter_capacity: 2\n  adapters:\n")
	spec.WriteString("version: \"2\"\ncategory: language\naggregate_rate: 30.0\nnum_requests: 240\nclients:\n")
	for i, a := range adapters {
		fmt.Fprintf(&lora, "    - id: %s\n      rank: 16\n", a)
		fmt.Fprintf(&spec, "  - id: c%d\n    tenant_id: t\n    slo_class: batch\n    adapter: %s\n    rate_fraction: %d\n"+
			"    arrival:\n      process: poisson\n"+
			"    input_distribution:\n      type: constant\n      params: { value: 512 }\n"+
			"    output_distribution:\n      type: constant\n      params: { value: 64 }\n", i, a, len(adapters)-i)
	}
	files := t.TempDir()
	loraPath, specPath := filepath.Join(files, "lora.yaml"), filepath.Join(files, "spec.yaml")
	for p, body := range map[string]string{loraPath: lora.String(), specPath: spec.String()} {
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	args := []string{"run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4",
		"--lora-config", loraPath, "--workload-spec", specPath, "--seed", "7", "--log", "debug"}
	t.Setenv(ContractCompatEnv, "0")
	want, wantErr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("the compatibility host loop: %v\nstderr:\n%s", err, lastLines(wantErr, 5))
	}
	if !strings.Contains(wantErr, "<< AdapterLoadCompletion") {
		t.Fatal("the run loaded no adapter; the comparison is vacuous")
	}
	t.Setenv(ContractCompatEnv, "")
	got, gotErr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
	}
	if strings.Contains(gotErr, "<< StepEvent") {
		t.Fatal("by default the compatibility engine's own host loop ran; the comparison is vacuous")
	}
	if got != want {
		t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
			lastLines(got, 40), lastLines(want, 40))
	}
}

// The contract path is the default for the shapes it carries; ContractCompatEnv=0 keeps the
// compatibility engine's own host loop.
func TestContractCompatIsTheDefaultPath(t *testing.T) {
	run := func(env string, extra ...string) string {
		t.Helper()
		t.Setenv(ContractCompatEnv, env)
		args := append([]string{"run", "--scenarios", filepath.Join("..", "examplescenarios"),
			"--scenario", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml", "--rate", "8", "--num-requests", "8", "--seed", "3",
			"--log", "debug"}, extra...)
		_, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("%v\nstderr:\n%s", err, lastLines(stderr, 5))
		}
		return stderr
	}
	if strings.Contains(run(""), "<< StepEvent") {
		t.Error("by default a single-instance 0.29.0 run must go through the engine contract")
	}
	if !strings.Contains(run("0"), "<< StepEvent") {
		t.Error("with the opt-out the compatibility engine's own host loop must run")
	}
	if strings.Contains(run("", "--num-instances", "4"), "<< StepEvent") {
		t.Error("colocated instances behind the router are carried by the contract")
	}
}

// A data-parallel replica behind vLLM's internal or hybrid balancer is one endpoint whose
// group picks each request's rank (the engine's PickRank, from the ranks' counts and the front
// end's in-flight count on each) and whose ranks render on one shared stage. Each mode runs both
// ways, byte for byte, and the pick spreads requests over both ranks of each replica. A third run
// routes on the replicas' load: the router scores each replica endpoint by its ranks' waiting
// requests and KV usage, summed and maxed over the ranks' published gauges.
func TestContractCompatInternalDPBalancerIsByteIdentical(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	loadRouted := routerVariant(t, filepath.Join(scenarios, dpScenario),
		"router:\n  endpoint_picker_config: load-only.yaml\n", map[string]string{"load-only.yaml": "../examples/llm-d/load-only.yaml"})
	for _, c := range []struct{ name, mode, dir, scenario string }{
		{"internal", "internal", "", dpScenario},
		{"hybrid", "hybrid", "", dpScenario},
		{"internal-load-routed", "internal", loadRouted, "variant.yaml"},
	} {
		t.Run(c.name, func(t *testing.T) {
			// Two logical instances of the dp=2 scenario: two replicas of two ranks each, under a
			// load whose requests queue for the replica's shared render stage, so the stage and the
			// pick's in-flight counts both shape the run.
			args := []string{"run", "--scenario", c.scenario, "--num-instances", "2", "--rate", "400",
				"--num-requests", "160", "--seed", "42", "--dp-load-balancing", c.mode, "--log", "debug"}
			if c.dir != "" {
				args = append(args, "--scenarios", c.dir)
			}
			t.Setenv(ContractCompatEnv, "0")
			want, wantErr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the compatibility host loop: %v\nstderr:\n%s", err, lastLines(wantErr, 5))
			}
			if !strings.Contains(wantErr, "<< StepEvent") {
				t.Fatal("with the opt-out the compatibility host loop logged no step events; the comparison is vacuous")
			}
			t.Setenv(ContractCompatEnv, "")
			got, gotErr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
			}
			if strings.Contains(gotErr, "<< StepEvent") {
				t.Fatal("by default the compatibility engine's own host loop ran; the comparison is vacuous")
			}
			if got != want {
				t.Errorf("the contract path's stdout differs from the compatibility host loop's:\ncontract:\n%s\nhost loop:\n%s",
					lastLines(got, 40), lastLines(want, 40))
			}
			served := 0
			for _, raw := range extractJSONObjects(got) {
				var obj map[string]any
				if json.Unmarshal([]byte(raw), &obj) != nil || obj["instance_id"] == "cluster" {
					continue
				}
				if n, ok := obj["completed_requests"].(float64); ok && n > 0 {
					served++
				}
			}
			if served < 4 {
				t.Errorf("%d of the 4 ranks completed requests; the replica's pick must spread them", served)
			}
		})
	}
}
