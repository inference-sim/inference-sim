package cmd

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// contractCompatCases are the byte gate's colocated cases (WS/tools/bytegate.cases):
// scenario directory, scenario and flags as the gate runs them. The two KV-pressure scenarios
// are vendored under testdata/contract-compat/scenarios.
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

// The contract path is the default for the shapes it carries; ContractCompatEnv=0 keeps the
// compatibility engine's own host loop, and a shape the contract does not carry yet (here
// P/D pools) keeps it regardless.
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
