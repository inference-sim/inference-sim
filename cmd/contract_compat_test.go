package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// contractCompatCases are the byte gate's single-instance cases (WS/tools/bytegate.cases):
// scenario directory, scenario and flags as the gate runs them. The two KV-pressure scenarios
// are vendored under testdata/contract-compat/scenarios, and every case's reference stdout under
// testdata/contract-compat, from the gate's reference (main 67f8017c).
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
}

const kvPressureLengths = "--prompt-tokens 6000 --prompt-tokens-stdev 1500 --prompt-tokens-max 12000 " +
	"--output-tokens 1500 --output-tokens-stdev 400 --output-tokens-max 3000"

// Phase 2c's first stage: a 0.29.0 or 0.31.0 run of one colocated instance, built through
// blis-vllm's compat/contract (the compatibility engine behind engine.Engine) and driven by
// sim.EngineDriver under ContractCompatEnv, prints exactly what the compatibility engine's own
// host loop prints: each of the byte gate's single-instance cases, stdout byte for byte against
// the gate's reference. The switched run must not be the old loop in disguise, so it must log
// none of the old loop's step events.
func TestContractCompatIsByteIdentical(t *testing.T) {
	for _, c := range contractCompatCases {
		t.Run(c.name, func(t *testing.T) {
			ref, err := os.ReadFile(filepath.Join("..", "testdata", "contract-compat", c.name+".stdout"))
			if err != nil {
				t.Fatal(err)
			}
			args := append([]string{"run", "--scenarios", filepath.Join("..", c.dir), "--scenario", c.scenario,
				"--log", "debug"}, strings.Fields(c.flags)...)
			t.Setenv(ContractCompatEnv, "1")
			got, gotErr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
			}
			if strings.Contains(gotErr, "<< StepEvent") {
				t.Fatal("under the switch the compatibility engine's own host loop ran; the comparison is vacuous")
			}
			if got != string(ref) {
				t.Errorf("the contract path's stdout differs from the byte gate's reference:\ncontract:\n%s\nreference:\n%s",
					lastLines(got, 30), lastLines(string(ref), 30))
			}
		})
	}
}

// Without the switch, the compatibility engine's own host loop runs, as before.
func TestContractCompatIsOffByDefault(t *testing.T) {
	t.Setenv(ContractCompatEnv, "")
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", filepath.Join("..", "examplescenarios"),
		"--scenario", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml", "--rate", "8", "--num-requests", "8", "--seed", "3",
		"--log", "debug")
	if err != nil {
		t.Fatalf("%v\nstderr:\n%s", err, lastLines(stderr, 5))
	}
	if !strings.Contains(stderr, "<< StepEvent") {
		t.Error("without the switch the compatibility engine's host loop must run")
	}
}
