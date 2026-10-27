package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Phase 2c's first stage: a 0.29.0 run of one colocated instance, built through blis-vllm's
// compat/contract (the compatibility engine behind engine.Engine) and driven by sim.EngineDriver
// under ContractCompatEnv, prints exactly what the compatibility engine's own host loop prints:
// the byte gate's gptoss-tp2 case, stdout byte for byte. The switched run must not be the old
// loop in disguise, so it must not log the old loop's step events.
func TestContractCompatIsByteIdentical(t *testing.T) {
	const scenario = "gpt-oss-120b-h100-fp4-vllm-tp2.yaml"
	args := []string{"run", "--scenarios", filepath.Join("..", "examplescenarios"), "--scenario", scenario,
		"--rate", "8", "--num-requests", "80", "--seed", "3", "--log", "debug"}
	t.Setenv(ContractCompatEnv, "")
	want, wantErr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("the compatibility path: %v\nstderr:\n%s", err, lastLines(wantErr, 5))
	}
	if !strings.Contains(wantErr, "<< StepEvent") {
		t.Fatal("the compatibility path logged no step events; the comparison is vacuous")
	}
	t.Setenv(ContractCompatEnv, "1")
	got, gotErr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("the contract path: %v\nstderr:\n%s", err, lastLines(gotErr, 5))
	}
	if strings.Contains(gotErr, "<< StepEvent") {
		t.Error("under the switch the compatibility engine's own host loop still ran")
	}
	if got != want {
		t.Errorf("the contract path's stdout differs from the compatibility path's:\ncontract:\n%s\ncompat:\n%s",
			lastLines(got, 30), lastLines(want, 30))
	}
	if ref, err := os.ReadFile(filepath.Join("..", "testdata", "contract-compat", "gptoss-tp2.stdout")); err == nil && string(ref) != got {
		t.Error("the contract path's stdout differs from the byte gate's reference for gptoss-tp2")
	}
}
