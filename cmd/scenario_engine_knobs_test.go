package cmd

import (
	"errors"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The scenario is the source of engine.long_prefill_token_threshold (blis-schemas#75) and
// parallel.data_parallel_load_balancing (#76), as it is of the offload tiers (#1910): a
// scenario stating a value runs byte-identically to the unstated scenario with the flag set
// to it, an agreeing flag changes nothing, and a disagreeing flag is refused naming both.
func TestRunCmd_ScenarioEngineKnobsAreTheSourceAndFlagsOnlyRestate(t *testing.T) {
	for _, tc := range []struct {
		name, scenario, marker, stated, flag, value, other string
		args                                               []string
	}{
		{"long_prefill_token_threshold", colocatedFixture, "      max_num_seqs: 256\n",
			"      max_num_seqs: 256\n      long_prefill_token_threshold: 256\n",
			"--long-prefill-token-threshold", "256", "512",
			[]string{"--num-requests", "20", "--rate", "8", "--prompt-tokens", "1500", "--seed", "5"}},
		{"data_parallel_load_balancing", dpScenario, "      dp: 2\n",
			"      dp: 2\n      data_parallel_load_balancing: external\n",
			"--dp-load-balancing", "external", "internal",
			[]string{"--num-instances", "2", "--rate", "10", "--num-requests", "40", "--seed", "42"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stated := capVariant(t, tc.scenario, tc.marker, tc.stated)
			unstated := capVariant(t, tc.scenario)
			run := func(dir string, extra ...string) (string, string, error) {
				args := append(append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml"}, tc.args...), extra...)
				return runKernelCLI(t, args...)
			}
			fromScenario, stderr, err := run(stated)
			if err != nil {
				t.Fatalf("scenario-stated run: %v\n%s", err, lastLines(stderr, 3))
			}
			fromFlag, stderr, err := run(unstated, tc.flag, tc.value)
			if err != nil {
				t.Fatalf("flag run: %v\n%s", err, lastLines(stderr, 3))
			}
			if fromScenario != fromFlag {
				t.Errorf("the scenario's %s and the equivalent %s differ", tc.name, tc.flag)
			}
			if plain, _, _ := run(unstated); plain == fromScenario {
				t.Errorf("non-vacuous: %s %s changed nothing on this fixture", tc.name, tc.value)
			}
			if both, stderr, err := run(stated, tc.flag, tc.value); err != nil || both != fromScenario {
				t.Errorf("an agreeing %s must change nothing: %v\n%s", tc.flag, err, lastLines(stderr, 3))
			}
			_, stderr, err = run(stated, tc.flag, tc.other)
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || !strings.Contains(stderr, "disagrees with the scenario's") ||
				!strings.Contains(stderr, tc.flag) || !strings.Contains(stderr, tc.name) {
				t.Errorf("a disagreeing %s must be refused naming the flag and the field: %v\n%s",
					tc.flag, err, lastLines(stderr, 3))
			}
		})
	}
}

// BLIS applies one threshold and one balancing mode to the whole run: pools stating
// different values are refused, and unstated pools defer to the stating ones.
func TestOnePoolValue_PoolsMustAgree(t *testing.T) {
	if v, err := onePoolValue([]int{0, 256, 256}, 0, "f"); err != nil || v != 256 {
		t.Errorf("agreeing pools: %v, %v", v, err)
	}
	if v, err := onePoolValue([]int{0, 0}, 0, "f"); err != nil || v != 0 {
		t.Errorf("unstated pools: %v, %v", v, err)
	}
	if _, err := onePoolValue([]deployment.DPLoadBalancing{deployment.DPLBExternal, deployment.DPLBHybrid}, "", "f"); err == nil {
		t.Error("pools stating different modes must be refused")
	}
}
