package cmd

import (
	"errors"
	"os/exec"
	"strings"
	"testing"

	"os"
	"path/filepath"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// The scenario is the source of parallel.data_parallel_load_balancing (#76), as it is of the
// offload tiers (#1910): a scenario stating a value runs byte-identically to the unstated
// scenario with the flag set to it, an agreeing flag changes nothing, and a disagreeing flag is
// refused naming both. engine.long_prefill_token_threshold has no flag at all (#2021 B6):
// TestRunCmd_LongPrefillThresholdHasNoFlag.
func TestRunCmd_ScenarioEngineKnobsAreTheSourceAndFlagsOnlyRestate(t *testing.T) {
	for _, tc := range []struct {
		name, scenario, marker, stated, flag, value, other string
		args                                               []string
	}{
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

// pdKnobVariant writes the dp=2 P/D fixture with extra lines appended to the prefill and the
// decode pool's engine (eng) and parallel (par) blocks.
func pdKnobVariant(t *testing.T, prefillEng, decodeEng, prefillPar, decodePar string) string {
	t.Helper()
	body := readFile(t, filepath.Join(pdScenarios, pdDPScenario))
	at := strings.Index(body, "  - role: decode\n")
	if at < 0 {
		t.Fatal("fixture has no decode pool")
	}
	edit := func(pool, eng, par string) string {
		pool = strings.Replace(pool, "      max_num_batched_tokens: 8192\n", "      max_num_batched_tokens: 8192\n"+eng, 1)
		return strings.Replace(pool, "      dp: 2\n", "      dp: 2\n"+par, 1)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"),
		[]byte(edit(body[:at], prefillEng, prefillPar)+edit(body[at:], decodeEng, decodePar)), 0o644); err != nil {
		t.Fatal(err)
	}
	return withPDRouter(t, dir)
}

// Each P/D pool runs its own engine.long_prefill_token_threshold (blis-schemas#75), as vLLM
// sets it per engine: a threshold stated on the prefill pool alone chunks the prefills, one
// stated on the decode pool alone reaches no prefill, and pools may state different values. The prompts are long enough to overlap in the
// prefill scheduler: vLLM applies the threshold only when another request is there (#57951).
func TestRunCmd_PDPoolsRunTheirOwnLongPrefillThreshold(t *testing.T) {
	const threshold = "      long_prefill_token_threshold: 256\n"
	run := func(dir string, extra ...string) (string, string, error) {
		return runKernelCLI(t, append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1",
			"--seed", "11", "--horizon", "600000000", "--num-requests", "20", "--rate", "100000", "--prompt-tokens", "16000",
			"--prompt-tokens-stdev", "0", "--prompt-tokens-min", "16000", "--prompt-tokens-max", "16000"},
			extra...)...)
	}
	must := func(out, stderr string, err error) string {
		t.Helper()
		if err != nil {
			t.Fatalf("%v\n%s", err, lastLines(stderr, 3))
		}
		return out
	}
	unstated := must(run(pdKnobVariant(t, "", "", "", "")))
	prefillOnly := must(run(pdKnobVariant(t, threshold, "", "", "")))
	if prefillOnly == unstated {
		t.Fatal("non-vacuous: a 256-token prefill threshold changed nothing on concurrent 16000-token prompts")
	}
	if decodeOnly := must(run(pdKnobVariant(t, "", threshold, "", ""))); decodeOnly != unstated {
		t.Error("a threshold stated on the decode pool alone changed the prefills")
	}
	differing := pdKnobVariant(t, threshold, "      long_prefill_token_threshold: 512\n", "", "")
	if out := must(run(differing)); out != prefillOnly {
		t.Error("the prefill pool's 256 must decide the prefills whatever the decode pool states")
	}
}

// engine.long_prefill_token_threshold comes from the scenario only (#2021 B6): run and replay
// have no flag for it, so the CLI cannot run an engine its scenario does not describe.
func TestRunCmd_LongPrefillThresholdHasNoFlag(t *testing.T) {
	for _, c := range []*cobra.Command{runCmd, replayCmd} {
		if f := c.Flags().Lookup("long-prefill-token-threshold"); f != nil {
			t.Errorf("%s still registers --long-prefill-token-threshold", c.Name())
		}
	}
	_, stderr, err := runKernelCLI(t, "run", "--scenario", colocatedFixture, "--num-requests", "4", "--rate", "4",
		"--long-prefill-token-threshold", "256")
	if err == nil || !strings.Contains(stderr, "unknown flag: --long-prefill-token-threshold") {
		t.Errorf("the flag must be unknown: %v\n%s", err, lastLines(stderr, 3))
	}
}

// Each P/D pool resolves its own parallel.data_parallel_load_balancing (blis-schemas#76):
// pools stating different modes run, each pool's overrides carry its own mode (the flag's
// when given), and a flag disagreeing with any stated pool mode is refused naming the pool.
func TestPDPoolsResolveTheirOwnDPLoadBalancing(t *testing.T) {
	dir := pdKnobVariant(t, "", "", "      data_parallel_load_balancing: external\n",
		"      data_parallel_load_balancing: hybrid\n")
	_, catalog, registry := kernelRepos(t)
	repos := kernelmodel.Repos{Scenarios: dir, Catalog: catalog, Registry: registry}
	savedFlag, savedMode := dpLoadBalancingFromFlag, dpLoadBalancing
	t.Cleanup(func() { dpLoadBalancingFromFlag, dpLoadBalancing = savedFlag, savedMode })
	for _, tc := range []struct {
		fromFlag             bool
		flag                 string
		wantPrefill, wantDec string
	}{
		{false, "", "external", "hybrid"},
		{true, "internal", "internal", "internal"},
	} {
		dpLoadBalancingFromFlag, dpLoadBalancing = tc.fromFlag, tc.flag
		got := map[deployment.Role]string{}
		for _, role := range []deployment.Role{deployment.RolePrefill, deployment.RoleDecode} {
			m, err := kernelmodel.OpenRole("variant.yaml", repos, role)
			if err != nil {
				t.Fatal(err)
			}
			got[role] = poolOverridesOf(m).DPLoadBalancing
		}
		if got[deployment.RolePrefill] != tc.wantPrefill || got[deployment.RoleDecode] != tc.wantDec {
			t.Errorf("flag %v %q: pools resolved %v, want prefill %s decode %s", tc.fromFlag, tc.flag, got,
				tc.wantPrefill, tc.wantDec)
		}
	}
	run := func(extra ...string) (string, error) {
		_, stderr, err := runKernelCLI(t, append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1",
			"--seed", "11", "--horizon", "600000000", "--num-requests", "10", "--rate", "8"}, extra...)...)
		return stderr, err
	}
	if stderr, err := run(); err != nil {
		t.Errorf("pools stating different modes must run: %v\n%s", err, lastLines(stderr, 3))
	}
	if stderr, err := run("--dp-load-balancing", "external"); err == nil ||
		!strings.Contains(stderr, "decode pool's parallel.data_parallel_load_balancing hybrid") {
		t.Errorf("a flag disagreeing with one pool must be refused naming it: %v\n%s", err, lastLines(stderr, 3))
	}
}
