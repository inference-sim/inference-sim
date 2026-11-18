package cmd

import (
	"math"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The scenarios that draft tokens run as shipped: with no acceptance flag, each resolves its
// acceptance from the registry's measured profile for its model, drafter and K, selected by the
// category and thinking mode its workload binding states. Before, the Muse-Spark DSpark
// scenarios pinned an acceptance length of 2.84 taken from an H100 sweep, which overrode the
// GB200 vLLM profile (2.72), and the DeepSeek-V4.1-Flash 1P1D scenarios stated no thinking mode,
// so the run was refused until --speculative-thinking and --speculative-category were given.
func TestShippedSpeculativeScenariosRunWithoutAcceptanceFlags(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the shipped scenarios")
	}
	pd := []string{"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1"}
	cases := []struct {
		dir, scenario string
		extra         []string
		// wantAL is the registry profile's acceptance length, 0 when only the layer is checked.
		wantAL float64
	}{
		{"../testdata/scenarios", "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml", pd, 0},
		{"../testdata/scenarios", "deepseek-v4.1-flash-h100-1p1d-dspark5.yaml", pd, 0},
		{"../testdata/scenarios/muse-spark", "muse-gb200-nvl-tep8-dspark.yaml", nil, 2.72},
		{"../testdata/scenarios/muse-spark", "muse-h200-ib-tep8-dspark.yaml", nil, 2.72},
		{"../testdata/scenarios/muse-spark", "muse-h100-roce-tp16-dspark.yaml", nil, 2.72},
	}
	for _, c := range cases {
		t.Run(c.scenario, func(t *testing.T) {
			args := append([]string{"run", "--scenarios", c.dir, "--scenario", c.scenario,
				"--rate", "2", "--num-requests", "4", "--seed", "1"}, c.extra...)
			stdout, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("the shipped scenario must run as shipped: %v\n%s", err, lastLines(stderr, 4))
			}
			cluster := resultDoc(t, stdout)["reports"].(map[string]any)["engine"].(map[string]any)["cluster"].(map[string]any)
			acc, _ := cluster["acceptance"].(map[string]any)
			if acc == nil {
				t.Fatal("the Result records no acceptance")
			}
			if acc["source_layer"] != "registry" {
				t.Errorf("source_layer %v, want registry: the scenario must not pin a value over the measured profile", acc["source_layer"])
			}
			entry, _ := acc["entry"].(map[string]any)
			if key, _ := entry["key"].(string); !strings.Contains(key, "thinking-on/coding") {
				t.Errorf("profile key %q, want the thinking-on coding profile the scenario's workload states", key)
			}
			if length, _ := acc["length"].(float64); c.wantAL > 0 && math.Abs(length-c.wantAL) > 1e-9 {
				t.Errorf("acceptance length %v, want the registry profile's %v", length, c.wantAL)
			}
		})
	}
}

// Every Muse-Spark scenario that drafts resolves the acceptance its README says. The 18 scenarios
// of the matrix take the registry's GB200 profile (2.72). The three H100 measured arms with DSpark
// (armC, armC-routed, armD) keep a scenario pin of 2.84: the registry has no H100 Muse profile, and
// 2.84 is the internal measurement on H100 PP2xTP8 those arms reproduce. The glob finds every
// drafting scenario, so none falls back to a pin unnoticed, and a new one must be stated here.
func TestEveryMuseSparkDraftingScenarioResolvesItsDocumentedAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip("runs every drafting Muse scenario")
	}
	dir := "../testdata/scenarios/muse-spark"
	files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	pinnedArms := map[string]bool{"muse-h100-roce-armC.yaml": true, "muse-h100-roce-armC-routed.yaml": true, "muse-h100-roce-armD.yaml": true}
	// vLLM refuses these on Hopper (data parallelism above one), before acceptance matters.
	refused := map[string]bool{"muse-h200-ib-dep8-dspark.yaml": true, "muse-h200-ib-dep8-dspark-128k.yaml": true}
	drafting := 0
	for _, f := range files {
		name := filepath.Base(f)
		if !strings.Contains(name, "dspark") && !pinnedArms[name] {
			continue
		}
		drafting++
		t.Run(name, func(t *testing.T) {
			args := []string{"run", "--scenarios", dir, "--scenario", name, "--rate", "2", "--num-requests", "4", "--seed", "1"}
			if strings.HasSuffix(name, "-routed.yaml") {
				args = append(args, "--num-instances", "2")
			}
			stdout, stderr, err := runKernelCLI(t, args...)
			if refused[name] {
				if err == nil || !strings.Contains(stderr, "requires data parallelism 1") {
					t.Fatalf("want vLLM's Hopper data-parallelism refusal, got err=%v\n%s", err, lastLines(stderr, 3))
				}
				return
			}
			if err != nil {
				t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
			}
			cluster := resultDoc(t, stdout)["reports"].(map[string]any)["engine"].(map[string]any)["cluster"].(map[string]any)
			acc, _ := cluster["acceptance"].(map[string]any)
			if acc == nil {
				t.Fatal("the Result records no acceptance")
			}
			wantLayer, wantAL := "registry", 2.72
			if pinnedArms[name] {
				wantLayer, wantAL = "scenario", 2.84
			}
			if acc["source_layer"] != wantLayer {
				t.Errorf("source_layer %v, want %s", acc["source_layer"], wantLayer)
			}
			if length, _ := acc["length"].(float64); math.Abs(length-wantAL) > 1e-9 {
				t.Errorf("acceptance length %v, want %v", length, wantAL)
			}
		})
	}
	if drafting != 21 {
		t.Errorf("%d drafting Muse scenarios, want 21 (18 resolved from the registry, 3 pinned H100 arms): state any new one in this test", drafting)
	}
}
