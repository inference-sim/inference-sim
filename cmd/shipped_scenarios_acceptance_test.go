package cmd

import (
	"math"
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
