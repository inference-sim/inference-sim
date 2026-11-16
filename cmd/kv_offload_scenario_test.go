package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// #1910, #2037: the scenario's Deployment.offload block is the run's KV offload configuration,
// and the only one.

// writeOffloadScenario writes scenario file name from dir with the offload block inserted
// before the marker (appended when the marker is ""), into a fresh directory.
func writeOffloadScenario(t *testing.T, dir, name, marker, offload string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	switch {
	case marker == "":
		body = strings.TrimRight(body, "\n") + "\n" + offload
	case strings.Contains(body, marker):
		body = strings.Replace(body, marker, offload+marker, 1)
	default:
		t.Fatalf("fixture %s has no %q", name, marker)
	}
	// A tier's rates are registry entries: the scenario that states a tier lists their set.
	if !strings.Contains(body, "cost-model-offload-tiers") {
		re := regexp.MustCompile(`(?m)^(coefficients: \[[^\]]*)\]`)
		body = re.ReplaceAllString(body, "${1}, cost-model-offload-tiers]")
	}
	out := t.TempDir()
	if err := os.WriteFile(filepath.Join(out, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(body, "endpoint_picker_config: "+pdAlwaysEPPConfig) {
		withPDRouter(t, out)
	}
	return out
}

// A scenario stating its CPU tier runs, and the exported trace header records the resolved
// kv_offload config. A scenario-offload run replays byte-identically; a replay under a scenario
// whose tier differs from the trace's is refused; the scenario's 0-byte tier is refused by the
// schema.
func TestRunCmd_ScenarioOffloadBlock(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	scenarios = scenarioWithOffloadSet(t, scenarios, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	const tier = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n  tiers:\n    - tier: cpu_dram\n      bytes: 1099511627776\n"
	for _, tc := range []struct {
		name, src, scenario, marker string
		args                        []string
		replayArgs                  []string
	}{
		{"colocated", scenarios, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml", "",
			[]string{"--num-requests", "20", "--rate", "4", "--prefix-tokens", "512", "--seed", "5", "--horizon", "600000000"},
			[]string{"--seed", "5", "--horizon", "600000000"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withBlock := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, tier)
			// pdTopology names its own --scenarios and --scenario; later flags win, so append ours.
			run := func(dir, prefix string, extra ...string) (string, string, error) {
				args := append(append([]string{"run", "--trace-output", prefix}, tc.args...), extra...)
				return runKernelCLI(t, append(args, "--scenarios", dir, "--scenario", tc.scenario)...)
			}
			tmp := t.TempDir()
			scen, stderr, err := run(withBlock, filepath.Join(tmp, "scen"))
			if err != nil {
				t.Fatalf("scenario offload run: %v\n%s", err, lastLines(stderr, 3))
			}
			if h := readFile(t, filepath.Join(tmp, "scen.yaml")); !strings.Contains(h, "cpu_bytes_to_use: 1099511627776") {
				t.Errorf("the trace header carries no kv_offload with the scenario's CPU tier:\n%s", h)
			}
			rep, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "scen.yaml"),
				"--trace-data", filepath.Join(tmp, "scen.csv")}, tc.replayArgs...), "--scenarios", withBlock, "--scenario", tc.scenario)...)
			if err != nil || inv13View(t, rep) != inv13View(t, scen) {
				t.Errorf("the scenario-offload run's replay differs: err=%v\n%s", err, lastLines(stderr, 3))
			}
			bigger := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, strings.Replace(tier, "bytes: 1099511627776", "bytes: 2199023255552", 1))
			if _, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "scen.yaml"),
				"--trace-data", filepath.Join(tmp, "scen.csv")}, tc.replayArgs...), "--scenarios", bigger, "--scenario", tc.scenario)...); err == nil ||
				!strings.Contains(stderr, "conflicts with the kv_offload config recorded in the trace header") {
				t.Errorf("replaying under a scenario whose tier differs from the trace's must be refused: err=%v\n%s", err, lastLines(stderr, 3))
			}
			zero := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, strings.Replace(tier, "bytes: 1099511627776", "bytes: 0", 1))
			if _, stderr, err := run(zero, filepath.Join(tmp, "zero")); err == nil || !strings.Contains(stderr, "must be positive") {
				t.Errorf("a 0-byte tier must be refused by the schema: err=%v\n%s", err, lastLines(stderr, 3))
			}
		})
	}
}

// The SimpleCPUOffloadConnector is accepted: one block-granular CPU tier, every block offloaded
// (no prompt-only clamp), eager by default and lazy when the spec says so; anything else it
// cannot be is refused.
func TestKVOffloadFromDeployment_TheSimpleConnector(t *testing.T) {
	cpu := []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 30}}
	for _, tc := range []struct {
		spec string
		lazy bool
	}{{"", false}, {"eager", false}, {"lazy", true}} {
		block, err := kvOffloadBlockFromDeployment(&deployment.Offload{Connector: simpleConnector, Spec: tc.spec, Tiers: cpu})
		if err != nil {
			t.Fatalf("spec %q: %v", tc.spec, err)
		}
		got, err := resolveKVOffload(block, nil, 64)
		if err != nil {
			t.Fatalf("spec %q: %v", tc.spec, err)
		}
		if !got.Simple || got.LazyOffload != tc.lazy || got.OffloadPromptOnly || got.BlocksPerChunk != 1 {
			t.Fatalf("spec %q resolves to %+v", tc.spec, got)
		}
		back := headerToSimOffload(simToHeaderOffload(got))
		if back.Simple != got.Simple || back.LazyOffload != got.LazyOffload {
			t.Fatalf("spec %q: the trace header loses the connector mode", tc.spec)
		}
	}
	for _, o := range []deployment.Offload{
		{Connector: simpleConnector, Spec: "TieringOffloadingSpec", Tiers: cpu},
		{Connector: simpleConnector, Tiers: append(cpu, deployment.Tier{Device: "nvme_gen4", Bytes: 1 << 40})},
	} {
		if _, err := kvOffloadBlockFromDeployment(&o); err == nil {
			t.Fatalf("%+v must be refused", o)
		}
	}
}
