package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// #1910: the scenario's Deployment.offload block is the run's KV offload configuration; a
// --kv-offload-config file may only restate it.

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
	out := t.TempDir()
	if err := os.WriteFile(filepath.Join(out, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(body, "endpoint_picker_config: "+pdAlwaysEPPConfig) {
		withPDRouter(t, out)
	}
	return out
}

// A scenario stating its CPU tier runs byte-identically to the same scenario without the
// block given the equivalent --kv-offload-config, colocated and P/D (every engine gets its own
// tier); the exported trace headers, which record the resolved kv_offload config, match too.
// Stating both, agreeing, is accepted; disagreeing is refused naming the field. The scenario's
// 0-byte tier is refused by the schema. A scenario-offload run replays byte-identically.
func TestRunCmd_ScenarioOffloadBlock(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	const tier = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n  tiers:\n    - tier: cpu_dram\n      bytes: 1099511627776\n"
	file := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n")
	other := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 2199023255552\n")
	for _, tc := range []struct {
		name, src, scenario, marker string
		args                        []string
		replayArgs                  []string
	}{
		{"colocated", scenarios, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml", "",
			[]string{"--num-requests", "20", "--rate", "4", "--prefix-tokens", "512", "--seed", "5", "--horizon", "600000000"},
			[]string{"--seed", "5", "--horizon", "600000000"}},
		{"P/D", writeScenarioVariant(t, "connector: NixlConnector", "connector: MultiConnector"), "variant.yaml", "pd_transfer:",
			append(append([]string(nil), pdTopology...), "--num-requests", "20", "--rate", "4", "--prefix-tokens", "512"),
			pdTopology},
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
			flag, stderr, err := run(tc.src, filepath.Join(tmp, "flag"), "--kv-offload-config", file)
			if err != nil {
				t.Fatalf("flag offload run: %v\n%s", err, lastLines(stderr, 3))
			}
			if simulated(t, scen) != simulated(t, flag) {
				t.Errorf("the scenario's offload block and the equivalent --kv-offload-config differ\n--- scenario\n%s\n--- flag\n%s", scen, flag)
			}
			hs, hf := readFile(t, filepath.Join(tmp, "scen.yaml")), readFile(t, filepath.Join(tmp, "flag.yaml"))
			if !strings.Contains(hs, "cpu_bytes_to_use: 1099511627776") || hs != hf {
				t.Errorf("trace headers differ or carry no kv_offload:\n--- scenario\n%s\n--- flag\n%s", hs, hf)
			}
			both, stderr, err := run(withBlock, filepath.Join(tmp, "both"), "--kv-offload-config", file)
			if err != nil || simulated(t, both) != simulated(t, scen) {
				t.Errorf("an agreeing flag beside the scenario block must change nothing: err=%v\n%s", err, lastLines(stderr, 3))
			}
			// The scenario cannot state blocks_per_chunk (no blis-schemas field), so a flag beside
			// the block supplies it, exactly as the flag alone does; it runs and replays.
			chunked := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n  blocks_per_chunk: 4\n")
			c4, stderr, err := run(withBlock, filepath.Join(tmp, "c4"), "--kv-offload-config", chunked)
			if err != nil {
				t.Fatalf("blocks_per_chunk 4 beside the scenario block: %v\n%s", err, lastLines(stderr, 3))
			}
			if h := readFile(t, filepath.Join(tmp, "c4.yaml")); !strings.Contains(h, "blocks_per_chunk: 4") {
				t.Errorf("the trace header does not record blocks_per_chunk 4:\n%s", h)
			}
			if alone, _, err := run(tc.src, filepath.Join(tmp, "c4flag"), "--kv-offload-config", chunked); err != nil || simulated(t, alone) != simulated(t, c4) {
				t.Errorf("blocks_per_chunk 4 from the flag alone differs from the flag beside the scenario block: err=%v", err)
			}
			if rep, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "c4.yaml"),
				"--trace-data", filepath.Join(tmp, "c4.csv")}, tc.replayArgs...), "--scenarios", withBlock, "--scenario", tc.scenario, "--kv-offload-config", chunked)...); err != nil || inv13View(t, rep) != inv13View(t, c4) {
				t.Errorf("the blocks_per_chunk 4 run's replay differs: err=%v\n%s", err, lastLines(stderr, 3))
			}
			if _, stderr, err := run(withBlock, filepath.Join(tmp, "bad"), "--kv-offload-config", other); err == nil ||
				!strings.Contains(stderr, "disagrees with scenario") || !strings.Contains(stderr, "CPUBytesToUse") {
				t.Errorf("a disagreeing flag must be refused naming the field: err=%v\n%s", err, lastLines(stderr, 3))
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
