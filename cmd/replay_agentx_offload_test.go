package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// cpuTier1T is a 1 TiB CPU tier as a scenario offload block and as the equivalent
// --kv-offload-config file body.
const (
	cpuTier1TBlock = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n" +
		"  tiers:\n    - tier: cpu_dram\n      bytes: 1099511627776\n"
	cpuTier1TFile = "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n"
	cpuTier2TFile = "kv_offload:\n  cpu_bytes_to_use: 2199023255552\n"
)

// agentXCLIArgs is an agentic replay over the committed six-trace slice.
func agentXCLIArgs(t *testing.T, dir, scenario string, extra ...string) []string {
	t.Helper()
	return append([]string{"replay", "--scenarios", dir, "--scenario", scenario,
		"--speculative-acceptance-rate", "0.502", "--agentic-scenario", "inferencex-agentx-mvp",
		"--corpus", agentXSlice, "--corpus-sha256", agentXSliceSHA(t), "--concurrency", "2",
		"--benchmark-duration", "900s"}, extra...)
}

// An agentic replay has no trace, so no recorded deployment: the scenario's offload block is
// its tiers, exactly as the equivalent --kv-offload-config is, an agreeing flag beside the
// block changes nothing, and a disagreeing one is refused (the block is read, not ignored).
// Colocated, and 1P1D with a MultiConnector handoff.
func TestReplayAgentX_ScenarioOffloadBlockIsTheTiers(t *testing.T) {
	scenarios := "../testdata/scenarios"
	file, other := writeOffloadYAML(t, cpuTier1TFile), writeOffloadYAML(t, cpuTier2TFile)
	for _, tc := range []struct {
		name, scenario, marker string
		withBlock              string // a scenario dir stating the tier
		topology               []string
	}{
		{name: "colocated", scenario: "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"},
		{name: "1P1D MultiConnector", scenario: "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml",
			withBlock: scenarios,
			topology:  []string{"--num-instances", "2", "--prefill-instances", "1", "--decode-instances", "1"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withBlock, plain := tc.withBlock, scenarios
			if withBlock == "" {
				withBlock = writeOffloadScenario(t, scenarios, tc.scenario, "", cpuTier1TBlock)
			} else {
				// The committed 1P1D fixture states the tier; its tier-free twin is the plain
				// 1P1D scenario with the same MultiConnector handoff.
				plain = writeScenarioWithout(t, scenarios, tc.scenario, cpuTier1TBlock)
			}
			run := func(dir string, extra ...string) (string, string, error) {
				return runKernelCLI(t, agentXCLIArgs(t, dir, tc.scenario, append(append([]string(nil), tc.topology...), extra...)...)...)
			}
			fromBlock, stderr, err := run(withBlock)
			if err != nil {
				t.Fatalf("agentic replay with the scenario's offload block: %v\n%s", err, lastLines(stderr, 3))
			}
			fromFlag, stderr, err := run(plain, "--kv-offload-config", file)
			if err != nil {
				t.Fatalf("agentic replay with --kv-offload-config: %v\n%s", err, lastLines(stderr, 3))
			}
			if a, b := simulated(t, fromBlock), simulated(t, fromFlag); a != b {
				t.Errorf("the scenario's offload block and the equivalent --kv-offload-config differ\n%s", firstDiff(a, b))
			}
			if both, stderr, err := run(withBlock, "--kv-offload-config", file); err != nil || simulated(t, both) != simulated(t, fromBlock) {
				t.Errorf("an agreeing flag beside the block must change nothing: %v\n%s", err, lastLines(stderr, 3))
			}
			if _, stderr, err := run(withBlock, "--kv-offload-config", other); err == nil ||
				!strings.Contains(stderr, "disagrees with scenario") {
				t.Errorf("a disagreeing flag must be refused: %v\n%s", err, lastLines(stderr, 3))
			}
		})
	}
}

// INV-13 still holds for a trace BLIS recorded: replaying a run captured without offload
// under a scenario that states an offload block is refused, since the block would add tiers
// the recorded run never had.
func TestReplay_ScenarioOffloadCannotBeAddedToARecordedSimTrace(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	prefix := filepath.Join(t.TempDir(), "trace")
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", scenarios, "--scenario", colocatedFixture,
		"--num-requests", "10", "--rate", "4", "--seed", "5", "--trace-output", prefix); err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	withBlock := writeOffloadScenario(t, scenarios, colocatedFixture, "", cpuTier1TBlock)
	for _, extra := range [][]string{nil, {"--kv-offload-config", writeOffloadYAML(t, cpuTier1TFile)}} {
		args := append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv",
			"--scenarios", withBlock, "--scenario", colocatedFixture, "--seed", "5"}, extra...)
		if _, stderr, err := runKernelCLI(t, args...); err == nil ||
			!strings.Contains(stderr, "cannot add one on replay of a sim-generated trace (INV-13)") {
			t.Errorf("%v: adding offload on replay of a recorded sim trace must be refused: %v\n%s",
				extra, err, lastLines(stderr, 3))
		}
	}
}

// writeScenarioWithout copies a scenario with block removed into a new directory.
func writeScenarioWithout(t *testing.T, dir, name, block string) string {
	t.Helper()
	body := readFile(t, filepath.Join(dir, name))
	if !strings.Contains(body, block) {
		t.Fatalf("fixture %s does not state %q", name, block)
	}
	out := t.TempDir()
	if err := os.WriteFile(filepath.Join(out, name), []byte(strings.Replace(body, block, "", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	return withPDRouter(t, out)
}
