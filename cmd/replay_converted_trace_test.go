package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// A trace blis convert wrote records external traffic, not a deployment BLIS ran, so its
// header mode is "converted" and replay takes the offload tiers from the scenario (#2067).
// The same file relabelled "generated" claims a recorded deployment and is refused, as
// TestReplay_ScenarioOffloadCannotBeAddedToARecordedSimTrace refuses a run's own trace.
func TestReplay_ConvertedTraceTakesTheScenarioOffloadBlock(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	dir := t.TempDir()
	prefix := filepath.Join(dir, "weka")
	if err := runConvertWeka(wekaSessions(t, 3), prefix, workload.WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 1}); err != nil {
		t.Fatalf("convert weka: %v", err)
	}
	header := readFile(t, prefix+".yaml")
	if !strings.Contains(header, "mode: converted\n") {
		t.Fatalf("blis convert weka must write mode converted, got:\n%s", header)
	}
	withBlock := writeOffloadScenario(t, scenarios, colocatedFixture, "", cpuTier1TBlock)
	replay := func(headerPath string) (string, string, error) {
		return runKernelCLI(t, "replay", "--trace-header", headerPath, "--trace-data", prefix+".csv",
			"--session-mode", "closed-loop", "--scenarios", withBlock, "--scenario", colocatedFixture, "--seed", "5")
	}
	if _, stderr, err := replay(prefix + ".yaml"); err != nil {
		t.Fatalf("a converted trace must replay with the scenario's offload block: %v\n%s", err, lastLines(stderr, 3))
	}
	relabelled := filepath.Join(dir, "generated.yaml")
	if err := os.WriteFile(relabelled, []byte(strings.Replace(header, "mode: converted\n", "mode: generated\n", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := replay(relabelled); err == nil ||
		!strings.Contains(stderr, "cannot add one on replay of a sim-generated trace (INV-13)") {
		t.Errorf("a generated trace with no kv_offload must still refuse the scenario's tiers: %v\n%s", err, lastLines(stderr, 3))
	}
}
