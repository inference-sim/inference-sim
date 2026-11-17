package cmd

import (
	"path/filepath"
	"strings"
	"testing"
)

// cpuTier1TBlock is a 1 TiB CPU tier as a scenario offload block.
const cpuTier1TBlock = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n" +
	"  tiers:\n    - tier: cpu_dram\n      bytes: 1099511627776\n"

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
	args := []string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv",
		"--scenarios", withBlock, "--scenario", colocatedFixture, "--seed", "5"}
	if _, stderr, err := runKernelCLI(t, args...); err == nil ||
		!strings.Contains(stderr, "cannot add one on replay of a trace blis run or blis replay wrote") {
		t.Errorf("adding offload on replay of a recorded sim trace must be refused: %v\n%s", err, lastLines(stderr, 3))
	}
}
