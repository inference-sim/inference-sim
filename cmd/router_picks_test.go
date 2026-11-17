package cmd

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// picksOf is a pick distribution of the Result's router.trace_summary (an instance id to its
// count), and the sum of its counts.
func picksOf(t *testing.T, stdout, name string) (map[string]int, int) {
	t.Helper()
	v, ok := resultValue(resultDoc(t, stdout), "router.trace_summary."+name)
	if !ok {
		t.Fatalf("the Result has no router.trace_summary.%s", name)
	}
	raw, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("router.trace_summary.%s is %T, not a table of picks", name, v)
	}
	picks, sum := map[string]int{}, 0
	for id, n := range raw {
		f, ok := n.(float64)
		if !ok {
			t.Fatalf("router.trace_summary.%s[%s] is %T, not a count", name, id, n)
		}
		picks[id], sum = int(f), sum+int(f)
	}
	return picks, sum
}

// In a disaggregated run the router makes two picks per request: the prefill instance and the
// decode instance. The Result's router.trace_summary lists each pool's picks under its own
// name, so a prefill-pool pick can be checked against the pool it must come from (blis-substrate
// #47): TargetDistribution names decode instances only, PrefillDistribution names prefill
// instances only, and each sums to the number of requests the run disaggregated.
func TestResult_TraceSummaryRecordsPrefillAndDecodePicksSeparately(t *testing.T) {
	const requests, prefill, decode = 30, 3, 1
	stdout, stderr, err := runKernelCLI(t, "run", "--trace-level", "decisions", "--summarize-trace", "--view", "summary",
		"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
		"--num-instances", fmt.Sprint(prefill+decode), "--prefill-instances", fmt.Sprint(prefill),
		"--decode-instances", fmt.Sprint(decode), "--seed", "1", "--horizon", "600000000",
		"--num-requests", fmt.Sprint(requests), "--rate", "50")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 4))
	}
	disaggregated := int(resultNum(t, stdout, "router.trace_summary.DisaggregatedCount"))
	if disaggregated != requests {
		t.Fatalf("the run disaggregated %d requests, want all %d (the test needs every request to take both picks)", disaggregated, requests)
	}
	pools := cluster.BuildPoolMembershipFromIndices(prefill+decode, prefill, decode, 0, 0)

	prefillPicks, prefillSum := picksOf(t, stdout, "PrefillDistribution")
	decodePicks, decodeSum := picksOf(t, stdout, "TargetDistribution")
	if prefillSum != disaggregated {
		t.Errorf("prefill picks sum to %d, want the %d disaggregated requests", prefillSum, disaggregated)
	}
	if decodeSum != disaggregated {
		t.Errorf("decode picks sum to %d, want the %d disaggregated requests", decodeSum, disaggregated)
	}
	for id := range prefillPicks {
		if pools[id] != cluster.PoolRolePrefill {
			t.Errorf("PrefillDistribution names %s, which is in the %v pool", id, pools[id])
		}
	}
	for id := range decodePicks {
		if pools[id] != cluster.PoolRoleDecode {
			t.Errorf("TargetDistribution names %s, which is in the %v pool", id, pools[id])
		}
	}
	// The text summary (--summarize-trace with --view summary) prints both tables.
	for _, want := range []string{"Target Distribution:", "Prefill Distribution:"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the --summarize-trace text has no %q table:\n%s", want, lastLines(stderr, 20))
		}
	}
}

// A colocated run makes no prefill pick: its Result records an empty PrefillDistribution next to
// the TargetDistribution, so a reader never mistakes a missing table for a missing pool.
func TestResult_TraceSummaryOfAColocatedRunHasNoPrefillPicks(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, resultRunArgs("--trace-level", "decisions", "--summarize-trace")...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if _, sum := picksOf(t, stdout, "PrefillDistribution"); sum != 0 {
		t.Errorf("a colocated run recorded %d prefill picks", sum)
	}
	if _, sum := picksOf(t, stdout, "TargetDistribution"); sum <= 0 {
		t.Errorf("a colocated run recorded no routing picks")
	}
}
