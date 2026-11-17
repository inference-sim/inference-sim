package cmd

// NOTE: mutates package-level CLI vars through runAgentXReplay; no t.Parallel().

import (
	"math"
	"testing"
)

// Each instance states how busy its engine was over the measured window, and on what
// (inference-sim#2081): the shares are fractions, the prefill and decode parts add up to the
// busy share, the cluster's is the instances' mean, and a heavier load keeps the engines busier.
func TestResultStatesBusyShares(t *testing.T) {
	run := func(conc string) (map[string]map[string]any, map[string]any) {
		stdout, _ := runAgentXReplay(t, "--concurrency", conc, "--scenarios", "../examples", "--scenario", "agentx-dsv41-flash-h100-2x-tp8-dspark5.yaml", "--num-instances", "2")
		c, _ := resultValue(resultDoc(t, string(stdout)), "reports.engine.cluster")
		return instanceFigures(t, string(stdout)), c.(map[string]any)
	}
	insts, cluster := run("4")
	var sum float64
	for id, b := range insts {
		mw := b["measured_window"].(map[string]any)
		busy, pf, dc := mw["busy_share"].(float64), mw["prefill_busy_share"].(float64), mw["decode_busy_share"].(float64)
		if busy <= 0 || busy > 1 {
			t.Errorf("instance %s busy_share %v, want in (0, 1]", id, busy)
		}
		if math.Abs(pf+dc-busy) > 1e-9 || pf <= 0 || dc <= 0 {
			t.Errorf("instance %s: prefill %v + decode %v != busy %v, or one is empty", id, pf, dc, busy)
		}
		sum += busy
		if mfu, ok := mw["mfu"].(float64); !ok || mfu <= 0 || mfu > busy {
			t.Errorf("instance %s mfu %v, want in (0, busy_share %v]: a step cannot run more than the peak", id, mw["mfu"], busy)
		}
	}
	if got := cluster["busy_share"].(float64); math.Abs(got-sum/float64(len(insts))) > 1e-9 {
		t.Errorf("cluster busy_share %v, want the instances' mean %v", got, sum/float64(len(insts)))
	}
	_, heavy := run("64")
	t.Logf("concurrency 4: busy %v mfu %v; concurrency 64: busy %v mfu %v", cluster["busy_share"], cluster["mfu"], heavy["busy_share"], heavy["mfu"])
	if heavy["busy_share"].(float64) <= cluster["busy_share"].(float64) {
		t.Errorf("busy_share %v at concurrency 64 is not above %v at 4", heavy["busy_share"], cluster["busy_share"])
	}
	if heavy["mfu"].(float64) <= cluster["mfu"].(float64) {
		t.Errorf("mfu %v at concurrency 64 is not above %v at 4: batching more work per weight read raises utilization", heavy["mfu"], cluster["mfu"])
	}
}
