package cmd

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// busyFigures states the busy-time figures of the metric registry (busy_share,
// prefill_busy_share, decode_busy_share; inference-sim#2081) over the measured window
// [startUs, endUs), for the given ledgers. With one ledger it is an instance's figure; with
// several, the instances' busy time over the window's length times their number, which is the
// pool's or the cluster's figure. A window of no length states none.
func busyFigures(ledgers []*sim.BusyLedger, startUs, endUs int64) map[string]any {
	if len(ledgers) == 0 || endUs <= startUs {
		return nil
	}
	var b sim.Busy
	for _, l := range ledgers {
		x := l.Over(startUs, endUs)
		b.BusyUs += x.BusyUs
		b.PrefillUs += x.PrefillUs
		b.DecodeUs += x.DecodeUs
	}
	span := float64(endUs-startUs) * float64(len(ledgers))
	return map[string]any{
		"busy_share":         b.BusyUs / span,
		"prefill_busy_share": b.PrefillUs / span,
		"decode_busy_share":  b.DecodeUs / span,
	}
}

// addBusy adds the cluster's busy figures to the engine section's cluster block, and each
// instance's to its measured_window block. A run with no measured window states none.
func addBusy(eng map[string]any, cs *cluster.ClusterSimulator, sum sim.MeasuredSummary) {
	if sum.MeasuredWindowS <= 0 {
		return
	}
	var all []*sim.BusyLedger
	for _, inst := range cs.Instances() {
		all = append(all, &inst.Metrics().Busy)
	}
	if c, ok := eng["cluster"].(map[string]any); ok {
		for k, v := range busyFigures(all, sum.WindowStartUs, sum.WindowEndUs) {
			c[k] = v
		}
	}
	insts, _ := eng["instances"].(map[string]any)
	for _, inst := range cs.Instances() {
		block, _ := insts[string(inst.ID())].(map[string]any)
		mw, _ := block["measured_window"].(map[string]any)
		if mw == nil {
			continue
		}
		for k, v := range busyFigures([]*sim.BusyLedger{&inst.Metrics().Busy}, sum.WindowStartUs, sum.WindowEndUs) {
			mw[k] = v
		}
	}
}
