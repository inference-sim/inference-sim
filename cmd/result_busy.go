package cmd

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// busyInstance is one instance's busy ledger and the GPUs it occupies.
type busyInstance struct {
	ledger *sim.BusyLedger
	gpus   int
}

// busyFigures states the busy-time and utilization figures of the metric registry (busy_share,
// prefill_busy_share, decode_busy_share, mfu; inference-sim#2081) over the measured window
// [startUs, endUs). With one instance it is that instance's figure; with several, the
// instances' busy time over the window's length times their number, which is the pool's or the
// cluster's. mfu is the model FLOPs run over the window's length times the GPUs times one GPU's
// peak, and is left out when the engine states no FLOPs. A window of no length states none.
func busyFigures(insts []busyInstance, startUs, endUs int64) map[string]any {
	if len(insts) == 0 || endUs <= startUs {
		return nil
	}
	var b sim.Busy
	var peakFLOPs float64 // the instances' peak, FLOP/s
	flopsStated := true
	for _, in := range insts {
		x := in.ledger.Over(startUs, endUs)
		b.BusyUs += x.BusyUs
		b.PrefillUs += x.PrefillUs
		b.DecodeUs += x.DecodeUs
		b.FLOPs += x.FLOPs
		flopsStated = flopsStated && x.PeakFLOPsPerGPU > 0
		peakFLOPs += x.PeakFLOPsPerGPU * float64(in.gpus)
	}
	span := float64(endUs-startUs) * float64(len(insts))
	out := map[string]any{
		"busy_share":         b.BusyUs / span,
		"prefill_busy_share": b.PrefillUs / span,
		"decode_busy_share":  b.DecodeUs / span,
	}
	if flopsStated && peakFLOPs > 0 {
		out["mfu"] = b.FLOPs / (float64(endUs-startUs) * 1e-6 * peakFLOPs)
	}
	return out
}

// addBusy adds the cluster's busy figures to the engine section's cluster block, and each
// instance's to its measured_window block. A run with no measured window states none.
func addBusy(eng map[string]any, cs *cluster.ClusterSimulator, dep *deployment.Deployment, sum sim.MeasuredSummary) {
	if sum.MeasuredWindowS <= 0 {
		return
	}
	gpus, _ := instanceGPUs(cs, dep)
	var all []busyInstance
	each := map[string]busyInstance{}
	for _, inst := range cs.Instances() {
		bi := busyInstance{ledger: &inst.Metrics().Busy, gpus: gpus[string(inst.ID())]}
		all = append(all, bi)
		each[string(inst.ID())] = bi
	}
	if c, ok := eng["cluster"].(map[string]any); ok {
		for k, v := range busyFigures(all, sum.WindowStartUs, sum.WindowEndUs) {
			c[k] = v
		}
	}
	insts, _ := eng["instances"].(map[string]any)
	for id, bi := range each {
		block, _ := insts[id].(map[string]any)
		mw, _ := block["measured_window"].(map[string]any)
		if mw == nil {
			continue
		}
		for k, v := range busyFigures([]busyInstance{bi}, sum.WindowStartUs, sum.WindowEndUs) {
			mw[k] = v
		}
	}
}
