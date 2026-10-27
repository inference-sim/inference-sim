// Moved from inference-sim/blis-schemas/rules/dplb.go at 70246460a379f8b1898fbf6c36a3091364118950.

package pack

import "github.com/inference-sim/blis-schemas/spec/deployment"

// ResolveDPLoadBalancing returns the load-balancing mode a pool runs, as both registered
// releases' EngineArgs.create_engine_config resolves it (v0.31.0 vllm/engine/arg_utils.py
// :2360-2399, v0.29.0 :2229-2268): hybrid with one rank per node runs external, and
// hybrid with every rank on one node runs internal. The other modes run as stated, and
// one rank runs internal whatever is stated.
func (p *Pack) ResolveDPLoadBalancing(pl deployment.Parallelism) deployment.DPLoadBalancing {
	m := pl.DataParallelLoadBalancing.Mode()
	if pl.DP <= 1 {
		return deployment.DPLBInternal
	}
	if m == deployment.DPLBHybrid {
		switch pl.DPLocal {
		case 1:
			return deployment.DPLBExternal
		case pl.DP:
			return deployment.DPLBInternal
		}
	}
	return m
}
