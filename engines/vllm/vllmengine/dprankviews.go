// Original to blis-vllm.
// H3 of the DP external load-balancing RFC (inference-sim#2699).

package vllmengine

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"

	vllmcore "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm/dprankview"
)

// DPRankViews exposes a built external data-parallel group as one engine.Engine per rank — the
// per-rank endpoint views the router sees under external load balancing (inference-sim#2699). It
// reports false for any engine that is not such a shared group. It is the engine module's
// host-facing entry for the view engine, so the host (cmd) builds the per-rank endpoints through
// engine.Engine alone and never imports the engine's core/vllm or dprankview packages
// (the "reach the engine only through the contract" boundary).
func DPRankViews(eng engine.Engine) ([]engine.Engine, bool) {
	g, ok := eng.(*vllmcore.DPGroup)
	if !ok {
		return nil, false
	}
	return dprankview.NewEndpointViews(g), true
}
