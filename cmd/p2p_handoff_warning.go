package cmd

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// warnP2PHandoffsWithoutWrite says, on stderr, that a disaggregated run's p2p handoffs moved no KV
// (inference-sim#2214, #2220): the run completed, but for those requests the decode engine held the
// prompt already or recomputed it, and the Result's kv_transfer figures describe only the WRITEs
// that happened (breakdowns.pd.p2p_handoffs_without_write counts the rest).
//
// With pools of unequal tensor parallelism the cause is vLLM's, which the engine reproduces and BLIS
// does not override: the p2p tier's connect handshake compares the run config, and the run config
// names the tensor-parallel size unless the KV layout is parallelism-agnostic. The layout is not
// agnostic under the V2 model runner, or for an MLA or hybrid model, or with a sliding window or
// several KV cache groups (offloading/config.py, is_parallelism_agnostic, vLLM v0.33.0). The peer
// connect is then rejected ("config fingerprint mismatch") and vLLM goes on without the prefill
// engine's KV, so no WRITE is priced because none is made. It is not a missing NIC cost: where the
// handshake succeeds, the WRITE is priced over the NIC whatever the tensor parallelism of either pool.
func warnP2PHandoffsWithoutWrite(pd *cluster.PDMetrics, dep *deployment.Deployment) {
	if pd == nil || pd.P2PHandoffsWithoutWrite == 0 {
		return
	}
	msg := fmt.Sprintf("%d of %d p2p handoffs moved no KV between the prefill and the decode engine: the decode engine "+
		"held those prompts already or recomputed them", pd.P2PHandoffsWithoutWrite, pd.P2PHandoffs)
	if unequal, p, d := unequalP2PPools(dep); unequal {
		msg += fmt.Sprintf(". The prefill pool is tp %d and the decode pool tp %d: vLLM's p2p connect handshake compares "+
			"the tensor-parallel size unless the KV layout is parallelism-agnostic (not under the V2 model runner, nor "+
			"for an MLA or hybrid model), and then rejects the peer (config fingerprint mismatch), so the decode "+
			"engine recomputes the prompt. Give both pools one tensor parallelism, or a parallelism-agnostic layout "+
			"(inference-sim#2214)", p, d)
	}
	logrus.Warn(msg)
}

// unequalP2PPools reports whether the deployment's prefill and decode pools differ in tensor
// parallelism, and the two sizes.
func unequalP2PPools(dep *deployment.Deployment) (unequal bool, prefill, decode int) {
	if dep == nil {
		return false, 0, 0
	}
	for _, p := range dep.Pools {
		switch p.Role {
		case deployment.RolePrefill:
			prefill = max(p.Parallel.TP, 1)
		case deployment.RoleDecode:
			decode = max(p.Parallel.TP, 1)
		}
	}
	return prefill > 0 && decode > 0 && prefill != decode, prefill, decode
}
