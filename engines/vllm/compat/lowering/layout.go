// Original to blis-vllm.
// The layout decisions moved from blis-substrate's kernel/internal/resolve/layout.go at
// 00817b10798cbc00f560e51921c30c7084fa5e0c, when the engine-settings resolution left the cost
// model (blis-substrate#58).

package lowering

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	subrules "github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// layout is the part of a pool's layout the settings resolution reads: the parallel widths and
// the MoE group, from the pool's parallel block and the cluster's packaging, and the engine
// decisions that follow from them.
type layout struct {
	TP, DP, PCP, DCP int
	ExpertWidth      int
	// MoEGroupWidth is how many ranks an MoE layer's dispatch and combine span: the
	// expert-parallel width with expert parallelism on, and with it off, tp x pcp x dp when dp
	// exceeds one, as vLLM v0.31.0 shares the MoE across the data-parallel ranks.
	MoEGroupWidth int
	GPUsPerNode   int

	SequenceParallelMoE   bool
	ModelSequenceParallel bool
	CustomAllReduce       bool
}

// geometry is the layout's widths, as the cost model derives them from the same documents.
func geometry(pk scenario.Packaging, pool deployment.Pool) (layout, error) {
	pl := pool.Parallel
	if pl.TP < 1 || pl.DP < 1 {
		return layout{}, fmt.Errorf("tensor- and data-parallel widths must be at least 1")
	}
	l := layout{TP: pl.TP, DP: pl.DP, PCP: pl.PCP, DCP: pl.DCP,
		ExpertWidth: pl.ExpertParallelWidth(), GPUsPerNode: pk.GPUsPerNode}
	l.MoEGroupWidth = l.ExpertWidth
	if !pl.EnableExpertParallel && pl.DP > 1 {
		l.MoEGroupWidth = pl.TP * max(pl.PCP, 1) * pl.DP
	}
	return l, nil
}

// moeDispatches reports whether an MoE layer runs a dispatch and combine at all. With expert
// parallelism on, vLLM v0.31.0 runs its all-to-all kernels only when use_all2all_kernels holds:
// use_ep and (dp > 1, pcp > 1, or a sequence-parallel MoE input)
// (vllm/model_executor/layers/fused_moe/config.py:1024-1027). With it off, the naive
// all-gather/reduce-scatter dispatch runs when dp exceeds one (moe_runner.py:797-800).
func (l layout) moeDispatches() bool {
	if l.ExpertWidth > 1 {
		return l.ExpertWidth > max(l.TP, 1) || l.SequenceParallelMoE || l.ModelSequenceParallel
	}
	return l.MoEGroupWidth > 1
}

// customAllReduce decides whether the SM-consuming reduction kernel runs, and records why when a
// request for it is declined.
func customAllReduce(e vllmsettings.Settings, l layout, rackIsOneDomain bool,
	rules subrules.EngineRules) (bool, []plan.Override) {
	requested := true
	if e.DisableCustomAllReduce != nil {
		requested = !*e.DisableCustomAllReduce
	}
	if e.AllReduceBackend == "nccl" {
		requested = false
	}
	if !requested {
		return false, nil
	}

	var overrides []plan.Override
	if !rules.CustomAllReduceSupportsWidth(l.TP) {
		return false, append(overrides, plan.Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"the kernel supports only certain rank counts and %d is not one, so the "+
					"reduction moves from SMs to the NIC", l.TP),
		})
	}
	// Crossing a node boundary disqualifies the kernel unless every rank has multi-node
	// NVLink. The catalog records whether a part does, through the rack tier.
	if l.GPUsPerNode > 0 && l.TP > l.GPUsPerNode && !rackIsOneDomain {
		return false, append(overrides, plan.Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"a tensor-parallel width of %d spans more than the %d GPUs in a node and "+
					"the part has no multi-node NVLink, so the reduction moves from SMs to the NIC",
				l.TP, l.GPUsPerNode),
		})
	}
	return true, overrides
}
