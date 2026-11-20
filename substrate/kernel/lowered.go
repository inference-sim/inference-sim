package latencykernel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// cloneLowered is a deep copy of the parts the kernel keeps: a caller editing the plan it passed does
// not change a kernel built from it.
func cloneLowered(l plan.Lowered) plan.Lowered {
	l.Overrides = slices.Clone(l.Overrides)
	l.Notes = slices.Clone(l.Notes)
	l.Plan.Graphs.Buckets = slices.Clone(l.Plan.Graphs.Buckets)
	l.Plan.Launches = slices.Clone(l.Plan.Launches)
	if w := l.ServedLayout; w != nil {
		c := *w
		l.ServedLayout = &c
	}
	return l
}

// validateLowered refuses a plan whose resolved settings no engine can mean, or that leaves out one the
// kernel needs and cannot price without.
func validateLowered(l plan.Lowered) error {
	p := l.Plan
	if err := p.ValidateResolved(); err != nil {
		return err
	}
	if w := p.Collectives.AllReduce; w != plan.AllReduceUnstated && w.UsesSMs() != l.Reported.SMAllReduce {
		return fmt.Errorf("the plan states the all-reduce kernel %q (Plan.Collectives.AllReduce) and "+
			"Reported.SMAllReduce %t, which disagree about whether it runs on the GPU's SMs; the "+
			"engine states one resolution (SUB-6)", w, l.Reported.SMAllReduce)
	}
	captured := p.Graphs.Decode != plan.LaunchEager || p.Graphs.Other != plan.LaunchEager
	switch {
	case p.WeightFormat != "" && model.DType(p.WeightFormat).Bytes() <= 0:
		return fmt.Errorf("the plan serves the weights as %q (Plan.WeightFormat), a format the cost "+
			"model cannot size: it is none of the catalog's dtypes, so its weights would be priced "+
			"at zero bytes", p.WeightFormat)
	case captured && p.Graphs.Ceiling <= 0:
		return fmt.Errorf("the plan launches steps from captured graphs (Plan.Graphs.Decode %q, "+
			"Other %q) and states no capture ceiling (Plan.Graphs.Ceiling); the engine resolves it, "+
			"and the cost model applies no default (SUB-6)", p.Graphs.Decode, p.Graphs.Other)
	case p.Batch.Requests <= 0:
		return fmt.Errorf("the plan states no request limit per step (Plan.Batch.Requests); the engine " +
			"resolves it, and the cost model applies no default (SUB-6)")
	case p.Batch.Tokens <= 0:
		return fmt.Errorf("the plan states no token limit per step (Plan.Batch.Tokens); the engine " +
			"resolves it, and the cost model applies no default (SUB-6)")
	case p.PageTokens <= 0:
		return fmt.Errorf("the plan states no KV page size (Plan.PageTokens); the engine resolves it, " +
			"and the cost model applies no default (SUB-6)")
	case p.Cache.ElementBytes <= 0:
		return fmt.Errorf("the plan states no KV element width (Plan.Cache.ElementBytes); the engine " +
			"resolves it, and the cost model applies no default (SUB-6)")
	case p.Cache.StateElementBytes <= 0:
		return fmt.Errorf("the plan states no recurrent state element width " +
			"(Plan.Cache.StateElementBytes); the engine resolves it, and the cost model applies no " +
			"default (SUB-6)")
	}
	return nil
}

// adoptLowered takes the engine's resolved settings into the kernel: the plan it prices from,
// the layout decisions that follow from it, and the record of how they were resolved, which the
// kernel reports ahead of its own.
func (k *Kernel) adoptLowered(l plan.Lowered) error {
	if err := validateLowered(l); err != nil {
		return err
	}
	k.low = l.Plan
	k.defaultCacheRecord = l.DefaultCacheRecord
	k.cascadeAttention = l.Reported.CascadeAttention
	c := l.Plan.Collectives
	k.layout.SequenceParallelMoE = c.SequenceParallelMoE
	k.layout.ModelSequenceParallel = c.SequenceParallelModel
	k.layout.CustomAllReduce = l.Reported.SMAllReduce
	k.allReduce = c.AllReduce
	if k.allReduce == plan.AllReduceUnstated {
		// An engine that predates Collectives.AllReduce states only whether the reduction runs on
		// SMs; custom and NCCL are the two kernels that statement can mean.
		k.allReduce = plan.AllReduceNCCL
		if l.Reported.SMAllReduce {
			k.allReduce = plan.AllReduceCustom
		}
	}
	k.overrides = l.Overrides
	for _, n := range l.Notes {
		k.assume(n.Name, n.Value, n.Reason)
	}
	return nil
}

// graphModeOfLaunch is the kernel's launch mode for a plan launch word.
func graphModeOfLaunch(l plan.Launch) graphMode {
	switch l {
	case plan.LaunchGraph:
		return graphModeFull
	case plan.LaunchPiecewise:
		return graphModePiecewise
	}
	return graphModeEager
}

// captureSetName names the set of graphs a pair of launches captures, as the registry's capture
// memory keys it (cudagraph_capture_bytes_<set>): a full graph for every step, a piecewise graph
// for every step, a full decode graph beside piecewise ones, a full decode graph alone, or none.
// A pair no engine captures has no figure, and is refused rather than priced at another set's.
func captureSetName(decode, other graphMode) (string, error) {
	switch {
	case decode == graphModeFull && other == graphModeFull:
		return "full", nil
	case decode == graphModePiecewise && other == graphModePiecewise:
		return "piecewise", nil
	case decode == graphModeFull && other == graphModePiecewise:
		return "full_and_piecewise", nil
	case decode == graphModeFull && other == graphModeEager:
		return "full_decode_only", nil
	case decode == graphModeEager && other == graphModeEager:
		return "none", nil
	}
	return "", fmt.Errorf("the plan launches a uniform decode step as %s and any other step as %s, "+
		"a pair of captures the registry states no memory figure for (Plan.Graphs.Decode, Other)",
		decode, other)
}
