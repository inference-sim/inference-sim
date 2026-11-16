package latencykernel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// cloneLowered is a deep copy of the parts the kernel keeps: a caller editing the plan it passed does
// not change a kernel built from it.
func cloneLowered(l plan.Lowered) plan.Lowered {
	l.Overrides = slices.Clone(l.Overrides)
	l.Notes = slices.Clone(l.Notes)
	l.Plan.Graphs.Buckets = slices.Clone(l.Plan.Graphs.Buckets)
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
	switch {
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
	k.cascadeAttention = l.Reported.CascadeAttention
	c := l.Plan.Collectives
	k.layout.SequenceParallelMoE = c.SequenceParallelMoE
	k.layout.ModelSequenceParallel = c.SequenceParallelModel
	k.layout.CustomAllReduce = l.Reported.SMAllReduce
	for _, o := range l.Overrides {
		k.layout.Overrides = append(k.layout.Overrides, resolve.Override{Field: o.Field,
			Requested: o.Requested, Resolved: o.Resolved, Reason: o.Reason})
	}
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
