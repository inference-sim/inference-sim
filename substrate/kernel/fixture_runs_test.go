package latencykernel

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/kernel/internal/resolve"
)

// SUB-6, SUB-15: the kernel runs the collectives the plan says run (Plan.Collectives.Runs) and
// evaluates no condition of its own. A test plan of plain values (fixtureLowered) therefore has
// to state them, as an engine's lowering does. completeRuns states one for every collective node
// of the graph the kernel will plan that the plan does not already state, by the rule below. The
// rule is this test's own and resolves nothing for any engine: it is the answer the cost model
// itself gave before the engine stated it, kept so that the cases which price one setting hold
// every other collective where it was.

// runsUnder answers a graph's emit condition for a layout.
func runsUnder(l resolve.Layout, c model.EmitCondition) bool {
	switch c {
	case model.EmitAlways:
		return true
	case model.EmitTensorParallel:
		return l.TP > 1
	case model.EmitExpertParallel:
		return l.MoEDispatches()
	case model.EmitTensorParallelUnlessSequenceParallelMoE:
		return l.TP > 1 && !l.SequenceParallelMoE
	case model.EmitSequenceParallel:
		return l.ModelSequenceParallel
	case model.EmitTensorParallelUnlessSequenceParallel:
		return l.TP > 1 && !l.ModelSequenceParallel
	case model.EmitTensorParallelWithoutExpertParallel:
		return l.TP > 1 && l.ExpertWidth <= 1
	case model.EmitExpertParallelWithoutDispatch:
		return l.ExpertWidth > 1 && l.TP > 1 && !l.MoEDispatches()
	}
	return false
}

// completeRuns is in with a plan that states every collective node of the graph the kernel will
// plan. A node the plan already states keeps its answer. An input the kernel would refuse for
// another reason is returned as it is, so New reports that reason.
func completeRuns(in Inputs) Inputs {
	if in.Lowered == nil || in.Model == nil || in.Deployment == nil || in.Chip == nil {
		return in
	}
	family, _ := in.Chip.CapabilityMajor()
	g, err := in.Model.ForCapability(family)
	if err != nil {
		return in
	}
	if in.Draft != nil {
		d, err := in.Draft.ForCapability(family)
		if err != nil {
			return in
		}
		if g, err = withDrafter(g, d); err != nil {
			return in
		}
	}
	pl := in.Deployment.Pools[in.PoolIndex].Parallel
	lay := resolve.Layout{TP: pl.TP, PP: pl.PP, DP: pl.DP, PCP: pl.PCP, DCP: pl.DCP,
		ExpertWidth: pl.ExpertParallelWidth(), MoEGroupWidth: pl.ExpertParallelWidth(),
		SequenceParallelMoE:   in.Lowered.Plan.Collectives.SequenceParallelMoE,
		ModelSequenceParallel: in.Lowered.Plan.Collectives.SequenceParallelModel}
	if !pl.EnableExpertParallel && pl.DP > 1 {
		lay.MoEGroupWidth = pl.TP * max(pl.PCP, 1) * pl.DP
	}
	l := cloneLowered(*in.Lowered)
	have := map[collectiveKey]bool{}
	for _, r := range l.Plan.Collectives.Runs {
		have[collectiveKey{r.Section, r.Op, r.Role}] = true
	}
	section := func(name string, nodes []model.Node) {
		for _, n := range nodes {
			k := collectiveKey{name, string(n.Op), n.Role}
			if !n.Op.Collective() || have[k] {
				continue
			}
			have[k] = true
			l.Plan.Collectives.Runs = append(l.Plan.Collectives.Runs,
				plan.CollectiveRun{Section: name, Op: k.op, Role: k.role, Runs: runsUnder(lay, n.Emit)})
		}
	}
	for _, lk := range g.LayerKinds {
		section(plan.LayerSection(lk.ID), lk.Nodes)
	}
	section(plan.SectionHead, g.Head)
	section(plan.SectionEmbedding, g.Embedding)
	if sp := g.Speculator; sp != nil {
		section(plan.SectionDraftEmbedding, sp.Embedding)
		section(plan.SectionDraftHead, sp.Head)
		section(plan.SectionDraftSeqHead, sp.SequentialHead)
	}
	in.Lowered = &l
	return in
}
