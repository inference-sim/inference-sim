package resolve

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// Layout is one pool's resolved parallelism and the engine decisions that follow from it.
//
// Resolution is where a requested setting becomes the setting that will run. A scenario
// states what a deployment asked for; the layout may not permit it. Recording both, and
// the reason, is what lets a prediction be read without re-deriving this logic.
type Layout struct {
	TP, PP, DP  int
	PCP, DCP    int
	ExpertWidth int
	// MoEGroupWidth is how many ranks an MoE layer's dispatch and combine span: the
	// expert-parallel width with expert parallelism on, and with it off, tp x pcp x dp
	// when dp exceeds one -- vLLM v0.31.0 then still shares the MoE across the data-parallel
	// ranks, sharding every expert over them and gathering their tokens to it
	// ("Detected DP deployment with no --enable-expert-parallel. Falling back to
	// AllGather+ReduceScatter dispatch/combine", vllm/model_executor/layers/fused_moe/
	// all2all_utils.py:202-214; the group is the EP group, distributed/parallel_state.py:
	// 2212-2220). One otherwise: a tensor-parallel MoE with no data parallelism dispatches
	// nothing.
	MoEGroupWidth int

	GPUsPerNode int
	GPUsPerRack int
	// NodesSpanned is how many nodes the widest collective group covers. It selects
	// which coefficients apply and whether a cross-node penalty is charged at all.
	NodesSpanned int

	// SequenceParallelMoE is whether the engine makes the MoE input sequence-parallel.
	// When it does, a layer's tensor-parallel reduction is replaced by a reduce-scatter
	// and all-gather pair, so the two are alternatives rather than both running.
	SequenceParallelMoE bool

	// ModelSequenceParallel is whether the model's own sequence parallelism is on, as the
	// rules pack decides it for the architecture the graph names. DeepSeek-V4.1 then shards
	// its residual stream by token across the tensor-parallel ranks, and replaces each
	// attention all-reduce with an all-gather and a reduce-scatter
	// (vllm/models/deepseek_v41/nvidia/model.py:239-247, _use_sequence_parallel, and
	// 579-586 at v0.31.0).
	ModelSequenceParallel bool

	// CustomAllReduce is whether the SM-consuming reduction kernel will run. A request
	// for it is declined when the group spans nodes without multi-node NVLink, or when
	// the rank count is one the kernel does not support — in which case the cost moves
	// from SMs to the NIC.
	CustomAllReduce bool

	// Overrides records each request the layout could not honour.
	Overrides []Override
}

// MoEGroup is the width an MoE layer's dispatch spans: MoEGroupWidth, which is never
// narrower than the expert group. Taking the larger of the two keeps a Layout built
// field by field, as a test does, consistent with one ResolveLayout returns.
func (l Layout) MoEGroup() int {
	if l.MoEGroupWidth > l.ExpertWidth {
		return l.MoEGroupWidth
	}
	return l.ExpertWidth
}

// MoEDispatches reports whether an MoE layer runs a dispatch and combine at all.
//
// With expert parallelism on, vLLM v0.31.0 runs its all-to-all kernels only when
// use_all2all_kernels holds: use_ep and (dp > 1, pcp > 1, or a sequence-parallel MoE input)
// (vllm/model_executor/layers/fused_moe/config.py:1024-1027). Otherwise
// maybe_make_prepare_finalize builds the no-DP prepare/finalize, which moves nothing
// (all2all_utils.py:188-214), the experts run on the replicated tokens, and the runner's
// tensor-parallel all-reduce combines them (fused_moe/runner/moe_runner.py:506-512) -- the
// graphs' mlp_out reduction. The input is sequence-parallel under the engine's rule
// (SequenceParallelMoE, which needs dp > 1) or the model's own (DeepSeek-V4.1 passes its
// _use_sequence_parallel to FusedMoE as is_sequence_parallel, so sp_size = tp,
// vllm/models/deepseek_v4/nvidia/model.py:912 and fused_moe/layer.py:55). With expert
// parallelism on, ExpertWidth is tp x pcp x dp, so dp or pcp above one is ExpertWidth
// above TP, which keeps a Layout built field by field, as a test does, answering the same.
//
// With it off, the naive all-gather/reduce-scatter dispatch runs when dp exceeds one
// (do_naive_dispatch_combine, moe_runner.py:797-800; MoEGroupWidth).
func (l Layout) MoEDispatches() bool {
	if l.ExpertWidth > 1 {
		return l.ExpertWidth > max(l.TP, 1) || l.SequenceParallelMoE || l.ModelSequenceParallel
	}
	return l.MoEGroupWidth > 1
}

// Override is one declined request, with the reason.
type Override struct {
	Field     string
	Requested string
	Resolved  string
	Reason    string
}

// Fabric is the hardware facts a layout needs. They come from the catalog rather than from
// an engine rules pack, because whether a rack is one NVLink domain is a property of the
// part and not of the software running on it.
type Fabric struct {
	// RackIsOneDomain is true where multi-node NVLink spans the rack, so a group inside
	// it crosses no boundary a collective must pay for.
	RackIsOneDomain bool
	// IntraNodeBwGBps and InterNodeBwGBps give the ratio a cross-node collective is
	// scaled by. The ratio belongs to the (chip, fabric) pairing rather than to either
	// alone, which is why both arrive together.
	IntraNodeBwGBps float64
	InterNodeBwGBps float64
	// InterNodeEfficiency is the measured fraction of InterNodeBwGBps a transfer over the
	// fabric achieves (blis-registry fabric_bandwidth_efficiency); zero reads as one, the
	// nominal rate.
	InterNodeEfficiency float64
}

// EffectiveInterNodeBwGBps is the fabric's nominal per-GPU rate corrected by its measured
// efficiency.
func (f Fabric) EffectiveInterNodeBwGBps() float64 {
	if f.InterNodeEfficiency > 0 {
		return f.InterNodeBwGBps * f.InterNodeEfficiency
	}
	return f.InterNodeBwGBps
}

// Ratio returns the intra-to-inter bandwidth ratio a collective's span is scaled by. One
// when no fabric is stated, which is the single-node case: nothing crosses, so nothing is
// scaled. The inter-node side is the effective rate, so a fabric that achieves less of its
// line rate scales a crossing collective's bytes by more.
func (f Fabric) Ratio() float64 {
	inter := f.EffectiveInterNodeBwGBps()
	if inter <= 0 || f.IntraNodeBwGBps <= 0 {
		return 1
	}
	return f.IntraNodeBwGBps / inter
}

// ResolveLayout derives a pool's layout from the scenario, the fabric it runs on, and the
// engine rules for its declared version.
func ResolveLayout(s *scenario.Scenario, pool deployment.Pool, eng enginesettings.Settings, fab Fabric,
	rules EngineRules) (Layout, error) {
	return ResolveLayoutFor(s, pool, eng, fab, rules, "")
}

// ResolveLayoutFor is ResolveLayout for a model whose graph names its architecture, which
// decides whether the model's own sequence parallelism is on.
func ResolveLayoutFor(s *scenario.Scenario, pool deployment.Pool, eng enginesettings.Settings, fab Fabric,
	rules EngineRules, architecture string) (Layout, error) {
	pl := pool.Parallel
	if pl.TP < 1 || pl.DP < 1 {
		return Layout{}, fmt.Errorf("tensor- and data-parallel widths must be at least 1")
	}
	l := Layout{
		TP: pl.TP, PP: pl.PP, DP: pl.DP, PCP: pl.PCP, DCP: pl.DCP,
		ExpertWidth: pl.ExpertParallelWidth(),
		GPUsPerNode: s.Cluster.GPUsPerNode,
		GPUsPerRack: s.Cluster.GPUsPerRack,
	}

	// The widest group decides how many nodes a collective spans. Expert parallelism is
	// usually the widest; tensor parallelism spans nodes only where it exceeds a node.
	//
	// A prefill-context group is narrow but SPREAD: its members are tp ranks apart, since
	// the engine numbers ranks DP x PP x PCP x TP with TP innermost
	// (vllm/distributed/parallel_state.py:2054-2060, :2155-2160 at v0.31.0), so it covers
	// the whole tp x pcp block. At tp=8 and pcp=2 that is two nodes for a two-rank group,
	// which counting its width alone would call one. The decode-context group never
	// covers more than that block either (parallel.py:563-578), so the block bounds both.
	l.MoEGroupWidth = l.ExpertWidth
	if !pl.EnableExpertParallel && pl.DP > 1 {
		l.MoEGroupWidth = pl.TP * max(pl.PCP, 1) * pl.DP
	}
	widest := max(l.TP, l.MoEGroupWidth, l.TP*max(l.PCP, 1))
	if l.GPUsPerNode > 0 {
		l.NodesSpanned = (widest + l.GPUsPerNode - 1) / l.GPUsPerNode
	} else {
		l.NodesSpanned = 1
	}
	// Multi-node NVLink makes a rack one domain, so a group inside it spans no node
	// boundary a collective has to pay for.
	if l.GPUsPerRack > 0 && widest <= l.GPUsPerRack && fab.RackIsOneDomain {
		l.NodesSpanned = 1
	}

	l.SequenceParallelMoE = rules.SequenceParallelMoE(
		eng.All2AllBackend, pl.EnableExpertParallel, pl.TP, pl.DP)
	if architecture != "" {
		l.ModelSequenceParallel = rules.ModelSequenceParallel(architecture, pl,
			eng.MoEBackend)
	}
	l.CustomAllReduce, l.Overrides = resolveCustomAllReduce(eng, l, fab, rules)
	return l, nil
}

// resolveCustomAllReduce decides whether the SM-consuming reduction kernel runs, and
// records why when a request for it is declined.
func resolveCustomAllReduce(e enginesettings.Settings, l Layout, fab Fabric,
	rules EngineRules) (bool, []Override) {
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

	var overrides []Override
	if !rules.CustomAllReduceSupportsWidth(l.TP) {
		return false, append(overrides, Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"the kernel supports only certain rank counts and %d is not one, so the "+
					"reduction moves from SMs to the NIC", l.TP),
		})
	}
	// Crossing a node boundary disqualifies the kernel unless every rank has multi-node
	// NVLink. The catalog records whether a part does, through the rack tier.
	if l.GPUsPerNode > 0 && l.TP > l.GPUsPerNode && !fab.RackIsOneDomain {
		return false, append(overrides, Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"a tensor-parallel width of %d spans more than the %d GPUs in a node and "+
					"the part has no multi-node NVLink, so the reduction moves from SMs to the NIC",
				l.TP, l.GPUsPerNode),
		})
	}
	return true, overrides
}

// Emits reports whether a conditional node is part of the graph under this layout.
//
// This is the whole of the condition evaluation, and it is a switch rather than an
// expression evaluator. The graph names which question to ask; the layout answers it. A
// condition the graph could state and this function could not answer is impossible,
// because the schema's enum and this switch are the same closed set.
func (l Layout) Emits(c model.EmitCondition) bool {
	switch c {
	case model.EmitAlways:
		return true
	case model.EmitTensorParallel:
		return l.TP > 1
	case model.EmitExpertParallel:
		// The MoE dispatch and combine. blis-schemas words the condition as the
		// expert-parallel width exceeding one; the engine is what this answers for, and
		// it dispatches only where MoEDispatches says.
		return l.MoEDispatches()
	case model.EmitTensorParallelUnlessSequenceParallelMoE:
		return l.TP > 1 && !l.SequenceParallelMoE
	case model.EmitSequenceParallel:
		return l.ModelSequenceParallel
	case model.EmitTensorParallelUnlessSequenceParallel:
		return l.TP > 1 && !l.ModelSequenceParallel
	case model.EmitTensorParallelWithoutExpertParallel:
		// The experts are sharded by tensor parallelism only while expert parallelism is
		// off; under it every expert is whole on one rank (FusedMoEParallelConfig.make,
		// vllm/model_executor/layers/fused_moe/config.py: ep_size = tp_size, tp_size = 1).
		return l.TP > 1 && l.ExpertWidth <= 1
	case model.EmitExpertParallelWithoutDispatch:
		// Expert parallelism over a group that holds every token: no all-to-all moves them
		// (MoEDispatches is false), so each rank's partial expert output is all-reduced
		// across the group, which is the tensor-parallel group here.
		return l.ExpertWidth > 1 && l.TP > 1 && !l.MoEDispatches()
	}
	// An unrecognized condition answers false here, but it never reaches this line from
	// the pricer: BuildPlan asks Recognizes first and rejects the graph, because silently
	// dropping a node would remove a cost with nothing reporting it.
	return false
}

// Recognizes reports whether Emits can answer a condition, so a resolver can reject a
// graph rather than price it with a node quietly missing.
func Recognizes(c model.EmitCondition) bool {
	switch c {
	case model.EmitAlways, model.EmitTensorParallel, model.EmitExpertParallel,
		model.EmitTensorParallelUnlessSequenceParallelMoE, model.EmitSequenceParallel,
		model.EmitTensorParallelUnlessSequenceParallel,
		model.EmitTensorParallelWithoutExpertParallel, model.EmitExpertParallelWithoutDispatch:
		return true
	}
	return false
}
