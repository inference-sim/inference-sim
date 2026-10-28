// Moved from blis-latency-kernel/deepep_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/vocab"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// dep8 is DeepSeek-V4.1-Flash as one DEP8 engine on the fixture's node: tp 1, dp 8, expert
// parallelism on, so the MoE dispatch runs over an 8-rank expert group inside the node.
// entries are extra moe_a2a_* coefficients for the fixture's part, as {name: value}; drop
// names registry entries to remove first.
func dep8(t *testing.T, backend, mode string, mnbt int, entries map[string]float64,
	drop ...string) *Kernel {
	t.Helper()
	names := map[string]bool{}
	for _, n := range drop {
		names[n] = true
	}
	in := withoutEntries(fixtureInputs(t, v41), names)
	pl := &in.Deployment.Pools[0].Parallel
	pl.TP, pl.DP, pl.EnableExpertParallel = 1, 8, true
	e := mustDecode(t, in.Deployment.Pools[0])
	e.All2AllBackend, e.CUDAGraphMode, e.MaxNumBatchedTokens = backend, mode, mnbt
	set := &coefficient.Set{Kind: "CoefficientSet", Name: "test-deepep"}
	for name, v := range entries {
		units := "bytes_per_us"
		if strings.HasPrefix(name, "moe_a2a_floor_") {
			units = "us_per_transfer"
		}
		set.Coefficients = append(set.Coefficients, coefficient.Entry{
			Name: name, Value: v, Units: vocab.Unit(units), Method: "measured", Rationale: "test",
			Scope: coefficient.Scope{Hardware: []string{"h200"}}})
	}
	if len(entries) > 0 {
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
	}
	setEngine(t, &in.Deployment.Pools[0], e)
	return build(t, in)
}

// moeLayers counts the MoE layers a step runs, the drafter's included while it speculates.
func moeLayers(k *Kernel) int {
	n := 0
	layers := k.plan.Layers
	if d := k.plan.Draft; d != nil && k.speculating {
		layers = append(append(layers[:len(layers):len(layers)], d.Layers...), d.Context...)
	}
	for _, l := range layers {
		if l.TopK > 0 {
			n += l.Count
		}
	}
	return n
}

func priced(variant string, floorUS, rate float64) map[string]float64 {
	return map[string]float64{
		"moe_a2a_floor_" + variant + "_8rank_h200": floorUS,
		"moe_a2a_rate_" + variant + "_8rank_h200":  rate,
	}
}

// A measured DeepEP backend prices each MoE layer's dispatch and combine as its own floor
// plus bytes over its own rate, not as NCCL's all-to-all (blis-latency-kernel#57): raising
// the floor by 100 us raises a decode step by 100 us per MoE layer.
func TestDeepEPDispatchIsPricedFromItsOwnMeasurement(t *testing.T) {
	b := decodeBatch(32, 1, 20000)
	lo := dep8(t, "deepep_low_latency", "FULL_AND_PIECEWISE", 256,
		priced("deepep_low_latency", 100, 1e5))
	hi := dep8(t, "deepep_low_latency", "FULL_AND_PIECEWISE", 256,
		priced("deepep_low_latency", 200, 1e5))
	want := time.Duration(moeLayers(lo)) * 100 * time.Microsecond
	got := hi.StepTime(b).NoOverlap - lo.StepTime(b).NoOverlap
	if d := got - want; d < -time.Microsecond || d > time.Microsecond {
		t.Errorf("100 us more floor moved the step by %v; %d MoE layers want %v",
			got, moeLayers(lo), want)
	}
}

// Without a measurement for the part, the routed all-to-all stands and that is recorded.
// blis-registry v0.2.0 measures deepep_high_throughput on h200, so the fixture's own
// measurements are stripped first.
func TestAnUnmeasuredDeepEPBackendIsRecorded(t *testing.T) {
	k := dep8(t, "deepep_high_throughput", "", 8192, nil,
		"moe_a2a_floor_deepep_high_throughput_8rank_h200",
		"moe_a2a_rate_deepep_high_throughput_8rank_h200")
	if k.deepEP != nil {
		t.Fatal("priced from a DeepEP measurement the scenario does not carry")
	}
	for _, o := range k.Provenance() {
		if o.Name == "moe_a2a_deepep_high_throughput" {
			return
		}
	}
	t.Error("the fallback to the generic routed all-to-all is not in Provenance")
}

// DeepEP v2 runs its capturable path whenever enforce_eager is off and its expanding eager
// path otherwise (all2all_utils.py:286-298), so the two take different measurements.
func TestDeepEPV2TakesItsPathFromTheGraphMode(t *testing.T) {
	entries := priced("deepep_v2_graph", 50, 1e5)
	for n, v := range priced("deepep_v2_eager", 500, 1e5) {
		entries[n] = v
	}
	for mode, want := range map[string]string{
		"FULL_AND_PIECEWISE": "deepep_v2_graph", "NONE": "deepep_v2_eager"} {
		k := dep8(t, "deepep_v2", mode, 2048, entries)
		if k.deepEP == nil || k.deepEP.variant != want {
			t.Errorf("cudagraph_mode %s: priced %v, want %s", mode, k.deepEP, want)
		}
	}
}

// DeepEP's buffers are made after vLLM's memory snapshot and so come out of the KV cache:
// HT holds VLLM_DEEPEP_BUFFER_SIZE_MB of NVLink buffer, LL that plus an RDMA buffer sized by
// max_num_batched_tokens, and the all-gather dispatch none.
func TestDeepEPBuffersComeOutOfTheKVCache(t *testing.T) {
	const mib = 1 << 20
	ag := dep8(t, "allgather_reducescatter", "", 256, nil).FixedBytes()
	ht := dep8(t, "deepep_high_throughput", "", 256, nil).FixedBytes()
	ll := dep8(t, "deepep_low_latency", "", 256, nil).FixedBytes()
	ll2 := dep8(t, "deepep_low_latency", "", 512, nil).FixedBytes()
	if ag.CommBuffers != 0 {
		t.Errorf("all-gather dispatch holds %d B of DeepEP buffer", ag.CommBuffers)
	}
	if ht.CommBuffers != 1024*mib {
		t.Errorf("HT holds %d B; want the 1024 MiB NVLink buffer", ht.CommBuffers)
	}
	rdma := lowLatencyRDMABytes(256, 5120, 8, 384)
	if ll.CommBuffers != 1024*mib+rdma {
		t.Errorf("LL holds %d B; want 1024 MiB + %d B RDMA", ll.CommBuffers, rdma)
	}
	if ll2.CommBuffers <= ll.CommBuffers {
		t.Error("LL's RDMA buffer does not grow with max_num_batched_tokens")
	}
}

// allgather_reducescatter gathers every DP rank's tokens to every rank and reduce-scatters
// them back (AgRsAll2AllManager, all2all.py:84-157), so at DEP8 a rank's dispatch moves the
// whole group's rows through an NCCL all-gather and reduce-scatter, not its own rows once.
func TestAllGatherDispatchMovesTheWholeGroupsTokens(t *testing.T) {
	k := dep8(t, "allgather_reducescatter", "", 8192, nil)
	const tokens = 4096
	rows := float64(tokens) * 5120 * 2
	ag := collKey{Op: model.OpAllGather, Group: price.GroupExpert}
	rs := collKey{Op: model.OpReduceScatter, Group: price.GroupExpert}
	perLayer := k.collectiveSeconds(ag, 8*rows) + k.collectiveSeconds(rs, 8*rows)
	got := k.StepTime(prefillOf(tokens, 0)).PerResource[kernel.ResourceNVLink]
	// The target's MoE layers: a prefill step runs no drafter forward over its tokens.
	layers := 0
	for _, l := range k.plan.Layers {
		if l.TopK > 0 {
			layers += l.Count
		}
	}
	want := time.Duration(float64(layers) * perLayer * float64(time.Second))
	// Within 0.5%: the step's other NVLink terms are a fraction of a millisecond.
	if r := float64(got) / float64(want); r < 1 || r > 1.005 {
		t.Errorf("a %d-token DEP8 prefill charges %v of NVLink; %d MoE layers of an 8-rank "+
			"all-gather and reduce-scatter of the group's %d rows want %v",
			tokens, got, layers, 8*tokens, want)
	}
}

// Unstated, max_num_batched_tokens is 256 under a batched data-parallel MoE
// (arg_utils.py:3046-3051), and that is what sizes the LL buffer. A layout that runs no
// dispatch -- a tp8 dp1 decode pool -- builds no DeepEP buffer at all.
func TestTheLowLatencyBufferFollowsTheEnginesTokenBudget(t *testing.T) {
	ll := dep8(t, "deepep_low_latency", "", 0, nil)
	if want := int64(1024<<20) + lowLatencyRDMABytes(256, 5120, 8, 384); ll.FixedBytes().CommBuffers != want {
		t.Errorf("unstated budget: LL holds %d B, want %d (256 tokens)",
			ll.FixedBytes().CommBuffers, want)
	}
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.All2AllBackend = "deepep_low_latency" })
	if b := build(t, in).FixedBytes().CommBuffers; b != 0 {
		t.Errorf("tp8 dp1 dispatches nothing yet holds %d B of DeepEP buffer", b)
	}
}

// DeepEP low-latency's kernels are captured into a decode step's graph, so a graphed step is
// priced from the graph-mode measurement, and an eager one from the eager measurement. Raising
// the graph floor by 100 us moves a captured decode step by 100 us per MoE layer and leaves
// an eager step (cudagraph_mode NONE) where it was.
func TestDeepEPLowLatencyTakesTheGraphPriceWhenTheStepIsCaptured(t *testing.T) {
	b := decodeBatch(32, 1, 20000)
	with := func(mode string, graphFloor float64) *Kernel {
		e := priced("deepep_low_latency", 300, 1e5)
		for k, v := range priced("deepep_low_latency_graph", graphFloor, 1e5) {
			e[k] = v
		}
		return dep8(t, "deepep_low_latency", mode, 256, e)
	}
	lo, hi := with("FULL_AND_PIECEWISE", 50), with("FULL_AND_PIECEWISE", 150)
	want := time.Duration(moeLayers(lo)) * 100 * time.Microsecond
	if got := hi.StepTime(b).NoOverlap - lo.StepTime(b).NoOverlap; got < want-time.Microsecond ||
		got > want+time.Microsecond {
		t.Errorf("100 us more graph floor moved a captured decode step by %v; want %v", got, want)
	}
	if d := with("NONE", 150).StepTime(b).NoOverlap - with("NONE", 50).StepTime(b).NoOverlap; d != 0 {
		t.Errorf("the graph floor moved an eager step by %v; it must not", d)
	}
	// A step past the capture ceiling runs eagerly whatever the mode.
	past := decodeBatch(lo.captureTokens+1, 1, 20000)
	if d := hi.StepTime(past).NoOverlap - lo.StepTime(past).NoOverlap; d != 0 {
		t.Errorf("the graph floor moved a step past the capture ceiling by %v; it must not", d)
	}
}
