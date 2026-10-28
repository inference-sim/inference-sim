// Moved from blis-latency-kernel/memory_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Fixed memory occupancy is the registry's magnitudes in NVIDIA's composition, with nothing
// the kernel hardcodes and nothing Provenance cannot see (inference-sim/blis-latency-kernel#22).
// liftMemory states the form and its sources.

// memoryFixture is the deployment #22 measured: deepseek-v3, tp=8, fp8 on h200, 8,192
// batched tokens, PIECEWISE.
const memoryFixture = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"

// THE FIGURES #22 STATES, against the pinned registry (blis-registry v0.1.1). Pinned
// because the issue's verification names them; a re-vendoring that changes them is
// supposed to break this, and the commit that does so says why.
//
//	at tp=8: activation 1,174,405,120  capture 858,783,744  communicator 411,041,792
//	         workspace 3,758,096,384 -- 2,033,188,864 of fixed occupancy beside weights
//	         inside the budget, and 4,169,138,176 held before vLLM's memory snapshot
//	at tp=2: the communicator is 358,612,992, not the 411,041,792 a constant gave
func TestFixedOccupancyMatchesTheFiguresTheIssueStates(t *testing.T) {
	k := fixture(t, memoryFixture)
	got := k.FixedBytes()
	if got.ActivationPeak != 1174405120 || got.CUDAGraph != 858783744 ||
		k.PreSnapshotBytes() != 411041792+3758096384 {
		t.Errorf("tp=8: activation %d, capture %d, communicator+workspace %d; want "+
			"1174405120, 858783744 and 4169138176", got.ActivationPeak, got.CUDAGraph,
			k.PreSnapshotBytes())
	}
	if nonWeight := got.Total() - got.Weights - got.EPLBRedundant; nonWeight != 2033188864 {
		t.Errorf("tp=8: %d bytes of fixed occupancy beside weights, want 2033188864", nonWeight)
	}
	in := fixtureInputs(t, memoryFixture)
	in.Deployment.Pools[0].Parallel.TP = 2
	at2, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	if comm := at2.PreSnapshotBytes() - 3758096384; comm != 358612992 {
		t.Errorf("tp=2: communicator %d bytes, want 358612992", comm)
	}
}

// OCCUPANCY ENTERS NO STEP TIME. Scaling every memory magnitude in the registry threefold
// must move FixedBytes and leave every step, transfer and overhead exactly where it was:
// these terms decide how many sequences fit, not how long a step takes. It is the
// metamorphic form of #22's requirement that cmd/shape and cmd/score do not move.
func TestMemoryMagnitudesMoveOccupancyAndNoStepTime(t *testing.T) {
	memory := func(name string) bool {
		for _, p := range []string{"nccl_communicator_bytes_", "engine_workspace_bytes",
			"cudagraph_capture_bytes_", "activation_buffer_count_",
			"activation_scratch_floor_bytes"} {
			if strings.HasPrefix(name, p) {
				return true
			}
		}
		return false
	}
	base := fixtureInputs(t, memoryFixture)
	scaled := fixtureInputs(t, memoryFixture)
	for i, s := range scaled.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if memory(c.Coefficients[j].Name) {
				c.Coefficients[j].Value *= 3
			}
		}
		scaled.Coefficients[i] = &c
	}
	a, err := newKernel(base)
	if err != nil {
		t.Fatal(err)
	}
	b, err := newKernel(scaled)
	if err != nil {
		t.Fatal(err)
	}
	fa, fb := a.FixedBytes(), b.FixedBytes()
	if fb.CommBuffers != 3*fa.CommBuffers || fb.CUDAGraph != 3*fa.CUDAGraph ||
		fb.ActivationPeak != 3*fa.ActivationPeak {
		t.Errorf("tripling the memory magnitudes gave %+v from %+v; each term is linear in "+
			"its coefficient above the activation floor", fb, fa)
	}
	if fb.Weights != fa.Weights || fb.EPLBRedundant != fa.EPLBRedundant {
		t.Error("the memory magnitudes moved the weights or the EPLB replicas")
	}
	for shape, batch := range batchShapes() {
		if x, y := a.StepTime(batch), b.StepTime(batch); x.Overlap != y.Overlap ||
			x.NoOverlap != y.NoOverlap {
			t.Errorf("%s: the memory magnitudes moved the step from %v/%v to %v/%v",
				shape, x.Overlap, x.NoOverlap, y.Overlap, y.NoOverlap)
		}
	}
	if a.SequenceVariableBytes(8192) != b.SequenceVariableBytes(8192) ||
		a.AdmissionOverhead(1024) != b.AdmissionOverhead(1024) {
		t.Error("the memory magnitudes moved per-sequence capacity or a host overhead")
	}
}

// ACTIVATION SCRATCH DOES NOT FOLLOW THE SERVED WEIGHT FORMAT. #22 proposed sizing it at the
// served width, which would halve it for fp8. blis-registry#33 established the opposite and
// this kernel follows it: vLLM's fp8 linears return out_dtype=x.dtype
// (vllm/model_executor/kernels/linear/scaled_mm/cutlass.py at v0.31.0), so the residual
// stream and the inter-layer buffers stay bf16 under fp8 weights.
//
// qwen3-30b-a3b is a bf16 checkpoint, so serving it fp8 is a genuine change of weight
// format; the activation scratch must not move while the weights do.
func TestActivationScratchDoesNotFollowTheServedWeightFormat(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(quant string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		in.Model, in.Scenario.Model = qwen, qwen.Name
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = quant })
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("quantization %q: %v", quant, err)
		}
		return k
	}
	bf16, fp8 := build(""), build("fp8")
	if bf16.FixedBytes().Weights <= fp8.FixedBytes().Weights {
		t.Fatalf("serving the bf16 checkpoint fp8 did not shrink its weights (%d against "+
			"%d); the comparison needs a real change of format",
			fp8.FixedBytes().Weights, bf16.FixedBytes().Weights)
	}
	if a, b := bf16.FixedBytes().ActivationPeak, fp8.FixedBytes().ActivationPeak; a != b {
		t.Errorf("activation scratch is %d bytes served bf16 and %d served fp8; activations "+
			"stay at the model's compute dtype", a, b)
	}
}

// THE MODE IS CHOSEN PER BATCH. vLLM v0.31.0 dispatches a uniform decode batch to a full
// graph under FULL_AND_PIECEWISE and FULL_DECODE_ONLY, and every other batch to piecewise or
// to no graph respectively (vllm/config/compilation.py:604-640,
// vllm/v1/cudagraph_dispatcher.py:233-310). So the host term of each mode must equal the
// host term of the single mode that batch actually runs under. A uniform decode batch with
// speculation verifies 1 + num_spec_tokens tokens per request (gpu_model_runner.py:883).
func TestTheGraphModeIsChosenPerBatch(t *testing.T) {
	host := func(mode string, spec int, b kernel.Batch) int64 {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		e := mustDecode(t, in.Deployment.Pools[0])
		e.CUDAGraphMode = mode
		if spec > 0 {
			e.Speculative = &engineconfig.Speculative{Method: "mtp", NumSpecTokens: spec}
		}
		setEngine(t, &in.Deployment.Pools[0], e)
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		return int64(k.StepTime(b).PerResource[kernel.ResourceHost])
	}
	decode := decodeBatch(32, 1, 4096)
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 4095, PromptLen: 4096},
		{Scheduled: 256, Computed: 0, PromptLen: 256}, // 257 tokens: inside the 512 capture
	}}
	for _, c := range []struct {
		mode        string
		decodeAs    string
		otherwiseAs string
	}{
		{"FULL_AND_PIECEWISE", "FULL", "PIECEWISE"},
		{"FULL_DECODE_ONLY", "FULL", "NONE"},
	} {
		if a, b := host(c.mode, 0, decode), host(c.decodeAs, 0, decode); a != b {
			t.Errorf("%s priced a uniform decode batch's host time at %d, want %s's %d",
				c.mode, a, c.decodeAs, b)
		}
		if a, b := host(c.mode, 0, mixed), host(c.otherwiseAs, 0, mixed); a != b {
			t.Errorf("%s priced a mixed batch's host time at %d, want %s's %d",
				c.mode, a, c.otherwiseAs, b)
		}
	}
	// The three modes differ on a mixed batch, or the comparison above proves nothing.
	if host("FULL", 0, mixed) == host("PIECEWISE", 0, mixed) ||
		host("PIECEWISE", 0, mixed) == host("NONE", 0, mixed) {
		t.Fatal("FULL, PIECEWISE and NONE price a mixed batch's host time alike")
	}
	// Under speculation a decode of 1 + k tokens per request is the uniform batch.
	spec := decodeBatch(32, 3, 4096)
	if a, b := host("FULL_AND_PIECEWISE", 2, spec), host("FULL", 2, spec); a != b {
		t.Errorf("a uniform speculative decode (3 tokens, 2 speculative) priced %d under "+
			"FULL_AND_PIECEWISE, want FULL's %d", a, b)
	}
}

// PROVENANCE SEES EVERY FIGURE FixedBytes REPORTS, and Evidence counts them. Before #22 the
// 1.3 GB of hardcoded occupancy had no entry at all, so Evidence's "measured of total"
// overstated the prediction's footing. Every memory coefficient the composition read must
// now appear, from the registry set that supplied it; and an engine default the kernel
// filled in must appear as a kernel assumption, which a fully stated deployment has none of
// -- except the cudagraph capture ceiling, which blis-schemas has no field to state, so any
// capturing deployment reports vLLM's default for it, and the engine release, which the
// fixture declares as 0.29.0 (a release the vLLM rules packs cover) where the kernel encodes
// 0.31.0.
func TestProvenanceSeesEveryFigureFixedBytesReports(t *testing.T) {
	k := fixture(t, memoryFixture)
	sets := map[string]string{}
	for _, o := range k.Provenance() {
		sets[o.Name] = o.Set
	}
	for _, name := range []string{
		"nccl_communicator_bytes_8rank", "engine_workspace_bytes",
		"cudagraph_capture_bytes_piecewise", "activation_buffer_count_moe_8rank",
		"activation_scratch_floor_bytes",
	} {
		set, ok := sets[name]
		if !ok {
			t.Errorf("Provenance has no entry for %s, which FixedBytes reports", name)
		}
		if set == KernelAssumptionSet {
			t.Errorf("%s is attributed to the kernel; it is a registry magnitude", name)
		}
	}
	for _, o := range k.Provenance() {
		if o.Set == KernelAssumptionSet && o.Name != "max_cudagraph_capture_size" &&
			o.Name != "engine_version" {
			t.Errorf("a deployment stating every engine setting reported the kernel "+
				"assumption %s", o.Name)
		}
	}

	// Leave the three settings unstated and each becomes a disclosed assumption, counted by
	// Evidence among the assumptions and in the total.
	in := fixtureInputs(t, memoryFixture)
	e := mustDecode(t, in.Deployment.Pools[0])
	e.MaxNumBatchedTokens, e.BlockSize, e.CUDAGraphMode = 0, 0, ""
	setEngine(t, &in.Deployment.Pools[0], e)
	open, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	_, totalStated, _ := k.Evidence()
	_, totalOpen, assumed := open.Evidence()
	for _, name := range []string{"max_num_batched_tokens", "block_size", "cudagraph_mode"} {
		if !hasAssumption(open, name) {
			t.Errorf("an unstated %s is not disclosed in Provenance", name)
		}
		found := false
		for _, a := range assumed {
			found = found || a == name
		}
		if !found {
			t.Errorf("Evidence does not list %s among the assumptions", name)
		}
	}
	if totalOpen != totalStated+3 {
		t.Errorf("Evidence counts %d entries with three settings unstated against %d "+
			"stated; each assumption is one entry", totalOpen, totalStated)
	}
}

// Every magnitude is required. A scenario that does not list cost-model-memory is refused,
// naming the coefficient, rather than priced with zero occupancy for the terms it lacks --
// which would overstate how many sequences fit by gigabytes per rank.
func TestAScenarioWithoutTheMemorySetIsRefused(t *testing.T) {
	in := fixtureInputs(t, memoryFixture)
	var kept []*coefficient.Set
	for _, s := range in.Coefficients {
		if s.Name != "cost-model-memory" {
			kept = append(kept, s)
		}
	}
	if len(kept) == len(in.Coefficients) {
		t.Fatal("the fixture does not list cost-model-memory")
	}
	in.Coefficients = kept
	in.Scenario.Coefficients = in.Scenario.Coefficients[:len(in.Scenario.Coefficients)-1]
	_, err := newKernel(in)
	if err == nil || !strings.Contains(err.Error(), "cost-model-memory") {
		t.Errorf("New without the memory set returned %v; want a refusal naming it", err)
	}
}

func hasAssumption(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && o.Set == KernelAssumptionSet &&
			o.Method == string(vocab.MethodAssumed) {
			return true
		}
	}
	return false
}

// AN MLA TOKEN IS ONE LATENT VECTOR, end to end. deepseek-v3 caches kv_lora_rank +
// qk_rope_head_dim = 576 elements per token per layer, over 61 layers, at one byte under an
// fp8 cache, with no value tensor (vllm/v1/kv_cache_interface.py:674-675 at v0.31.0) and no
// tensor-parallel reduction of its single head. One 16-token page is therefore
// 16 x 61 x 576 bytes on every rank, at tp=8 as at tp=1.
func TestAnMLATokenCachesOneLatentVector(t *testing.T) {
	for _, tp := range []int{1, 8} {
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Parallel.TP = tp
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := k.SequenceVariableBytes(16), int64(16*61*576); got != want {
			t.Errorf("tp=%d: a 16-token page holds %d bytes, want %d (one 576-wide latent "+
				"vector per token per layer, fp8)", tp, got, want)
		}
	}
}

// EPLB'S REPLICAS ARE COUNTED ONCE. A rank holds its share of the model's experts plus its
// share of the redundant ones, and that physical total is what FixedBytes().Total() must
// report: enabling EPLB with R redundant experts may raise Total by exactly the replicas'
// bytes, which is what EPLBRedundant reports. An earlier form reported the replicas in
// EPLBRedundant AND inside Weights, so Total rose by twice that.
//
// minimax-m2.5 at EP 8 holds 256 experts as 32 per rank; 8 redundant ones make it 33, one
// expert per layer more, which is EPLBRedundant's figure too.
func TestEPLBReplicasAreCountedOnce(t *testing.T) {
	build := func(redundant int) kernel.MemoryBreakdown {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
		if redundant > 0 {
			editEngine(t, &in, func(e *engineconfig.Engine) {
				e.EPLB = &engineconfig.EPLB{Enabled: true, NumRedundantExperts: redundant}
			})
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.FixedBytes()
	}
	off, on := build(0), build(8)
	if on.EPLBRedundant <= 0 {
		t.Fatalf("EPLB with 8 redundant experts reported %d replica bytes", on.EPLBRedundant)
	}
	if grew := on.Total() - off.Total(); grew != on.EPLBRedundant {
		t.Errorf("enabling 8 redundant experts raised Total by %d bytes, want the replicas' "+
			"%d: a replica is held once", grew, on.EPLBRedundant)
	}
	if on.Weights != off.Weights {
		t.Errorf("enabling EPLB moved Weights from %d to %d; the replicas belong in "+
			"EPLBRedundant", off.Weights, on.Weights)
	}
}

// AN UNSTATED RECURRENT CACHE MODE IS WHAT THE ENGINE RUNS. The config default is "none"
// (vllm/config/cache.py:190 at v0.31.0), but vLLM sets a hybrid model's mode to "align"
// whenever prefix caching is on (vllm/model_executor/models/config.py:640-642), and prefix
// caching is on by default (cache.py:142). The hook cannot tell a stated "none" from the
// default, so a stated "none" runs as "align" too, and is reported in Resolved. So on
// Nemotron-3-Ultra an unstated or stated-none mode must hold what "align" holds, and "none"
// only once prefix caching is turned off.
func TestAnUnstatedRecurrentCacheModeIsTheEnginesEffectiveDefault(t *testing.T) {
	build := func(mode string, prefixCaching *bool) int64 {
		t.Helper()
		in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
		e := mustDecode(t, in.Deployment.Pools[0])
		e.MambaCacheMode, e.EnablePrefixCaching = mode, prefixCaching
		setEngine(t, &in.Deployment.Pools[0], e)
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.SequenceFixedBytes()
	}
	off := false
	align, none := build("align", nil), build("none", &off)
	if align <= none {
		t.Fatalf("align holds %d bytes per sequence against none's %d; the comparison needs "+
			"them to differ", align, none)
	}
	if got := build("none", nil); got != align {
		t.Errorf("a stated none with prefix caching on holds %d bytes, want align's %d: the "+
			"engine overrides it", got, align)
	}
	in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MambaCacheMode = "none" })
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	reported := false
	for _, o := range k.Resolved().Overrides {
		reported = reported || (o.Field == "mamba_cache_mode" && o.Resolved == "align")
	}
	if !reported {
		t.Errorf("a stated none run as align is not reported in Resolved: %+v",
			k.Resolved().Overrides)
	}
	if got := build("", nil); got != align {
		t.Errorf("an unstated mode with prefix caching on holds %d bytes, want align's %d",
			got, align)
	}
	if got := build("", &off); got != none {
		t.Errorf("an unstated mode with prefix caching off holds %d bytes, want none's %d",
			got, none)
	}
}

// COLLECTIVES MOVE ACTIVATIONS, AND ACTIVATIONS ARE BF16. Serving a bf16 checkpoint fp8
// halves its weight bytes and must leave every collective exactly where it was: a quantized
// linear returns its input's dtype (vllm/model_executor/kernels/linear/scaled_mm/
// cutlass.py:147,153 at v0.31.0), so the hidden state a tensor-parallel all-reduce carries is
// bf16 either way. An earlier form sized it at the served width and halved it under fp8.
func TestServingFP8LeavesTheCollectivesWhereTheyWere(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(quant string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		in.Model, in.Scenario.Model = qwen, qwen.Name
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = quant })
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	bf16, fp8 := build(""), build("fp8")
	for shape, b := range batchShapes() {
		x, y := bf16.StepTime(b).PerResource, fp8.StepTime(b).PerResource
		if x[kernel.ResourceNVLink] != y[kernel.ResourceNVLink] {
			t.Errorf("%s: collectives cost %v served bf16 and %v served fp8", shape,
				x[kernel.ResourceNVLink], y[kernel.ResourceNVLink])
		}
	}
}

// A DATA-PARALLEL MoE WITH EXPERT PARALLELISM OFF IS ONE SHARED MoE, NOT dp REPLICAS.
// vLLM v0.31.0 shards every expert over the dp x pcp x tp ranks
// (flatten_tp_across_dp_and_pcp, vllm/model_executor/layers/fused_moe/config.py:1090-1098;
// pcp is 1 here, so dp x tp) and gathers every rank's tokens to it with an all-gather/reduce-scatter dispatch, whatever all2all backend is named
// ("Detected DP deployment with no --enable-expert-parallel. Falling back to
// AllGather+ReduceScatter", all2all_utils.py:202-214). Four consequences, each asserted on
// minimax-m2.5 at tp=8 with dp 1, 2 and 4:
//
//   - a rank's expert weights divide by dp: the expert part of Weights halves per doubling;
//   - per-rank routed compute does not move: dp times the tokens, each on a 1/dp slice;
//   - a dispatch now runs over the MoE group, so inflating the all-to-all floors moves a
//     step at dp > 1 and not at dp = 1, where no dispatch exists;
//   - a routed backend named with expert parallelism off prices as the fallback it is.
func TestADataParallelMoEWithoutExpertParallelismIsShared(t *testing.T) {
	build := func(dp int, edit func(*Inputs)) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		pool := &in.Deployment.Pools[0]
		pool.Parallel.EnableExpertParallel = false
		pool.Parallel.DP = dp
		pool.Nodes, in.Scenario.Cluster.Nodes = dp, dp
		if dp > 1 {
			in.Scenario.Cluster.Fabric = "ib-400g"
			f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			in.Fabric = f
		}
		if edit != nil {
			edit(&in)
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("dp=%d: %v", dp, err)
		}
		return k
	}
	w1, w2, w4 := build(1, nil).FixedBytes().Weights, build(2, nil).FixedBytes().Weights,
		build(4, nil).FixedBytes().Weights
	if d12, d24 := float64(w1-w2), float64(w2-w4); d12 <= 0 || math.Abs(d12/d24-2) > 1e-6 {
		t.Errorf("expert weights per rank did not halve per doubling of dp: Weights %d, %d, "+
			"%d at dp 1, 2, 4", w1, w2, w4)
	}

	// Per-rank routed compute: the SM term of a large decode is the same at dp 1 and 2,
	// since the dense work per rank is unchanged and the routed work is dp tokens on 1/dp.
	big := decodeBatch(256, 1, 4096)
	if a, b := build(1, nil).StepTime(big).PerResource[kernel.ResourceSM],
		build(2, nil).StepTime(big).PerResource[kernel.ResourceSM]; a != b {
		t.Errorf("per-rank compute moved from %v at dp=1 to %v at dp=2; the gathered tokens "+
			"each reach a half-size slice", a, b)
	}

	inflate := func(in *Inputs) { in.Coefficients = scaleFloors(in.Coefficients, "alltoall", 10) }
	decode := decodeBatch(32, 1, 4096)
	if a, b := build(1, nil).StepTime(decode).NoOverlap,
		build(1, inflate).StepTime(decode).NoOverlap; a != b {
		t.Errorf("at dp=1 inflating the all-to-all floors moved the step from %v to %v; a "+
			"tensor-parallel MoE dispatches nothing", a, b)
	}
	// At dp=2 the fallback dispatch is an all-gather and a reduce-scatter over the
	// data-parallel ranks (AgRsAll2AllManager, all2all.py:84-157), so inflating the
	// all-gather floors moves the step by more than the tensor-parallel all-gathers alone
	// do at dp=1.
	gather := func(in *Inputs) { in.Coefficients = scaleFloors(in.Coefficients, "all_gather", 10) }
	d1 := build(1, gather).StepTime(decode).NoOverlap - build(1, nil).StepTime(decode).NoOverlap
	d2 := build(2, gather).StepTime(decode).NoOverlap - build(2, nil).StepTime(decode).NoOverlap
	if d2 <= d1 {
		t.Errorf("inflating the all-gather floors moved the step %v at dp=2 and %v at dp=1; "+
			"the MoE's dispatch gathers over the data-parallel ranks", d2, d1)
	}

	named := build(2, func(in *Inputs) {
		editEngine(t, in, func(e *engineconfig.Engine) { e.All2AllBackend = "deepep_low_latency" })
	})
	if a, b := build(2, nil).StepTime(decode).NoOverlap, named.StepTime(decode).NoOverlap; a != b {
		t.Errorf("naming deepep_low_latency with expert parallelism off priced %v against %v "+
			"for the default; the engine falls back to all-gather/reduce-scatter", b, a)
	}
}

// PAST THE CAPTURE CEILING NO GRAPH RUNS. vLLM v0.31.0 captures token counts up to
// max_cudagraph_capture_size -- unstated, min(max_num_seqs x decode_query_len x 2, 512) off
// data-center Blackwell (vllm/config/vllm.py:2442-2460) -- and dispatches a larger batch with
// no graph whatever the mode (vllm/v1/worker/gpu/cudagraph_utils.py:499-529). The fixture
// states max_num_seqs 256, so its ceiling is 512 tokens: a 512-token prefill runs piecewise,
// a 513-token one runs eager, and under every capturing mode the two sides of the ceiling
// must price like PIECEWISE and NONE respectively.
func TestABatchPastTheCaptureCeilingRunsWithNoGraph(t *testing.T) {
	host := func(mode string, tokens int) time.Duration {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = mode })
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.StepTime(decodeBatch(1, tokens, tokens)).PerResource[kernel.ResourceHost]
	}
	for _, mode := range []string{"PIECEWISE", "FULL_AND_PIECEWISE"} {
		if a, b := host(mode, 512), host("PIECEWISE", 512); a != b {
			t.Errorf("%s: a 512-token prefill priced %v of host time, want PIECEWISE's %v",
				mode, a, b)
		}
		if a, b := host(mode, 513), host("NONE", 513); a != b {
			t.Errorf("%s: a 513-token prefill priced %v of host time, want NONE's %v; it is "+
				"past every captured size", mode, a, b)
		}
	}
	if host("PIECEWISE", 512) == host("NONE", 512) {
		t.Fatal("PIECEWISE and NONE price alike, so this test cannot discriminate")
	}
}

// UNDER DCP A MIXED STEP IS TWO LAUNCHES EVEN ON FLASHATTENTION. With decode-context
// parallelism FlashAttention runs _forward_with_dcp instead of its one varlen kernel
// (vllm/v1/attention/backends/flash_attn.py:1344-1357 at v0.31.0): a context pass and a
// query pass over the whole step, then a merge (:1543-1717). So the one-launch rebate the
// previous test asserts must not apply: at dcp=2 the same mixed step prices identically on
// the h200 and on the h200 marked data-center Blackwell.
func TestUnderDCPAMixedStepGetsNoOneLaunchRebate(t *testing.T) {
	const fixture = "minimax-m25-h200-tp8.yaml"
	build := func(blackwell bool) *Kernel {
		t.Helper()
		return mustDCPVariant(t, fixture, 8, 1, 2, func(in *Inputs) {
			if blackwell {
				chip := *in.Chip
				chip.NVFP4Peak = 1
				in.Chip = &chip
			}
		})
	}
	fa, fi := build(false), build(true)
	busy := func(k *Kernel, b kernel.Batch) time.Duration {
		p := k.StepTime(b).PerResource
		return p[kernel.ResourceSM] + p[kernel.ResourceHBM]
	}
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: append(append([]kernel.ReqShape{},
		decodeBatch(8, 1, 4096).Reqs...), decodeBatch(1, 256, 256).Reqs...)}
	if a, c := busy(fa, mixed), busy(fi, mixed); a != c {
		t.Errorf("at dcp=2 a mixed step priced %v on FlashAttention and %v on FlashInfer; "+
			"_forward_with_dcp is two launches on both", a, c)
	}
}

// The buffers scale as vLLM sizes them: the indexer's gather with the context bound, at 40 x
// 132 B a token of context on a ratio-1 layer, and the drafter's hidden-state and top-k
// buffers and its workspace lane only while the deployment speculates.
func TestV41EngineBuffersFollowTheContextBoundAndSpeculation(t *testing.T) {
	with := func(edit func(*engineconfig.Engine)) *Kernel {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, edit)
		return build(t, in)
	}
	base := with(func(*engineconfig.Engine) {})
	longer := with(func(e *engineconfig.Engine) { e.MaxModelLen = 262144 })
	if d := longer.FixedBytes().ActivationPeak - base.FixedBytes().ActivationPeak; d != 40*131072*132 {
		t.Errorf("doubling the context bound to 262,144 moved the activation by %d B, want "+
			"the indexer gather's 40 x 131,072 x 132 = %d", d, 40*131072*132)
	}
	plain := with(func(e *engineconfig.Engine) { e.Speculative = nil })
	const drafter = 8192*20480*2 + 8192*512*4 + 38305792
	if d := base.FixedBytes().ActivationPeak - plain.FixedBytes().ActivationPeak; d != drafter {
		t.Errorf("speculation adds %d B of buffers, want DSpark's hidden-state and top-k "+
			"buffers and its workspace lane, %d", d, drafter)
	}
}

// PROPERTY: fixed occupancy never falls as the token budget or the context bound grows,
// since every buffer the engine sizes from either is sized at least as large.
func TestFixedOccupancyIsMonotoneInTheTokenBudgetAndContextBound(t *testing.T) {
	for _, scenario := range []string{v41, memoryFixture} {
		total := func(mnbt, mml int) int64 {
			in := fixtureInputs(t, scenario)
			e := mustDecode(t, in.Deployment.Pools[0])
			e.MaxNumBatchedTokens, e.MaxModelLen = mnbt, engineconfig.MaxModelLen(mml)
			setEngine(t, &in.Deployment.Pools[0], e)
			return build(t, in).FixedBytes().Total()
		}
		prev := int64(0)
		for _, mnbt := range []int{2048, 4096, 8192, 16384, 32768, 65536} {
			if got := total(mnbt, 65536); got < prev {
				t.Errorf("%s: %d batched tokens hold %d B, fewer than a smaller budget's %d",
					scenario, mnbt, got, prev)
			}
			prev = total(mnbt, 65536)
		}
		prev = 0
		for _, mml := range []int{8192, 65536, 131072, 262144, 1048576} {
			if mml > 163840 && scenario == memoryFixture {
				continue // past deepseek-v3's context window
			}
			if got := total(8192, mml); got < prev {
				t.Errorf("%s: a %d-token bound holds %d B, fewer than a shorter one's %d",
					scenario, mml, got, prev)
			}
			prev = total(8192, mml)
		}
	}
}

// METAMORPHIC: stating the default capture size prices every step as leaving it unstated;
// only Provenance changes, as nothing was assumed.
func TestStatingTheDefaultCaptureSizeChangesNoPrice(t *testing.T) {
	unstated := build(t, fixtureInputs(t, v41))
	def := unstated.CaptureTokens()
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxCUDAGraphCaptureSize = &def })
	stated := build(t, in)
	for _, b := range []kernel.Batch{decodeBatch(1, 6, 8192), decodeBatch(85, 6, 8192),
		decodeBatch(86, 6, 8192), prefillOf(512, 0), prefillOf(8192, 90000)} {
		if a, s := unstated.StepTime(b), stated.StepTime(b); !reflect.DeepEqual(a, s) {
			t.Errorf("%d tokens: stating the default %d priced %v against %v", b.Tokens(), def,
				s.NoOverlap, a.NoOverlap)
		}
	}
	if !hasAssumption(unstated, "max_cudagraph_capture_size") {
		t.Error("an unstated capture size is not recorded as an assumption")
	}
}

// The stated knobs resolve as the engine resolves them: a maximum above the token budget is
// capped at it, and reported; a maximum that is not the largest of the stated sizes is
// refused, as vLLM refuses it at startup.
func TestStatedCaptureSizesResolveAsTheEngineDoes(t *testing.T) {
	in := fixtureInputs(t, v41)
	big := 100000
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxCUDAGraphCaptureSize = &big })
	k := build(t, in)
	if got := k.CaptureTokens(); got != 8192 {
		t.Errorf("a stated 100,000 runs as %d, want the 8,192-token budget", got)
	}
	reported := false
	for _, o := range k.Resolved().Overrides {
		reported = reported || (o.Field == "max_cudagraph_capture_size" && o.Resolved == "8192")
	}
	if !reported {
		t.Error("the cap at max_num_batched_tokens is not reported as an override")
	}
	in = fixtureInputs(t, v41)
	m := 64
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxCUDAGraphCaptureSize = &m })
	editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphCaptureSizes = []int{8, 16, 32} })
	if _, err := newKernel(in); err == nil {
		t.Error("a maximum of 64 over sizes up to 32 was priced; vLLM refuses it")
	}
}

// GraphModes reports the modes that run, after the engine's own overrides.
func TestGraphModesAreTheResolvedOnes(t *testing.T) {
	for _, c := range []struct {
		mode, decode, other string
		edit                func(*Inputs)
	}{
		{"", "FULL", "PIECEWISE", nil},
		{"PIECEWISE", "PIECEWISE", "PIECEWISE", nil},
		{"NONE", "NONE", "NONE", nil},
		{"FULL_AND_PIECEWISE", "NONE", "NONE", func(in *Inputs) {
			pl := &in.Deployment.Pools[0].Parallel
			pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
			editEngine(t, in, func(e *engineconfig.Engine) { e.All2AllBackend = "deepep_high_throughput" })
		}},
	} {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = c.mode })
		if c.edit != nil {
			c.edit(&in)
		}
		k := build(t, in)
		if d, o := k.GraphModes(); d != c.decode || o != c.other {
			t.Errorf("mode %q: GraphModes() = %s, %s; want %s, %s", c.mode, d, o, c.decode,
				c.other)
		}
		if (c.decode == "NONE") != (k.CaptureTokens() == 0) {
			t.Errorf("mode %q: CaptureTokens() = %d", c.mode, k.CaptureTokens())
		}
	}
}

// A RECURRENT STATE IS SHARDED BY TENSOR PARALLELISM. vLLM divides a recurrent layer's
// conv and temporal state by the tensor-parallel width (mamba2_state_shape,
// gated_delta_net_state_shape, kda_state_shape, vllm/model_executor/layers/mamba/
// mamba_utils.py:195-326 at v0.31.0), so a sequence's fixed state halves each time the width
// doubles. The graph states the whole model's heads. An earlier form charged every rank
// the whole state: about 377 MB a sequence at every width on Qwen3.5-397B.
func TestRecurrentStateIsShardedByTensorParallelism(t *testing.T) {
	at := func(tp int) int64 {
		in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
		in.Deployment.Pools[0].Parallel.TP = tp
		return build(t, in).SequenceFixedBytes()
	}
	two, four, eight := at(2), at(4), at(8)
	if two <= 0 || two != 2*four || four != 2*eight {
		t.Errorf("a sequence's recurrent state is %d, %d and %d B at tp 2, 4 and 8; each "+
			"doubling of the width must halve it", two, four, eight)
	}
}

// captureContract is docs/COEFF_CONTRACT.md v2.3's capture entries for FULL_AND_PIECEWISE,
// at test magnitudes: a floor and a per-token cost for each graph family.
func captureContract(drop ...string) []coefficient.Entry {
	var out []coefficient.Entry
	for name, v := range map[string]float64{
		"cudagraph_capture_floor_bytes_full_and_piecewise":            256 << 20,
		"cudagraph_capture_bytes_per_mixed_token_full_and_piecewise":  1 << 20,
		"cudagraph_capture_bytes_per_decode_token_full_and_piecewise": 8 << 20,
	} {
		skip := false
		for _, d := range drop {
			skip = skip || d == name
		}
		if !skip {
			out = append(out, coefficient.Entry{Name: name, Value: v, Units: "bytes_per_rank",
				Method: "assumed", Rationale: "test", Scope: coefficient.Scope{
					Hardware: []string{"h200"}, Model: []string{"deepseek-v4.1-flash"}}})
		}
	}
	return out
}

// v41Engine is the V4.1 fixture at the stated engine knobs, less the registry's capture
// entries, with entries appended as their own set.
func v41Engine(t *testing.T, entries []coefficient.Entry, seqs, mnbt, capture int) *Kernel {
	t.Helper()
	// blis-registry v0.2.0 publishes the three capture entries for V4.1 on h100 and h200;
	// strip them so each test here states its own (or none, for the flat fallback).
	names := map[string]bool{}
	for _, e := range captureContract() {
		names[e.Name] = true
	}
	in := withoutEntries(fixtureInputs(t, v41), names)
	e := mustDecode(t, in.Deployment.Pools[0])
	e.MaxNumSeqs, e.MaxNumBatchedTokens, e.MaxCUDAGraphCaptureSize = seqs, mnbt, &capture
	if len(entries) > 0 {
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
			&coefficient.Set{Kind: "CoefficientSet", Name: "test-capture", Coefficients: entries})
	}
	setEngine(t, &in.Deployment.Pools[0], e)
	return build(t, in)
}

// WITHOUT THE v2.3 ENTRIES THE CAPTURE IS THE FLAT FIGURE, whatever the capture size or
// max_num_seqs, so a registry without them prices byte-identically; a partial set falls back
// to it and says so.
func TestCaptureMemoryFallsBackToTheFlatFigure(t *testing.T) {
	flat := v41Engine(t, nil, 256, 8192, 512).FixedBytes().CUDAGraph
	for _, c := range [][2]int{{2, 16}, {56, 512}, {1024, 1024}} {
		if got := v41Engine(t, nil, c[0], 8192, c[1]).FixedBytes().CUDAGraph; got != flat {
			t.Errorf("max_num_seqs %d, capture %d: %d B without the entries, want the flat %d",
				c[0], c[1], got, flat)
		}
	}
	k := v41Engine(t, captureContract("cudagraph_capture_bytes_per_decode_token_full_and_piecewise"),
		56, 8192, 512)
	if got := k.FixedBytes().CUDAGraph; got != flat {
		t.Errorf("a partial set charges %d B, want the flat %d", got, flat)
	}
	if !hasFallback(k, "cudagraph_capture_floor_bytes_full_and_piecewise") {
		t.Error("a partial capture set fell back with nothing recorded")
	}
}

// THE FORM. floor + per-mixed-token x capture ceiling + per-decode-token x decode ceiling:
// at 56 requests and a 512 ceiling, 256 MiB + 512 MiB + 336 x 8 MiB.
func TestCaptureMemoryIsTheContractsForm(t *testing.T) {
	got := v41Engine(t, captureContract(), 56, 8192, 512).FixedBytes().CUDAGraph
	if want := int64(256<<20 + 512<<20 + 336*8<<20); got != want {
		t.Errorf("capture %d B, want %d", got, want)
	}
}

// PROPERTY: the KV pool never grows as max_num_seqs, the capture ceiling or the token
// budget grows, with the v2.3 entries present and without them: every term vLLM sizes from
// them is at least as large (decodeCaptureTokens, samplerPeakBytes, the activation form).
func TestTheKVPoolIsNonIncreasingInSeqsCaptureAndBudget(t *testing.T) {
	for _, entries := range [][]coefficient.Entry{nil, captureContract()} {
		total := func(seqs, mnbt, capture int) int64 {
			return v41Engine(t, entries, seqs, mnbt, capture).FixedBytes().Total()
		}
		check := func(what string, xs []int, at func(int) int64) {
			prev := int64(0)
			for _, x := range xs {
				got := at(x)
				if got < prev {
					t.Errorf("entries %v: %s %d holds %d B, fewer than a smaller one's %d",
						entries != nil, what, x, got, prev)
				}
				prev = got
			}
		}
		check("max_num_seqs", []int{1, 2, 8, 32, 56, 128, 1024, 4096},
			func(x int) int64 { return total(x, 8192, 512) })
		check("capture", []int{8, 16, 64, 256, 512, 1024, 2048},
			func(x int) int64 { return total(256, 8192, x) })
		check("max_num_batched_tokens", []int{1024, 2048, 4096, 8192, 32768},
			func(x int) int64 { return total(256, x, min(x, 512)) })
	}
	k := func(seqs, capture int) int64 {
		return v41Engine(t, captureContract(), seqs, 8192, capture).FixedBytes().CUDAGraph
	}
	if k(56, 512) <= k(48, 512) {
		t.Error("with the entries, more requests under the same ceiling capture no more")
	}
	if k(1024, 512) <= k(1024, 256) {
		t.Error("with the entries, a higher ceiling captures no more")
	}
}

// THE DUMMY SAMPLER can set the profile peak: at a small token budget and many requests its
// logits outgrow the forward's scratch, and then the peak follows max_num_seqs. DSpark's K
// query rows a request bound the dummy batch at max_num_batched_tokens / K requests.
func TestTheDummySamplerRunCanSetTheActivationPeak(t *testing.T) {
	act := func(seqs int) int64 { return v41Engine(t, nil, seqs, 2048, 512).FixedBytes().ActivationPeak }
	if act(16) != act(64) {
		t.Errorf("at 16 and 64 requests the forward's scratch binds, yet the peak moves: %d, %d",
			act(16), act(64))
	}
	if act(409) <= act(64) {
		t.Errorf("409 requests' logits do not raise the peak: %d against %d", act(409), act(64))
	}
	if act(409) != act(4096) {
		t.Errorf("past 2,048 / 5 = 409 dummy requests the peak still moves: %d, %d",
			act(409), act(4096))
	}
}

// THE PUBLISHED V4.1 CAPTURE FORM REPRODUCES THE START-UPS IT WAS FITTED TO. blis-registry
// c5e24a7 fits floor, per-mixed-token and per-decode-token to five InferenceX H200 TP8
// start-ups (K 5, max_num_seqs 1024, mnbt 8192); vLLM logged "CUDA graph pool memory: ...
// E GiB (estimated)" per rank at each capture ceiling. At the published values this kernel's
// form must give each E within 0.15 GiB, and with max_num_seqs cut to 8 at the 512 ceiling
// (C_decode 48) it must charge less: the decode family follows max_num_seqs, as vLLM's
// estimate does (_init_candidates, vllm/v1/worker/gpu/cudagraph_utils.py:313-327).
func TestThePublishedV41CaptureFormReproducesItsStartUps(t *testing.T) {
	published := []coefficient.Entry{}
	for name, v := range map[string]float64{
		"cudagraph_capture_floor_bytes_full_and_piecewise":            849150826,
		"cudagraph_capture_bytes_per_mixed_token_full_and_piecewise":  9972071,
		"cudagraph_capture_bytes_per_decode_token_full_and_piecewise": 6772854,
	} {
		published = append(published, coefficient.Entry{Name: name, Value: v,
			Units: "bytes_per_rank", Method: "measured", Fitted: true, Rationale: "test",
			Scope: coefficient.Scope{Hardware: []string{"h200"},
				Model: []string{"deepseek-v4.1-flash"}}})
	}
	logged := map[int]float64{64: 1.69, 128: 2.71, 256: 4.90, 512: 8.61, 1024: 16.67}
	for capture, want := range logged {
		got := float64(v41Engine(t, published, 1024, 8192, capture).FixedBytes().CUDAGraph) /
			(1 << 30)
		if math.Abs(got-want) > 0.15 {
			t.Errorf("capture %d: %.2f GiB, vLLM estimated %.2f", capture, got, want)
		}
	}
	few := v41Engine(t, published, 8, 8192, 512).FixedBytes().CUDAGraph
	if want := int64(849150826 + 9972071*512 + 6772854*48); few != want {
		t.Errorf("max_num_seqs 8, capture 512: %d B, want %d (decode ceiling 48)", few, want)
	}
}
