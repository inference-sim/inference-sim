// Moved from blis-latency-kernel/memory_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Fixed memory occupancy is the registry's magnitudes in NVIDIA's composition, with nothing
// the kernel hardcodes and nothing Provenance cannot see (inference-sim/blis-latency-kernel#22).
// liftMemory states the form and its sources.

// memoryFixture is the deployment #22 measured: deepseek-v3, tp=8, fp8 on h200, 8,192
// batched tokens, PIECEWISE.
const memoryFixture = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"

// registryValue reads one coefficient the way the kernel resolved it for this deployment,
// so a test can state the composition against the registry rather than against literals a
// refit would invalidate.
func registryValue(t *testing.T, in Inputs, k *Kernel, name string) float64 {
	t.Helper()
	c, err := resolve.Load(in.Coefficients, resolve.Scope{
		Hardware: in.Chip.Name, Model: in.Model.Name, TP: k.layout.TP,
		EP: k.layout.ExpertWidth, NodesSpanned: k.layout.NodesSpanned,
	})
	if err != nil {
		t.Fatal(err)
	}
	v, err := c.Value(name)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// THE COMPOSITION, as an oracle over the registry, at every tensor-parallel width the
// registry declares a communicator for and below it.
//
//	CommBuffers    = nccl_communicator_bytes_<tp>rank (none at one rank) + engine_workspace_bytes
//	CUDAGraph      = cudagraph_capture_bytes_<mode>
//	ActivationPeak = max(activation_buffer_count_moe_<tp>rank * batched * hidden * 2,
//	                     activation_scratch_floor_bytes)
//
// deepseek-v3 is MoE and latent, so its multiple is the MoE row and its width is the hidden
// size. Every term must vary with tp exactly where its coefficient does -- which is the
// defect #22 opens with: the communicator was 392 MiB at every width.
func TestFixedOccupancyIsTheRegistrysComposition(t *testing.T) {
	for _, tp := range []int{1, 2, 4, 8} {
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Parallel.TP = tp
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("tp=%d: %v", tp, err)
		}
		got := k.FixedBytes()
		v := func(name string) float64 { return registryValue(t, in, k, name) }

		comm := v("engine_workspace_bytes")
		if tp > 1 {
			comm += v(fmt.Sprintf("nccl_communicator_bytes_%drank", tp))
		}
		// Held before vLLM's memory snapshot, so outside the budget and out of Total.
		if pre := k.PreSnapshotBytes(); pre != int64(comm) || got.CommBuffers != 0 {
			t.Errorf("tp=%d: PreSnapshotBytes %d and CommBuffers %d, want %d "+
				"(communicator + workspace) and 0", tp, pre, got.CommBuffers, int64(comm))
		}
		if want := int64(v("cudagraph_capture_bytes_piecewise")); got.CUDAGraph != want {
			t.Errorf("tp=%d: CUDAGraph %d, want the piecewise capture %d", tp, got.CUDAGraph, want)
		}
		batched := float64(poolEngine(t, in).MaxNumBatchedTokens)
		act := math.Max(v(fmt.Sprintf("activation_buffer_count_moe_%drank", tp))*batched*
			float64(in.Model.Global.HiddenSize)*2, v("activation_scratch_floor_bytes"))
		if got.ActivationPeak != int64(act) {
			t.Errorf("tp=%d: ActivationPeak %d, want %d", tp, got.ActivationPeak, int64(act))
		}
	}
}

// THE FLOOR. A small token budget is charged NVIDIA's minimum scratch, not its product
// (AISimulate clamps from below at MIN_ACTIVATION_BYTES), and above the floor the scratch
// is linear in the budget.
func TestActivationScratchIsLinearInTheBudgetAboveItsFloor(t *testing.T) {
	at := func(batched int) int64 {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumBatchedTokens = batched })
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.FixedBytes().ActivationPeak
	}
	in := fixtureInputs(t, memoryFixture)
	k := fixture(t, memoryFixture)
	floor := int64(registryValue(t, in, k, "activation_scratch_floor_bytes"))
	if got := at(1); got != floor {
		t.Errorf("a one-token budget gave %d bytes of scratch, want the floor %d", got, floor)
	}
	if a, b := at(4096), at(8192); b != 2*a || a <= floor {
		t.Errorf("4,096 and 8,192 batched tokens gave %d and %d bytes; above the floor the "+
			"scratch doubles with the budget", a, b)
	}
}

// h IS HEADS x HEAD_DIM, NOT HIDDEN_SIZE, for a model whose config declares a head_dim that
// differs. qwen3-30b-a3b runs 32 heads of 128 over a 2,048 hidden size, so its scratch per
// token is twice what hidden_size gives. The kernel reads it from the graph's attention
// node; a latent layer's d_h is a cache width, not a head_dim, so a latent model uses its
// hidden size (see activationWidth for the one place that can differ from NVIDIA's figure).
func TestActivationWidthIsHeadsTimesHeadDim(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
	in.Model, in.Scenario.Model = qwen, qwen.Name
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = "" })
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := k.activationWidth, 4096.0; got != want {
		t.Errorf("qwen3-30b-a3b's activation width is %v, want 32 x 128 = %v, not its "+
			"hidden size %d", got, want, qwen.Global.HiddenSize)
	}
	if got := fixture(t, memoryFixture).activationWidth; got != 7168 {
		t.Errorf("deepseek-v3's activation width is %v, want its hidden size 7168: its "+
			"config states no head_dim", got)
	}
}

// THE CAPTURE IS KEYED BY THE MODE THAT RUNS, and an unstated mode is vLLM v0.31.0's
// default, FULL_AND_PIECEWISE (vllm/config/compilation.py:614) -- for memory and for host
// time alike, and disclosed as an assumption because the engine downgrades it where a
// backend cannot capture a full graph.
func TestTheCaptureIsTheResolvedModesAndAnUnstatedModeIsTheDefault(t *testing.T) {
	build := func(mode string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = mode })
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("mode %q: %v", mode, err)
		}
		return k
	}
	in := fixtureInputs(t, memoryFixture)
	ref := fixture(t, memoryFixture)
	for _, mode := range []string{"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"} {
		want := int64(registryValue(t, in, ref, "cudagraph_capture_bytes_"+strings.ToLower(mode)))
		if got := build(mode).FixedBytes().CUDAGraph; got != want {
			t.Errorf("%s: capture %d, want %d", mode, got, want)
		}
	}
	if build("NONE").FixedBytes().CUDAGraph != 0 {
		t.Error("an eager deployment holds capture memory")
	}

	unstated, stated := build(""), build("FULL_AND_PIECEWISE")
	if unstated.FixedBytes() != stated.FixedBytes() {
		t.Error("an unstated mode's occupancy differs from FULL_AND_PIECEWISE's")
	}
	for shape, b := range batchShapes() {
		if unstated.StepTime(b).NoOverlap != stated.StepTime(b).NoOverlap {
			t.Errorf("%s: an unstated mode priced differently from FULL_AND_PIECEWISE", shape)
		}
	}
	if !hasAssumption(unstated, "cudagraph_mode") || hasAssumption(stated, "cudagraph_mode") {
		t.Error("an unstated mode must be disclosed as a kernel assumption, and a stated one " +
			"must not be")
	}
}

// An unstated token budget is vLLM v0.31.0's own default for the part in the API-server
// context (vllm/engine/arg_utils.py:2858-2887), which the kernel can compute because it holds
// the chip: the activation scratch it yields equals stating that figure.
func TestAnUnstatedTokenBudgetIsTheEnginesDefaultForThePart(t *testing.T) {
	want := map[string]int{
		"a100-80": 2048, "a100-sxm": 2048, "l40s": 2048,
		"h100": 8192, "h200": 8192,
		"b200": 16384, "b300": 16384, "gb200-nvl72": 16384, "gb300": 16384,
	}
	for name, budget := range want {
		chip, err := schemas.LoadChip(filepath.Join(catalogRoot, "hardware", name+".yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if got := defaultMaxNumBatchedTokens(*chip); got != budget {
			t.Errorf("%s (%.1f GiB): default budget %d, want %d", name, chip.MemoryGiB, got,
				budget)
		}
	}
	in := fixtureInputs(t, memoryFixture)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumBatchedTokens = 0 })
	unstated, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	stated := fixture(t, memoryFixture) // states 8,192, h200's default
	if unstated.FixedBytes().ActivationPeak != stated.FixedBytes().ActivationPeak {
		t.Error("an unstated budget on h200 did not size scratch as 8,192 tokens")
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

// A SPARSE-MLA LAYER IS TWO SPLIT POINTS. PIECEWISE splits at every op in
// CompilationConfig._attention_ops (vllm/config/compilation.py:764-782 at v0.31.0), and a DSA
// layer runs two of them: its indexer (vllm::sparse_attn_indexer) and its attention. The
// graph states the indexer as a second, cache-less attention node, so the segment count is
// the layer count, plus one per indexed sparse-MLA layer, plus one -- on glm5, every layer
// indexed; on deepseek-v3, none.
func TestASparseMLALayersIndexerIsAPiecewiseSplitPoint(t *testing.T) {
	for _, c := range []struct {
		fixture string
		indexed bool
	}{
		{dcpSparseFixture, true},
		{dcpMLAFixture, false},
	} {
		k := fixture(t, c.fixture)
		want := k.plan.TotalLayers + 1
		if c.indexed {
			want += k.plan.TotalLayers
		}
		if got := k.plan.PiecewiseSegments(); got != want {
			t.Errorf("%s: %d piecewise segments, want %d", c.fixture, got, want)
		}
	}
}

// A PACKED SPARSE-MLA CACHE IS ITS OWN SIZE. fp8_ds_mla stores 656 bytes a token per layer
// and nvfp4_ds_mla 352 (vllm/model_executor/layers/attention/mla_attention.py:1361-1363 at
// v0.31.0), where the per-element formula gives 576 at one byte. Stated on glm5 (a 576-wide
// head), a 64-token page -- the block a DSA model runs -- must hold exactly that times its
// layers.
//
// A plain "fp8" is whatever the chosen backend makes of it (sparseMLABackend): on h200,
// FlashMLA-sparse serves it as fp8_ds_mla, 656; on the same chip marked data-center
// Blackwell, FlashInfer serves it as stated, 576. "fp8_e4m3" stays 576 on h200, because
// FlashMLA-sparse does not list it and FlashInfer's SM90 sparse MLA does.
func TestAPackedSparseMLACacheIsItsOwnSize(t *testing.T) {
	page := func(cache string, blackwell bool) int64 {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = cache })
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.SequenceVariableBytes(dsaBlockSize)
	}
	layers := int64(fixture(t, dcpSparseFixture).kvLayers)
	for _, c := range []struct {
		cache     string
		blackwell bool
		cell      int64
	}{
		{"fp8_ds_mla", false, 656},
		{"nvfp4_ds_mla", true, 352},
		{"fp8", false, 656},
		{"fp8", true, 576},
		{"fp8_e4m3", false, 576},
	} {
		if got, want := page(c.cache, c.blackwell), dsaBlockSize*c.cell*layers; got != want {
			t.Errorf("%s (blackwell %v): a 64-token page holds %d bytes, want %d (%d a "+
				"token per layer)", c.cache, c.blackwell, got, want, c.cell)
		}
	}
}

// A MIXED STEP IS ONE ATTENTION LAUNCH ON FLASHATTENTION. Without DCP, FlashAttention runs
// one dense varlen kernel over every row of a mixed batch
// (vllm/v1/attention/backends/flash_attn.py:1326-1478 at v0.31.0), so it pays one launch per
// layer where FlashInfer, which splits decode from prefill, pays two. v0.31.0 picks
// FlashAttention for causal full attention everywhere but data-center Blackwell
// (vllm/platforms/cuda.py:154-179).
//
// Asserted as a contrast on one deployment: minimax-m2.5 on h200, and the same chip marked as
// data-center Blackwell (native NVFP4, which nothing else in this fp8-served deployment
// reads). Decode-only and prefill-only steps must price identically on both; a mixed step
// must be cheaper on the FlashAttention part by exactly the smaller floor per attention layer.
func TestAMixedStepIsOneAttentionLaunchOnFlashAttention(t *testing.T) {
	build := func(blackwell bool) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	fa, fi := build(false), build(true)
	busy := func(k *Kernel, b kernel.Batch) time.Duration {
		p := k.StepTime(b).PerResource
		return p[kernel.ResourceSM] + p[kernel.ResourceHBM]
	}
	decode := decodeBatch(8, 1, 4096)
	prefill := decodeBatch(1, 256, 256)
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: append(append([]kernel.ReqShape{},
		decode.Reqs...), prefill.Reqs...)}
	for name, b := range map[string]kernel.Batch{"decode": decode, "prefill": prefill} {
		if a, c := busy(fa, b), busy(fi, b); a != c {
			t.Errorf("a %s-only step priced %v on FlashAttention and %v on FlashInfer", name, a, c)
		}
	}
	// The decode floor is the layer kind's own where the registry carries one, the
	// part-wide one otherwise; the smaller of it and the prefill floor is what one launch
	// saves.
	decodeFloor := fa.attentionFloor
	if fr, ok := fa.attentionByKind[model.AttentionGQA]; ok {
		decodeFloor = fr.floor
	}
	smaller := min(decodeFloor, fa.attentionPrefillFloor)
	var attnLayers int
	for _, l := range fa.plan.Layers {
		if l.AttnQHeads > 0 {
			attnLayers += l.Count
		}
	}
	saved := busy(fi, mixed) - busy(fa, mixed)
	want := time.Duration(float64(smaller) * float64(attnLayers))
	if d := saved - want; d < -2 || d > 2 {
		t.Errorf("a mixed step saved %v on FlashAttention, want one smaller floor per "+
			"attention layer, %v", saved, want)
	}
}

// DeepSeek-V4.1's engine buffers are charged beside the registry's activation scratch, at
// the figures its model code allocates (engineBufferBytes; inference-sim/blis-latency-kernel
// #52). At the fixture -- tp 8, 8,192 batched tokens, a 131,072-token context, DSpark -- they
// are:
//   - the target's workspace, the indexer's 693,108,736 B request (40 x 131,072 entries of
//     128 FP8 values and 4 scale bytes, and 1 MiB of radix top-k), which exceeds the
//     attention's 591,953,920 B (4 x (131,072 + 128 + 8,192) x 512 BF16 and its indices);
//   - DSpark's workspace in its own lane, 38,305,792 B (4 x (128 + 8,192) x 512 BF16,
//     8,192 x 128 and 8,192 int32);
//   - the indexer's 536,870,912 B logits stand-in;
//   - two 16,777,216 B top-k buffers (target and DSpark), a 67,108,864 B candidate-block
//     buffer and DSpark's 335,544,320 B hidden-state buffer;
//   - two 268,435,456 B rotary caches, plain and YaRN, at the model's 1,048,576-position
//     window x 64 FP32 values, whatever the context bound.
func TestV41EngineBuffersAreChargedAtTheEnginesFigures(t *testing.T) {
	in := fixtureInputs(t, v41)
	k := build(t, in)
	generic := math.Max(
		registryValue(t, in, k, "activation_buffer_count_moe_8rank")*8192*5120*2,
		registryValue(t, in, k, "activation_scratch_floor_bytes"))
	const buffers = 693108736 + 38305792 + 536870912 + 2*16777216 + 67108864 + 335544320 +
		2*268435456
	if got := k.FixedBytes().ActivationPeak; got != int64(generic)+buffers {
		t.Errorf("activation %d B, want the registry's %d plus %d of engine buffers",
			got, int64(generic), buffers)
	}
}

// A STATED CAPTURE SIZE IS THE CEILING (inference-sim/blis-latency-kernel#54). vLLM v0.31.0
// captures up to max_cudagraph_capture_size and runs a larger step with no graph
// (vllm/v1/worker/gpu/cudagraph_utils.py:499-529). Stated at 64 with DSpark's K = 5, a
// uniform decode of n requests verifies 6n tokens, so ten requests replay a graph and eleven
// run the target eager. The drafter has its own graphs of 5n tokens
// (dflash/speculator.py:126-150), so at eleven and twelve it still replays one, and only at
// thirteen does the whole step price like cudagraph_mode NONE.
func TestAStatedCaptureSizeIsTheCeiling(t *testing.T) {
	with := func(edit func(*engineconfig.Engine)) *Kernel {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, edit)
		return build(t, in)
	}
	size := 64
	stated := with(func(e *engineconfig.Engine) { e.MaxCUDAGraphCaptureSize = &size })
	none := with(func(e *engineconfig.Engine) { e.CUDAGraphMode = "NONE" })
	if got := stated.CaptureTokens(); got != 64 {
		t.Fatalf("CaptureTokens() = %d with 64 stated", got)
	}
	if hasAssumption(stated, "max_cudagraph_capture_size") {
		t.Error("a stated capture size is still recorded as an assumption")
	}
	host := func(k *Kernel, n int) time.Duration {
		return k.StepTime(decodeBatch(n, 6, 8192)).PerResource[kernel.ResourceHost]
	}
	if host(stated, 10) == host(none, 10) {
		t.Error("ten requests verify 60 tokens, within the stated 64, yet run eager")
	}
	d := stated.plan.Draft
	draft := time.Duration(d.Kernels + d.ContextKernels + 5*d.SequentialKernels)
	for _, n := range []int{11, 12} {
		want := host(none, n) - draft*stated.launchEager + stated.replayPerStep
		if got := host(stated, n); got != want {
			t.Errorf("%d requests verify %d tokens, past the stated 64, and draft %d, "+
				"within it: %v of host time, want NONE's less the drafter's eager launches "+
				"plus its replay, %v", n, 6*n, 5*n, got, want)
		}
	}
	if a, b := host(stated, 13), host(none, 13); a != b {
		t.Errorf("thirteen requests draft 65 tokens, past the stated 64: %v of host time, "+
			"want NONE's %v", a, b)
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

// THE UNIFORM-DECODE CEILING IS vLLM's. Each capture size is rounded up to a multiple of
// 1 + K and kept within max_num_seqs requests and the capture ceiling
// (vllm/v1/worker/gpu/cudagraph_utils.py:313-327 at v0.31.0). K = 5: at 56 requests and a
// 512 ceiling the largest is 336 tokens; with requests to spare it is 498, from the 496
// size, since 512 rounds to 516 and is dropped.
func TestTheDecodeCaptureCeilingIsTheEnginesRounding(t *testing.T) {
	for _, c := range []struct{ seqs, capture, want int }{
		{56, 512, 336}, {48, 512, 288}, {256, 512, 498}, {2, 16, 12}, {16, 64, 60},
		{1024, 1024, 1008},
	} {
		if got := v41Engine(t, nil, c.seqs, 8192, c.capture).decodeCaptureTokens(); got != c.want {
			t.Errorf("max_num_seqs %d, capture %d: decode ceiling %d tokens, want %d",
				c.seqs, c.capture, got, c.want)
		}
	}
}

// METAMORPHIC, TP: per GPU, the weights are R + S/tp, and nothing else this occupancy
// charges divides by tensor parallelism -- the capture is measured per rank, and the
// engine buffers are per rank -- so from TP 8 to TP 4 the non-weight occupancy moves only
// by the activation multiple's own TP key.
func TestTPScalesOnlyTheShardedWeightsAndTheActivationKey(t *testing.T) {
	at := func(tp int) (kernel.MemoryBreakdown, *Kernel, *model.Graph) {
		in := fixtureInputs(t, v41)
		in.Deployment.Pools[0].Parallel.TP = tp
		e := mustDecode(t, in.Deployment.Pools[0])
		capture := 512
		e.MaxNumSeqs, e.MaxCUDAGraphCaptureSize = 56, &capture
		setEngine(t, &in.Deployment.Pools[0], e)
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
			&coefficient.Set{Kind: "CoefficientSet", Name: "test-capture",
				Coefficients: captureContract()})
		k := build(t, in)
		return k.FixedBytes(), k, in.Model
	}
	f4, k4, g := at(4)
	f8, k8, _ := at(8)
	if f4.CUDAGraph != f8.CUDAGraph {
		t.Errorf("capture %d B at TP 4 and %d at TP 8; it is per rank", f4.CUDAGraph, f8.CUDAGraph)
	}
	if s := float64(f4.Weights - f8.Weights); s <= 0 {
		t.Errorf("TP 4 holds no more weights a rank than TP 8: %d, %d", f4.Weights, f8.Weights)
	} else if r := float64(8*f8.Weights-4*f4.Weights) / 4; r < 0 || r > float64(f8.Weights) {
		t.Errorf("the replicated part R = %.0f B is outside [0, W8]", r)
	}
	scratch := func(k *Kernel) float64 {
		return math.Max(math.Max(k.activationBuffers*float64(k.batchedTokens)*
			k.activationWidth*2, k.activationFloor), k.samplerPeakBytes(g))
	}
	b4 := float64(f4.Total()-f4.Weights) - scratch(k4)
	b8 := float64(f8.Total()-f8.Weights) - scratch(k8)
	if math.Abs(b4-b8) > 1 {
		t.Errorf("non-weight, non-scratch occupancy %.0f B at TP 4 and %.0f at TP 8; it is per rank", b4, b8)
	}
}

// THE DRAFTER'S CEILING IS ITS OWN ROUNDED SIZE. Its manager rounds each capture size up to
// K and drops what passes the ceiling (_init_candidates, cudagraph_utils.py:313-327), so at
// K = 5 under 512 the largest graph is 500 tokens, 100 requests, not floor(512 / 5) = 102;
// at K = 3 under 64 it is 57 tokens, 19 requests, not 21.
func TestTheDrafterCeilingIsTheLargestRoundedSize(t *testing.T) {
	for _, c := range []struct{ k, capture, last int }{{5, 512, 100}, {3, 64, 19}} {
		at := func(mode string) *Kernel {
			in := fixtureInputs(t, v41)
			e := mustDecode(t, in.Deployment.Pools[0])
			size := c.capture
			e.MaxCUDAGraphCaptureSize, e.MaxNumSeqs = &size, 256
			e.Speculative.NumSpecTokens = c.k
			if mode != "" {
				e.CUDAGraphMode = mode
			}
			setEngine(t, &in.Deployment.Pools[0], e)
			return build(t, in)
		}
		stated, none := at(""), at("NONE")
		host := func(k *Kernel, n int) time.Duration {
			return k.StepTime(decodeBatch(n, c.k+1, 8192)).PerResource[kernel.ResourceHost]
		}
		d := stated.plan.Draft
		draft := time.Duration(d.Kernels + d.ContextKernels + c.k*d.SequentialKernels)
		if c.last*(c.k+1) <= c.capture {
			t.Fatalf("K %d: the target still graphs %d requests; pick a wider batch", c.k, c.last)
		}
		want := host(none, c.last) - draft*stated.launchEager + stated.replayPerStep
		if got := host(stated, c.last); got != want {
			t.Errorf("K %d, capture %d: %d requests draft %d tokens, within the largest rounded "+
				"graph: %v of host time, want NONE's less the drafter's eager launches plus "+
				"its replay, %v", c.k, c.capture, c.last, c.k*c.last, got, want)
		}
		for _, n := range []int{c.last + 1, c.last + 2} {
			if a, b := host(stated, n), host(none, n); a != b {
				t.Errorf("K %d, capture %d: %d requests draft %d tokens, past every rounded "+
					"graph: %v of host time, want NONE's %v", c.k, c.capture, n, c.k*n, a, b)
			}
		}
	}
}
