package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// A piecewise-captured stage replays its own layers' segments: one per layer plus one per
// sparse-MLA indexer it holds, plus one. Two stages of one model, one holding the sparse
// layers, replay different counts, and neither its bare layer count.
func TestAPiecewiseStageReplaysItsOwnSegments(t *testing.T) {
	k := &Kernel{graphDecode: graphModePiecewise, graphOther: graphModePiecewise,
		replayPerStep: time.Microsecond, captureTokens: 512}
	dense := price.PlannedLayer{Count: 4, AttnKind: model.AttentionGQA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}}}
	sparse := price.PlannedLayer{Count: 3, AttnKind: model.AttentionSparseMLA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}, {Role: "block_index_scores"}}}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 9, PromptLen: 10}}}
	for _, c := range []struct {
		what string
		st   pipelineStage
		want time.Duration
	}{
		{"dense stage", pipelineStage{layers: []price.PlannedLayer{dense}, totalLayers: 4}, 5 * time.Microsecond},
		{"sparse stage", pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}, 7 * time.Microsecond},
	} {
		if got := k.stageHost(b, &c.st, 0); got != c.want {
			t.Errorf("%s: host %v, want %v", c.what, got, c.want)
		}
	}
	// Under a breakable capture the stage replays one piece per break it holds, plus one,
	// even where the whole model's count is larger.
	k.breakableCapture, k.captureBreaks = true, 99
	st := pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}
	if got := k.stageHost(b, &st, 0); got != 4*time.Microsecond {
		t.Errorf("breakable stage: host %v, want 4us", got)
	}
}

// The binding pipeline rank is the one whose own budget less its own fixed bytes over its own
// KV a token holds the fewest tokens, and FixedBytes and SequenceVariableBytes are that one
// rank's (get_kv_cache_configs step 5, vllm/v1/core/kv_cache_utils.py:2660-2680@5df63a72). Here
// the weight-heaviest stage is not the KV-heaviest: pairing the one's weights with the other's
// KV, as the kernel did, describes no worker and understates the pool.
func TestTheBindingStageIsOneWorkersPool(t *testing.T) {
	const gb = 1e9
	k := &Kernel{chip: hardware.Chip{MemoryGiB: 80}, graph: &model.Graph{}, blockSize: 16,
		engine: enginesettings.Settings{GPUMemoryUtilization: 0.92},
		stages: []pipelineStage{
			{memory: stageMemory{weights: 60 * gb, kvPerToken: 1000}},
			{memory: stageMemory{weights: 40 * gb, kvPerToken: 3000}},
		}}
	if err := k.bindStage(); err != nil {
		t.Fatal(err)
	}
	k.fixed = k.computeFixedBytes(k.graph)
	if k.ppKVStage != 1 || k.FixedBytes().Weights != 40*gb || k.SequenceVariableBytes(16) != 16*3000 {
		t.Errorf("binding stage %d with %d weight bytes and %d KV bytes a block; vLLM's smallest pool is "+
			"stage 1's, 40 GB and 48000 B", k.ppKVStage, k.FixedBytes().Weights, k.SequenceVariableBytes(16))
	}
	if sm := k.PipelineStageMemory(); sm[0].Fixed.Weights != 60*gb || sm[0].PerBlockBytes != 16000 || sm[0].Binding || !sm[1].Binding {
		t.Errorf("PipelineStageMemory %+v", sm)
	}
	// The other way round: the KV-heaviest stage has the room. 80 GiB x 0.92 is 79.0 GB, so
	// stage 0 leaves 3.0 GB for 3,000 B tokens (1.0M) and stage 1 39.0 GB for 3,100 B (12.6M).
	k.stages[0].memory = stageMemory{weights: 76 * gb, kvPerToken: 3000}
	k.stages[1].memory = stageMemory{weights: 40 * gb, kvPerToken: 3100}
	if err := k.bindStage(); err != nil {
		t.Fatal(err)
	}
	if k.ppKVStage != 0 {
		t.Errorf("binding stage %d; stage 0's pool is the smaller, 1.0M tokens against 12.6M", k.ppKVStage)
	}
}

// A tensor-parallel width that does not divide the KV heads of an attention that cannot
// replicate a head is refused, in the drafter's layers too; one that can replicate is priced.
func TestAnUnsplittableKVHeadCountIsRefused(t *testing.T) {
	g := func(notReplicated bool) *model.Graph {
		return &model.Graph{LayerKinds: []model.LayerKind{{ID: "a", Nodes: []model.Node{
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA, NumQHeads: 128,
				NumKVHeads: 16, HeadDim: 64, KVHeadsNotReplicated: notReplicated}}}}}
	}
	for _, c := range []struct {
		tp      int
		fused   bool
		refused bool
	}{{8, true, false}, {16, true, false}, {32, true, true}, {12, true, true}, {32, false, false}} {
		err := refuseUnsplittableKVHeads(g(c.fused), c.tp)
		if (err != nil) != c.refused {
			t.Errorf("tp %d fused %v: %v, want refused %v", c.tp, c.fused, err, c.refused)
		}
	}
	draft := g(false)
	draft.Speculator = &model.Speculator{Head: []model.Node{{Op: model.OpGEMM}}}
	draft.LayerKinds = append(draft.LayerKinds, model.LayerKind{ID: "d", Nodes: []model.Node{
		{Op: model.OpAttention, NumKVHeads: 16, KVHeadsNotReplicated: true}}})
	if refuseUnsplittableKVHeads(draft, 32) == nil {
		t.Error("a drafter layer's unsplittable heads at tp 32 were priced")
	}
}

// The binding stage is compared at the deployment's gpu_memory_utilization, not vLLM's
// default: each worker's pool is total x utilization less its own fixed bytes
// (gpu_worker.py:535-540@v0.32.0), so a stage heavy on weights binds at a low utilization
// and the KV-heavy one at a high one.
func TestTheBindingStageIsReadAtTheDeploymentsUtilization(t *testing.T) {
	const gb = 1e9
	k := &Kernel{chip: hardware.Chip{MemoryGiB: 80}, graph: &model.Graph{}, blockSize: 16}
	stages := func() []pipelineStage {
		return []pipelineStage{
			{memory: stageMemory{weights: 60 * gb, kvPerToken: 1000}},
			{memory: stageMemory{weights: 20 * gb, kvPerToken: 3000}},
		}
	}
	// 80 GiB is 85.9 GB. At 0.92 (79.0 GB) stage 0 holds 19.0M tokens and stage 1 19.7M: stage
	// 0 binds. At 0.75 (64.4 GB) stage 0 holds 4.4M and stage 1 14.8M: stage 0. At 0.99
	// (85.0 GB) stage 0 holds 25.0M and stage 1 21.7M: stage 1.
	for _, c := range []struct {
		util float64
		want int
	}{{0.92, 0}, {0.99, 1}} {
		k.stages = stages()
		k.engine.GPUMemoryUtilization = c.util
		if err := k.bindStage(); err != nil {
			t.Fatal(err)
		}
		if k.ppKVStage != c.want {
			t.Errorf("gpu_memory_utilization %v: binding stage %d, want %d", c.util, k.ppKVStage, c.want)
		}
	}
}

// SIM-9, the refusal and its closest neighbour. The cost model resolves no default (SUB-6): a
// pipeline plan whose gpu_memory_utilization is zero is refused, naming the field and saying
// the engine resolves it; the same plan at 0.92 and at the smallest positive value is priced.
// This is the refusal's pair: the kernel never supplies vLLM's 0.92 (it would also have been
// wrong for SGLang's mem_fraction_static and TensorRT-LLM's free_gpu_memory_fraction).
func TestAZeroMemoryUtilizationIsRefusedNotDefaulted(t *testing.T) {
	const gb = 1e9
	k := &Kernel{chip: hardware.Chip{MemoryGiB: 80}, graph: &model.Graph{}, blockSize: 16,
		layout: resolve.Layout{PP: 2}}
	stages := []pipelineStage{
		{memory: stageMemory{weights: 60 * gb, kvPerToken: 1000}},
		{memory: stageMemory{weights: 20 * gb, kvPerToken: 3000}},
	}
	for _, util := range []float64{0, -0.1} {
		k.stages, k.engine.GPUMemoryUtilization = stages, util
		err := k.bindStage()
		if err == nil {
			t.Fatalf("gpu_memory_utilization %v: priced; want a refusal", util)
		}
		for _, want := range []string{"gpu_memory_utilization", "SUB-6", "engine"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("gpu_memory_utilization %v: refusal %q does not name %q", util, err, want)
			}
		}
		for _, a := range k.assumptions {
			if a.Name == "pipeline_binding_utilization" {
				t.Errorf("an assumption records a utilization the kernel defaulted: %+v", a)
			}
		}
	}
	for _, util := range []float64{0.92, 0.01} {
		k.stages, k.engine.GPUMemoryUtilization = stages, util
		if err := k.bindStage(); err != nil {
			t.Errorf("gpu_memory_utilization %v: %v", util, err)
		}
	}
}

// A boundary carries each auxiliary state the drafter reads that was captured upstream of it.
// One captured on the sending stage crosses at the width the graph states (Muse Spark's fp32
// stream); one captured on an earlier stage is relayed out of the receive slot the stage before
// allocated at the model dtype (reserve_aux_intermediate_tensor_slots,
// vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:89-95@v0.32.0, sized at
// model_config.dtype, gpu/model_runner.py:537-541, copied in at :1980, relayed as received,
// gpu/pp_utils.py:109-123), so it crosses at one hidden-width activation.
func TestARelayedAuxStateCrossesAtTheModelDType(t *testing.T) {
	aux := []int{14, 30, 46, 58, 62}
	const own, act = 32768.0, 16384.0
	spans := ppIndices(62, 3) // [0,21) [21,42) [42,62)
	for _, c := range []struct {
		stage int
		want  float64
	}{
		{0, 1 * own},       // 14 captured on stage 0
		{1, 1*act + 1*own}, // 14 relayed, 30 captured on stage 1
		{2, 0},             // the last stage sends nothing
	} {
		if c.stage == len(spans)-1 {
			continue
		}
		if got := auxCarriedBytes(aux, spans[c.stage], own, act); got != c.want {
			t.Errorf("stage %d %v: carries %v B of aux states a token, want %v", c.stage,
				spans[c.stage], got, c.want)
		}
	}
	// At pp 2 every carried state is the sending stage's own.
	if got := auxCarriedBytes(aux, ppIndices(62, 2)[0], own, act); got != 2*own {
		t.Errorf("pp 2 stage 0: %v B, want %v", got, 2*own)
	}
}
