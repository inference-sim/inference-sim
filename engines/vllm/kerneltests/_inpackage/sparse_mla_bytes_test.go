// Moved from blis-latency-kernel/sparse_mla_bytes_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// A sparse-MLA layer does not read its whole context, and these pin that behaviourally.
//
// Before this, `sparse_mla` fell through to the full-attention byte count, so every layer
// of deepseek-v4-pro was priced as if it read the entire KV cache. That is wrong by up to
// 128x at long context, and it is not a small constant error: fitting a rate against the
// full-context reading forces it to 1.00 of datasheet peak -- physically impossible, which
// is how the wrong byte count announces itself (blis-registry
// scripts/fit_attention_sparse_mla.py reports 1.698x error clamped against 1.301x
// interior for the bounded form).
//
// Measured on h200 (AISimulate dsv4_hca_attn_module_perf, FLASHMLA_SPARSE_DSV4): a batch-1
// decode costs 9.8-13.7us flat from a 0 to a 16,384-token context and only 30.6us at
// 1,048,575. End-to-end on the InferenceX corpus the bound moved deepseek-v4-pro's TPOT
// error from 10.42% to 8.72%, against AISimulate's 12.82%.

// The headline property: step time must grow far more slowly in context for a sparse layer
// than the context grows. A model whose byte count is the full context would scale roughly
// linearly once past the floor.
func TestSparseMLADecodeDoesNotScaleWithTheWholeContext(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b200-fp4-vllm-tp8.yaml")

	var sparse *price.PlannedLayer
	for i := range k.plan.Layers {
		if k.plan.Layers[i].AttnKind == model.AttentionSparseMLA {
			sparse = &k.plan.Layers[i]
			break
		}
	}
	if sparse == nil {
		t.Skip("this fixture has no sparse_mla layer")
	}
	if sparse.AttnIndexTopK <= 0 && sparse.AttnWindow <= 0 {
		t.Fatalf("sparse layer carries neither index_topk nor window, so the plan is "+
			"dropping the geometry the catalog states: %+v", sparse)
	}

	at := func(ctx int) int64 {
		b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
			{Scheduled: 1, Computed: ctx, PromptLen: ctx},
		}}
		return k.StepTime(b).NoOverlap.Microseconds()
	}

	// A 64x context increase. A full-context byte count would push the attention term up
	// by about 64x; a bounded one leaves it nearly flat, so the step is dominated by the
	// terms that do not depend on context.
	short, long := at(16384), at(1048576)
	if short <= 0 || long <= 0 {
		t.Fatalf("both step times must be positive, got %d and %d", short, long)
	}
	// 1.8x, and the threshold is MEASURED rather than chosen for looseness. On this
	// deployment the bounded step moves 1.301x across that 64x context increase and the
	// unbounded one moves 2.330x, so 1.8 sits between them with room either side.
	//
	// Neither figure is near 64x because a step is not all attention: the GEMM and MoE
	// terms dominate here and are untouched by this bound, which compresses the whole
	// range. An earlier draft of this test used 8x on the reasoning that a full-context
	// read "would be near 64x" -- it passed under a mutation that ignored the bound
	// entirely, because the attention term was never 64x of the step to begin with. The
	// lesson is that a threshold argued from the term has to be checked against the
	// STEP, which is what the test actually measures.
	const maxRatio = 1.8
	if ratio := float64(long) / float64(short); ratio > maxRatio {
		t.Errorf("a 64x context increase moved the step %.3fx (%d -> %d us), above the "+
			"%.1fx bound; a sparse layer's read is capped at its selected top-k plus a "+
			"compressed remainder, so the step must grow far sub-linearly in context. "+
			"Measured: %.3fx with the bound, %.3fx without it",
			ratio, short, long, maxRatio, 1.301, 2.330)
	}
}

// Removing the geometry must change the price. This is the structural check the MLA
// dispatch test makes for its coefficient, in the form this change needs: if the plan
// stopped carrying index_topk and compress_ratio, the layer would silently revert to the
// full-context read while every other test still passed.
func TestSparseMLAGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b200-fp4-vllm-tp8.yaml")

	batch := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 262144, PromptLen: 262144},
	}}
	with := k.StepTime(batch).NoOverlap

	// Strip the bound from every sparse layer and re-price.
	stripped := 0
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		if l.AttnKind != model.AttentionSparseMLA {
			continue
		}
		l.AttnIndexTopK, l.AttnWindow, l.AttnCompressRatio = 0, 0, 0
		stripped++
	}
	if stripped == 0 {
		t.Skip("this fixture has no sparse_mla layer")
	}
	without := k.StepTime(batch).NoOverlap
	if without <= with {
		t.Errorf("stripping the sparse geometry from %d layers did not RAISE the step "+
			"(%v with the bound, %v without); the unbounded read is the whole context and "+
			"must cost more, so the geometry is not reaching the pricer",
			stripped, with, without)
	}
}
