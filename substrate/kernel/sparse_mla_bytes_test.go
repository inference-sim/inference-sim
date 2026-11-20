package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
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

// The bound must be applied PER REQUEST, not to the batch-wide context total, because
// min(sum) != sum(min): a batch holding requests on both sides of the top-k gets both
// wrong if the cap is applied to the aggregate.
//
// Asserted on selectedKVTokens directly rather than through StepTime, and that choice is
// the finding. Through StepTime the distinction is UNOBSERVABLE on this model: an
// 8,600 us step is dominated by the GEMM and MoE terms, so a few hundred selected tokens
// across 60 layers does not move the microsecond-resolution total at all. Every shape pair
// tried -- 4x1024 against 4093+3x1, 4x128 against 509+3x1, 4x512 against 2045+3x1, at
// equal totals and equal request counts -- priced IDENTICALLY both with the correct
// per-request bound and with a deliberately broken batch-total one.
//
// So an end-to-end assertion here would be a test that cannot fail, which is worse than no
// test: it would report the property as protected while a batch-total regression shipped.
// The arithmetic is where the difference lives, so that is where it is pinned.
//
// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. This pins the FUNCTION, not the call
// site. A mutation replacing the pricer's `decodeContexts` with a one-element slice
// holding the batch total still passes every test in this file, because the resulting
// price difference is below microsecond resolution on every fixture here. Closing that
// would need either a fixture whose step is attention-dominated or an injection seam in
// the pricer, and neither exists today -- so the gap is recorded here instead of being
// hidden behind an assertion that happens to pass.
func TestSparseMLABoundsEachRequestNotTheBatchTotal(t *testing.T) {
	// csa4_moe's geometry: a 128-token window, and a top-k of 1,024 from a stream
	// compressed 4:1.
	l := &price.PlannedLayer{
		AttnKind: model.AttentionSparseMLA, AttnWindow: 128, AttnIndexTopK: 1024,
		AttnCompressRatio: 4,
	}

	// Four requests of 1,024 tokens: each reads its window and the 256 compressed
	// positions it has, all of which the top-k admits: 4 x (128 + 256).
	perRequest := selectedKVTokens(l, []int{1024, 1024, 1024, 1024})
	if want := 4 * (128.0 + 256.0); perRequest != want {
		t.Errorf("4 requests of 1,024 selected %v tokens, want %v", perRequest, want)
	}

	// The same 4,096 tokens of context in ONE request: its window and a top-k that its
	// 1,024 compressed positions just fill, 128 + 1,024. Applying the bounds to the batch
	// total would give this same answer for the case above, which is how the two differ.
	concentrated := selectedKVTokens(l, []int{4096})
	if want := 128.0 + 1024.0; concentrated != want {
		t.Errorf("one request of 4,096 selected %v tokens, want %v", concentrated, want)
	}
	if perRequest == concentrated {
		t.Errorf("4x1,024 and 1x4,096 tokens of context both selected %v; the bound is "+
			"being applied to the batch total rather than per request", perRequest)
	}

	// And the direction matters: spreading the same context over more requests reads MORE,
	// because each request reads its own window.
	if perRequest <= concentrated {
		t.Errorf("spreading context over 4 requests selected %v, not more than the %v one "+
			"request reads; each request pays its own window", perRequest, concentrated)
	}

	// Past the top-k nothing grows with context: a top-k layer reads its window plus at
	// most index_topk positions (vllm/models/deepseek_v4/nvidia/flashmla.py:187-262 at
	// v0.31.0). The form this replaced added (ctx - topk) / ratio, which grew without bound.
	if long := selectedKVTokens(l, []int{1 << 20}); long != concentrated {
		t.Errorf("a 1M-token request selected %v tokens, want the saturated %v",
			long, concentrated)
	}
}

// A layer that is not sparse, or that states no bound, must report -1 so the caller keeps
// the full-context read. Returning 0 instead would make a non-sparse layer's attention
// free, which is silently cheap rather than visibly wrong.
func TestSelectedKVTokensRefusesWhatItCannotBound(t *testing.T) {
	for _, c := range []struct {
		name string
		l    price.PlannedLayer
	}{
		{"full attention", price.PlannedLayer{AttnKind: model.AttentionGQA}},
		{"dense MLA", price.PlannedLayer{AttnKind: model.AttentionMLA}},
		{"sparse with neither topk nor window", price.PlannedLayer{
			AttnKind: model.AttentionSparseMLA}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := selectedKVTokens(&c.l, []int{65536}); got != -1 {
				t.Errorf("got %v, want -1 so the caller charges the whole context", got)
			}
		})
	}

	// A compressing layer that selects nothing IS a bound: its window plus the whole
	// compressed stream, which is how blis-catalog states csa128_moe.
	l := price.PlannedLayer{
		AttnKind: model.AttentionSparseMLA, AttnWindow: 128, AttnCompressRatio: 128,
	}
	if got := selectedKVTokens(&l, []int{65536}); got != 128.0+65536.0/128.0 {
		t.Errorf("a window-selected sparse layer bounded to %v, want %v",
			got, 128.0+65536.0/128.0)
	}

	// A window bounds every kind, not sparse_mla alone: a sliding-window or windowed MLA
	// layer reads min(ctx, window), as the registry's swa rate was fitted
	// (blis-registry scripts/fit_attention_by_kind.py:98-109).
	for _, kind := range []model.AttentionKind{model.AttentionSWA, model.AttentionMLA} {
		w := price.PlannedLayer{AttnKind: kind, AttnWindow: 128}
		if got := selectedKVTokens(&w, []int{65536, 64}); got != 128+64 {
			t.Errorf("a %s layer with a 128-token window read %v, want %v", kind, got, 128+64)
		}
	}
}
