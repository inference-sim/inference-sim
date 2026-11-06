package price

import (
	"math/rand"
	"testing"
)

// vllmPageBytesPerToken transcribes vLLM v0.31.0's per-token, per-layer page cost on one
// rank: num_heads * (head_size + head_size_v) * dtype_size
// (AttentionSpec.state_content_size_bytes and unpadded_page_size_bytes,
// vllm/v1/kv_cache_interface.py:521-529), where an attention spec stores a value of the
// key's width and an MLA spec stores none ("MLA stores a single latent vector per state;
// there is no separate V", head_size_v = 0, :674-675). The heads are the rank's: KV heads
// divided by tensor parallelism, never fewer than one.
func vllmPageBytesPerToken(nKV, tp, headDim int, dtype float64, latent bool) float64 {
	heads := max(nKV/max(tp, 1), 1)
	headSizeV := headDim
	if latent {
		headSizeV = 0
	}
	return float64(heads) * float64(headDim+headSizeV) * dtype
}

// KVBytesPerToken must equal the engine's own page arithmetic, layer for layer, for every
// geometry whose cache is a plain per-element dtype -- full attention and latent, any head
// count and tensor-parallel width, any element width. The layouts that state their own size
// are applied by the kernel, not this law; see KVBytesPerToken.
//
// Property-checked over random geometries against the transcription above. The case that
// motivated it is the latent one: an earlier form charged an MLA token a key AND a value,
// twice the single latent vector the engine stores.
func TestKVBytesPerTokenIsTheEnginesPageArithmetic(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 20000; i++ {
		nKV := 1 + r.Intn(128)
		tp := 1 << r.Intn(5)
		headDim := 32 * (1 + r.Intn(20))
		layers := 1 + r.Intn(128)
		dtype := []float64{0.5, 1, 2, 4}[r.Intn(4)]
		latent := r.Intn(2) == 0
		if latent {
			nKV = 1 // a latent cache has one head, which is what the catalog records
		}
		want := vllmPageBytesPerToken(nKV, tp, headDim, dtype, latent) * float64(layers)
		if got := KVBytesPerToken(nKV, tp, headDim, layers, dtype, latent); got != want {
			t.Fatalf("nKV=%d tp=%d headDim=%d layers=%d dtype=%v latent=%v: got %v, want %v",
				nKV, tp, headDim, layers, dtype, latent, got, want)
		}
	}
}

// The metamorphic half, stated directly: at one geometry, a latent cache is half a
// key-and-value one, and no tensor-parallel width reduces a one-head cache.
func TestALatentCacheIsOneVectorAndNoTensorParallelWidthReducesIt(t *testing.T) {
	kv := KVBytesPerToken(1, 1, 576, 61, 1, false)
	latent := KVBytesPerToken(1, 1, 576, 61, 1, true)
	if latent*2 != kv {
		t.Errorf("a latent cache is %v bytes per token against %v for a key and a value of "+
			"the same width; it holds one vector, not two", latent, kv)
	}
	for _, tp := range []int{2, 4, 8, 16} {
		if got := KVBytesPerToken(1, tp, 576, 61, 1, true); got != latent {
			t.Errorf("tp=%d shrank a one-head latent cache to %v from %v", tp, got, latent)
		}
	}
}
