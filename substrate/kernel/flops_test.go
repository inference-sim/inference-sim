package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// For a dense model the count has a closed form from its config. Qwen3-14B: hidden 5120, 40
// layers of 40 query heads x 128, qkv 5120x7168, o 5120x5120, gate-up 5120x34816, down
// 17408x5120, and a 151936-wide head. A 1024-token prompt run whole pays every layer's GEMMs
// for 1024 tokens, the head for one row, and attention over 1024*1025/2 causal pairs.
func TestStepFLOPsOfADenseModelMatchesTheClosedForm(t *testing.T) {
	k := stepKernel(t)
	const n = 1024
	got := k.StepFLOPs(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: n, PromptLen: n}}})
	perToken := 2.0 * 40 * (5120*7168 + 5120*5120 + 5120*34816 + 17408*5120)
	head := 2.0 * 151936 * 5120
	attn := 40 * 4.0 * 40 * 128 * (n * (n + 1) / 2)
	want := perToken*n + head + attn
	if math.Abs(got-want) > 1e-9*want {
		t.Errorf("StepFLOPs = %.6g, closed form %.6g", got, want)
	}
	// A decode token past a 4096 context: the GEMMs for one token, the head for its row, and
	// attention over the 4097 keys it reads.
	got = k.StepFLOPs(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 4096, PromptLen: 100}}})
	want = perToken + head + 40*4.0*40*128*4097
	if math.Abs(got-want) > 1e-9*want {
		t.Errorf("decode StepFLOPs = %.6g, closed form %.6g", got, want)
	}
}

// A window bounds the keys a query reads: past the window every query reads w, below it all
// that came before.
func TestQueryKeyPairsAreBoundedByTheWindow(t *testing.T) {
	for _, c := range []struct {
		computed, n, window int
		want                float64
	}{
		{0, 4, 0, 10}, // 1+2+3+4
		{0, 4, 2, 1 + 2 + 2 + 2},
		{10, 3, 4, 12}, // all past the window: 4 each
		{2, 4, 4, 3 + 4 + 4 + 4},
		{5, 2, 0, 6 + 7},
	} {
		if got := queryKeyPairs(c.computed, c.n, c.window); got != c.want {
			t.Errorf("queryKeyPairs(%d, %d, %d) = %v, want %v", c.computed, c.n, c.window, got, c.want)
		}
	}
}

// A prefill-bound batch keeps the tensor cores far busier than a decode-only one, which waits
// on memory: the model-FLOPs utilization (FLOPs over step time times peak) of 4096 prompt
// tokens is several times that of 32 single-token decodes, and neither exceeds the peak.
func TestPrefillBatchHasHigherMFUThanADecodeBatch(t *testing.T) {
	k := stepKernel(t)
	mfu := func(b kernel.Batch) float64 {
		secs := k.StepTime(b).NoOverlap.Seconds()
		return k.StepFLOPs(b) / secs / k.PeakFLOPsPerGPU()
	}
	prefill := mfu(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096}}})
	var reqs []kernel.ReqShape
	for i := 0; i < 32; i++ {
		reqs = append(reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024})
	}
	decode := mfu(kernel.Batch{DecodeThreshold: 1, Reqs: reqs})
	if prefill <= 0 || prefill > 1 || decode <= 0 || decode > 1 {
		t.Fatalf("mfu prefill %.3f, decode %.3f; want both in (0, 1]", prefill, decode)
	}
	if prefill < 3*decode {
		t.Errorf("prefill mfu %.3f is not at least 3x decode mfu %.3f", prefill, decode)
	}
}
