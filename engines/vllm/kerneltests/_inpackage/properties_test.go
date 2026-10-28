// Moved from blis-latency-kernel/properties_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
)

// Properties that must hold for EVERY deployment the catalog can price, checked across a
// spread of batch shapes rather than on one fixture at one shape.
//
// These are metamorphic and invariant tests: they relate one call's output to another's
// without knowing either absolute value. That is the only kind of assertion available here
// — the right step time for a deployment is an empirical question answered by the scoring
// corpora, not by a unit test — and it is the kind that survives a recoefficienting. A test
// pinning a microsecond figure breaks whenever the registry is refitted and tells you
// nothing about whether the model still behaves like a cost model.
//
// They complement methods_test.go, whose assertions are pinned to one fixture. A term that
// is wired up correctly for minimax-m2.5 on H200 and wrong for a dense model on one GPU
// passes there and fails here.

// propertyFixtures are deployments spanning the shapes that change which terms fire:
// dense vs MoE, expert-parallel vs tensor-parallel only, single- vs multi-node, a
// recurrent stack, and speculative decoding.
//
// Named rather than discovered so the list is reviewable, and kept small because each
// builds a kernel against the real catalog.
func propertyFixtures(t *testing.T) map[string]*Kernel {
	t.Helper()
	out := map[string]*Kernel{}
	for _, name := range []string{
		"glm5-h200-tp8.yaml",               // MoE, tensor-parallel only, one node
		"minimax-m25-h200-tp8.yaml",        // MoE, one node, no expert parallelism
		"nemotron3-ultra-h100-agg.yaml",    // three replicas, fabric declared
		"kimi-k3-h100-nospec.yaml",         // multi-node, speculation disabled
		"direct/m27-h200-puretp4-fpm.yaml", // four GPUs per node
		v41,                                // stated caches, replicated nodes, a drafter
	} {
		out[name] = fixture(t, name)
	}
	return out
}

// batchShapes spans the regimes a step can be in: empty, pure decode at several widths and
// contexts, pure prefill, and mixed.
func batchShapes() map[string]kernel.Batch {
	return map[string]kernel.Batch{
		"empty":                {},
		"1 decode ctx 1":       decodeBatch(1, 1, 1),
		"8 decode ctx 1k":      decodeBatch(8, 1, 1024),
		"256 decode ctx 8k":    decodeBatch(256, 1, 8192),
		"512 decode ctx 32k":   decodeBatch(512, 1, 32768),
		"1 prefill 2k":         decodeBatch(1, 2048, 2048),
		"4 prefill 8k":         decodeBatch(4, 8192, 8192),
		"speculative q=2":      decodeBatch(64, 2, 4096),
		"speculative q=4 wide": decodeBatch(256, 4, 16384),
	}
}

// DATA-PARALLEL LOCKSTEP PADDING (Batch.PadTokens). A pad runs the forward's GEMMs,
// collectives and experts but no attention and no logits row
// (vllm/v1/worker/gpu_model_runner.py:4407-4413, logits_indices :2231 at v0.31.0). So: no
// pad prices exactly as before; a step never gets cheaper as its pad grows; the pad leaves
// the logits rows and the attention work where they were; and a pad of n tokens costs no
// more than n more one-token decode requests, which also run attention and the LM head.
func TestPadTokensProperties(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			zero := b
			zero.PadTokens = 0
			if a, z := k.StepTime(b), k.StepTime(zero); !reflect.DeepEqual(a, z) {
				t.Errorf("%s / %s: PadTokens 0 priced %v against %v", name, shape,
					z.NoOverlap, a.NoOverlap)
			}
			prev := k.StepTime(b).NoOverlap
			for _, pad := range []int{1, 7, 64, 500, 4096} {
				p := b
				p.PadTokens = pad
				got := k.StepTime(p).NoOverlap
				if got < prev {
					t.Errorf("%s / %s: a %d-token pad prices %v, below a smaller pad's %v",
						name, shape, pad, got, prev)
				}
				prev = got
				ps, bs := k.shapeOf(p), k.shapeOf(b)
				if ps.logitsRows != bs.logitsRows || ps.decodeKVTokens != bs.decodeKVTokens ||
					ps.causalFLOPs != bs.causalFLOPs || ps.attnTokensF != bs.attnTokensF ||
					ps.tokensF != bs.tokensF+float64(pad) {
					t.Errorf("%s / %s: a %d-token pad moved the logits rows or attention "+
						"(%v, %v) or ran %v tokens, want %v", name, shape, pad, ps.logitsRows,
						ps.decodeKVTokens, ps.tokensF, bs.tokensF+float64(pad))
				}
			}
			n := 1 + r.Intn(64)
			padded, requests := b, b
			padded.PadTokens = n
			requests.Reqs = append(append([]kernel.ReqShape{}, b.Reqs...),
				decodeBatch(n, 1, 1).Reqs...)
			if requests.DecodeThreshold == 0 {
				requests.DecodeThreshold = 8
			}
			if p, q := k.StepTime(padded).NoOverlap, k.StepTime(requests).NoOverlap; p > q {
				t.Errorf("%s / %s: a %d-token pad prices %v, above %d decode requests' %v",
					name, shape, n, p, n, q)
			}
		}
	}
}
