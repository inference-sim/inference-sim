package latencykernel

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// Per-layer caches, as a graph states them (blis-schemas model.Cache), checked through
// what a simulator reads: SequenceVariableBytes, SequenceFixedBytes, PDTransferDemand and
// StepTime.

// The selection bound, as properties over random layers and contexts: the read never
// shrinks as context grows, and a layer that selects a top-k reads at most its window plus
// that top-k whatever the context.
func TestSelectionBoundProperties(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		l := &price.PlannedLayer{AttnKind: model.AttentionSparseMLA}
		if r.Intn(2) == 0 {
			l.AttnWindow = 1 + r.Intn(4096)
		}
		if r.Intn(2) == 0 {
			l.AttnIndexTopK = 1 + r.Intn(4096)
		}
		if r.Intn(2) == 0 {
			l.AttnCompressRatio = 1 + r.Intn(128)
		}
		if !readBounded(l) {
			continue
		}
		prev := 0.0
		for _, ctx := range []int{1, 7, 128, 1000, 4096, 65536, 1 << 20} {
			got := selectedKVTokensFor(l, ctx)
			if got < prev {
				t.Fatalf("%+v: read %v at %d, less than %v at a shorter context", *l, got, ctx, prev)
			}
			if l.AttnIndexTopK > 0 &&
				got > float64(l.AttnWindow)+float64(l.AttnIndexTopK)+1e-9 {
				t.Fatalf("%+v: read %v at %d, more than window plus top-k", *l, got, ctx)
			}
			prev = got
		}
	}
}
