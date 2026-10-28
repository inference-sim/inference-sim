// Moved from blis-latency-kernel/provenance_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// dropCoefficient returns the sets without the named entry.
func dropCoefficient(sets []*coefficient.Set, name string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if e.Name != name {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		out[i] = &c
	}
	return out
}

// A DECODE READS THE CACHE IT HOLDS. Growing a one-request decode's context must add exactly
// the bytes the cache grew by (SequenceVariableBytes) over the attention decode rate to the
// HBM term -- on a hybrid stack as on a pure one. Nemotron-3-Ultra holds KV on 12 of its 108
// layers; dividing the whole-model KV figure by every layer, as the decode read once did,
// charged a ninth of it. Both fixtures' attention is full (gqa), so the part-wide rate
// applies; contexts are block-aligned, so the cache grows by whole pages. Equal to within
// the 2 ns that rounding to time.Duration allows.
func TestADecodeReadsTheCacheItHolds(t *testing.T) {
	for _, f := range []string{"nemotron3-ultra-h100-agg.yaml", "minimax-m25-h200-tp8.yaml"} {
		in, k := fixtureInputs(t, f), fixture(t, f)
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(1, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		const short, long = 4096, 8192
		grew := float64(k.SequenceVariableBytes(long) - k.SequenceVariableBytes(short))
		want := grew / (registryValue(t, in, k, "attention_decode_rate") * 1e6)
		if got := hbm(long) - hbm(short); math.Abs(got-want) > 2e-9 {
			t.Errorf("%s: %d more tokens of context added %.3f us of HBM; the cache grew %.0f "+
				"bytes, which the attention rate reads in %.3f us", f, long-short, got*1e6,
				grew, want*1e6)
		}

		// With no measured attention rate the read falls back to HBM bandwidth -- the
		// chip's, derated by the registry's hbm_derate -- and must still be the cache's
		// growth, not a ninth of it.
		in.Coefficients = dropCoefficient(in.Coefficients, "attention_decode_rate")
		fb, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		fbHBM := func(ctx int) float64 {
			return fb.StepTime(decodeBatch(1, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		bw := in.Chip.MemoryBandwidthTBs * 1e12 * registryValue(t, in, fb, "hbm_derate")
		if got, want := fbHBM(long)-fbHBM(short), grew/bw; math.Abs(got-want) > 2e-9 {
			t.Errorf("%s without an attention rate: %d more tokens added %.3f us of HBM; the "+
				"cache grew %.0f bytes, which HBM reads in %.3f us", f, long-short, got*1e6,
				grew, want*1e6)
		}
	}
}
