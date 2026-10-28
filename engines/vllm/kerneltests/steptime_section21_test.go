// Moved from blis-latency-kernel/steptime_section21_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
)

func decodeBatch(requests, queryLen, context int) kernel.Batch {
	b := kernel.Batch{Reqs: make([]kernel.ReqShape, requests), DecodeThreshold: 8}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: queryLen, Computed: context - queryLen, PromptLen: context,
		}
	}
	return b
}

func TestStepTimeAgainstDesignSection21(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		scenario                    string
		requests, queryLen, context int
	}{
		{"256 req q=2 ctx8k EP16", "minimax-m25-h200-ep8.yaml", 256, 2, 8192},
		{"256 req q=2 ctx8k EP72", "minimax-m25-h200-ep72.yaml", 256, 2, 8192},
		{"32 req q=2 ctx8k EP16", "minimax-m25-h200-ep8.yaml", 32, 2, 8192},
		{"1 req 2048 prefill EP16", "minimax-m25-h200-ep8.yaml", 1, 2048, 2048},
		{"4 req 2048 prefill EP72", "minimax-m25-h200-ep72.yaml", 4, 2048, 2048},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := fixture(t, tc.scenario)
			e := k.StepTime(decodeBatch(tc.requests, tc.queryLen, tc.context))
			t.Logf("overlap=%.2fms nooverlap=%.2fms bottleneck=%s",
				float64(e.Overlap.Microseconds())/1000,
				float64(e.NoOverlap.Microseconds())/1000, e.Bottleneck)
			for r, d := range e.PerResource {
				t.Logf("    %-8s %.3f ms", r, float64(d.Microseconds())/1000)
			}
		})
	}
}
