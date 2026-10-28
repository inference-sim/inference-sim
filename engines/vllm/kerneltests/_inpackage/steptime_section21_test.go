// Moved from blis-latency-kernel/steptime_section21_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
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
