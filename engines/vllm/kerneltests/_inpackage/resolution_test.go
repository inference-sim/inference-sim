// Moved from blis-latency-kernel/resolution_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"
)

// A consumer reaches what this kernel resolved through kernel.Kernel alone: Deployment for
// the pool as stated, Resolved for what resolution settled. Nothing a simulator needs is on
// the concrete type, so a second cost model implementing the interface can serve the same
// consumer. (The two context-parallel widths are the interim exception; see
// DecodeContextParallelWidth.)
//
// Each test asserts what THIS pool states or resolved, not merely something non-zero. A
// kernel answering for pool 0 whatever index it was opened at would satisfy a non-zero
// check and still be the bug -- a decode pool priced at a prefill pool's parallelism --
// that holding a pool index alongside the kernel used to cause.

// TestResolvedReportsTheWidthsThisKernelPriced is the property Resolution's widths exist
// for: each is the width the kernel actually priced against, and each has its floor.
//
// Checked over the property fixtures and both pools of the disaggregated one, so dense and
// expert-parallel, single- and multi-node, and a decode pool with dp > 1 are all covered.
// The widths are compared with the resolved layout -- the one place the pricer reads them
// -- so a Resolution filled from the document, or left at zero, fails.
func TestResolvedReportsTheWidthsThisKernelPriced(t *testing.T) {
	kernels := propertyFixtures(t)
	r := writeDisaggregated(t)
	for i, name := range []string{"disagg prefill", "disagg decode"} {
		k, err := OpenPool("disagg.yaml", r, i)
		if err != nil {
			t.Fatal(err)
		}
		kernels[name] = k
	}
	for name, k := range kernels {
		res := k.Resolved()
		for _, w := range []struct {
			field        string
			got, floored int
			priced       int
		}{
			{"tensor", res.TensorParallelWidth, res.TensorParallel(), k.layout.TP},
			{"data", res.DataParallelWidth, res.DataParallel(), k.layout.DP},
			{"expert", res.ExpertParallelWidth, res.ExpertParallel(), k.layout.ExpertWidth},
		} {
			if w.got < 1 {
				t.Errorf("%s: %s-parallel width %d; an implementation fills every width, "+
					"and a scheduler multiplies its budgets by these", name, w.field, w.got)
			}
			if w.got != w.floored {
				t.Errorf("%s: %s-parallel width %d disagrees with its floored reading %d",
					name, w.field, w.got, w.floored)
			}
			if w.got != max(w.priced, 1) {
				t.Errorf("%s: Resolved reports a %s-parallel width of %d, the pricer used %d",
					name, w.field, w.got, w.priced)
			}
		}
	}
}
