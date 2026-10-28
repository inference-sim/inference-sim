// Moved from blis-latency-kernel/sparse_mla_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"testing"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// A REPACKED CACHE IS REPORTED AS RESOLVED. Where the backend serves a stated "fp8" cache as
// fp8_ds_mla (h200), Resolved says so, since the priced layout is not the one requested;
// where it serves it as stated (the same chip marked data-center Blackwell), nothing is
// reported.
func TestARepackedCacheIsReportedAsResolved(t *testing.T) {
	repack := func(blackwell bool) string {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = "fp8" })
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range k.Resolved().Overrides {
			if o.Field == "cache_dtype" {
				return o.Resolved
			}
		}
		return ""
	}
	if got := repack(false); got != "fp8_ds_mla" {
		t.Errorf("fp8 on h200: Resolved reports the cache as %q, want fp8_ds_mla", got)
	}
	if got := repack(true); got != "" {
		t.Errorf("fp8 on Blackwell: Resolved reports a repack to %q; FlashInfer serves it "+
			"as stated", got)
	}
}
