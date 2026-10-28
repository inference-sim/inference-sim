// Moved from blis-latency-kernel/skip_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/coefficient"

	"github.com/inference-sim/blis-latency-kernel/internal/artifacttest"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// skipUntilSixteenRankAllToAllExists skips a test that can only be written against a
// 16-wide expert group, for as long as no 16-rank alltoall coefficient exists to price one.
//
// Exactly two tests need this, and both assert cross-node behaviour a narrower group
// cannot exercise: that Resolved() reports a 16-wide expert group, and that such a group
// spanning two 8-GPU nodes pays the NIC where a TP=8 reduction stays on NVLink. Every
// other test that used to ride on that fixture asserts something a measured width shows
// just as well, and uses minimax-m25-h200-ep8.yaml instead.
//
// A SKIP rather than a deletion, because the assertions are correct and wanted — it is the
// data to run them that is missing, tracked in inference-sim/blis-registry#27. A skip
// rather than a standing failure, because a permanently red suite is how a reader learns
// to stop reading failures, and that is the pathology this package has already suffered
// twice (see testdata/minimax-m25-h200-ep16.yaml's header, and the 41 tests that skipped
// for the life of the pseudo-version pin).
//
// It is CONDITIONAL ON THE REGISTRY rather than a blanket skip, and that distinction is
// the whole point. It probes for the coefficient and skips only while it is genuinely
// absent, so the day the triple lands these two tests run again with no edit here — and if
// the registry is present but unreadable, RequireArtifact fails rather than skipping. A
// skip that cannot turn itself off is the kind that hid those 41.
func skipUntilSixteenRankAllToAllExists(t *testing.T) {
	t.Helper()
	path := filepath.Join(registryRoot, "coefficients", "cost-model-collectives.yaml")
	set, err := schemas.LoadCoefficientSet(path)
	artifacttest.RequireArtifact(t, registryRoot, path, "registry", err)

	c, err := resolve.Load([]*coefficient.Set{set}, resolve.Scope{Hardware: "h200"})
	if err != nil {
		t.Fatalf("resolving the collectives set: %v", err)
	}
	for _, w := range measuredWidths(c, "alltoall", "fp16", "h200") {
		if w == 16 {
			return // the coefficient landed; run the test
		}
	}
	t.Skip("needs a 16-rank alltoall coefficient for h200, which the registry carries " +
		"for no part (inference-sim/blis-registry#27). This skip clears itself when the " +
		"coefficient lands.")
}
