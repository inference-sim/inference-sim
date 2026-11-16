// Original to blis-sglang.

package blissglang

import "testing"

// The pin is exact (design §14.3): the release this build models is named by version, tag and full
// commit, and a run's provenance reports that record and the engine kind.
func TestProvenanceReportsTheExactUpstreamPin(t *testing.T) {
	for _, p := range Upstream.Problems() {
		t.Error(p)
	}
	got := ProvenanceOf()
	if got.EngineKind != "sglang" || got.UpstreamVersion != "0.5.23" || got.UpstreamCommit != "297e54fd105e974cdf401e5b4d20253a22c16f90" {
		t.Errorf("ProvenanceOf() = %+v, want engine sglang at SGLang 0.5.23, commit 297e54fd105e974cdf401e5b4d20253a22c16f90", got)
	}
	if got.Module == "" {
		t.Error("ProvenanceOf().Module is empty; a run must record this module's revision")
	}
}
