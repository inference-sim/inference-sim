// Original to blis-sglang.

package blissglang

import "testing"

// The release this build models is a well-formed record (a release version, a tag, a full commit),
// and a run's provenance names this module's revision: in a build or test of this repository
// "(devel)", never "unknown", which would mean the build info could not be read.
func TestProvenanceReportsAWellFormedPinAndThisModulesRevision(t *testing.T) {
	for _, p := range Upstream.Problems() {
		t.Error(p)
	}
	if got := ProvenanceOf(); got.Module == "unknown" || got.Module == "" {
		t.Errorf("ProvenanceOf().Module = %q; a run must record this module's revision", got.Module)
	}
}
