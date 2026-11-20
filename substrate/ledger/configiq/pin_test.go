package configiq_test

import (
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/ledger/configiq"
)

// PinnedRevision is the dataset revision the ledger names for configiq (M3.2 Table W row 17),
// and PinnedUnits the units its ingest produces.
const (
	pinnedRevision = "21cc6eb06b1934485a83b17a93e634ef38b5799e"
	pinnedUnits    = 1702
)

// The ledger names revision 21cc6eb and nothing else: sources.yaml, the ingest report and the
// unit list agree on it, and the unit list holds the 1,702 units that revision produces. A move
// to another revision is a deliberate change of this test, in the pull request that moves it.
func TestConfigiqPinnedRevision(t *testing.T) {
	l := committed(t)
	src := l.Sources.Lookup(configiq.SourceID)
	if src == nil {
		t.Fatal("sources.yaml has no configiq source")
	}
	if src.Revision.Kind != "hf_dataset_revision" || src.Revision.Ref != pinnedRevision {
		t.Errorf("sources.yaml pins %s %q, want hf_dataset_revision %q", src.Revision.Kind, src.Revision.Ref, pinnedRevision)
	}
	if !strings.Contains(src.Revision.URL, pinnedRevision) {
		t.Errorf("the revision url %q does not name %s", src.Revision.URL, pinnedRevision)
	}
	raw, err := os.ReadFile("../../catalog/ledger/reports/configiq.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var rep configiq.Report
	if err := yaml.Unmarshal(raw, &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Revision != pinnedRevision || rep.Units != pinnedUnits {
		t.Errorf("the ingest report names revision %s with %d units, want %s with %d", rep.Revision, rep.Units, pinnedRevision, pinnedUnits)
	}
	if got := len(l.Units[configiq.SourceID]); got != pinnedUnits {
		t.Errorf("the unit list holds %d units, want %d", got, pinnedUnits)
	}
}
