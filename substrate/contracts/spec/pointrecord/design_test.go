package pointrecord

import (
	"sort"
	"testing"
)

// designRevision is the revision of the published design (inference-sim/blis-design, the file
// blis-target-design.html) that designG5 below was read from. The lists restate the design
// because the design is another repository; a change to either must change the other, and a
// reader checks the list against the section at this revision.
const designRevision = "bc25a5c86584fa73daedacda098fa4b38a98d3c0"

// designG5 is the list of section 11.4, "Reasons a point is not scored (G5)", in the order the
// section gives it. COV-5 makes it closed: any other reason fails CI.
var designG5 = []G5Code{
	"code_not_visible", "router_unsupported", "value_not_recorded", "field_not_modelled",
	"field_unclassified", "engine_unsupported", "topology_unsupported", "not_in_catalog",
	"field_not_in_release", "value_conflict", "build_unparseable",
}

// TestTheG5ReasonsAreTheDesignsClosedList holds g5Codes equal to section 11.4's list, in both
// directions, so neither a missing reason nor an extra one passes (COV-5).
func TestTheG5ReasonsAreTheDesignsClosedList(t *testing.T) {
	if len(designG5) != 11 {
		t.Fatalf("design %s section 11.4 lists eleven reasons; the test restates %d", designRevision, len(designG5))
	}
	want := map[G5Code]bool{}
	for _, c := range designG5 {
		want[c] = true
		if !c.Valid() {
			t.Errorf("%s is a reason of design section 11.4 and the record refuses it", c)
		}
	}
	var extra []string
	for c := range g5Codes {
		if !want[c] {
			extra = append(extra, string(c))
		}
	}
	sort.Strings(extra)
	if len(extra) > 0 || len(g5Codes) != len(designG5) {
		t.Errorf("the record accepts reasons that design section 11.4 does not list: %v (%d accepted, %d listed)", extra, len(g5Codes), len(designG5))
	}
}

// Each reason is written by a G5 record that validates; the two the code lacked before are the
// ones the scoring harness writes for a split deployment and for sources that disagree.
func TestAG5RecordCarriesEachReasonOfTheDesign(t *testing.T) {
	for _, c := range designG5 {
		r := valid()
		r.Grade, r.Figures, r.Substitutions = G5, nil, nil
		r.G5Reason = &G5Reason{Code: c, Subject: "x"}
		if p := r.Validate(); !p.OK() {
			t.Errorf("a G5 record with reason %s was refused:\n%s", c, p.Error())
		}
	}
}
