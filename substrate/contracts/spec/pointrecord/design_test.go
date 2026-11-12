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

// designG2 is the list of section 11.3 under "G2 estimated", "As G1, except that one or more
// of these holds, each named on the point", in the order the section gives it: thirteen
// conditions, each with the kind that records it. The text is the section's own, shortened.
var designG2 = []struct {
	condition string
	kind      SubstitutionKind
}{
	{"some coefficients are transferred, literature, vendor_spec, roofline or assumed", SubCoefficientMethod},
	{"a coefficient is extrapolated or unidentified", SubCoefficientRange},
	{"an execution mode is substituted", SubExecutionMode},
	{"a recorded library's kernel hash differs from the one priced", SubKernelHash},
	{"a modelled field is set to a value outside every value class its evidence covers", SubValueClass},
	{"a recorded value was missing and replaced by the engine's estimation rule", SubEstimated},
	{"a fact of the cluster the run's own record lacks was taken from the benchmark's later record of the same runner", SubLaterRecord},
	{"a workload-dependent value the run did not record was taken from a substrate profile (D8)", SubProfileValue},
	{"a resolved value differs from the engine's resolution", SubResolvedDiffers},
	{"a field the run sets has a status allowed only during a release plan", SubInterimStatus},
	{"the simulated requests differ from the ones the run's client sent", SubClientDifference},
	{"the run was measured behind a router BLIS does not host, and the router table maps it", SubRouterDiffers},
	{"a condition of G1 could not be evaluated, because the data it needs is not recorded for the point", SubUnevaluated},
}

// TestTheG2KindsAreTheDesignsConditions holds the record's G2 kinds to section 11.3's list:
// thirteen conditions, one kind each, and no G2 kind without a condition. A condition the
// record cannot name cannot be scored, and a G1 is then published for a point that fails it.
func TestTheG2KindsAreTheDesignsConditions(t *testing.T) {
	if len(designG2) != 13 {
		t.Fatalf("design %s section 11.3 lists thirteen G2 conditions; the test restates %d", designRevision, len(designG2))
	}
	seen := map[SubstitutionKind]string{}
	for _, c := range designG2 {
		if prev, dup := seen[c.kind]; dup {
			t.Errorf("kind %s records two conditions: %q and %q", c.kind, prev, c.condition)
		}
		seen[c.kind] = c.condition
		if g, ok := substitutionGrade[c.kind]; !ok {
			t.Errorf("no kind records the condition %q", c.condition)
		} else if g != G2 {
			t.Errorf("kind %s records %q and sets grade %s, not G2", c.kind, c.condition, g)
		}
	}
	var extra []string
	for k, g := range substitutionGrade {
		if _, ok := seen[k]; !ok && g == G2 {
			extra = append(extra, string(k))
		}
	}
	sort.Strings(extra)
	if len(extra) > 0 {
		t.Errorf("G2 kinds that record no condition of design section 11.3: %v", extra)
	}
}
