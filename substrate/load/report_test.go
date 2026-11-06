package load

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// TestReportConvertsToTheResultWithoutLoss checks the one conversion from the loader's
// report to the Result's selected_entries: every kind of entry lands in its own list with
// every identifying field, and the unread files carry over. A new Kind with no case in
// Report.Result would be dropped from the Result; the last entry below makes that fail.
func TestReportConvertsToTheResultWithoutLoss(t *testing.T) {
	scope := coefficient.Scope{Hardware: []string{"h200"}}
	r := Report{
		Selected: []Selected{
			{Kind: KindCoefficient, File: "coefficients/a.yaml", Set: "a", Entry: "rate", Scope: scope, Method: "measured"},
			{Kind: KindAcceptance, File: "acceptance/b.yaml", Set: "b", Entry: "k1", Method: "measured"},
			{Kind: KindRouting, File: "routing/c.yaml", Set: "c", Entry: "k2", Method: "assumed"},
		},
		Unloaded: []string{"coefficients/z.yaml"},
	}
	got := r.Result()
	want := result.SelectedEntries{
		Coefficients:       []result.SelectedCoefficient{{Set: "a", Name: "rate", Method: "measured", Scope: scope, File: "coefficients/a.yaml"}},
		AcceptanceProfiles: []result.SelectedProfile{{Set: "b", Key: "k1", File: "acceptance/b.yaml"}},
		RoutingProfiles:    []result.SelectedProfile{{Set: "c", Key: "k2", File: "routing/c.yaml"}},
		UnreadFiles:        []string{"coefficients/z.yaml"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Report.Result() = %+v, want %+v", got, want)
	}
	n := len(got.Coefficients) + len(got.AcceptanceProfiles) + len(got.RoutingProfiles)
	if n != len(r.Selected) {
		t.Fatalf("%d of %d selected entries reached the Result", n, len(r.Selected))
	}
}

// TestEveryKindHasAResultList fails when a Kind is added without a case in Report.Result.
func TestEveryKindHasAResultList(t *testing.T) {
	for _, k := range []Kind{KindCoefficient, KindAcceptance, KindRouting} {
		out := Report{Selected: []Selected{{Kind: k, Set: "s", Entry: "e"}}}.Result()
		if len(out.Coefficients)+len(out.AcceptanceProfiles)+len(out.RoutingProfiles) != 1 {
			t.Errorf("kind %q does not reach the Result", k)
		}
	}
}

// TestProvenanceConvertsToTheResult checks the provenance conversion field by field.
func TestProvenanceConvertsToTheResult(t *testing.T) {
	p := Provenance{Source: SourcePath, Path: "/x", Hash: "h", CombinedHash: "c",
		Overlays: []Overlay{{Path: "/o", Hash: "oh"}}}
	want := result.DataSource{Source: result.SourcePath, Hash: "h", CombinedHash: "c",
		Overlays: []result.Overlay{{Hash: "oh"}}}
	if got := p.Result(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Provenance.Result() = %+v, want %+v", got, want)
	}
	if result.Source(SourceEmbedded) != result.SourceEmbedded || result.Source(SourcePath) != result.SourcePath {
		t.Fatal("load.Source and result.Source name the sources differently")
	}
}

// TestReportOrderSatisfiesTheResultValidator records one entry from each of two sets where
// one name is a prefix of the other. Their files sort one way ("-" is 0x2d, "." is 0x2e) and
// the sets the other; the Result validator requires the set order, so Report must use it.
func TestReportOrderSatisfiesTheResultValidator(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	d.RecordCoefficient("cost-model-collectives", "a", coefficient.Scope{}, "measured")
	d.RecordCoefficient("cost-model-collectives-inferencex", "a", coefficient.Scope{}, "measured")
	res := result.Result{SelectedEntries: d.Report().Result()}
	for _, p := range res.Validate().All() {
		if strings.Contains(p.Message, "sorted by set") {
			t.Fatalf("Report order is refused by the Result validator: %s", p.Message)
		}
	}
}
