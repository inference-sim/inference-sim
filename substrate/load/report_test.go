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

// TestRecordEntryCarriesWhatTheG1ConditionsNeed records an entry with a kernel implementation,
// a fit range and an interval wider than half its value, and reads the facts back from the
// Result's conversion. A second record of the entry replaces the first, so a run that records
// at its start and again at its end reports extrapolation.
func TestRecordEntryCarriesWhatTheG1ConditionsNeed(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	impl := coefficient.KernelImpl{Library: "attn-lib", Variant: "decode_v2", CodeHash: "0123456789ab"}
	rng := coefficient.Bound{Dim: "tokens", Low: 1, High: 512}
	e := coefficient.Entry{Name: "gemm_efficiency", Value: 10, Method: "measured",
		Scope: coefficient.Scope{Hardware: []string{"h100"}}, KernelImpls: []coefficient.KernelImpl{impl},
		FitRange: []coefficient.Bound{rng}, CI95: &coefficient.Interval{Low: 0, High: 100}}

	d.RecordEntry("cost-model-primitives", e, false)
	got := d.Report().Result().Coefficients
	if len(got) != 1 {
		t.Fatalf("got %d coefficients, want 1", len(got))
	}
	c := got[0]
	if !reflect.DeepEqual(c.KernelImpls, []coefficient.KernelImpl{impl}) ||
		!reflect.DeepEqual(c.FitRange, []coefficient.Bound{rng}) || c.Extrapolated || !c.Unidentified {
		t.Fatalf("recorded %+v, want the implementation, the range, not extrapolated, unidentified", c)
	}

	d.RecordEntry("cost-model-primitives", e, true)
	got = d.Report().Result().Coefficients
	if len(got) != 1 || !got[0].Extrapolated {
		t.Fatalf("a second record did not replace the first: %+v", got)
	}

	// An entry that states nothing, and a narrow interval, report nothing.
	plain := coefficient.Entry{Name: "plain", Value: 10, Method: "measured", CI95: &coefficient.Interval{Low: 9, High: 11}}
	d.RecordEntry("cost-model-primitives", plain, false)
	for _, c := range d.Report().Result().Coefficients {
		if c.Name == "plain" && (c.KernelImpls != nil || c.FitRange != nil || c.Extrapolated || c.Unidentified) {
			t.Errorf("a plain entry reports %+v", c)
		}
	}
}

// TestRecordEntryCopiesTheEntry changes the entry's slices after recording; the report
// must not follow them.
func TestRecordEntryCopiesTheEntry(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	e := coefficient.Entry{Name: "x", Method: "measured",
		KernelImpls: []coefficient.KernelImpl{{Library: "l", Variant: "v", CodeHash: "0123456789ab"}},
		FitRange:    []coefficient.Bound{{Dim: "tokens", Low: 1, High: 2}}}
	d.RecordEntry("cost-model-primitives", e, false)
	e.KernelImpls[0].Library, e.FitRange[0].High = "changed", 99
	c := d.Report().Result().Coefficients[0]
	if c.KernelImpls[0].Library != "l" || c.FitRange[0].High != 2 {
		t.Fatalf("the report follows the caller's slices: %+v", c)
	}
}
