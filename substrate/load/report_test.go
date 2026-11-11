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

// The report is marked as carrying the closeness data exactly when it holds a coefficient and
// every coefficient was recorded whole. One recorded by identity alone leaves the report
// unmarked and strips the data from every coefficient, so the conversion validates.
func TestTheReportIsMarkedWhenEveryCoefficientWasRecordedWhole(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	if got := d.Report().Result().ClosenessVersion; got != 0 {
		t.Errorf("an empty report is marked %d; it says nothing about the pricing", got)
	}
	rich := coefficient.Entry{Name: "a", Method: "measured", Value: 10,
		KernelImpls: []coefficient.KernelImpl{{Library: "l", Variant: "v", CodeHash: "0123456789ab"}},
		FitRange:    []coefficient.Bound{{Dim: "tokens", Low: 1, High: 8}}, CI95: &coefficient.Interval{Low: 0, High: 100}}
	d.RecordEntry("cost-model-primitives", rich, true)
	got := d.Report().Result()
	if got.ClosenessVersion != result.ClosenessVersion || !got.Coefficients[0].Extrapolated || !got.Coefficients[0].Unidentified ||
		got.Coefficients[0].CI95 == nil {
		t.Errorf("a report of whole entries: %+v", got)
	}
	d.RecordCoefficient("cost-model-primitives", "b", coefficient.Scope{}, "measured")
	mixed := d.Report().Result()
	if mixed.ClosenessVersion != 0 {
		t.Errorf("a report with an identity-only coefficient is marked %d, want unmarked", mixed.ClosenessVersion)
	}
	for _, c := range mixed.Coefficients {
		if c.KernelImpls != nil || c.FitRange != nil || c.CI95 != nil || c.Extrapolated || c.Unidentified {
			t.Errorf("%s keeps closeness data in an unmarked report: %+v", c.Name, c)
		}
	}
	res := result.Result{SelectedEntries: mixed}
	for _, p := range res.Validate().All() {
		if strings.Contains(p.Message, "closeness") {
			t.Errorf("the report's own conversion is refused for its closeness data: %s", p.Message)
		}
	}
}

type fakeObserver struct{ yes map[string]bool }

func (f fakeObserver) Extrapolated(set, name string, _ coefficient.Scope, _ string) bool {
	return f.yes[set+"/"+name]
}

// An observer is asked when the report is read, so a step priced after the entry was recorded
// still reaches the Result; it is registered once however often it is added.
func TestAnObserverMarksEntriesExtrapolatedWhenTheReportIsRead(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	d.RecordEntry("cost-model-primitives", coefficient.Entry{Name: "a", Method: "measured"}, false)
	d.RecordEntry("cost-model-primitives", coefficient.Entry{Name: "b", Method: "measured"}, false)
	o := &fakeObserver{yes: map[string]bool{}}
	d.Observe(o)
	d.Observe(o)
	if len(d.observers) != 1 {
		t.Errorf("the observer was registered %d times", len(d.observers))
	}
	flags := func() map[string]bool {
		out := map[string]bool{}
		for _, c := range d.Report().Result().Coefficients {
			out[c.Name] = c.Extrapolated
		}
		return out
	}
	if got := flags(); got["a"] || got["b"] {
		t.Fatalf("before any step: %v", got)
	}
	o.yes["cost-model-primitives/b"] = true
	if got := flags(); got["a"] || !got["b"] {
		t.Fatalf("after a step left b's range: %v, want only b", got)
	}
}

// Several cost models record into one Data. A flag any of them raised stays raised, whatever
// the order. All of them read one registry entry, so their implementations, ranges and intervals
// agree; if they do not, the record states none of them and the report is not marked.
func TestRecordsOfOneEntryAccumulate(t *testing.T) {
	impl := coefficient.KernelImpl{Library: "lib", Variant: "a", CodeHash: "0123456789ab"}
	entry := func(hi float64) coefficient.Entry {
		return coefficient.Entry{Name: "gemm", Value: 10, Method: "measured", CI95: &coefficient.Interval{Low: 9, High: 11},
			KernelImpls: []coefficient.KernelImpl{impl},
			FitRange:    []coefficient.Bound{{Dim: "tokens", Low: 1, High: hi}}}
	}
	record := func(first, second bool, secondHi float64) result.SelectedEntries {
		d, err := Embedded()
		if err != nil {
			t.Fatal(err)
		}
		d.RecordEntry("cost-model-primitives", entry(8), first)
		d.RecordEntry("cost-model-primitives", entry(secondHi), second)
		return d.Report().Result()
	}
	for name, c := range map[string][2]bool{"the second extrapolated": {false, true}, "the first, then a clean call": {true, false}} {
		got := record(c[0], c[1], 8)
		if len(got.Coefficients) != 1 || !got.Coefficients[0].Extrapolated || got.ClosenessVersion != 1 {
			t.Errorf("%s: %+v; a raised flag must stay raised and the report stay marked", name, got)
		}
		if cf := got.Coefficients[0]; len(cf.KernelImpls) != 1 || len(cf.FitRange) != 1 || cf.CI95 == nil {
			t.Errorf("%s: agreeing records lost their facts: %+v", name, cf)
		}
	}
	got := record(true, false, 16)
	cf := got.Coefficients[0]
	if got.ClosenessVersion != 0 || cf.KernelImpls != nil || cf.FitRange != nil || cf.CI95 != nil {
		t.Errorf("records that disagree on the range: %+v, want the report unmarked and no facts", got)
	}
	// An identity-only record joining a whole one leaves the report unmarked, in either order.
	for name, order := range map[string]bool{"whole first": true, "identity first": false} {
		d, _ := Embedded()
		if order {
			d.RecordEntry("cost-model-primitives", entry(8), false)
			d.RecordCoefficient("cost-model-primitives", "gemm", coefficient.Scope{}, "measured")
		} else {
			d.RecordCoefficient("cost-model-primitives", "gemm", coefficient.Scope{}, "measured")
			d.RecordEntry("cost-model-primitives", entry(8), false)
		}
		if got := d.Report().Result().ClosenessVersion; got != 0 {
			t.Errorf("%s: an entry also recorded by identity alone leaves the report marked %d", name, got)
		}
	}
}

// A nil observer is ignored rather than a panic at the next report.
func TestANilObserverIsIgnored(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	d.Observe(nil)
	d.RecordEntry("cost-model-primitives", coefficient.Entry{Name: "a", Method: "measured"}, false)
	_ = d.Report()
}

// An entry that states no interval and no range reaches the Result with neither: false flags
// there are "not evaluated", and the Result must not invent an interval or a range.
func TestAnEntryThatStatesNothingReachesTheResultStatingNothing(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	d.RecordEntry("cost-model-primitives", coefficient.Entry{Name: "bare", Method: "measured", Value: 5}, false)
	c := d.Report().Result().Coefficients[0]
	if c.CI95 != nil || c.FitRange != nil || c.KernelImpls != nil || c.Unidentified || c.Extrapolated {
		t.Errorf("a bare entry carries %+v", c)
	}
}

// The registry's validator and the Result's validator of a selected entry must agree about the
// committed data: every committed coefficient entry is recorded as the cost model records it
// (RecordEntry, with extrapolated raised wherever it could be) and the Result built from the
// report validates. A rule one validator holds and the other loosens fails here, on real data.
func TestEveryCommittedCoefficientEntryRoundTripsThroughTheResultValidator(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	n, withRange, withCI := 0, 0, 0
	for _, stem := range sortedKeys(d.registry.Sets) {
		for _, e := range d.registry.Sets[stem].Coefficients {
			d.RecordEntry(stem, e, len(e.FitRange) > 0)
			n++
			if len(e.FitRange) > 0 {
				withRange++
			}
			if e.CI95 != nil {
				withCI++
			}
		}
	}
	if n < 100 || withRange == 0 || withCI == 0 {
		t.Fatalf("the round trip saw %d entries, %d with a fit range, %d with an interval: the registry shape this test relies on is gone", n, withRange, withCI)
	}
	res := result.Result{SelectedEntries: d.Report().Result()}
	for _, p := range res.Validate().Errors() {
		if strings.HasPrefix(p.Path, "selected_entries.coefficients") {
			t.Errorf("%s: %s", p.Path, p.Message)
		}
	}
}

// A method that truly may not state a range or implementations is refused by both validators,
// through the same predicate.
func TestAMethodThatMayNotStateARangeIsRefusedInTheResult(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	e := coefficient.Entry{Name: "x", Method: "assumed", Value: 1,
		FitRange:    []coefficient.Bound{{Dim: "tokens", Low: 1, High: 2}},
		KernelImpls: []coefficient.KernelImpl{{Library: "l", Variant: "v", CodeHash: "0123456789ab"}}}
	d.RecordEntry("cost-model-primitives", e, false)
	res := result.Result{SelectedEntries: d.Report().Result()}
	var got []string
	for _, p := range res.Validate().Errors() {
		got = append(got, p.Path)
	}
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "coefficients[0].fit_range") || !strings.Contains(joined, "coefficients[0].kernel_impls") {
		t.Errorf("problems %v, want the fit range and the implementations of an assumed entry refused", got)
	}
	if coefficient.MayStateFitRange("assumed") || !coefficient.MayStateFitRange("transferred") ||
		coefficient.MayStateKernelImpls("assumed") || !coefficient.MayStateKernelImpls("transferred") ||
		!coefficient.MayStateKernelImpls("measured") {
		t.Error("the shared predicates changed")
	}
}
