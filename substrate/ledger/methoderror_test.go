package ledger

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// series builds one family's points on the given chips, one key, values chip by chip.
func series(family, key string, v map[string]float64) []Point {
	var out []Point
	for c, x := range v {
		out = append(out, Point{Family: family, Chip: c, Key: key, Value: x})
	}
	return out
}

// manySeries builds n series (keys k0..) of a family where every chip's value is base*factor[chip]
// of the key's base, so the right answer of any estimator is known by arithmetic.
func manySeries(family string, n int, factor map[string]float64) []Point {
	var out []Point
	for i := 0; i < n; i++ {
		base := float64(10 + i)
		for c, f := range factor {
			out = append(out, Point{Family: family, Chip: c, Key: string(rune('a' + i)), Value: base * f})
		}
	}
	return out
}

var gen = map[string]string{"h100": "hopper", "h200": "hopper", "b200": "blackwell", "b300": "blackwell", "a100": "ampere"}

func rowFor(rows []MethodErrorRow, method, family string) (MethodErrorRow, bool) {
	for _, r := range rows {
		if r.Method == method && r.Family == family {
			return r, true
		}
	}
	return MethodErrorRow{}, false
}

// The same-generation median is exact where siblings agree and off by the stated ratio where
// they do not, leaving each chip out in turn: known answers.
func TestSameGenerationLeavesOneChipOut(t *testing.T) {
	// Hopper twins differ by 2x in every series; Blackwell twins agree.
	pts := manySeries("attn_rate", 12, map[string]float64{"h100": 1, "h200": 2, "b200": 3, "b300": 3})
	rows, err := MethodError(pts, []Estimator{SameGeneration("assumed", gen)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, ok := rowFor(rows, "assumed", "attn_rate")
	if !ok {
		t.Fatalf("no row: %+v", rows)
	}
	// Points: 12 series x 4 chips = 48. Hopper chips (24 points) are off by exactly 2 (the sibling
	// is the other twin: 2/1 and 1/2 both read as ratio 2). Blackwell chips (24) are exact (1).
	if r.Points != 48 || r.WorstRatio != 2 || r.MedianRatio != 1.5 {
		t.Errorf("row = %+v; want 48 points, median 1.5 (half exact, half 2x), worst 2", r)
	}
}

// A chip with no sibling gets no estimate and adds no point: a method with no formula for the
// case is not scored as if it had one.
func TestAChipWithNoSiblingAddsNoPoint(t *testing.T) {
	pts := manySeries("f", 12, map[string]float64{"h100": 1, "h200": 1, "a100": 5})
	rows, err := MethodError(pts, []Estimator{SameGeneration("assumed", gen)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, _ := rowFor(rows, "assumed", "f")
	if r.Points != 24 { // the two Hopper chips only; the lone Ampere chip has no sibling
		t.Errorf("points = %d, want 24 (a100 has no same-generation sibling)", r.Points)
	}
	if r.WorstRatio != 1 {
		t.Errorf("worst = %v", r.WorstRatio)
	}
}

// A chip is never estimated from its own value. Two parts that carry identical values would
// otherwise predict each other exactly and report an error of one; leaving the group out
// together measures the transfer to a part with no twin.
func TestNearCopiesAreLeftOutTogether(t *testing.T) {
	g := map[string]string{"a100-80": "ampere", "a100-sxm": "ampere", "l40s": "ampere"}
	pts := manySeries("f", 12, map[string]float64{"a100-80": 4, "a100-sxm": 4, "l40s": 1})
	est := []Estimator{SameGeneration("assumed", g)}
	alone, err := MethodError(pts, est, nil)
	if err != nil {
		t.Fatal(err)
	}
	grouped, err := MethodError(pts, est, map[string]string{"a100-80": "a100", "a100-sxm": "a100"})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := rowFor(alone, "assumed", "f")
	b, _ := rowFor(grouped, "assumed", "f")
	// Alone, an A100 held out still has its twin and the l40s as siblings, so it reads their median
	// (2.5x the l40s, 1.6 off its own 4) while the l40s reads the twins' 4 (a factor of 4).
	// Grouped, each A100 reads only the l40s (a factor of 4) and so does everything else.
	if a.Points != 36 || a.WorstRatio != 4 || math.Abs(a.MedianRatio-1.6) > 1e-9 {
		t.Errorf("without grouping: %+v; want 36 points, median 1.6, worst 4", a)
	}
	if b.MedianRatio != 4 || b.WorstRatio != 4 {
		t.Errorf("with the twins grouped: %+v; every estimate crosses to the other part", b)
	}
	if !(b.MedianRatio > a.MedianRatio) {
		t.Error("grouping near-copies did not raise the error; a twin is still predicting its twin")
	}
}

// D7: a family with too few points is merged into "other", and "other" too small is dropped.
func TestSmallFamiliesMergeAndThenVanish(t *testing.T) {
	pts := append(manySeries("big", 12, map[string]float64{"h100": 1, "h200": 1}), // 24 points
		manySeries("tiny1", 3, map[string]float64{"h100": 1, "h200": 1})...) // 6 points
	rows, err := MethodError(pts, []Estimator{SameGeneration("assumed", gen)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := rowFor(rows, "assumed", "big"); !ok {
		t.Errorf("big family missing: %+v", rows)
	}
	if _, ok := rowFor(rows, "assumed", "tiny1"); ok {
		t.Error("a family of 6 points was reported on its own")
	}
	if _, ok := rowFor(rows, "assumed", "other"); ok {
		t.Error("6 points were reported as 'other'; fewer than 10 are never reported")
	}
	// Two small families together reach 10 and are reported as "other".
	pts = append(manySeries("big", 12, map[string]float64{"h100": 1, "h200": 1}),
		append(manySeries("t1", 3, map[string]float64{"h100": 1, "h200": 1}), manySeries("t2", 3, map[string]float64{"h100": 1, "h200": 1})...)...)
	rows, _ = MethodError(pts, []Estimator{SameGeneration("assumed", gen)}, nil)
	if o, ok := rowFor(rows, "assumed", "other"); !ok || o.Points != 12 {
		t.Errorf("other = %+v, %v; want the 12 small points merged", o, ok)
	}
	for _, r := range rows {
		if r.Points < MinAggregateUnits {
			t.Errorf("row %+v covers fewer than %d points", r, MinAggregateUnits)
		}
	}
}

// The figures depend on the measured values alone: order and duplicates of nothing, and a
// permutation of the points gives the same rows.
func TestMethodErrorDoesNotDependOnOrder(t *testing.T) {
	pts := manySeries("f", 12, map[string]float64{"h100": 1, "h200": 3, "b200": 2, "b300": 2})
	est := []Estimator{SameGeneration("assumed", gen), MedianOfOthers("baseline")}
	want, err := MethodError(pts, est, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, j := 0, len(pts)-1; i < j; i, j = i+1, j-1 {
		pts[i], pts[j] = pts[j], pts[i]
	}
	got, _ := MethodError(pts, est, nil)
	if len(got) != len(want) {
		t.Fatalf("%d rows against %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d: %+v against %+v", i, got[i], want[i])
		}
	}
}

// The baseline reads every other chip, so against a series with a clear generation structure
// the same-generation method beats it; this is the comparison the design's per-method figure
// exists to make.
func TestSameGenerationBeatsTheGlobalMedianOnAGenerationStructure(t *testing.T) {
	pts := manySeries("f", 12, map[string]float64{"h100": 1, "h200": 1.02, "b200": 5, "b300": 5.1})
	rows, err := MethodError(pts, []Estimator{SameGeneration("assumed", gen), MedianOfOthers("baseline")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := rowFor(rows, "assumed", "f")
	b, _ := rowFor(rows, "baseline", "f")
	if !(a.MedianRatio < b.MedianRatio) {
		t.Errorf("same-generation %v is not better than the global median %v", a.MedianRatio, b.MedianRatio)
	}
	if a.MedianRatio > 1.1 {
		t.Errorf("same-generation median ratio %v; the twins agree to 1.02", a.MedianRatio)
	}
}

func TestMethodErrorRefusesWhatItCannotScore(t *testing.T) {
	good := series("f", "k", map[string]float64{"h100": 1})
	est := []Estimator{MedianOfOthers("m")}
	for name, pts := range map[string][]Point{
		"zero value":      {{Family: "f", Chip: "h100", Value: 0}},
		"negative value":  {{Family: "f", Chip: "h100", Value: -1}},
		"NaN":             {{Family: "f", Chip: "h100", Value: math.NaN()}},
		"infinite":        {{Family: "f", Chip: "h100", Value: math.Inf(1)}},
		"no family":       {{Chip: "h100", Value: 1}},
		"no chip":         {{Family: "f", Value: 1}},
		"a chip's second": append(good, Point{Family: "f", Chip: "h100", Key: "k", Value: 2}),
	} {
		if _, err := MethodError(pts, est, nil); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := MethodError(good, []Estimator{{Method: "m"}}, nil); err == nil {
		t.Error("an estimator with no estimate function was accepted")
	}
	bad := Estimator{Method: "m", Estimate: func(string, string, string, map[string]float64) (float64, bool) { return 0, true }}
	pts := series("f", "k", map[string]float64{"h100": 1, "h200": 2})
	if _, err := MethodError(pts, []Estimator{bad}, nil); err == nil || !strings.Contains(err.Error(), "estimated") {
		t.Errorf("an estimate of zero was accepted: %v", err)
	}
}

// The report is aggregates only, over at least MinAggregateUnits points, and passes the same
// check as every other seed-holding job's output (D7).
func TestMethodErrorReportNamesNoUnit(t *testing.T) {
	pts := manySeries("attn_rate", 12, map[string]float64{"h100": 1, "h200": 2})
	rows, _ := MethodError(pts, []Estimator{SameGeneration("assumed", gen)}, nil)
	rep := MethodErrorReport(strings.Repeat("a", 40), rows)
	if !rep.Evaluated || len(rep.Aggregates) != 2 {
		t.Fatalf("report = %+v", rep)
	}
	raw := mustJSON(t, rep)
	if err := CheckReport(raw, []string{"deadbeefdeadbeefdeadbeef"}); err != nil {
		t.Errorf("a method-error report failed the seed-holding job's own check: %v", err)
	}
	empty := MethodErrorReport("c", nil)
	if empty.Evaluated {
		t.Error("an empty report claims to have evaluated something")
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
