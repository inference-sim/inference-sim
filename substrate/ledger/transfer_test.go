package ledger

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
)

const testAxis = "bw"

func transferSpec() TransferSpec {
	return TransferSpec{
		Axis: func(family string) (string, bool) {
			if strings.HasPrefix(family, "rate_") {
				return testAxis, true
			}
			return "", false
		},
		Datasheet: map[string]map[string]float64{testAxis: {"h1": 2, "h2": 4, "b1": 8, "b2": 16}},
		Generation: map[string]string{
			"h1": "Hopper", "h2": "Hopper", "b1": "Blackwell", "b2": "Blackwell",
		},
	}
}

// transferPoints builds n series of one family. In series i the efficiency (value over the
// datasheet figure) of a Hopper chip is hopperEff(i) and of a Blackwell chip blackwellEff(i).
func transferPoints(family string, n int, hopperEff, blackwellEff func(i int) float64) []Point {
	spec := transferSpec()
	var out []Point
	for i := 0; i < n; i++ {
		for chip, ds := range spec.Datasheet[testAxis] {
			eff := hopperEff(i)
			if spec.Generation[chip] == "Blackwell" {
				eff = blackwellEff(i)
			}
			out = append(out, Point{Family: family, Chip: chip, Key: string(rune('a' + i)), Value: eff * ds})
		}
	}
	return out
}

func transferRowFor(t *testing.T, rows []TransferRow, from, to string) TransferRow {
	t.Helper()
	for _, r := range rows {
		if r.From == from && r.To == to && r.Class == testAxis {
			return r
		}
	}
	t.Fatalf("no row for %s to %s in %+v", from, to, rows)
	return TransferRow{}
}

// A rule that carries a series' efficiency across the datasheet ratio is exact when the
// efficiency is the generation's, and the family's pooled default efficiency is not, so the
// rule beats roofline and is used. Both are scored on the same held-out points.
func TestTransferRuleBeatsRooflineWhenEfficiencyIsAConstantOfTheSeries(t *testing.T) {
	eff := func(i int) float64 { return 0.2 + 0.1*float64(i) }
	rows, err := TransferError(transferPoints("rate_x", 6, eff, eff), transferSpec())
	if err != nil {
		t.Fatal(err)
	}
	r := transferRowFor(t, rows, "Hopper", "Blackwell")
	if r.Points != 12 {
		t.Errorf("%d points, want 6 series on 2 Blackwell chips = 12", r.Points)
	}
	if math.Abs(r.RuleMedian-1) > 1e-9 || math.Abs(r.RuleWorst-1) > 1e-9 {
		t.Errorf("rule ratio median %v worst %v, want exactly 1", r.RuleMedian, r.RuleWorst)
	}
	if !(r.RooflineMedian > 1.2) {
		t.Errorf("roofline median %v; the pooled efficiency across series should miss by more than 20%%", r.RooflineMedian)
	}
	if !r.UseRule {
		t.Error("the rule beat roofline on the same points and was not used")
	}
}

// When the target generation's efficiency in a series has nothing to do with the source's, the
// rule loses to the family's pooled efficiency and transfers rank as roofline.
func TestTransferRuleLosesWhenTheTargetsEfficiencyDiffers(t *testing.T) {
	hopper := func(i int) float64 { return 0.2 + 0.2*float64(i%2) }
	blackwell := func(i int) float64 { return 0.3 + 0.2*float64((i+1)%2) } // the opposite series is high
	rows, err := TransferError(transferPoints("rate_x", 6, hopper, blackwell), transferSpec())
	if err != nil {
		t.Fatal(err)
	}
	r := transferRowFor(t, rows, "Hopper", "Blackwell")
	if !(r.RuleMedian > r.RooflineMedian) {
		t.Fatalf("setup: rule %v should be worse than roofline %v", r.RuleMedian, r.RooflineMedian)
	}
	if r.UseRule {
		t.Error("the rule lost to roofline on the same points and was used")
	}
	if got := TransferMethod(rows, "Hopper", "Blackwell", testAxis); got != vocab.MethodRoofline {
		t.Errorf("a losing rule gives %q, want roofline", got)
	}
}

// The target generation's own values never enter either estimate: if they did, the rule would
// read them and the error here would shrink toward 1. Every Blackwell value is three times the
// efficiency-preserving value, and both estimators must miss by exactly that factor.
func TestTransferEstimatesNeverReadTheHeldOutGeneration(t *testing.T) {
	hopper := func(i int) float64 { return 0.25 }
	blackwell := func(i int) float64 { return 0.75 }
	rows, err := TransferError(transferPoints("rate_x", 6, hopper, blackwell), transferSpec())
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.From == row.To {
			t.Errorf("%+v: a chip's own generation is a source for the rule, but the generation is held out", row)
		}
	}
	r := transferRowFor(t, rows, "Hopper", "Blackwell")
	if math.Abs(r.RuleMedian-3) > 1e-9 || math.Abs(r.RooflineMedian-3) > 1e-9 {
		t.Errorf("rule %v roofline %v, want both 3: an estimate that reads the held-out generation would be nearer 1", r.RuleMedian, r.RooflineMedian)
	}
	if r.UseRule {
		t.Error("a tie is not a win: the rule is used only if it beats roofline")
	}
}

// A figure covers at least MinAggregateUnits points (D7). Below that nothing is published, and
// a transfer with no published validation ranks as roofline.
func TestTransferRowsBelowTheAggregationFloorAreNotPublished(t *testing.T) {
	eff := func(i int) float64 { return 0.2 + 0.1*float64(i) }
	rows, err := TransferError(transferPoints("rate_x", 4, eff, eff), transferSpec()) // 4 series x 2 chips = 8
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		if r.Points < MinAggregateUnits {
			t.Errorf("row %+v covers fewer than %d points", r, MinAggregateUnits)
		}
	}
	if got := TransferMethod(rows, "Hopper", "Blackwell", testAxis); got != vocab.MethodRoofline {
		t.Errorf("an unvalidated cross-generation transfer ranks %q, want roofline", got)
	}
}

// Within one generation a transfer is not a cross-generation transfer, and ranks as transferred.
// A validated, winning direction ranks as transferred; the other direction, unvalidated, does not.
func TestTransferMethodByDirection(t *testing.T) {
	rows := []TransferRow{{From: "Hopper", To: "Blackwell", Class: testAxis, Points: 12, UseRule: true}}
	for name, c := range map[string]struct {
		from, to string
		want     vocab.Method
	}{
		"validated and winning":    {"Hopper", "Blackwell", vocab.MethodTransferred},
		"the other direction":      {"Blackwell", "Hopper", vocab.MethodRoofline},
		"within a generation":      {"Hopper", "Hopper", vocab.MethodTransferred},
		"a generation not covered": {"Ampere", "Blackwell", vocab.MethodRoofline},
	} {
		if got := TransferMethod(rows, c.from, c.to, testAxis); got != c.want {
			t.Errorf("%s: %q, want %q", name, got, c.want)
		}
	}
	if got := TransferMethod(rows, "Hopper", "Blackwell", "other_axis"); got != vocab.MethodRoofline {
		t.Errorf("a class with no row: %q, want roofline", got)
	}
}

// Only points both methods estimate are compared. A family with no datasheet axis, and a
// series with no source-generation value, give the rule or roofline nothing to say; they are not
// counted for either.
func TestTransferComparesOnlyPointsBothMethodsEstimate(t *testing.T) {
	eff := func(i int) float64 { return 0.2 + 0.1*float64(i) }
	pts := transferPoints("rate_x", 6, eff, eff)
	pts = append(pts, transferPoints("floor_x", 6, eff, eff)...) // no axis
	// a series present only on Blackwell: roofline can estimate it, the rule cannot
	pts = append(pts, Point{Family: "rate_x", Chip: "b1", Key: "solo", Value: 3}, Point{Family: "rate_x", Chip: "b2", Key: "solo", Value: 5})
	rows, err := TransferError(pts, transferSpec())
	if err != nil {
		t.Fatal(err)
	}
	if r := transferRowFor(t, rows, "Hopper", "Blackwell"); r.Points != 12 {
		t.Errorf("%d points, want the 12 that have an axis and a Hopper source", r.Points)
	}
}

func TestTransferRefusesBadInput(t *testing.T) {
	spec := transferSpec()
	pts := transferPoints("rate_x", 6, func(int) float64 { return 1 }, func(int) float64 { return 1 })
	bad := append([]Point{}, pts...)
	bad[0].Value = 0
	if _, err := TransferError(bad, spec); err == nil {
		t.Error("a non-positive measured value was accepted")
	}
	spec.Datasheet[testAxis]["h1"] = -1
	if _, err := TransferError(pts, spec); err == nil {
		t.Error("a non-positive datasheet figure was accepted")
	}
	dup := append(append([]Point{}, pts...), pts[0])
	if _, err := TransferError(dup, transferSpec()); err == nil {
		t.Error("two points for one family, key and chip were accepted")
	}
}
