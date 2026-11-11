package result

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
)

// withG1 returns the valid Result with the first coefficient carrying everything the G1
// conditions read, .
func withG1() *Result {
	r := valid()
	r.SelectedEntries.ClosenessVersion = ClosenessVersion
	c := &r.SelectedEntries.Coefficients[0]
	c.KernelImpls = []coefficient.KernelImpl{{Library: "attn-lib", Variant: "decode_v2", CodeHash: "0123456789abcdef"}}
	c.FitRange = []coefficient.Bound{{Dim: "tokens", Low: 1, High: 8192}}
	c.CI95 = &coefficient.Interval{Low: 0, High: 100}
	c.Extrapolated = true
	c.Unidentified = true
	return r
}

func TestAResultWithTheG1DataValidatesAndRoundTrips(t *testing.T) {
	r := withG1()
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	for _, f := range []Format{FormatJSON, FormatYAML} {
		b, err := Marshal(r, f)
		if err != nil {
			t.Fatal(err)
		}
		back := &Result{}
		if err := Unmarshal(b, back); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if !reflect.DeepEqual(back.SelectedEntries, r.SelectedEntries) {
			t.Errorf("%s: the G1 data did not round-trip:\n got %+v\nwant %+v", f,
				back.SelectedEntries, r.SelectedEntries)
		}
	}
}

// Known-bad cases (SUB-2): each breaks one rule of the G1 data and expects a problem at the
// named path.
func TestTheG1DataIsRefusedWhenMalformed(t *testing.T) {
	co := func(f func(c *SelectedCoefficient)) func(*Result) {
		return func(r *Result) { f(&r.SelectedEntries.Coefficients[0]) }
	}
	cases := []struct {
		name   string
		break_ func(r *Result)
		path   string
		msg    string
	}{
		{"implementation without a library", co(func(c *SelectedCoefficient) { c.KernelImpls[0].Library = "" }),
			"selected_entries.coefficients[0].kernel_impls[0].library", "required"},
		{"implementation without a variant", co(func(c *SelectedCoefficient) { c.KernelImpls[0].Variant = "" }),
			"selected_entries.coefficients[0].kernel_impls[0].variant", "required"},
		{"implementation with a short hash", co(func(c *SelectedCoefficient) { c.KernelImpls[0].CodeHash = "abc" }),
			"selected_entries.coefficients[0].kernel_impls[0].code_hash", "hexadecimal"},
		{"implementation with an upper-case hash", co(func(c *SelectedCoefficient) { c.KernelImpls[0].CodeHash = "0123456789ABCDEF" }),
			"selected_entries.coefficients[0].kernel_impls[0].code_hash", "hexadecimal"},
		{"the same implementation twice", co(func(c *SelectedCoefficient) {
			c.KernelImpls = append(c.KernelImpls, c.KernelImpls[0])
		}), "selected_entries.coefficients[0].kernel_impls[1]", "duplicate"},
		{"fit range on an unknown dimension", co(func(c *SelectedCoefficient) { c.FitRange[0].Dim = "batch" }),
			"selected_entries.coefficients[0].fit_range[0].dim", "is not one of"},
		{"fit range low above high", co(func(c *SelectedCoefficient) { c.FitRange[0].Low = 9000 }),
			"selected_entries.coefficients[0].fit_range[0]", "exceeds"},
		{"fit range twice on one dimension", co(func(c *SelectedCoefficient) {
			c.FitRange = append(c.FitRange, c.FitRange[0])
		}), "selected_entries.coefficients[0].fit_range[1].dim", "duplicate"},
		{"closeness data in an unmarked report", func(r *Result) { r.SelectedEntries.ClosenessVersion = 0 },
			"selected_entries.coefficients[0]", "closeness_version is absent"},
		{"an unknown closeness version", func(r *Result) { r.SelectedEntries.ClosenessVersion = 2 },
			"selected_entries.closeness_version", "must be 1 or absent"},
		{"unidentified with no interval", co(func(c *SelectedCoefficient) { c.CI95 = nil }),
			"selected_entries.coefficients[0].unidentified", "no ci95"},
		{"an interval in an unmarked report", func(r *Result) {
			r.SelectedEntries.ClosenessVersion = 0
			c := &r.SelectedEntries.Coefficients[0]
			c.KernelImpls, c.FitRange, c.Extrapolated, c.Unidentified = nil, nil, false, false
		}, "selected_entries.coefficients[0]", "closeness_version is absent"},
		{"an interval backward", co(func(c *SelectedCoefficient) { c.CI95 = &coefficient.Interval{Low: 5, High: 1} }),
			"selected_entries.coefficients[0].ci95", "low not above high"},
		{"implementations on an assumed entry", co(func(c *SelectedCoefficient) { c.Method = "assumed" }),
			"selected_entries.coefficients[0].kernel_impls", "states none"},
		{"a fit range on a literature entry", co(func(c *SelectedCoefficient) { c.KernelImpls = nil; c.Method = "literature" }),
			"selected_entries.coefficients[0].fit_range", "not fitted"},
		{"extrapolated with no fit range", co(func(c *SelectedCoefficient) { c.FitRange = nil }),
			"selected_entries.coefficients[0].extrapolated", "no fit_range"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := withG1()
			c.break_(r)
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

// A report written before the closeness data existed carries none of it, is unmarked, and
// validates: its absent flags mean "not said", which a marked report's absent flags do not.
func TestAnUnmarkedReportWithoutClosenessDataStillValidates(t *testing.T) {
	r := valid()
	if r.SelectedEntries.ClosenessVersion != 0 {
		t.Fatal("the fixture is meant to be unmarked")
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	r.SelectedEntries.ClosenessVersion = ClosenessVersion
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a marked report with every flag false: %s", p.Error())
	}
}
