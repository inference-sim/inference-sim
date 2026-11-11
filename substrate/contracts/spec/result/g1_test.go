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
	c.Name, c.Primitive = "gemm_eps_max_bf16", "gemm"
	c.Priced, c.KernelMatch = c.KernelImpls, coefficient.KernelExact
	r.SelectedEntries.Implementations = map[string][]coefficient.KernelImpl{"gemm": c.KernelImpls}
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
		{"an unknown closeness version", func(r *Result) { r.SelectedEntries.ClosenessVersion = 3 },
			"selected_entries.closeness_version", "must be 2 or absent"},
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
		// The implementation the entry priced (design VAL-8, G1).
		{"a sibling build priced, which the entry is not valid for", co(func(c *SelectedCoefficient) {
			c.Priced = []coefficient.KernelImpl{{Library: "attn-lib", Variant: "decode_v2", CodeHash: "fedcba9876543210"}}
		}), "selected_entries.coefficients[0].priced[0]", "not among kernel_impls"},
		{"a priced build the run did not run", func(r *Result) {
			r.SelectedEntries.Implementations = map[string][]coefficient.KernelImpl{"all_reduce": r.SelectedEntries.Implementations["gemm"]}
		},
			"selected_entries.coefficients[0].priced[0]", "did not run it for that primitive"},
		{"an exact entry that names nothing priced", co(func(c *SelectedCoefficient) { c.Priced = nil }),
			"selected_entries.coefficients[0].priced", "required for an exact entry"},
		{"a priced build with no code hash", co(func(c *SelectedCoefficient) {
			c.Priced = []coefficient.KernelImpl{{Library: "attn-lib", Variant: "decode_v2"}}
		}),
			"selected_entries.coefficients[0].priced[0].code_hash", "hexadecimal"},
		{"an unkeyed entry that lists implementations", co(func(c *SelectedCoefficient) { c.KernelMatch = coefficient.KernelUnkeyed }),
			"selected_entries.coefficients[0].kernel_match", "unkeyed, but the entry lists"},
		{"an unkeyed entry that names a priced build", co(func(c *SelectedCoefficient) {
			c.KernelMatch, c.KernelImpls = coefficient.KernelUnkeyed, nil
		}), "selected_entries.coefficients[0].priced", "names no implementation"},
		{"an exact entry that lists nothing", co(func(c *SelectedCoefficient) { c.KernelImpls, c.Priced = nil, nil }),
			"selected_entries.coefficients[0].kernel_match", "lists no kernel_impls"},
		{"a marked entry with no kernel match", co(func(c *SelectedCoefficient) { c.KernelMatch = "" }),
			"selected_entries.coefficients[0].kernel_match", "required"},
		{"a refused match", co(func(c *SelectedCoefficient) { c.KernelMatch = coefficient.KernelRefused }),
			"selected_entries.coefficients[0].kernel_match", "does not return"},
		{"measured_on on a measured entry", co(func(c *SelectedCoefficient) { c.MeasuredOn = c.KernelImpls }),
			"selected_entries.coefficients[0].measured_on", "not transferred"},
		{"the run's implementations out of order", func(r *Result) {
			a := coefficient.KernelImpl{Library: "attn-lib", Variant: "decode_v2", CodeHash: "0123456789abcdef"}
			b := coefficient.KernelImpl{Library: "attn-lib", Variant: "decode_v2", CodeHash: "fedcba9876543210"}
			r.SelectedEntries.Implementations = map[string][]coefficient.KernelImpl{"gemm": {b, a}}
		}, "selected_entries.implementations.gemm[1]", "sorted"},
		{"an implementation list of a word that is not a primitive", func(r *Result) {
			r.SelectedEntries.Implementations["warp_drive"] = r.SelectedEntries.Implementations["gemm"]
		}, "selected_entries.implementations.warp_drive", "not a lowered primitive"},
		{"a primitive that is not the name's", co(func(c *SelectedCoefficient) { c.Primitive = "all_reduce" }),
			"selected_entries.coefficients[0].primitive", "prices"},
		{"an exact entry with no primitive", co(func(c *SelectedCoefficient) { c.Primitive = "" }),
			"selected_entries.coefficients[0].primitive", "required for an exact entry"},
		{"the run's implementations in an unmarked report", func(r *Result) {
			r.SelectedEntries.ClosenessVersion = 0
			r.SelectedEntries.Coefficients[0] = SelectedCoefficient{Set: "s", Name: "n", Method: "measured", File: "coefficients/s.yaml"}
		}, "selected_entries.implementations", "closeness_version is absent"},
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
	r.SelectedEntries.Coefficients[0].KernelMatch = coefficient.KernelUnkeyed
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a marked report with every flag false: %s", p.Error())
	}
}

// The operations that read an entry are a subset of the primitive's operations, so a run may have
// run another build of the primitive for operations the entry is not read by (full attention on
// one kernel, the sliding-window layers on another): the entry's priced list is then narrower than
// the primitive's, and valid. A priced build the run did not run is still refused.
func TestAnEntryMayBePricedOnFewerBuildsThanThePrimitiveRan(t *testing.T) {
	r := withG1()
	b := coefficient.KernelImpl{Library: "attn-lib", Variant: "decode_v2", CodeHash: "fedcba9876543210"}
	r.SelectedEntries.Implementations["gemm"] = append(r.SelectedEntries.Implementations["gemm"], b)
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a primitive that ran another build for other operations is refused: %s", p.Error())
	}
	c := &r.SelectedEntries.Coefficients[0]
	c.Priced = append(c.Priced, coefficient.KernelImpl{Library: "attn-lib", Variant: "decode_v2", CodeHash: "00000000000000ff"})
	if p := r.Validate(); p.OK() {
		t.Fatal("a priced build the run did not run was accepted")
	}
}
