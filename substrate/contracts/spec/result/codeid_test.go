package result

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
)

// withCodeID returns the valid Result whose first coefficient is an engine-path entry valid for
// one CodeID and scoped to the host CPUs it was carried for.
func withCodeID() *Result {
	r := valid()
	r.SelectedEntries.ClosenessVersion = ClosenessVersion
	c := &r.SelectedEntries.Coefficients[0]
	c.Name, c.KernelMatch = "host_output_token", coefficient.KernelUnkeyed
	c.KernelImpls, c.Priced, c.Primitive = nil, nil, ""
	c.Scope = coefficient.Scope{Hardware: []string{"h200"}, HostCPU: []string{"x86"}}
	c.CodeIDs = []string{"0123456789abcdef"}
	return r
}

func TestAResultNamesTheCodeAnEngineCostWasPricedFor(t *testing.T) {
	r := withCodeID()
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	for _, f := range []Format{FormatJSON, FormatYAML} {
		b, err := Marshal(r, f)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), "code_ids") || !strings.Contains(string(b), "host_cpu") {
			t.Errorf("%s: the report omits the code ids or the host cpu:\n%s", f, b)
		}
		back := &Result{}
		if err := Unmarshal(b, back); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if !reflect.DeepEqual(back.SelectedEntries, r.SelectedEntries) {
			t.Errorf("%s: did not round-trip:\n got %+v\nwant %+v", f, back.SelectedEntries, r.SelectedEntries)
		}
	}
}

func TestTheCodeIDsOfAnEntryAreRefusedWhenMalformed(t *testing.T) {
	co := func(f func(c *SelectedCoefficient)) func(*Result) {
		return func(r *Result) { f(&r.SelectedEntries.Coefficients[0]) }
	}
	for _, c := range []struct {
		name   string
		break_ func(r *Result)
		path   string
		msg    string
	}{
		{"not a digest", co(func(c *SelectedCoefficient) { c.CodeIDs = []string{"HOST"} }),
			"selected_entries.coefficients[0].code_ids", "lower-case hexadecimal"},
		{"a repeated id", co(func(c *SelectedCoefficient) { c.CodeIDs = []string{"0123456789abcdef", "0123456789abcdef"} }),
			"selected_entries.coefficients[0].code_ids", "duplicate"},
		{"on a name that prices no engine-path code", co(func(c *SelectedCoefficient) { c.Name = "hbm_derate" }),
			"selected_entries.coefficients[0].code_ids", "prices no engine-path code"},
		{"in an unmarked report", func(r *Result) {
			r.SelectedEntries.ClosenessVersion = 0
			c := &r.SelectedEntries.Coefficients[0]
			c.KernelMatch = ""
		}, "selected_entries.coefficients[0]", "closeness_version is absent"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := withCodeID()
			c.break_(r)
			p := r.Validate()
			if p.OK() || !strings.Contains(p.Error(), c.path) || !strings.Contains(p.Error(), c.msg) {
				t.Errorf("want a problem at %s containing %q, got:\n%s", c.path, c.msg, p.Error())
			}
		})
	}
}

// Two entries of one name that differ only in the code they are valid for are both listed, in a
// stable order.
func TestEntriesThatDifferOnlyInTheirCodeIDsAreListedOnceEach(t *testing.T) {
	r := withCodeID()
	a := r.SelectedEntries.Coefficients[0]
	b := a
	b.CodeIDs = []string{"fedcba9876543210"}
	r.SelectedEntries.Coefficients = []SelectedCoefficient{a, b}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("%s", p.Error())
	}
	r.SelectedEntries.Coefficients = []SelectedCoefficient{b, a}
	if p := r.Validate(); p.OK() {
		t.Error("entries out of order were accepted")
	}
}
