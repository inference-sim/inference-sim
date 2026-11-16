package main

import (
	"strings"
	"testing"
)

// The comparison fails on any difference in a price, a figure or a refusal, and on a record one
// side lacks; a reordering passes only for a field the caller names.
func TestCompareFailsOnAnyDifferenceButANamedReordering(t *testing.T) {
	base := `{"fixture":"a.yaml","pool":0,"step_ns":{"d":10},"origins":[{"Name":"x"},{"Name":"y"}]}
{"fixture":"b.yaml","pool":0,"error":"refused: no rules"}
`
	order := map[string]bool{"origins": true, "prov": true}
	for _, c := range []struct {
		name, new string
		allow     map[string]bool
		fail      bool
	}{
		{"identical", base, order, false},
		{"origins reordered and allowed", strings.Replace(base, `[{"Name":"x"},{"Name":"y"}]`, `[{"Name":"y"},{"Name":"x"}]`, 1), order, false},
		{"origins reordered and not allowed", strings.Replace(base, `[{"Name":"x"},{"Name":"y"}]`, `[{"Name":"y"},{"Name":"x"}]`, 1), nil, true},
		{"origins changed", strings.Replace(base, `{"Name":"y"}`, `{"Name":"z"}`, 1), order, true},
		{"a price changed", strings.Replace(base, `"d":10`, `"d":11`, 1), order, true},
		{"a refusal changed", strings.Replace(base, "no rules", "no chip", 1), order, true},
		{"a refusal became a price", strings.Replace(base, `"error":"refused: no rules"`, `"step_ns":{"d":1}`, 1), order, true},
		{"a record is missing", strings.SplitAfter(base, "\n")[0], order, true},
		{"a field only the new side has", strings.Replace(base, `"origins":[{"Name":"x"},{"Name":"y"}]`, `"origins":[{"Name":"x"},{"Name":"y"}],"prov":{"A":[1,2]}`, 1), order, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			r, err := compare(strings.NewReader(base), strings.NewReader(c.new), c.allow)
			if err != nil {
				t.Fatal(err)
			}
			if r.failed() != c.fail {
				t.Errorf("failed %v, want %v (differs %v)", r.failed(), c.fail, r.Differs)
			}
		})
	}
}

// A provenance object whose lists hold the same items in another order passes where its field is
// allowed to be reordered.
func TestCompareReadsNestedListsAsMultisets(t *testing.T) {
	old := `{"fixture":"a.yaml","pool":0,"prov":{"Assumptions":[{"Name":"x"},{"Name":"y"}],"Total":3}}` + "\n"
	reordered := strings.Replace(old, `[{"Name":"x"},{"Name":"y"}]`, `[{"Name":"y"},{"Name":"x"}]`, 1)
	changed := strings.Replace(old, `"Total":3`, `"Total":4`, 1)
	for _, c := range []struct {
		name, new string
		fail      bool
	}{{"reordered", reordered, false}, {"a count changed", changed, true}} {
		r, err := compare(strings.NewReader(old), strings.NewReader(c.new), map[string]bool{"prov": true})
		if err != nil {
			t.Fatal(err)
		}
		if r.failed() != c.fail {
			t.Errorf("%s: failed %v, want %v (differs %v)", c.name, r.failed(), c.fail, r.Differs)
		}
	}
}
