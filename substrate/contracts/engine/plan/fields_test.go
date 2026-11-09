package plan

import (
	"bufio"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// planFields lists every field of PoolPlan, StepShape and Transfer and of the types they hold,
// as Type.Field, in sorted order. It walks the types, so a field added to one appears here
// without anyone remembering to list it.
func planFields() []string {
	seen := map[reflect.Type]bool{}
	var out []string
	var walk func(t reflect.Type)
	walk = func(t reflect.Type) {
		for t.Kind() == reflect.Pointer || t.Kind() == reflect.Slice || t.Kind() == reflect.Array {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct || seen[t] || !strings.Contains(t.PkgPath(), "contracts/engine/plan") {
			return
		}
		seen[t] = true
		for i := 0; i < t.NumField(); i++ {
			out = append(out, t.Name()+"."+t.Field(i).Name)
			walk(t.Field(i).Type)
		}
	}
	for _, v := range []any{PoolPlan{}, StepShape{}, Transfer{}} {
		walk(reflect.TypeOf(v))
	}
	sort.Strings(out)
	return out
}

// SUB-11: every field of PoolPlan, StepShape and Transfer changes some price when it is
// changed, or is listed as not yet priced with the reason. The cost model that reads a plan
// is the next step of the boundary transition (design §3.3): until it does, no field is
// priced, and testdata/unpriced_plan_fields.txt lists them all. The list may only shrink, and
// the test fails on a field that is not on it and on a listed field that no longer exists. A
// field is deleted from the list in the change that prices it, with the test that proves it.
func TestEveryPlanFieldIsPricedOrListed(t *testing.T) {
	listed := map[string]string{}
	f, err := os.Open("testdata/unpriced_plan_fields.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		field, why, _ := strings.Cut(l, " # ")
		if _, dup := listed[field]; dup {
			t.Errorf("%s is listed twice", field)
		}
		if strings.TrimSpace(why) == "" {
			t.Errorf("%s is listed with no reason", field)
		}
		listed[field] = why
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, fl := range planFields() {
		have[fl] = true
		if _, ok := listed[fl]; !ok {
			t.Errorf("%s is neither priced by a test nor on testdata/unpriced_plan_fields.txt; a new field is priced in the change that adds it", fl)
		}
	}
	for fl := range listed {
		if !have[fl] {
			t.Errorf("%s is listed but no longer exists; remove the line", fl)
		}
	}
	if len(have) < 40 {
		t.Fatalf("the walk found %d fields; it is not reaching the plan's parts", len(have))
	}
}
