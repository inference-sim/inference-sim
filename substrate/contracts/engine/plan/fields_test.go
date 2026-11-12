package plan

import (
	"bufio"
	"os"
	"path/filepath"
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
	priced := pricedFields(t)
	have := map[string]bool{}
	for _, fl := range planFields() {
		have[fl] = true
		_, isListed := listed[fl]
		_, isPriced := priced[fl]
		switch {
		case isListed && isPriced:
			t.Errorf("%s is on both testdata/unpriced_plan_fields.txt and testdata/priced_plan_fields.txt; delete it from the unpriced list", fl)
		case !isListed && !isPriced:
			t.Errorf("%s is neither priced by a test (testdata/priced_plan_fields.txt) nor on testdata/unpriced_plan_fields.txt; a new field is priced in the change that adds it", fl)
		}
	}
	for fl := range listed {
		if !have[fl] {
			t.Errorf("%s is listed but no longer exists; remove the line", fl)
		}
	}
	for fl := range priced {
		if !have[fl] {
			t.Errorf("%s is listed as priced but no longer exists; remove the line", fl)
		}
	}
	if len(have) < 40 {
		t.Fatalf("the walk found %d fields; it is not reaching the plan's parts", len(have))
	}
}

// pricedFields reads testdata/priced_plan_fields.txt: each field a test prices, with that test.
// The test must exist, as a function of the package it names, so a line cannot name a proof
// that was deleted or never written.
func pricedFields(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	f, err := os.Open("testdata/priced_plan_fields.txt")
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
		field, proof, ok := strings.Cut(l, " # ")
		pkg, name, ok2 := strings.Cut(strings.TrimSpace(proof), ": ")
		if !ok || !ok2 || pkg == "" || name == "" {
			t.Errorf("%q: want \"Type.Field # package: TestName\"", l)
			continue
		}
		if _, dup := out[field]; dup {
			t.Errorf("%s is listed twice", field)
		}
		out[field] = proof
		dir := map[string]string{"kernel": "../../../kernel"}[pkg]
		if dir == "" {
			t.Errorf("%s: the proof is in package %q, which this check does not know where to find", field, pkg)
			continue
		}
		if !testExists(t, dir, name) {
			t.Errorf("%s: the proof %s: %s does not exist", field, pkg, name)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// testExists reports whether dir holds a test function called name.
func testExists(t *testing.T, dir, name string) bool {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*_test.go"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no test files in %s (%v)", dir, err)
	}
	for _, p := range files {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), "func "+name+"(t *testing.T)") {
			return true
		}
	}
	return false
}
