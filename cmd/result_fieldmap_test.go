package cmd

import (
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// TestResultFieldMap_EveryMetricsOutputFieldHasAHome is #1871's acceptance criterion that
// every field of the MetricsOutput JSON has a home in the Result or is listed as dropped with
// its reason: the docs/guide/results.md table names each field, and a field added to
// MetricsOutput without a row fails here.
func TestResultFieldMap_EveryMetricsOutputFieldHasAHome(t *testing.T) {
	data, err := os.ReadFile("../docs/guide/results.md")
	if err != nil {
		t.Fatal(err)
	}
	doc := string(data)
	start, end := strings.Index(doc, "<!-- result-field-map -->"), strings.Index(doc, "<!-- /result-field-map -->")
	if start < 0 || end < start {
		t.Fatal("docs/guide/results.md has no result-field-map table")
	}
	mapped := map[string]bool{}
	cell := regexp.MustCompile("`([a-z0-9_]+)`")
	for _, line := range strings.Split(doc[start:end], "\n") {
		cols := strings.Split(line, "|")
		if len(cols) < 3 {
			continue
		}
		for _, m := range cell.FindAllStringSubmatch(cols[1], -1) {
			mapped[m[1]] = true
		}
		if home := strings.TrimSpace(cols[2]); home == "" {
			t.Errorf("row %q names no home and no reason", strings.TrimSpace(line))
		}
	}
	fields := jsonFields(reflect.TypeOf(sim.MetricsOutput{}))
	if len(fields) < 40 {
		t.Fatalf("non-vacuity: found only %d MetricsOutput fields", len(fields))
	}
	for _, f := range fields {
		if !mapped[f] {
			t.Errorf("MetricsOutput field %q has no row in the result-field-map table", f)
		}
	}
}

// jsonFields are the JSON keys a struct type marshals to, following embedded structs.
func jsonFields(typ reflect.Type) []string {
	var out []string
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		ft := f.Type
		if ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		tag := f.Tag.Get("json")
		if f.Anonymous && tag == "" && ft.Kind() == reflect.Struct {
			out = append(out, jsonFields(ft)...)
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name == "" || name == "-" || !f.IsExported() {
			continue
		}
		out = append(out, name)
	}
	return out
}
