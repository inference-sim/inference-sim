package cmd

import (
	"os"
	"path/filepath"
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

// #1871 review N2: every home the field-map table names exists in an emitted Result. One
// document exercises everything but the conditional sections: a multi-instance P/D run with SLO
// targets, a detector, the deprecated --metrics-path file and a policy override. The homes only
// a different kind of run states are listed with the run that states them.
func TestResultFieldMap_EveryNamedHomeExistsInAnEmittedResult(t *testing.T) {
	conditional := map[string]string{
		"engine.cluster.acceptance":    "a speculating run",
		"engine.cluster.adapters":      "a LoRA run",
		"breakdowns.per_adapter":       "a LoRA run",
		"engine.cluster.agentic":       "an agentic replay",
		"engine.cluster.server_series": "--server-series-interval-s",
		"overrides":                    "a run that sets a flag or overrides its scenario",
		"profiling_start":              "an agentic replay's explicit window (window.start)",
		"profiling_end":                "an agentic replay's explicit window (window.stop)",
	}
	data, err := os.ReadFile("../docs/guide/results.md")
	if err != nil {
		t.Fatal(err)
	}
	doc := string(data)
	table := doc[strings.Index(doc, "<!-- result-field-map -->"):strings.Index(doc, "<!-- /result-field-map -->")]
	path := regexp.MustCompile("`((?:requests|metrics|engine|breakdowns|goodput|window|revisions|artifacts|saturation|kernel)[a-z0-9_.*]*)`")
	var homes []string
	for _, line := range strings.Split(table, "\n") {
		if cols := strings.Split(line, "|"); len(cols) >= 3 {
			for _, m := range path.FindAllStringSubmatch(cols[2], -1) {
				homes = append(homes, m[1])
			}
		}
	}
	if len(homes) < 40 {
		t.Fatalf("non-vacuity: found %d homes in the table", len(homes))
	}
	metricsFile := filepath.Join(t.TempDir(), "m.json")
	out, stderr, err := runKernelCLI(t, append([]string{"run", "--metrics-path", metricsFile, "--slo-ttft", "default=500ms",
		"--detectors", "composite"}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	emitted := resultDoc(t, out)
	for _, h := range homes {
		if _, ok := conditional[h]; ok {
			continue
		}
		if !homeExists(emitted, strings.Split(h, ".")) {
			t.Errorf("the field map names %q, which the emitted Result does not carry", h)
		}
	}
}

// homeExists reports whether the dotted path, with * matching any key part, is in doc.
func homeExists(v any, segs []string) bool {
	if len(segs) == 0 {
		return true
	}
	m, ok := v.(map[string]any)
	if !ok {
		return false
	}
	if !strings.Contains(segs[0], "*") {
		next, ok := m[segs[0]]
		return ok && homeExists(next, segs[1:])
	}
	re := regexp.MustCompile("^" + strings.ReplaceAll(regexp.QuoteMeta(segs[0]), `\*`, ".+") + "$")
	for k, next := range m {
		if re.MatchString(k) && homeExists(next, segs[1:]) {
			return true
		}
	}
	return false
}
