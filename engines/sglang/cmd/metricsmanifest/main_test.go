// Original to blis-sglang.

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tree(t *testing.T, py string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "python", "sglang", "srt")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "m.py"), []byte(py), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func classes(t *testing.T, js string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.json")
	if err := os.WriteFile(p, []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const gauge = "x = Gauge(\n    name=\"sglang:alpha\",\n    documentation=\"d\",\n    labelnames=labels.keys(),\n)\n"

func TestAFamilyWithoutAClassFailsTheRun(t *testing.T) {
	err := run(tree(t, gauge), classes(t, `{"exact_series":[],"rules":[],"not_families":{}}`), filepath.Join(t.TempDir(), "o.json"), false)
	if err == nil || !strings.Contains(err.Error(), "no class for sglang:alpha") {
		t.Fatalf("got %v, want a missing class for sglang:alpha", err)
	}
}

func TestEveryFamilyIsListedWithItsClass(t *testing.T) {
	out := filepath.Join(t.TempDir(), "o.json")
	js := `{"exact_series":["sglang:alpha"],"rules":[{"match":"beta","class":"later","why":"w"}],"not_families":{}}`
	src := gauge + "y = Counter(\"sglang:beta_total\", \"d\")\n"
	if err := run(tree(t, src), classes(t, js), out, false); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(out)
	s := string(b)
	if !strings.Contains(s, `"name": "sglang:alpha"`) || !strings.Contains(s, `"class": "exact"`) || !strings.Contains(s, `"class": "later"`) {
		t.Fatalf("manifest lacks the families or classes:\n%s", s)
	}
}

func TestAnUnaccountedMetricNameFailsTheRun(t *testing.T) {
	js := `{"exact_series":["sglang:alpha"],"rules":[],"not_families":{}}`
	err := run(tree(t, gauge+"print(\"sglang:gamma\")\n"), classes(t, js), filepath.Join(t.TempDir(), "o.json"), false)
	if err == nil || !strings.Contains(err.Error(), "sglang:gamma") {
		t.Fatalf("got %v, want the stray name reported", err)
	}
}

func TestACheckFailsWhenTheManifestIsStale(t *testing.T) {
	js := `{"exact_series":["sglang:alpha"],"rules":[],"not_families":{}}`
	out := filepath.Join(t.TempDir(), "o.json")
	if err := os.WriteFile(out, []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(tree(t, gauge), classes(t, js), out, true); err == nil {
		t.Fatal("a stale manifest passed -check")
	}
}
