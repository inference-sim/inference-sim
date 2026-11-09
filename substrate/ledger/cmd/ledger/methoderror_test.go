package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/load"
)

const methodErrorPagePath = "../../../docs/registry/reference/method-error.md"

// The committed reference page is the page the committed registry produces (VAL-5): a figure
// on it cannot disagree with the data.
func TestMethodErrorPageIsCurrent(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"method-error", "-check", methodErrorPagePath}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s%s", code, out.String(), errb.String())
	}
}

func TestAStaleMethodErrorPageIsRefused(t *testing.T) {
	raw, err := os.ReadFile(methodErrorPagePath)
	if err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(t.TempDir(), "page.md")
	if err := os.WriteFile(stale, []byte(strings.Replace(string(raw), "| 1.04 |", "| 9.99 |", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := run([]string{"method-error", "-check", stale}, &out, &errb); code != 1 || !strings.Contains(errb.String(), "not the page the registry produces") {
		t.Errorf("exit %d: %s", code, errb.String())
	}
}

func TestMethodErrorWritesAReportThatNamesNoUnit(t *testing.T) {
	dir := t.TempDir()
	rep := filepath.Join(dir, "report.json")
	var out, errb bytes.Buffer
	sha := strings.Repeat("a", 40)
	if code := run([]string{"method-error", "-candidate", sha, "-out", rep}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	raw, err := os.ReadFile(rep)
	if err != nil {
		t.Fatal(err)
	}
	var r ledger.Report
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	if r.Job != "error" || !r.Evaluated || r.Candidate != sha || len(r.Aggregates) < 20 {
		t.Errorf("report = job %q evaluated %v candidate %q, %d aggregates", r.Job, r.Evaluated, r.Candidate, len(r.Aggregates))
	}
	for _, a := range r.Aggregates {
		if a.Units < ledger.MinAggregateUnits {
			t.Errorf("%+v covers fewer than %d points", a, ledger.MinAggregateUnits)
		}
	}
	if err := ledger.CheckReport(raw, nil); err != nil {
		t.Errorf("the seed-holding check refuses the report: %v", err)
	}
	// A report needs both a commit and a place to go.
	for _, args := range [][]string{{"method-error"}, {"method-error", "-out", rep}, {"method-error", "-candidate", sha}} {
		if code := run(args, &out, &errb); code != 2 {
			t.Errorf("%v: exit %d, want 2", args, code)
		}
	}
}

// The generation table equals the one the registry's by-generation writer uses, so the error is
// measured along the axis the values are written along.
func TestGenerationsEqualTheWritersTable(t *testing.T) {
	raw, err := os.ReadFile("../../../registry/scripts/extrapolate_by_generation.py")
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)GENERATION = \{(.*?)\n\}`).FindStringSubmatch(string(raw))
	if block == nil {
		t.Fatal("the writer has no GENERATION table")
	}
	got := map[string]string{}
	for _, m := range regexp.MustCompile(`"([a-z0-9-]+)":\s*"([A-Za-z]+)"`).FindAllStringSubmatch(block[1], -1) {
		got[m[1]] = m[2]
	}
	if len(got) != len(generations) {
		t.Fatalf("writer %v, here %v", got, generations)
	}
	for c, g := range got {
		if generations[c] != g {
			t.Errorf("%s: writer %q, here %q", c, g, generations[c])
		}
	}
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range d.Chips() {
		if _, ok := generations[c]; !ok {
			t.Errorf("catalog chip %s has no generation; add it to generations and to the writer", c)
		}
	}
}

// Every measured, positive coefficient the cost model's sets carry becomes a point, and a
// point is never the same chip twice in a series.
func TestRegistryPointsAreWellFormed(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	pts, err := registryPoints(d, methodErrorSets)
	if err != nil {
		t.Fatal(err)
	}
	if len(pts) < 800 {
		t.Fatalf("%d points; the registry carries about 900 measured values", len(pts))
	}
	type k struct{ f, key, chip string }
	seen := map[k]bool{}
	for _, p := range pts {
		if !(p.Value > 0) {
			t.Errorf("%+v", p)
		}
		key := k{p.Family, p.Key, p.Chip}
		if seen[key] {
			t.Errorf("two points for %v", key)
		}
		seen[key] = true
	}
}
