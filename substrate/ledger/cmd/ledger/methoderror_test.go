package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/load"
)

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

// A coefficient scoped to a chip and a model is one series per model: the series key drops the
// hardware term only. Before, the hardware pattern ran to the end of the scope and dropped the
// model with it, so two models' measured entries of one name on one chip were refused as two
// points of one series, and one model's entry on h200 and another's on h100 were compared as
// the same quantity.
func TestSeriesKeyKeepsTheModelScope(t *testing.T) {
	entry := func(models ...string) coefficient.Entry {
		return coefficient.Entry{Name: "attention_prefill_floor_sparse_mla",
			Scope: coefficient.Scope{Hardware: []string{"h200"}, Model: models}}
	}
	a, b := seriesKey("cost-model-attention", entry("deepseek-v4.1-flash")), seriesKey("cost-model-attention", entry("glm-5.3"))
	if a == b {
		t.Errorf("two models' entries share the series key %q", a)
	}
	if strings.Contains(a, "h200") || !strings.Contains(a, "deepseek-v4.1-flash") {
		t.Errorf("series key %q should drop the chip and keep the model", a)
	}
	h100 := coefficient.Entry{Name: "attention_prefill_floor_sparse_mla",
		Scope: coefficient.Scope{Hardware: []string{"h100"}, Model: []string{"deepseek-v4.1-flash"}}}
	if k := seriesKey("cost-model-attention", h100); k != a {
		t.Errorf("one model's entry on two chips is two series: %q and %q", a, k)
	}
	if k := seriesKey("s", coefficient.Entry{Name: "gemm_eps_max_bf16_h200", Scope: coefficient.Scope{Hardware: []string{"h200"}}}); k != "s|gemm_eps_max_bf16|" {
		t.Errorf("a hardware-only entry's key is %q, want %q", k, "s|gemm_eps_max_bf16|")
	}
}
