package vllmperf_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/vllmperf"
)

var (
	c1 = strings.Repeat("a", 40)
	c2 = strings.Repeat("b", 40)
)

func row(date, model, device, conc, image, tput string) string {
	t := "null"
	if tput != "" {
		t = fmt.Sprintf("%q", tput)
	}
	return fmt.Sprintf(`{"date":%q,"model":%q,"device":%q,"tp":"8","conc":%q,"isl":"8192","osl":"1024","precision":"bf16","image":%q,`+
		`"tput_per_gpu":%s,"input_tput_per_gpu":null,"output_tput_per_gpu":null,"mean_ttft":"0.5","mean_tpot":null,"mean_itl":null,"mean_e2el":null,`+
		`"p99_ttft":null,"p99_tpot":null,"p99_itl":null,"p99_e2el":null,"median_ttft":null,"median_tpot":null,"median_itl":null,"median_e2el":null}`,
		date, model, device, conc, image, t)
}

func recipe(wl, gpu, model, bench string, conc int) string {
	return fmt.Sprintf(`{"workload":%q,"gpu":%q,"model":%q,"serve_args":"--tp 8","benchmark":%q,"input_len":8192,"output_len":1024,"max_concurrency":%d}`,
		wl, gpu, model, bench, conc)
}

func img(tag string) string { return "registry/vllm-release-repo:" + tag }

func input(t *testing.T, rows, recipes []string) *vllmperf.Input {
	t.Helper()
	in, err := vllmperf.ReadInput(strings.NewReader(strings.Join(rows, "\n")+"\n"), strings.NewReader(strings.Join(recipes, "\n")+"\n"))
	if err != nil {
		t.Fatal(err)
	}
	return in
}

// A unit is a benchmark configuration at one commit. Rows of one configuration on different
// days are measurements of one unit, a row with another commit is another unit, and each row
// that cannot be placed is left out under its reason, so every row is accounted for.
func TestIngestUnitsAndLeftOut(t *testing.T) {
	rs := []string{recipe("w-h200", "H200", "m", "8k-in-1k-out", 128)}
	rows := []string{
		row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "100.5"),
		row("2026-06-02 00:00:00", "m", "h200", "128", img(c1), "101.5"),
		row("2026-06-02 00:00:00", "m", "h200", "128", img(c2), "102.5"),
		row("2026-06-03 00:00:00", "m", "h200", "128", img("nightly"), "103.5"),
		row("2026-06-03 00:00:00", "m", "h200", "128", img(c1[:12]), "104.5"),
		row("2026-06-03 00:00:00", "m", "h200", "64", img(c1), "105.5"),
		row("2026-06-03 00:00:00", "other", "h200", "128", img(c1), "106.5"),
		row("2026-06-03 00:00:00", "m", "h200", "128", img(c2), ""),
		row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "100.5"),
	}
	units, rep, err := vllmperf.Ingest(input(t, rows, rs), "rev")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Records != 9 || rep.Units != 2 || rep.RoleKeys != 2 || rep.Duplicates != 1 || rep.JoinedUnits != 1 {
		t.Errorf("records %d units %d role keys %d duplicates %d joined %d, want 9 2 2 1 1", rep.Records, rep.Units, rep.RoleKeys, rep.Duplicates, rep.JoinedUnits)
	}
	for why, n := range map[string]int{vllmperf.LeftOutNoCommit: 2, vllmperf.LeftOutNoRecipe: 2, vllmperf.LeftOutEmptyMetric: 1} {
		if len(rep.LeftOut[why]) != n {
			t.Errorf("left out as %s: %v, want %d", why, rep.LeftOut[why], n)
		}
	}
	for _, u := range units {
		if u.Identity["benchmark"] != "8k-in-1k-out" || u.Identity["hardware"] != "h200" {
			t.Errorf("identity %v", u.Identity)
		}
		rk, err := vllmperf.RoleKey(u.Identity)
		if err != nil {
			t.Fatal(err)
		}
		if id, _ := ledger.ID(rk); id != u.RoleKey {
			t.Errorf("unit %s: role key %s, want %s", u.UnitID, u.RoleKey, id)
		}
	}
}

// A configuration that two recipes name differently has no single benchmark name, so its rows
// are left out; two recipes that agree are one.
func TestAmbiguousRecipeIsLeftOut(t *testing.T) {
	rows := []string{row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "1.5")}
	_, rep, err := vllmperf.Ingest(input(t, rows, []string{recipe("a", "H200", "m", "x", 128), recipe("b", "H200", "m", "y", 128)}), "rev")
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.LeftOut[vllmperf.LeftOutAmbiguousRecipe]) != 1 || rep.Units != 0 {
		t.Errorf("units %d left out %v", rep.Units, rep.LeftOut)
	}
	_, rep, err = vllmperf.Ingest(input(t, rows, []string{recipe("a", "H200", "m", "x", 128), recipe("a", "H200", "m", "x", 128)}), "rev")
	if err != nil || rep.Units != 1 {
		t.Errorf("identical recipes: units %d, %v", rep.Units, err)
	}
}

// The role key keeps the commit, so rows of one configuration at two commits do not share one,
// and the configuration hash separates the same name at another tensor-parallel size.
func TestRoleKeySeparatesCommitsAndConfigurations(t *testing.T) {
	rs := []string{recipe("w", "H200", "m", "b", 128)}
	other := strings.Replace(row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "3.5"), `"tp":"8"`, `"tp":"4"`, 1)
	rows := []string{
		row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "1.5"),
		row("2026-06-01 00:00:00", "m", "h200", "128", img(c2), "2.5"),
		other,
	}
	_, rep, err := vllmperf.Ingest(input(t, rows, rs), "rev")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Units != 3 || rep.RoleKeys != 3 {
		t.Errorf("units %d role keys %d, want 3 3", rep.Units, rep.RoleKeys)
	}
}

func TestImageCommit(t *testing.T) {
	for in, want := range map[string]string{
		"r/x:" + c1:                  c1,
		"r/x:nightly":                "",
		"r/x:" + c1[:12]:             "",
		"r/x":                        "",
		"host:5000/x":                "",
		"host:5000/x:" + c1:          c1,
		"r/x@sha256:" + c1 + c1[:24]: "",
		"r/x:" + strings.ToUpper(c1): "",
	} {
		if _, got := vllmperf.ImageCommit(in); got != want {
			t.Errorf("ImageCommit(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestReadInputIsStrict(t *testing.T) {
	good := row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "1.5")
	rc := recipe("w", "H200", "m", "b", 128)
	for name, c := range map[string][2]string{
		"unknown row field":    {strings.Replace(good, `"tp"`, `"extra":1,"tp"`, 1), rc},
		"unknown recipe field": {good, strings.Replace(rc, `"gpu"`, `"extra":1,"gpu"`, 1)},
		"two documents":        {good + good, rc},
		"not JSON":             {"x", rc},
	} {
		if _, err := vllmperf.ReadInput(strings.NewReader(c[0]+"\n"), strings.NewReader(c[1]+"\n")); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// D3: the rows and the recipes are the pinned ones, as many, and none is dated after the cutoff.
func TestCheckRevision(t *testing.T) {
	rows := []string{row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "1.5")}
	in := input(t, rows, []string{recipe("w", "H200", "m", "b", 128)})
	good := func() *ledger.Source {
		return &ledger.Source{ID: vllmperf.SourceID, Revision: ledger.Revision{Ref: in.RowsSHA256, Rows: 1, Cutoff: "2026-06-01",
			Assets: []ledger.Asset{{Name: "recipes.jsonl", SHA256: in.RecipesSHA256}}}}
	}
	if err := vllmperf.CheckRevision(good(), in); err != nil {
		t.Fatalf("the pinned input was refused: %v", err)
	}
	for name, mut := range map[string]func(s *ledger.Source){
		"other rows":    func(s *ledger.Source) { s.Revision.Ref = strings.Repeat("0", 64) },
		"other count":   func(s *ledger.Source) { s.Revision.Rows = 2 },
		"other recipes": func(s *ledger.Source) { s.Revision.Assets[0].SHA256 = strings.Repeat("0", 64) },
		"after cutoff":  func(s *ledger.Source) { s.Revision.Cutoff = "2026-05-31" },
	} {
		s := good()
		mut(s)
		if err := vllmperf.CheckRevision(s, in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
