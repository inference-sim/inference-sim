package ledger_test

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

func apiRowsSources() *ledger.Sources {
	return &ledger.Sources{Kind: "LedgerSources", Sources: []ledger.Source{{
		ID: "s", Title: "t", Status: "no units yet",
		Revision: ledger.Revision{Kind: "api_rows", URL: "https://example.org/api", Ref: strings.Repeat("a", 64),
			Cutoff: "2026-11-19", Rows: 5, Commit: strings.Repeat("b", 40),
			Assets: []ledger.Asset{{Name: "recipes.jsonl", SHA256: strings.Repeat("c", 64)}}},
		Licence: ledger.Licence{SPDX: "NOASSERTION", Evidence: []ledger.LicenceQuote{{
			URL: "https://github.com/o/r/blob/" + strings.Repeat("b", 40) + "/LICENSE", SHA256: strings.Repeat("d", 64), Lines: "1", Quote: "none"}}},
	}}}
}

// D3 for an API with no revision: the rows are pinned by hash, day and count, and the recipes
// they are joined with by commit and file hash. Each known-bad pin is refused.
func TestAPIRowsPinIsExact(t *testing.T) {
	if err := apiRowsSources().Validate(); err != nil {
		t.Fatalf("a pinned snapshot was refused: %v", err)
	}
	for name, c := range map[string]struct {
		mut  func(r *ledger.Revision)
		want string
	}{
		"ref is a name":       {func(r *ledger.Revision) { r.Ref = "latest" }, "SHA-256 of the exported rows"},
		"no cutoff":           {func(r *ledger.Revision) { r.Cutoff = "" }, "YYYY-MM-DD"},
		"cutoff with time":    {func(r *ledger.Revision) { r.Cutoff = "2026-11-19T00:00:00Z" }, "YYYY-MM-DD"},
		"no row count":        {func(r *ledger.Revision) { r.Rows = 0 }, "how many rows"},
		"recipes on a branch": {func(r *ledger.Revision) { r.Commit = "main" }, "40-digit commit"},
		"no recipes pin":      {func(r *ledger.Revision) { r.Assets = nil }, "recipes file"},
		"two assets":          {func(r *ledger.Revision) { r.Assets = append(r.Assets, r.Assets[0]) }, "and nothing else"},
		"short recipes sha":   {func(r *ledger.Revision) { r.Assets[0].SHA256 = "abc" }, "recipes file"},
	} {
		s := apiRowsSources()
		c.mut(&s.Sources[0].Revision)
		if err := s.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}
}

func trainingConfig(source string) *ledger.Config {
	return &ledger.Config{Kind: "Ledger", PartitionVersion: 1, Rules: map[string]ledger.SourceRule{
		source: {CutPoints: ledger.CutPoints{{Role: ledger.RoleFit, From: 0, To: 69}, {Role: ledger.RoleSelect, From: 70, To: 84}, {Role: ledger.RoleValidate, From: 85, To: 99}}},
	}}
}

// D9, T-D2: a new calibration source is training data. Its rule may assign fit, select and
// validate only, and has no legacy side; the same cut-points are fine for any other source.
func TestTrainingOnlySources(t *testing.T) {
	for _, src := range ledger.TrainingOnlySources {
		if err := trainingConfig(src).Validate(); err != nil {
			t.Fatalf("%s: a training-only rule was refused: %v", src, err)
		}
		withHeld := trainingConfig(src)
		withHeld.Rules[src] = ledger.SourceRule{CutPoints: ledger.CutPoints{{Role: ledger.RoleFit, From: 0, To: 49}, {Role: ledger.RoleHeld, From: 50, To: 99}}}
		if err := withHeld.Validate(); err == nil || !strings.Contains(err.Error(), "training-only") {
			t.Errorf("%s: a held role: %v", src, err)
		}
		withAnchor := trainingConfig(src)
		withAnchor.Rules[src] = ledger.SourceRule{CutPoints: ledger.CutPoints{{Role: ledger.RoleFit, From: 0, To: 49}, {Role: ledger.RoleAnchor, From: 50, To: 99}}}
		if err := withAnchor.Validate(); err == nil || !strings.Contains(err.Error(), "training-only") {
			t.Errorf("%s: an anchor role: %v", src, err)
		}
		withLegacy := trainingConfig(src)
		r := withLegacy.Rules[src]
		r.Legacy = map[ledger.Side]ledger.CutPoints{ledger.SideFit: {{Role: ledger.RoleFit, From: 0, To: 74}, {Role: ledger.RoleSelect, From: 75, To: 99}}}
		withLegacy.Rules[src] = r
		if err := withLegacy.Validate(); err == nil || !strings.Contains(err.Error(), "legacy") {
			t.Errorf("%s: a legacy side: %v", src, err)
		}
	}
	other := trainingConfig("another")
	other.Rules["another"] = ledger.SourceRule{CutPoints: ledger.CutPoints{{Role: ledger.RoleFit, From: 0, To: 49}, {Role: ledger.RoleHeld, From: 50, To: 99}}}
	if err := other.Validate(); err != nil {
		t.Errorf("another source may hold units out: %v", err)
	}
}

// The committed rules of the three new sources are training-only, and each has units.
func TestCommittedNewSourcesAreTrainingOnly(t *testing.T) {
	cfg, err := ledger.LoadConfig("../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	for _, src := range ledger.TrainingOnlySources {
		r, ok := cfg.Rules[src]
		if !ok {
			t.Errorf("no rule for %s", src)
			continue
		}
		if len(r.Legacy) != 0 {
			t.Errorf("%s has a legacy side", src)
		}
		got := map[ledger.Role]bool{}
		for _, p := range r.CutPoints {
			got[p.Role] = true
		}
		if len(got) != 3 || !got[ledger.RoleFit] || !got[ledger.RoleSelect] || !got[ledger.RoleValidate] {
			t.Errorf("%s assigns %v, want fit, select and validate only", src, got)
		}
	}
}
