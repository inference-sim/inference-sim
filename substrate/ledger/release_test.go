package ledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// releaseLedger writes a small ledger directory: ledger.yaml with a recorded seed, a sources.yaml
// whose source "s" has a unit list, and the unit list. It returns the directory, the seed and
// the units.
func releaseLedger(t *testing.T, n int) (string, string, []Unit) {
	t.Helper()
	seed := strings.Repeat("5a", 32)
	raw, _ := hex.DecodeString(seed)
	sum := sha256.Sum256(raw)
	cfg := testConfig()
	cfg.Seed = SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "test", GeneratedOn: "2026-11-04", RotationOwner: "test"}
	units := make([]Unit, n)
	for i := range units {
		units[i] = testUnit(i, i)
	}
	sort.Slice(units, func(i, j int) bool { return units[i].UnitID < units[j].UnitID })
	src := goodSources()
	src.Sources[0].Status = ""
	src.Sources[0].Units = DirUnits + "/s.jsonl.gz"

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, DirUnits), 0o755); err != nil {
		t.Fatal(err)
	}
	for path, v := range map[string]any{FileLedger: cfg, FileSources: src} {
		b, err := yaml.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, path), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var gz bytes.Buffer
	if err := EncodeUnits(&gz, units); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DirUnits, "s.jsonl.gz"), gz.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir, seed, units
}

func writeLists(t *testing.T, dir string, lists map[Role]*Published) {
	t.Helper()
	pub := filepath.Join(dir, DirPublished)
	if err := os.RemoveAll(pub); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(pub, 0o755); err != nil {
		t.Fatal(err)
	}
	for r, p := range lists {
		b, err := yaml.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(pub, string(r)+".yaml"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// The release gate (D7, COV-2, D1): a release carries all three published lists, and the fit and
// anchor lists each hold a unit. A tree with no lists, with an empty set of lists, or with an
// empty anchor or fit list is not releasable, and the release job refuses to write one. Each
// case is a known-bad input; the good case is the lists the release job itself wrote.
func TestReleaseRefusesAnEmptyPublishedSet(t *testing.T) {
	dir, seed, units := releaseLedger(t, 200)
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	byUnit := map[string][]Unit{"s": units}
	a, err := NewAssigner(cfg, nil, testRoleKeys, seed)
	if err != nil {
		t.Fatal(err)
	}
	lists, err := a.Release(cfg, byUnit)
	if err != nil {
		t.Fatal(err)
	}
	writeLists(t, dir, lists)
	if err := CheckRelease(dir); err != nil {
		t.Fatalf("the lists the release job wrote are not releasable: %v", err)
	}

	clone := func(mut func(map[Role]*Published)) map[Role]*Published {
		out := map[Role]*Published{}
		for r, p := range lists {
			q := *p
			q.Units = append([]string(nil), p.Units...)
			out[r] = &q
		}
		mut(out)
		return out
	}
	for name, c := range map[string]struct {
		mut  func(map[Role]*Published)
		want string
	}{
		"no list at all":  {func(m map[Role]*Published) { clear(m) }, "fit list is not published"},
		"an empty anchor": {func(m map[Role]*Published) { m[RoleAnchor].Units = []string{} }, "anchor list holds no unit"},
		"an empty fit":    {func(m map[Role]*Published) { m[RoleFit].Units = nil }, "fit list holds no unit"},
		"three empty lists": {func(m map[Role]*Published) {
			m[RoleFit].Units, m[RoleAnchor].Units, m[RoleReport].Units = []string{}, []string{}, []string{}
		}, "anchor list holds no unit"},
		"a missing anchor":    {func(m map[Role]*Published) { delete(m, RoleAnchor) }, "holds 2 of the three"},
		"a missing report":    {func(m map[Role]*Published) { delete(m, RoleReport) }, "holds 2 of the three"},
		"a missing fit alone": {func(m map[Role]*Published) { delete(m, RoleFit) }, "holds 2 of the three"},
	} {
		d, _, _ := releaseLedger(t, 200)
		writeLists(t, d, clone(c.mut))
		err := CheckRelease(d)
		if len(clone(c.mut)) == 0 {
			// No file at all: loading succeeds (a tree may merge before it releases), the gate refuses.
			err = CheckReleasable(map[Role]*Published{})
		}
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}

	// A tree that has published nothing yet loads, and is refused for release.
	bare, _, _ := releaseLedger(t, 200)
	if _, err := LoadPublished(bare, cfg, byUnit); err != nil {
		t.Errorf("a tree with no lists must still load: %v", err)
	}
	if err := CheckRelease(bare); err == nil || !strings.Contains(err.Error(), "is not published") {
		t.Errorf("a tree with no lists was releasable: %v", err)
	}

	// An empty report list is allowed: the partition gives the report role no cut-points.
	if err := CheckReleasable(clone(func(m map[Role]*Published) { m[RoleReport].Units = []string{} })); err != nil {
		t.Errorf("an empty report list was refused: %v", err)
	}

	// The release job writes no empty fit or anchor list: units that all draw one role.
	var onlyFit []Unit
	for i := 0; len(onlyFit) < 5; i++ {
		u := testUnit(i, i)
		if r, err := a.Role("s", &u); err == nil && r == RoleFit {
			onlyFit = append(onlyFit, u)
		}
	}
	if _, err := a.Release(cfg, map[string][]Unit{"s": onlyFit}); err == nil || !strings.Contains(err.Error(), "anchor list holds no unit") {
		t.Errorf("the release job wrote lists with no anchor: %v", err)
	}
}

// A published file that is present but empty fails the load itself, not only the release gate:
// three empty lists drawn under the right seed were valid before.
func TestPublishedCheckRefusesEmptyFitAndAnchor(t *testing.T) {
	dir, _, units := releaseLedger(t, 10)
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	mk := func(r Role, ids ...string) *Published {
		return &Published{Kind: PublishedKind, Role: r, PartitionVersion: 1, SeedSHA256: cfg.Seed.SHA256, Units: ids}
	}
	for _, r := range []Role{RoleFit, RoleAnchor} {
		if err := mk(r).Check(r, cfg); err == nil || !strings.Contains(err.Error(), "holds no unit") {
			t.Errorf("an empty %s list passed Check: %v", r, err)
		}
		if err := mk(r, units[0].UnitID).Check(r, cfg); err != nil {
			t.Errorf("a %s list of one unit was refused: %v", r, err)
		}
	}
	if err := mk(RoleReport).Check(RoleReport, cfg); err != nil {
		t.Errorf("an empty report list was refused: %v", err)
	}
}
