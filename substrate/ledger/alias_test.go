package ledger

import (
	"strings"
	"testing"
)

func withRecords(u Unit, recs ...string) Unit {
	u.Measurements = []Measurement{{Record: recs[0], Measured: strings.Repeat("1", 64), Duplicates: recs[1:], Merged: []MergedRecord{}, Recorded: map[string]any{}}}
	return u
}

// D4: the alias table pairs the identities two ingest tools give the same records, and a
// change that would let one new unit or role key inherit two role keys is refused.
func TestGenerateAliases(t *testing.T) {
	base := []Unit{withRecords(testUnit(1, 10), "1", "2"), withRecords(testUnit(2, 10), "3"), withRecords(testUnit(3, 11), "4")}
	// Unchanged: no aliases.
	tab, err := GenerateAliases("s", "r", base, base)
	if err != nil || len(tab.Alias) != 0 {
		t.Fatalf("unchanged lists: %v %v", tab, err)
	}
	// Unit 1 is renamed (same role key): one alias.
	renamed := []Unit{withRecords(testUnit(101, 10), "1", "2"), base[1], base[2]}
	tab, err = GenerateAliases("s", "r", base, renamed)
	if err != nil || len(tab.Alias) != 1 || tab.Alias[0].OldUnitID != base[0].UnitID || tab.Alias[0].NewUnitID != renamed[0].UnitID {
		t.Fatalf("one rename: %+v %v", tab, err)
	}
	// Units 1 and 3 (role keys 10 and 11) both land in one new unit: refused.
	merged := []Unit{withRecords(testUnit(101, 12), "1", "2", "4"), base[1]}
	if _, err := GenerateAliases("s", "r", base, merged); err == nil {
		t.Error("one unit inheriting two role keys was accepted")
	}
	// Unit 3's role key 11 is renamed into role key 10, which survives at the base: refused.
	into := []Unit{base[0], base[1], withRecords(testUnit(3, 10), "4")}
	if _, err := GenerateAliases("s", "r", base, into); err == nil {
		t.Error("a role key folded into an existing one was accepted")
	}
	// Unit 1's records split over two new units: refused.
	split := []Unit{withRecords(testUnit(101, 10), "1"), withRecords(testUnit(102, 10), "2"), base[1], base[2]}
	if _, err := GenerateAliases("s", "r", base, split); err == nil {
		t.Error("a unit split in two was accepted")
	}

	// The committed table must equal the generated one.
	gen, _ := GenerateAliases("s", "r", base, renamed)
	if err := CheckAliasesGenerated("s", []*AliasTable{gen}, gen); err != nil {
		t.Errorf("the generated table was refused: %v", err)
	}
	wrong := *gen
	wrong.Alias = []Alias{{OldUnitID: gen.Alias[0].OldUnitID, NewUnitID: base[1].UnitID, OldRoleKey: gen.Alias[0].OldRoleKey, NewRoleKey: gen.Alias[0].NewRoleKey}}
	if err := CheckAliasesGenerated("s", []*AliasTable{&wrong}, gen); err == nil {
		t.Error("a hand-written alias that differs from the generated one was accepted")
	}
	if err := CheckAliasesGenerated("s", nil, gen); err == nil {
		t.Error("a missing alias table was accepted")
	}
}
