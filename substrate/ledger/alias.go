package ledger

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
)

// records returns every source record a unit covers: each measurement's kept record, its
// duplicates and the records merged into it.
func records(u *Unit) []string {
	var out []string
	for _, m := range u.Measurements {
		out = append(out, m.Record)
		out = append(out, m.Duplicates...)
		for _, x := range m.Merged {
			out = append(out, x.Record)
		}
	}
	return out
}

// GenerateAliases pairs the identities two ingest tools give the same records of one source
// revision (design §10.2). base is the merge base's unit list and cand the candidate's, both
// generated from the same pinned revision. For each base unit whose unit_id or role key the
// candidate does not keep, it finds the candidate units holding its records.
//
// It returns the alias table, or an error naming the units that cannot be aliased: a base
// unit whose records fall in two candidate units, or a candidate unit or role key that would
// inherit two base role keys (and so possibly two roles). Such a change needs a new partition
// version, which a person approves.
func GenerateAliases(source, reason string, base, cand []Unit) (*AliasTable, error) {
	inCand := map[string]*Unit{}
	unitOfRecord := map[string]string{}
	for i := range cand {
		inCand[cand[i].UnitID] = &cand[i]
		for _, r := range records(&cand[i]) {
			unitOfRecord[r] = cand[i].UnitID
		}
	}
	t := &AliasTable{Kind: "AliasTable", Source: source, Reason: reason}
	var errs []error
	newKeyFrom := map[string]string{}  // new role key -> the base role key it inherits
	newUnitFrom := map[string]string{} // new unit_id -> the base role key it inherits
	for i := range base {
		b := &base[i]
		if c, ok := inCand[b.UnitID]; ok && c.RoleKey == b.RoleKey {
			continue
		}
		targets := map[string]bool{}
		for _, r := range records(b) {
			if u, ok := unitOfRecord[r]; ok {
				targets[u] = true
			}
		}
		switch len(targets) {
		case 0:
			errs = append(errs, fmt.Errorf("unit %s: none of its records is in the candidate's unit list", b.UnitID))
			continue
		case 1:
		default:
			errs = append(errs, fmt.Errorf("unit %s: its records fall in %d candidate units; one unit cannot keep one role as several", b.UnitID, len(targets)))
			continue
		}
		var target string
		for u := range targets {
			target = u
		}
		c := inCand[target]
		for k, m := range map[string]map[string]string{c.RoleKey: newKeyFrom, c.UnitID: newUnitFrom} {
			if prev, ok := m[k]; ok && prev != b.RoleKey {
				errs = append(errs, fmt.Errorf("unit %s: %s would inherit role keys %s and %s, which can have two roles; this needs a new partition version", b.UnitID, k, prev, b.RoleKey))
			}
			m[k] = b.RoleKey
		}
		t.Alias = append(t.Alias, Alias{OldUnitID: b.UnitID, NewUnitID: c.UnitID, OldRoleKey: b.RoleKey, NewRoleKey: c.RoleKey})
	}
	// A candidate unit or role key that kept a base identity must not also inherit another.
	baseKeys := map[string]bool{}
	for i := range base {
		baseKeys[base[i].RoleKey] = true
	}
	for k, from := range newKeyFrom {
		if baseKeys[k] && k != from {
			errs = append(errs, fmt.Errorf("role key %s exists at the merge base and would also inherit %s; this needs a new partition version", k, from))
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	sort.Slice(t.Alias, func(i, j int) bool { return t.Alias[i].OldUnitID < t.Alias[j].OldUnitID })
	return t, nil
}

// CheckAliasesGenerated compares the candidate's committed alias tables for a source with the
// table generated from the merge-base and candidate ingest tools (D4). Only the aliases are
// compared; the reason is free text. A generated table with no alias needs no committed table.
func CheckAliasesGenerated(source string, committed []*AliasTable, generated *AliasTable) error {
	var have []Alias
	for _, t := range committed {
		if t.Source == source {
			have = append(have, t.Alias...)
		}
	}
	sort.Slice(have, func(i, j int) bool { return have[i].OldUnitID < have[j].OldUnitID })
	a, _ := CanonicalStruct(have)
	b, _ := CanonicalStruct(generated.Alias)
	if len(have) == 0 && len(generated.Alias) == 0 {
		return nil
	}
	if !bytes.Equal(a, b) {
		return fmt.Errorf("%s: the committed aliases (%d) differ from those the merge-base and candidate ingest tools generate (%d); commit the generated table", source, len(have), len(generated.Alias))
	}
	return nil
}
