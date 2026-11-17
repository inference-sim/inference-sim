package configiq_test

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/configiq"
	"github.com/inference-sim/blis-substrate/ledger/inferencex"
)

var roleKeys = map[string]ledger.RoleKeyFunc{
	inferencex.SourceID: inferencex.RoleKey,
	configiq.SourceID:   configiq.RoleKey,
}

func committed(t *testing.T) *ledger.Ledger {
	t.Helper()
	l, err := ledger.Load("../../catalog/ledger", roleKeys)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// The committed ledger holds the configiq source: pinned, redistributable (CC-BY-4.0), with
// cut-points and no legacy side, and a unit list whose units all check.
func TestCommittedConfigiqSource(t *testing.T) {
	l := committed(t)
	src := l.Sources.Lookup(configiq.SourceID)
	if src == nil {
		t.Fatal("sources.yaml has no configiq source")
	}
	if !src.Licence.Redistributable() {
		t.Errorf("configiq licence %q is not redistributable", src.Licence.SPDX)
	}
	rule, ok := l.Config.Rules[configiq.SourceID]
	if !ok || len(rule.Legacy) != 0 {
		t.Errorf("rules.configiq: present %v, legacy sides %d; want cut-points and no legacy side", ok, len(rule.Legacy))
	}
	if len(l.Units[configiq.SourceID]) == 0 {
		t.Fatal("the configiq unit list is empty (X22)")
	}
}

// release draws the three published lists from the given units under cfg, with a test seed.
func release(t *testing.T, cfg ledger.Config, legacy *ledger.LegacyMap, units map[string][]ledger.Unit, seed string) map[ledger.Role]map[string]bool {
	t.Helper()
	raw, _ := hex.DecodeString(seed)
	sum := sha256.Sum256(raw)
	cfg.Seed = ledger.SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "test", GeneratedOn: "2026-11-16", RotationOwner: "test"}
	a, err := ledger.NewAssigner(&cfg, legacy, roleKeys, seed)
	if err != nil {
		t.Fatal(err)
	}
	lists, err := a.Release(&cfg, units)
	if err != nil {
		t.Fatal(err)
	}
	out := map[ledger.Role]map[string]bool{}
	for r, p := range lists {
		out[r] = map[string]bool{}
		for _, id := range p.Units {
			out[r][id] = true
		}
	}
	return out
}

// D7, D4: adding configiq's cut-points under the current partition version moves no
// InferenceX unit's role. The release job's fit, anchor and report lists, restricted to
// InferenceX units, are the same with and without the configiq source, for two seeds; and
// configiq's own units are drawn into the lists too.
func TestAddingConfigiqMovesNoInferenceXRole(t *testing.T) {
	l := committed(t)
	ix := l.Units[inferencex.SourceID]
	if len(ix) == 0 {
		t.Fatal("no InferenceX units (X22)")
	}
	before := *l.Config
	before.Rules = map[string]ledger.SourceRule{}
	for src, r := range l.Config.Rules {
		if src != configiq.SourceID {
			before.Rules[src] = r
		}
	}
	for _, seed := range []string{strings.Repeat("5a", 32), strings.Repeat("c3", 32)} {
		was := release(t, before, l.Legacy, map[string][]ledger.Unit{inferencex.SourceID: ix}, seed)
		now := release(t, *l.Config, l.Legacy, l.Units, seed)
		cq := 0
		for _, u := range l.Units[configiq.SourceID] {
			for _, r := range []ledger.Role{ledger.RoleFit, ledger.RoleAnchor, ledger.RoleReport} {
				if now[r][u.UnitID] {
					cq++
				}
			}
		}
		if cq == 0 {
			t.Errorf("seed %s…: no configiq unit is on a published list", seed[:4])
		}
		for _, r := range []ledger.Role{ledger.RoleFit, ledger.RoleAnchor, ledger.RoleReport} {
			moved := 0
			var first string
			for _, u := range ix {
				if was[r][u.UnitID] != now[r][u.UnitID] {
					if moved == 0 {
						first = u.UnitID
					}
					moved++
				}
			}
			if moved > 0 {
				t.Errorf("seed %s…: %d InferenceX units changed membership of the %s list, first %s", seed[:4], moved, r, first)
			}
		}
		if len(was[ledger.RoleFit]) == 0 {
			t.Fatalf("seed %s…: the InferenceX fit list is empty", seed[:4])
		}
	}
}
