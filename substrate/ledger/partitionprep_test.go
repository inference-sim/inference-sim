package ledger

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
)

// committedConfig is the ledger.yaml in the tree: partition version 1, the version the move
// to version 2 starts from. It holds no secret; its seed is a hash.
func committedConfig(t *testing.T) *Config {
	t.Helper()
	c, err := LoadConfig("../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	if c.PartitionVersion != 1 {
		t.Skipf("the tree is at partition version %d; these tests start from version 1", c.PartitionVersion)
	}
	return c
}

func prepRecords(source string, reads map[string]Role) []*Reads {
	rec := &Reads{Kind: "UnitReads", Consumer: "fit-step", Revision: "abc"}
	for _, id := range sortedKeys(reads) {
		rec.Reads = append(rec.Reads, UnitRead{Source: source, UnitID: id, Role: reads[id]})
	}
	return []*Reads{rec}
}

// Table DC "Partition version": only InferenceX has a held cut-point and only InferenceX and
// configiq an anchor cut-point, and the proposal the tool writes holds them.
func TestRoleHoldersFromVersionTwo(t *testing.T) {
	cur := committedConfig(t)
	cur.Seed = SeedRecord{SHA256: strings.Repeat("a", 64), GeneratedBy: "x", GeneratedOn: "2026-01-01", RotationOwner: "x"}
	units := []Unit{{UnitID: hexID(1)}, {UnitID: hexID(2)}, {UnitID: hexID(3)}}
	rep, err := PrepareAnchorSource("configiq", units, prepRecords("configiq", map[string]Role{hexID(2): RoleFit}), nil)
	if err != nil {
		t.Fatal(err)
	}
	next, err := ProposeNextPartition(cur, map[string]*PrepReport{"configiq": rep})
	if err != nil {
		t.Fatalf("the proposal was refused: %v", err)
	}
	if err := next.Validate(); err != nil {
		t.Fatalf("the proposal is not valid: %v", err)
	}
	for src, r := range next.Rules {
		if hasRole(r.CutPoints, RoleHeld) != contains(HeldSources, src) {
			t.Errorf("%s: held cut-point is %v, want %v", src, hasRole(r.CutPoints, RoleHeld), contains(HeldSources, src))
		}
		if hasRole(r.CutPoints, RoleAnchor) != contains(AnchorSources, src) {
			t.Errorf("%s: anchor cut-point is %v, want %v", src, hasRole(r.CutPoints, RoleAnchor), contains(AnchorSources, src))
		}
	}

	bad := map[string]func(c *Config){
		"held on AISimulate": func(c *Config) { c.Rules["aisimulate"] = cur.Rules["aisimulate"] },
		"anchor on AISimulate": func(c *Config) {
			c.Rules["aisimulate-fpm"] = SourceRule{CutPoints: CutPoints{{RoleFit, 0, 79}, {RoleAnchor, 80, 99}}}
		},
		"held on configiq": func(c *Config) { c.Rules["configiq"] = SourceRule{CutPoints: CutPoints{{RoleHeld, 0, 99}}} },
		"configiq trains": func(c *Config) {
			c.Rules["configiq"] = SourceRule{CutPoints: CutPoints{{RoleFit, 0, 49}, {RoleAnchor, 50, 99}}}
		},
		"configiq half anchor": func(c *Config) {
			c.Rules["configiq"] = SourceRule{CutPoints: CutPoints{{RoleAnchor, 0, 49}, {RoleSelect, 50, 99}}}
		},
		"InferenceX loses held": func(c *Config) {
			c.Rules["inferencex"] = SourceRule{CutPoints: CutPoints{{RoleFit, 0, 79}, {RoleAnchor, 80, 99}}}
		},
		"anchor on a calibration": func(c *Config) {
			c.Rules["vllm-ci-perf"] = SourceRule{CutPoints: CutPoints{{RoleFit, 0, 79}, {RoleAnchor, 80, 99}}}
		},
		"left out on InferenceX": func(c *Config) {
			r := c.Rules["inferencex"]
			r.LeftOut = []string{hexID(1)}
			c.Rules["inferencex"] = r
		},
		"left out unsorted": func(c *Config) {
			r := c.Rules["configiq"]
			r.LeftOut = []string{hexID(2), hexID(1)}
			c.Rules["configiq"] = r
		},
		"left out twice": func(c *Config) {
			r := c.Rules["configiq"]
			r.LeftOut = []string{hexID(1), hexID(1)}
			c.Rules["configiq"] = r
		},
		"left out is not a unit id": func(c *Config) {
			r := c.Rules["configiq"]
			r.LeftOut = []string{"abc"}
			c.Rules["configiq"] = r
		},
	}
	for name, mutate := range bad {
		c, err := ProposeNextPartition(cur, map[string]*PrepReport{"configiq": rep})
		if err != nil {
			t.Fatal(err)
		}
		mutate(c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// Version 1 states its roles as it did, and cannot leave units out.
	v1 := committedConfig(t)
	r := v1.Rules["configiq"]
	r.LeftOut = []string{hexID(1)}
	v1.Rules["configiq"] = r
	if err := v1.Validate(); err == nil {
		t.Error("left_out in partition version 1 was accepted")
	}
}

// The D2 join report: a unit read under fit, select or validate is left out; a unit read under
// held, a unit no record mentions, and units of other sources are not.
func TestPrepareAnchorSource(t *testing.T) {
	units := []Unit{{UnitID: hexID(1)}, {UnitID: hexID(2)}, {UnitID: hexID(3)}, {UnitID: hexID(4)}, {UnitID: hexID(5)}}
	recs := []*Reads{
		{Kind: "UnitReads", Consumer: "fit-step", Revision: "abc", Reads: []UnitRead{{"configiq", hexID(1), RoleFit}, {"other", hexID(9), RoleFit}}},
		{Kind: "UnitReads", Consumer: "select", Revision: "abc", Reads: []UnitRead{{"configiq", hexID(3), RoleSelect}, {"configiq", hexID(4), RoleHeld}}},
		{Kind: "UnitReads", Consumer: "error", Revision: "abc", Reads: []UnitRead{{"configiq", hexID(1), RoleValidate}, {"configiq", hexID(5), RoleValidate}}},
	}
	rep, err := PrepareAnchorSource("configiq", units, recs, []string{"fit-step", "select", "error"})
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(rep.LeftOut); got != fmt.Sprint([]string{hexID(1), hexID(3), hexID(5)}) {
		t.Errorf("left out %v, want units 1, 3 and 5", rep.LeftOut)
	}
	if rep.Units != 5 || rep.Anchor != 2 || rep.Conflicts != 1 {
		t.Errorf("units %d anchor %d conflicts %d, want 5, 2, 1 (unit 1 was read as fit and validate)", rep.Units, rep.Anchor, rep.Conflicts)
	}
	if len(rep.Consumers) != 3 || rep.Consumers[0].Consumer != "error" || rep.Consumers[1].Reads != 1 {
		t.Errorf("consumers %+v", rep.Consumers)
	}

	for name, c := range map[string]struct {
		recs []*Reads
		req  []string
		want string
	}{
		"no records":           {nil, nil, "no consumer record"},
		"a required one lacks": {recs[:1], []string{"fit-step", "error"}, `"error" is missing`},
		"unknown unit":         {[]*Reads{{Kind: "UnitReads", Consumer: "c", Revision: "abc", Reads: []UnitRead{{"configiq", hexID(77), RoleFit}}}}, nil, "does not hold"},
	} {
		if _, err := PrepareAnchorSource("configiq", units, c.recs, c.req); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}
}

// A version that keeps the seed is accepted only when the seed record is the base's whole and
// InferenceX's rule is unchanged; any other seed-kept bump is still refused.
func TestSeedKeptVersionBump(t *testing.T) {
	base := committedConfig(t)
	baseL := &Ledger{Config: base}
	rep := &PrepReport{LeftOut: []string{}}
	next, err := ProposeNextPartition(base, map[string]*PrepReport{"configiq": rep})
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckAgainstBase(baseL, &Ledger{Config: next}); err != nil {
		t.Fatalf("the seed-kept version the proposal states was refused: %v", err)
	}
	for name, mutate := range map[string]func(c *Config){
		"InferenceX cut-point moved": func(c *Config) {
			r := c.Rules["inferencex"]
			r.CutPoints = append(CutPoints(nil), r.CutPoints...)
			r.CutPoints[3].To, r.CutPoints[4].From = 74, 75
			c.Rules["inferencex"] = r
		},
		"InferenceX legacy side moved": func(c *Config) {
			r := c.Rules["inferencex"]
			r.Legacy = map[Side]CutPoints{SideFit: r.Legacy[SideFit], SideHeld: {{RoleHeld, 0, 49}, {RoleAnchor, 50, 99}}}
			c.Rules["inferencex"] = r
		},
		"InferenceX rule dropped": func(c *Config) { delete(c.Rules, "inferencex") },
		"seed generator changed":  func(c *Config) { c.Seed.GeneratedBy = "someone else" },
		"two versions ahead":      func(c *Config) { c.PartitionVersion = 3 },
	} {
		c, err := ProposeNextPartition(base, map[string]*PrepReport{"configiq": rep})
		if err != nil {
			t.Fatal(err)
		}
		mutate(c)
		if err := CheckAgainstBase(baseL, &Ledger{Config: c}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The done-when of row 26, on synthetic keys: under one seed, no InferenceX unit changes role
// from version 1 to the proposal; a left-out configiq unit has no role and every other
// configiq unit is an anchor; AISimulate draws only training roles. Nothing here is a real
// seed: the secret is a repeated byte pattern.
func TestProposalMovesNoInferenceXUnit(t *testing.T) {
	seed := strings.Repeat("5a", 32)
	raw, _ := hex.DecodeString(seed)
	sum := sha256.Sum256(raw)
	cur := committedConfig(t)
	cur.Seed.SHA256 = hex.EncodeToString(sum[:])

	srcKeys := func(id map[string]any) (map[string]any, error) { return map[string]any{"k": id["k"]}, nil }
	keys := map[string]RoleKeyFunc{}
	for src := range cur.Rules {
		keys[src] = srcKeys
	}
	legacy := &LegacyMap{Source: "inferencex", Units: map[string]Side{}, RoleKeySides: map[string]Side{}}
	units := map[string][]Unit{}
	for _, src := range []string{"inferencex", "configiq", "aisimulate"} {
		for i := 0; i < 600; i++ {
			u := testUnit(i, i/3)
			units[src] = append(units[src], u)
			if src == "inferencex" && i%4 == 0 {
				side := SideFit
				if i%8 == 0 {
					side = SideHeld
				}
				legacy.Units[u.UnitID] = side
				legacy.RoleKeySides[u.RoleKey] = side
			}
		}
	}
	trained := map[string]Role{}
	for i, u := range units["configiq"] {
		if i%5 == 0 {
			trained[u.UnitID] = RoleFit
		}
	}
	rep, err := PrepareAnchorSource("configiq", units["configiq"], prepRecords("configiq", trained), nil)
	if err != nil {
		t.Fatal(err)
	}
	next, err := ProposeNextPartition(cur, map[string]*PrepReport{"configiq": rep})
	if err != nil {
		t.Fatal(err)
	}
	a1, err := NewAssigner(cur, legacy, keys, seed)
	if err != nil {
		t.Fatal(err)
	}
	a2, err := NewAssigner(next, legacy, keys, seed)
	if err != nil {
		t.Fatal(err)
	}
	roles := func(a *Assigner, src string, i int) Role {
		r, err := a.Role(src, &units[src][i])
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	seen := map[Role]int{}
	for i := range units["inferencex"] {
		before, after := roles(a1, "inferencex", i), roles(a2, "inferencex", i)
		if before != after {
			t.Fatalf("InferenceX unit %d moved from %s to %s", i, before, after)
		}
		seen[after]++
	}
	if seen[RoleHeld] == 0 || seen[RoleAnchor] == 0 {
		t.Errorf("the synthetic InferenceX units hold no held or anchor unit (%v); the check above proves nothing", seen)
	}
	left := map[string]bool{}
	for _, id := range rep.LeftOut {
		left[id] = true
	}
	for i, u := range units["configiq"] {
		got, want := roles(a2, "configiq", i), RoleAnchor
		if left[u.UnitID] {
			want = RoleNone
		}
		if got != want {
			t.Fatalf("configiq unit %d is %q, want %q", i, got, want)
		}
	}
	for i := range units["aisimulate"] {
		if r := roles(a2, "aisimulate", i); !contains(trainingRoles, r) {
			t.Fatalf("AISimulate unit %d is %q, want a training role", i, r)
		}
	}
	// The release carries no left-out configiq unit on any list. AISimulate units with ids
	// apart from configiq's give it the fit list a release requires.
	var distinctUnits []Unit
	for i := 0; i < 100; i++ {
		distinctUnits = append(distinctUnits, testUnit(5000+i, i))
	}
	lists, err := a2.Release(next, map[string][]Unit{"configiq": units["configiq"], "aisimulate": distinctUnits})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range lists {
		for _, id := range p.Units {
			if left[id] {
				t.Errorf("left-out unit %s is on the %s list", id, p.Role)
			}
		}
	}
}

// A unit with no role is held to the one-role check alone: reading it as fit and as validate is
// still a conflict, and reading it as fit alone is not.
func TestJoinOfAUnitWithNoRole(t *testing.T) {
	recs := []*Reads{
		{Kind: "UnitReads", Consumer: "a", Revision: "abc", Reads: []UnitRead{{"s", hexID(1), RoleFit}, {"s", hexID(2), RoleFit}}},
		{Kind: "UnitReads", Consumer: "b", Revision: "abc", Reads: []UnitRead{{"s", hexID(2), RoleValidate}}},
	}
	c, err := Join(recs, func(string, string) (Role, error) { return RoleNone, nil })
	if err != nil || len(c) != 1 || c[0].UnitID != hexID(2) {
		t.Fatalf("conflicts %v, err %v; want only unit 2", c, err)
	}
}
