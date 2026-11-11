package ledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The canonical encoding equals Python's json.dumps(sort_keys=True, separators=(",", ":"),
// ensure_ascii=False), which accuracy-dashboard's contract.py uses. The expected strings were
// printed by Python 3.13 for these inputs.
func TestCanonicalMatchesPython(t *testing.T) {
	cases := []struct {
		in   any
		want string
		hash string
	}{
		{map[string]any{"s": "é\"\\\n\t\x01\x7f /<>&", "b": true, "n": nil, "i": -5,
			"l": []any{1, map[string]any{"z": 0, "a": "x"}}},
			"{\"b\":true,\"i\":-5,\"l\":[1,{\"a\":\"x\",\"z\":0}],\"n\":null,\"s\":\"é\\\"\\\\\\n\\t\\u0001\x7f /<>&\"}",
			"3f2ea68321a3d97d6d31d20b35c8af35d187531ae9ecdab4789d1c7863814c91"},
		// The dashboard's point identifier of an eight-field identity.
		{map[string]any{"model": "dsr1", "hardware": "b200", "precision": "fp4", "framework": "vllm",
			"benchmark_type": "single_turn", "recipe": "row:1", "conc": 4, "measured_date": "2026-01-01"},
			"", "50df96d1427ab98acac86fc620ce8e0f770f99a81264530b45ab1167d1b88d4b"},
	}
	for _, c := range cases {
		got, err := Canonical(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if c.want != "" && string(got) != c.want {
			t.Errorf("Canonical = %s\nwant        %s", got, c.want)
		}
		id, _ := ID(c.in)
		if id != c.hash {
			t.Errorf("ID = %s, want %s", id, c.hash)
		}
	}
}

// Known-bad inputs: a float, a non-UTF-8 string and an unsupported type are refused.
func TestCanonicalRefusesFloatsAndBadStrings(t *testing.T) {
	for _, v := range []any{
		map[string]any{"conc": 4.0},
		map[string]any{"x": 0.5},
		map[string]any{"x": json.Number("1.5")},
		map[string]any{"x": "\xff"},
		map[string]any{"x": struct{}{}},
	} {
		if _, err := Canonical(v); err == nil {
			t.Errorf("Canonical(%#v) accepted", v)
		}
	}
}

// MeasuredID keeps each number's text, so the same literal hashes equal and a re-formatted one
// (1e-7 against 0.0000001) does not.
func TestMeasuredIDKeepsNumberText(t *testing.T) {
	a, _ := MeasuredID(map[string]any{"x": json.Number("0.0000001"), "y": json.Number("2")})
	b, _ := MeasuredID(map[string]any{"y": json.Number("2"), "x": json.Number("0.0000001")})
	c, _ := MeasuredID(map[string]any{"x": json.Number("1e-7"), "y": json.Number("2")})
	if a != b {
		t.Error("key order changed the hash")
	}
	if a == c {
		t.Error("different literals hashed equal")
	}
	if _, err := MeasuredID(map[string]any{"x": json.Number("1.")}); err == nil {
		t.Error("a malformed number was accepted")
	}
}

// The HMAC draw equals HMAC-SHA-256(seed, canonical JSON of the role key) mod 100 as Python
// computes it: int.from_bytes(hmac.new(seed, msg, sha256).digest(), "big") % 100.
func TestDrawMatchesHMACReference(t *testing.T) {
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}
	sum := sha256.Sum256(seed)
	s, err := ParseSeed(hex.EncodeToString(seed), hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatal(err)
	}
	for msg, want := range map[string]int{`{"k":"a"}`: 21, `{"conc":4,"model":"m","source":"inferencex"}`: 55} {
		if got := s.Draw([]byte(msg)); got != want {
			t.Errorf("Draw(%s) = %d, want %d", msg, got, want)
		}
	}
}

// Known-bad seeds: wrong length, not hex, wrong hash.
func TestParseSeedRefuses(t *testing.T) {
	seed := strings.Repeat("ab", 32)
	raw, _ := hex.DecodeString(seed)
	sum := sha256.Sum256(raw)
	good := hex.EncodeToString(sum[:])
	if _, err := ParseSeed(seed, good); err != nil {
		t.Fatalf("a good seed was refused: %v", err)
	}
	for name, c := range map[string][2]string{
		"short":      {strings.Repeat("ab", 16), good},
		"not hex":    {strings.Repeat("zz", 32), good},
		"wrong hash": {seed, strings.Repeat("0", 64)},
	} {
		if _, err := ParseSeed(c[0], c[1]); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func testConfig() *Config {
	return &Config{Kind: "Ledger", PartitionVersion: 1, Rules: map[string]SourceRule{
		"s": {
			CutPoints: CutPoints{{RoleFit, 0, 29}, {RoleSelect, 30, 39}, {RoleValidate, 40, 49}, {RoleHeld, 50, 79}, {RoleAnchor, 80, 99}},
			Legacy: map[Side]CutPoints{
				SideFit:  {{RoleFit, 0, 74}, {RoleSelect, 75, 99}},
				SideHeld: {{RoleHeld, 0, 59}, {RoleAnchor, 60, 99}},
			},
		},
	}}
}

// Known-bad cut-points: a gap, an overlap, a short range, an unknown role, and a legacy side
// that would give a role outside its side (FIT giving held).
func TestCutPointsRefused(t *testing.T) {
	if err := testConfig().Validate(); err != nil {
		t.Fatalf("the good config was refused: %v", err)
	}
	bad := map[string]func(c *Config){
		"gap":      func(c *Config) { r := c.Rules["s"]; r.CutPoints[1].From = 31; c.Rules["s"] = r },
		"overlap":  func(c *Config) { r := c.Rules["s"]; r.CutPoints[1].From = 29; c.Rules["s"] = r },
		"short":    func(c *Config) { r := c.Rules["s"]; r.CutPoints[4].To = 98; c.Rules["s"] = r },
		"unknown":  func(c *Config) { r := c.Rules["s"]; r.CutPoints[0].Role = "train"; c.Rules["s"] = r },
		"FIT→held": func(c *Config) { c.Rules["s"].Legacy[SideFit][1] = CutPoint{RoleHeld, 75, 99} },
		"HELD→fit": func(c *Config) { c.Rules["s"].Legacy[SideHeld][0] = CutPoint{RoleFit, 0, 59} },
		"version":  func(c *Config) { c.PartitionVersion = 0 },
		"seed who": func(c *Config) { c.Seed.SHA256 = strings.Repeat("a", 64) },
		"three on a side": func(c *Config) {
			c.Rules["s"].Legacy[SideFit] = CutPoints{{RoleFit, 0, 49}, {RoleSelect, 50, 79}, {RoleFit, 80, 99}}
		},
	}
	for name, mutate := range bad {
		c := testConfig()
		mutate(c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func assignerFor(t *testing.T, legacy *LegacyMap) *Assigner {
	t.Helper()
	seed := strings.Repeat("5a", 32)
	raw, _ := hex.DecodeString(seed)
	sum := sha256.Sum256(raw)
	c := testConfig()
	c.Seed = SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "test", GeneratedOn: "2026-11-04", RotationOwner: "test"}
	a, err := NewAssigner(c, legacy, testRoleKeys, seed)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// testRoleKeys is the role-key rule of the test source "s": the identity's "k".
var testRoleKeys = map[string]RoleKeyFunc{"s": func(id map[string]any) (map[string]any, error) {
	return map[string]any{"k": id["k"]}, nil
}}

// testUnit returns a unit of source "s" with identity {"n": n, "k": k}.
func testUnit(n, k int) Unit {
	id := map[string]any{"n": n, "k": k}
	uid, _ := ID(id)
	rk, _ := ID(map[string]any{"k": k})
	return Unit{UnitID: uid, RoleKey: rk, Identity: id}
}

func hexID(n int) string {
	return strings.Repeat("0", 60) + hex.EncodeToString([]byte{byte(n >> 8), byte(n)})
}

// D4: every unit of one role key gets one role, and a legacy unit stays on its side over many
// role keys: FIT gives only fit or select, HELD only held or anchor. A unit outside the table
// whose role key holds legacy units takes their side.
func TestRolesPerRoleKeyAndLegacySide(t *testing.T) {
	legacy := &LegacyMap{Source: "s", Units: map[string]Side{}, RoleKeySides: map[string]Side{}}
	for i := 0; i < 400; i++ {
		side := SideFit
		if i%2 == 1 {
			side = SideHeld
		}
		u := testUnit(i, 5000+i)
		legacy.Units[u.UnitID] = side
		legacy.RoleKeySides[u.RoleKey] = side
	}
	a := assignerFor(t, legacy)
	seen := map[Role]bool{}
	for i := 0; i < 400; i++ {
		u := testUnit(i, 5000+i)
		r, err := a.Role("s", &u)
		if err != nil {
			t.Fatal(err)
		}
		seen[r] = true
		want := legacy.Units[u.UnitID].roles()
		if !contains(want, r) {
			t.Fatalf("legacy %s unit got %s", legacy.Units[u.UnitID], r)
		}
		// A new unit of the same role key takes the same side, and so the same role.
		v := testUnit(1000+i, 5000+i)
		if r2, _ := a.Role("s", &v); r2 != r {
			t.Fatalf("two units of one role key: %s and %s", r, r2)
		}
	}
	for _, r := range []Role{RoleFit, RoleSelect, RoleHeld, RoleAnchor} {
		if !seen[r] {
			t.Errorf("no legacy unit got %s over 400 draws", r)
		}
	}
	// A role-key field that disagrees with the unit's role_key is refused, not drawn.
	bad := testUnit(1, 5001)
	bad.RoleKey = hexID(1)
	if _, err := a.Role("s", &bad); err == nil {
		t.Error("a unit whose role_key is not its role key's hash was drawn")
	}
	// Without the legacy table the same units draw from the full cut-points, validate included.
	b := assignerFor(t, nil)
	any := map[Role]bool{}
	for i := 0; i < 400; i++ {
		u := testUnit(i, 5000+i)
		r, _ := b.Role("s", &u)
		any[r] = true
	}
	if !any[RoleValidate] {
		t.Error("new units never drew validate")
	}
}

// The release job publishes only fit, anchor and report.
func TestReleasePublishesOnlyThreeLists(t *testing.T) {
	a := assignerFor(t, nil)
	var units []Unit
	byID := map[string]Unit{}
	for i := 0; i < 200; i++ {
		u := testUnit(i, i)
		units = append(units, u)
		byID[u.UnitID] = u
	}
	lists, err := a.Release(testConfig(), map[string][]Unit{"s": units})
	if err != nil {
		t.Fatal(err)
	}
	if len(lists) != 3 || lists[RoleFit] == nil || lists[RoleAnchor] == nil || lists[RoleReport] == nil {
		t.Fatalf("published %v", lists)
	}
	for r, p := range lists {
		for _, id := range p.Units {
			u := byID[id]
			if got, _ := a.Role("s", &u); got != r {
				t.Errorf("%s list holds a %s unit", r, got)
			}
		}
	}
}

// No role can be drawn before a ledger owner records the seed.
func TestNoSeedNoRoles(t *testing.T) {
	if _, err := NewAssigner(testConfig(), nil, testRoleKeys, strings.Repeat("ab", 32)); err == nil {
		t.Error("an assigner was made with no recorded seed")
	}
}

// D3: "latest", "main" and "unknown" are refused, in any case, as are empty revisions and
// branch URLs in licence evidence.
func TestUnpinnedRevisionsRefused(t *testing.T) {
	for _, ref := range []string{"latest", "main", "unknown", "MAIN", " Latest ", ""} {
		if err := CheckPinned("ref", ref); err == nil {
			t.Errorf("%q accepted", ref)
		}
	}
	if err := CheckPinned("ref", "2f0f4cd8606d580567b253d02f3d699fe36db865"); err != nil {
		t.Errorf("a commit was refused: %v", err)
	}
	src := goodSources()
	if err := src.Validate(); err != nil {
		t.Fatalf("good sources refused: %v", err)
	}
	for name, mutate := range map[string]func(s *Sources){
		"ref main":       func(s *Sources) { s.Sources[0].Revision.Ref = "main" },
		"short commit":   func(s *Sources) { s.Sources[0].Revision.Ref = "2f0f4cd" },
		"branch licence": func(s *Sources) { s.Sources[0].Licence.Evidence[0].URL = "https://github.com/o/r/blob/main/LICENSE" },
		"no spdx":        func(s *Sources) { s.Sources[0].Licence.SPDX = "" },
		"no evidence":    func(s *Sources) { s.Sources[0].Licence.Evidence = nil },
		"half sign-off":  func(s *Sources) { s.Sources[0].Licence.SignOff.By = "someone" },
		"release no sum": func(s *Sources) {
			s.Sources[0].Revision = Revision{Kind: "release_assets", URL: "https://x/y", Ref: "tag", Commit: strings.Repeat("a", 40)}
		},
	} {
		s := goodSources()
		mutate(s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func goodSources() *Sources {
	return &Sources{Kind: "LedgerSources", Sources: []Source{{
		ID: "s", Title: "t", Status: "no units yet",
		Revision: Revision{Kind: "git_commit", URL: "https://github.com/o/r/tree/2f0f4cd8606d580567b253d02f3d699fe36db865", Ref: "2f0f4cd8606d580567b253d02f3d699fe36db865"},
		Licence: Licence{SPDX: "Apache-2.0", Evidence: []LicenceQuote{{
			URL: "https://github.com/o/r/blob/2f0f4cd8606d580567b253d02f3d699fe36db865/LICENSE", SHA256: strings.Repeat("a", 64), Lines: "1", Quote: "Apache"}}},
	}}}
}

// X12, D6: a source is redistributable only with an allow-listed licence, no restriction and
// a ledger owner's sign-off.
func TestRedistributable(t *testing.T) {
	l := goodSources().Sources[0].Licence
	if l.Redistributable() {
		t.Error("redistributable without a sign-off")
	}
	l.SignOff = SignOff{By: "owner", On: "2026-11-04"}
	if !l.Redistributable() {
		t.Error("an allow-listed, signed-off licence is not redistributable")
	}
	r := l
	r.Restrictions = []LicenceQuote{l.Evidence[0]}
	if r.Redistributable() {
		t.Error("redistributable despite a stated restriction")
	}
	g := l
	g.SPDX = "GPL-3.0"
	if g.Redistributable() {
		t.Error("GPL-3.0 is not on the allow-list")
	}
}

// D4: the layout check refuses a file the ledger does not define, such as a role list under
// another name, and the role scan refuses a line that pairs an identifier with a role.
func TestLayoutAndRoleScan(t *testing.T) {
	dir := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("catalog/ledger/ledger.yaml", "kind: Ledger\n")
	write("catalog/ledger/published/fit.yaml", "units:\n  - "+hexID(1)+"\n")
	if err := CheckLayout(filepath.Join(dir, "catalog/ledger")); err != nil {
		t.Fatalf("a good layout was refused: %v", err)
	}
	write("catalog/ledger/published/held.yaml", "units: []\n")
	if err := CheckLayout(filepath.Join(dir, "catalog/ledger")); err == nil {
		t.Error("published/held.yaml was accepted")
	}

	good := []string{"catalog/ledger/published/fit.yaml", "notes.yaml"}
	write("notes.yaml", "cite: \"the report (not redistributed), sha256 f250d35ee0fb\"\nrole: primary\n")
	if err := CheckNoRoles(dir, good, nil); err != nil {
		t.Errorf("prose and the published list were refused: %v", err)
	}
	for name, body := range map[string]string{
		"yaml":  hexID(7) + ": held\n",
		"json":  `{"unit_id":"` + hexID(7) + `","role":"select"}` + "\n",
		"csv":   hexID(7) + ",validate\n",
		"list":  "held:\n  - " + hexID(7) + "\n",
		"label": "- {unit: " + hexID(7) + ", side: HELD}\n",
	} {
		write("bad.yaml", body)
		if err := CheckNoRoles(dir, []string{"bad.yaml"}, nil); err == nil {
			t.Errorf("%s: a role list was accepted: %q", name, body)
		}
	}
}

// D2: the join fails on a unit read under two roles, and, given the assignment, on a unit read
// under a role the seed does not give it.
func TestJoin(t *testing.T) {
	recs := []*Reads{
		{Kind: "UnitReads", Consumer: "fit-attention", Revision: "abc", Reads: []UnitRead{{"s", hexID(1), RoleFit}, {"s", hexID(2), RoleFit}}},
		{Kind: "UnitReads", Consumer: "gate", Revision: "abc", Reads: []UnitRead{{"s", hexID(3), RoleHeld}}},
	}
	c, err := Join(recs, nil)
	if err != nil || len(c) != 0 {
		t.Fatalf("clean records: %v %v", c, err)
	}
	recs = append(recs, &Reads{Kind: "UnitReads", Consumer: "select", Revision: "abc", Reads: []UnitRead{{"s", hexID(1), RoleSelect}}})
	c, _ = Join(recs, nil)
	if len(c) != 1 || c[0].UnitID != hexID(1) {
		t.Fatalf("a unit read as fit and select was not caught: %v", c)
	}
	// One consumer reads unit 2 as fit, but the seed gives it held.
	c, _ = Join(recs[:2], func(src, id string) (Role, error) {
		if id == hexID(2) {
			return RoleHeld, nil
		}
		if id == hexID(3) {
			return RoleHeld, nil
		}
		return RoleFit, nil
	})
	if len(c) != 1 || c[0].UnitID != hexID(2) {
		t.Fatalf("a held unit read as fit was not caught: %v", c)
	}
}

// Known-bad reads records: unknown field, trailing data, unpinned revision, invalid role.
func TestDecodeReadsStrict(t *testing.T) {
	good := `{"kind":"UnitReads","consumer":"c","revision":"abc","reads":[{"source":"s","unit_id":"` + hexID(1) + `","role":"fit"}]}`
	if _, err := DecodeReads(strings.NewReader(good)); err != nil {
		t.Fatalf("good record refused: %v", err)
	}
	for name, body := range map[string]string{
		"unknown":  strings.Replace(good, `"kind"`, `"extra":1,"kind"`, 1),
		"trailing": good + " {}",
		"latest":   strings.Replace(good, `"abc"`, `"latest"`, 1),
		"role":     strings.Replace(good, `"fit"`, `"train"`, 1),
	} {
		if _, err := DecodeReads(strings.NewReader(body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// D7: the predictions loader refuses what a candidate could use to smuggle data or overflow
// the job, and the report check refuses per-unit rows and aggregates under 10 units.
func TestPredictionsAndReports(t *testing.T) {
	want := map[string]bool{"s/" + hexID(1): true, "s/" + hexID(2): true}
	good := `{"kind":"Predictions","candidate":"` + strings.Repeat("a", 40) + `","predictions":[{"source":"s","unit_id":"` + hexID(1) + `","values":{"ttft_ms":1.5}},{"source":"s","unit_id":"` + hexID(2) + `","reason":"refused"}]}`
	if _, err := ReadPredictions(strings.NewReader(good), want, strings.Repeat("a", 40)); err != nil {
		t.Fatalf("good predictions refused: %v", err)
	}
	for name, body := range map[string]string{
		"unknown field": strings.Replace(good, `"kind"`, `"note":"x","kind"`, 1),
		"unasked unit":  strings.Replace(good, hexID(2), hexID(9), 1),
		"twice":         strings.Replace(good, hexID(2), hexID(1), 1),
		"bad name":      strings.Replace(good, "ttft_ms", "TTFT ms", 1),
		"short commit":  strings.Replace(good, strings.Repeat("a", 40), "abc", 1),
		"trailing":      good + "{}",
		"omitted unit":  strings.Replace(good, `,{"source":"s","unit_id":"`+hexID(2)+`","reason":"refused"}`, "", 1),
		"none at all":   `{"kind":"Predictions","candidate":"` + strings.Repeat("a", 40) + `","predictions":[]}`,
		"too large":     good + strings.Repeat(" ", MaxPredictionsBytes),
	} {
		if _, err := ReadPredictions(strings.NewReader(body), want, strings.Repeat("a", 40)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}

	// X22/D7: the artifact's candidate must be the commit of the run it came from.
	if _, err := ReadPredictions(strings.NewReader(good), want, strings.Repeat("b", 40)); err == nil {
		t.Error("predictions for another commit accepted")
	}
	if _, err := ReadPredictions(strings.NewReader(good), want, ""); err == nil {
		t.Error("no expected commit accepted")
	}

	ids := []string{hexID(1), hexID(2)}
	ok := `{"kind":"AggregateReport","job":"gate","candidate":"x","aggregates":[{"lane":"h200","metric":"ttft_mape","units":10,"value":3.5}]}`
	if err := CheckReport([]byte(ok), ids); err != nil {
		t.Fatalf("good report refused: %v", err)
	}
	for name, body := range map[string]string{
		"nine units": strings.Replace(ok, `"units":10`, `"units":9`, 1),
		"unit named": strings.Replace(ok, `"lane":"h200"`, `"lane":"`+hexID(2)+`"`, 1),
		"prefix":     strings.Replace(ok, `"lane":"h200"`, `"lane":"`+hexID(2)[:16]+`"`, 1),
		"per unit":   strings.Replace(ok, `"aggregates"`, `"rows":[],"aggregates"`, 1),
	} {
		if err := CheckReport([]byte(body), ids); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// Units: the unit list round-trips byte for byte, and a line with a role, a wrong unit_id or
// a measurement in two units is refused.
func TestUnitList(t *testing.T) {
	rk := func(id map[string]any) (map[string]any, error) { return map[string]any{"k": id["k"]}, nil }
	mk := func(k string, conc int, measured string) Unit {
		id := map[string]any{"k": k, "conc": conc}
		uid, _ := ID(id)
		r, _ := rk(id)
		rkid, _ := ID(r)
		return Unit{UnitID: uid, RoleKey: rkid, Identity: id, Measurements: []Measurement{{Record: "1", Measured: measured,
			Duplicates: []string{}, Merged: []MergedRecord{}, Recorded: map[string]any{}}}}
	}
	units := []Unit{mk("a", 1, strings.Repeat("1", 64)), mk("a", 2, strings.Repeat("2", 64))}
	for i := range units {
		if err := units[i].Check(rk); err != nil {
			t.Fatal(err)
		}
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "u.jsonl.gz")
	var b bytes.Buffer
	if err := EncodeUnits(&b, units); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	got, raw, err := ReadUnitList(p)
	if err != nil || len(got) != 2 {
		t.Fatalf("round trip: %v", err)
	}
	again, _ := UnitListBytes(got)
	if !bytes.Equal(raw, again) {
		t.Error("the list did not round-trip byte for byte")
	}

	withRole := bytes.Replace(raw, []byte(`"measurements"`), []byte(`"held":true,"measurements"`), 1)
	var gz bytes.Buffer
	if err := Gzip(&gz, withRole); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, gz.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadUnitList(p); err == nil {
		t.Error("a unit line with a role was accepted")
	}

	bad := units[0]
	bad.UnitID = strings.Repeat("f", 64)
	if err := bad.Check(rk); err == nil {
		t.Error("a unit_id that is not its identity's hash was accepted")
	}
	bad = units[0]
	bad.RoleKey = strings.Repeat("f", 64)
	if err := bad.Check(rk); err == nil {
		t.Error("a role key that is not its identity's was accepted")
	}
	dup := []Unit{units[0], mk("b", 1, strings.Repeat("1", 64))}
	if err := CheckMeasurementsUnique(dup); err == nil {
		t.Error("one measurement in two units was accepted")
	}
}

// D4 against the merge base: a unit that disappears without an alias is refused, an alias
// with the right role keys is accepted, and a change of cut-points is refused.
func TestCheckAgainstBase(t *testing.T) {
	u := func(id, rk string) Unit { return Unit{UnitID: id, RoleKey: rk} }
	base := &Ledger{Config: testConfig(), Units: map[string][]Unit{"s": {u(hexID(1), hexID(100)), u(hexID(2), hexID(100))}}}
	same := &Ledger{Config: testConfig(), Units: map[string][]Unit{"s": {u(hexID(1), hexID(100)), u(hexID(2), hexID(100)), u(hexID(3), hexID(101))}}}
	if err := CheckAgainstBase(base, same); err != nil {
		t.Fatalf("an appended unit was refused: %v", err)
	}
	gone := &Ledger{Config: testConfig(), Units: map[string][]Unit{"s": {u(hexID(1), hexID(100)), u(hexID(9), hexID(100))}}}
	if err := CheckAgainstBase(base, gone); err == nil {
		t.Error("a unit that vanished was accepted")
	}
	gone.Aliases = []*AliasTable{{Kind: "AliasTable", Source: "s", Reason: "r", Alias: []Alias{{hexID(2), hexID(9), hexID(100), hexID(100)}}}}
	if err := CheckAgainstBase(base, gone); err != nil {
		t.Errorf("an aliased unit was refused: %v", err)
	}
	gone.Aliases[0].Alias[0].NewRoleKey = hexID(555)
	if err := CheckAgainstBase(base, gone); err == nil {
		t.Error("an alias naming the wrong role key was accepted")
	}
	rekeyed := &Ledger{Config: testConfig(), Units: map[string][]Unit{"s": {u(hexID(1), hexID(100)), u(hexID(2), hexID(102))}}}
	if err := CheckAgainstBase(base, rekeyed); err == nil {
		t.Error("a role key changed without an alias was accepted")
	}
	moved := &Ledger{Config: testConfig(), Units: same.Units}
	moved.Config.Rules["s"].CutPoints[0].To = 30
	moved.Config.Rules["s"].CutPoints[1].From = 31
	if err := CheckAgainstBase(base, moved); err == nil {
		t.Error("changed cut-points were accepted")
	}
}

// D4, D7: a new partition version is a seed rotation the pull request states, and nobody
// approves it there (no check waits on a person). It is the next number, with a new seed
// record, and may carry the cut-points of the new version. Before this change every bump was
// refused, and the refusal said a ledger owner overrides it through a review question that no
// longer exists, so a rotation could not merge without an administrator's bypass.
func TestANewPartitionVersionIsASeedRotationTheCheckAccepts(t *testing.T) {
	u := func(id, rk string) Unit { return Unit{UnitID: id, RoleKey: rk} }
	units := map[string][]Unit{"s": {u(hexID(1), hexID(100)), u(hexID(2), hexID(100))}}
	seed := func(c *Config, tag string, version int) *Config {
		sum := sha256.Sum256([]byte(tag))
		c.PartitionVersion = version
		c.Seed = SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "owner", GeneratedOn: "2026-11-04", RotationOwner: "owner"}
		return c
	}
	base := &Ledger{Config: seed(testConfig(), "seed one", 1), Units: units}
	ledgerOf := func(c *Config) *Ledger { return &Ledger{Config: c, Units: units} }

	rotated := seed(testConfig(), "seed two", 2)
	rotated.Rules["s"].CutPoints[0].To = 24 // the new version states its own cut-points
	rotated.Rules["s"].CutPoints[1].From = 25
	if err := CheckAgainstBase(base, ledgerOf(rotated)); err != nil {
		t.Errorf("the next partition version with a new seed and its cut-points was refused: %v", err)
	}
	// The first seed ever recorded needs no version change: nothing was drawn before it.
	if err := CheckAgainstBase(&Ledger{Config: testConfig(), Units: units}, ledgerOf(seed(testConfig(), "first", 1))); err != nil {
		t.Errorf("recording the first seed was refused: %v", err)
	}

	for name, c := range map[string]*Config{
		"version two ahead":               seed(testConfig(), "seed two", 3),
		"version back":                    seed(testConfig(), "seed zero", 0),
		"new version with the same seed":  seed(testConfig(), "seed one", 2),
		"new version with no seed record": func() *Config { c := testConfig(); c.PartitionVersion = 2; return c }(),
		"new seed with no new version":    seed(testConfig(), "seed two", 1),
	} {
		if err := CheckAgainstBase(base, ledgerOf(c)); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	// Cut-points still change only with the new version.
	moved := seed(testConfig(), "seed one", 1)
	moved.Rules["s"].CutPoints[0].To = 24
	moved.Rules["s"].CutPoints[1].From = 25
	if err := CheckAgainstBase(base, ledgerOf(moved)); err == nil {
		t.Error("changed cut-points with no new partition version were accepted")
	}
}

// The legacy map: a unit whose records carry both labels takes FIT, and a role key holding
// both FIT and HELD units gives FIT to its new units.
func TestBuildLegacy(t *testing.T) {
	units := []Unit{{UnitID: hexID(1), RoleKey: hexID(100)}, {UnitID: hexID(2), RoleKey: hexID(100)}, {UnitID: hexID(3), RoleKey: hexID(100)}, {UnitID: hexID(4), RoleKey: hexID(101)}}
	labels := map[string]Side{"p1": SideHeld, "p2": SideFit, "p3": SideHeld, "p4": SideHeld, "p5": SideFit}
	pu := map[string][]string{"p1": {hexID(1)}, "p2": {hexID(1)}, "p3": {hexID(2)}, "p4": {}}
	m, pts := BuildLegacy("s", LegacyPartition{}, labels, pu, units, "f")
	if m.Units[hexID(1)] != SideFit || len(m.BothLabels) != 1 || m.BothLabels[0] != hexID(1) {
		t.Errorf("a unit with both labels: %v %v", m.Units, m.BothLabels)
	}
	if m.Units[hexID(2)] != SideHeld {
		t.Errorf("a HELD unit: %v", m.Units[hexID(2)])
	}
	if m.RoleKeySides[hexID(100)] != SideFit {
		t.Errorf("a role key holding FIT and HELD gives %v, want FIT", m.RoleKeySides[hexID(100)])
	}
	if len(m.NewUnitsTakingSide) != 1 || m.NewUnitsTakingSide[0] != hexID(3) {
		t.Errorf("new units taking a side: %v", m.NewUnitsTakingSide)
	}
	if m.Points != (LegacyCounts{Total: 5, Mapped: 3, NotInSource: 1, AllLeftOut: 1}) {
		t.Errorf("counts %+v", m.Points)
	}
	if len(pts) != 5 {
		t.Errorf("%d points", len(pts))
	}
}

// The reduction reads only the units of its own role, so predictions on other units change
// nothing, and a lane under 10 units is merged into "other" or dropped.
func TestAggregatePredictions(t *testing.T) {
	p := &Predictions{Kind: "Predictions", Candidate: strings.Repeat("a", 40)}
	m := &Measured{Values: map[string]map[string]float64{}, Lanes: map[string]string{}}
	roles := map[string]Role{}
	for i := 0; i < 30; i++ {
		id := hexID(i)
		k := "s/" + id
		m.Values[k] = map[string]float64{"ttft": 100}
		m.Lanes[k] = "h200"
		if i >= 15 {
			m.Lanes[k] = "b200"
		}
		roles[id] = RoleHeld
		if i%3 == 0 {
			roles[id] = RoleFit
		}
		p.Predictions = append(p.Predictions, Prediction{Source: "s", UnitID: id, Values: map[string]float64{"ttft": 110}})
	}
	roleOf := func(_, id string) (Role, error) { return roles[id], nil }
	rep, err := AggregatePredictions("gate", "held", p, m, roleOf, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Evaluated || len(rep.Aggregates) != 2 {
		t.Fatalf("aggregates %+v", rep.Aggregates)
	}
	for _, a := range rep.Aggregates {
		if a.Units != 10 || a.Value != 10 {
			t.Errorf("%+v, want 10 units at 10%%", a)
		}
	}
	// Corrupting every fit prediction moves no held aggregate.
	for i := range p.Predictions {
		if roles[p.Predictions[i].UnitID] == RoleFit {
			p.Predictions[i].Values["ttft"] = 1e9
		}
	}
	again, _ := AggregatePredictions("gate", "held", p, m, roleOf, nil)
	if fmt.Sprint(again.Aggregates) != fmt.Sprint(rep.Aggregates) {
		t.Error("predictions on fit units moved a held aggregate")
	}
	// Too few units in all: nothing reported, and the report says it decides nothing.
	fewHeld := func(_, id string) (Role, error) {
		if id == hexID(1) || id == hexID(2) || id == hexID(4) {
			return RoleHeld, nil
		}
		return RoleFit, nil
	}
	none, _ := AggregatePredictions("gate", "held", p, m, fewHeld, nil)
	if none.Evaluated || len(none.Aggregates) != 0 {
		t.Errorf("small lane reported: %+v", none.Aggregates)
	}
	// Held units on a legacy side are reported apart, marked not blind, never pooled with the
	// blind ones. Half the held units of each lane are on a legacy side, so each lane holds 5
	// of each kind; pooled, each lane would report 10. Apart, each kind falls into "other".
	onSide := map[string]bool{}
	for i := 0; i < 30; i += 2 {
		onSide[hexID(i)] = true
	}
	split, _ := AggregatePredictions("gate", "held", p, m, roleOf, func(_, id string) bool { return onSide[id] })
	if len(split.Aggregates) != 2 || split.Aggregates[0].Lane != "other" || split.Aggregates[1].Lane != "other" ||
		split.Aggregates[0].NotBlind == split.Aggregates[1].NotBlind {
		t.Fatalf("aggregates %+v, want one blind and one not-blind figure in lane other", split.Aggregates)
	}
}

// D7: what a seed-holding report covers is fixed by the seed and the measured data. A candidate
// that gives a reason for a unit, or a metric for only some units, changes the figure but not
// the units it covers, so it cannot choose the population by omission (probing the partition).
func TestAggregateCoverageIsNotCandidateChosen(t *testing.T) {
	m := &Measured{Values: map[string]map[string]float64{}, Lanes: map[string]string{}}
	roles := map[string]Role{}
	full := &Predictions{Kind: "Predictions", Candidate: strings.Repeat("a", 40)}
	for i := 0; i < 30; i++ {
		id := hexID(i)
		m.Values["s/"+id] = map[string]float64{"ttft": 100, "tpot": 10}
		m.Lanes["s/"+id] = "h200"
		roles[id] = RoleFit
		if i%2 == 0 {
			roles[id] = RoleHeld
		}
		full.Predictions = append(full.Predictions, Prediction{Source: "s", UnitID: id, Values: map[string]float64{"ttft": 100, "tpot": 10}})
	}
	roleOf := func(_, id string) (Role, error) { return roles[id], nil }
	base, err := AggregatePredictions("gate", "held", full, m, roleOf, nil)
	if err != nil || len(base.Aggregates) != 2 {
		t.Fatalf("base %+v, %v", base, err)
	}
	units := func(r *Report) string {
		var s string
		for _, a := range r.Aggregates {
			s += fmt.Sprintf("%s/%s=%d ", a.Lane, a.Metric, a.Units)
		}
		return s
	}
	// Variant 1: a reason for one held unit. Variant 2: tpot for only one held unit.
	// Variant 3: ttft supplied by no held unit at all, tpot by every one.
	reason := &Predictions{Kind: full.Kind, Candidate: full.Candidate, Predictions: append([]Prediction(nil), full.Predictions...)}
	reason.Predictions[0] = Prediction{Source: "s", UnitID: hexID(0), Reason: "refused"}
	some := &Predictions{Kind: full.Kind, Candidate: full.Candidate}
	noTTFT := &Predictions{Kind: full.Kind, Candidate: full.Candidate}
	for i, x := range full.Predictions {
		y := Prediction{Source: x.Source, UnitID: x.UnitID, Values: map[string]float64{"ttft": 100}}
		if i != 2 {
			y.Values["tpot"] = 10
		}
		some.Predictions = append(some.Predictions, y)
		z := Prediction{Source: x.Source, UnitID: x.UnitID, Values: map[string]float64{"tpot": 10}}
		noTTFT.Predictions = append(noTTFT.Predictions, z)
	}
	for name, p := range map[string]*Predictions{"reason": reason, "partial metric": some} {
		got, err := AggregatePredictions("gate", "held", p, m, roleOf, nil)
		if err != nil {
			t.Fatal(err)
		}
		if units(got) != units(base) {
			t.Errorf("%s: covers %s, want %s", name, units(got), units(base))
		}
		if fmt.Sprint(got.Aggregates) == fmt.Sprint(base.Aggregates) {
			t.Errorf("%s: a missing prediction scored as if it were exact", name)
		}
	}
	got, _ := AggregatePredictions("gate", "held", noTTFT, m, roleOf, nil)
	for _, a := range got.Aggregates {
		if a.Units != 15 {
			t.Errorf("%+v: covers %d units, want 15", a, a.Units)
		}
	}
	// Reasons for every held unit: nothing was predicted, so nothing is evaluated.
	allReasons := &Predictions{Kind: full.Kind, Candidate: full.Candidate}
	for _, x := range full.Predictions {
		allReasons.Predictions = append(allReasons.Predictions, Prediction{Source: x.Source, UnitID: x.UnitID, Reason: "refused"})
	}
	if none, _ := AggregatePredictions("gate", "held", allReasons, m, roleOf, nil); none.Evaluated || len(none.Aggregates) != 0 {
		t.Errorf("all reasons reported %+v", none.Aggregates)
	}
}

// D1: a coefficient citing a unit off the fit list is refused.
func TestCheckCitesOnlyFit(t *testing.T) {
	fit := &Published{Role: RoleFit, Units: []string{hexID(1), hexID(2)}}
	if err := CheckCitesOnlyFit(map[string][]string{"a": {hexID(1)}, "b": {hexID(2), hexID(1)}}, fit); err != nil {
		t.Errorf("fit citations refused: %v", err)
	}
	if err := CheckCitesOnlyFit(map[string][]string{"a": {hexID(1), hexID(3)}}, fit); err == nil {
		t.Error("a citation of a unit off the fit list was accepted")
	}
	if err := CheckCitesOnlyFit(nil, &Published{Role: RoleAnchor}); err == nil {
		t.Error("the anchor list was accepted as the fit list")
	}
}

// The published lists are read back and checked against the ledger that drew them (D7, COV-2):
// a list from another seed or partition, a role the release does not publish, a unit no source
// has, a unit on two lists, an unsorted list, and a directory holding only some of the three are
// each refused. Each case is a known-bad directory.
func TestPublishedListsAreCheckedAgainstTheLedger(t *testing.T) {
	a := assignerFor(t, nil)
	cfg := testConfig()
	rawSeed, _ := hex.DecodeString(strings.Repeat("5a", 32))
	sum := sha256.Sum256(rawSeed)
	cfg.Seed = SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "test", GeneratedOn: "2026-11-04", RotationOwner: "test"}
	var units []Unit
	for i := 0; i < 200; i++ {
		units = append(units, testUnit(i, i))
	}
	byUnit := map[string][]Unit{"s": units}
	lists, err := a.Release(cfg, byUnit)
	if err != nil {
		t.Fatal(err)
	}
	write := func(t *testing.T, mut func(map[Role]*Published)) string {
		t.Helper()
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, DirPublished), 0o755); err != nil {
			t.Fatal(err)
		}
		cp := map[Role]*Published{}
		for r, p := range lists {
			q := *p
			q.Units = append([]string(nil), p.Units...)
			cp[r] = &q
		}
		mut(cp)
		for r, p := range cp {
			b, err := yaml.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, DirPublished, string(r)+".yaml"), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}
	if len(lists[RoleAnchor].Units) < 2 || len(lists[RoleFit].Units) < 2 {
		t.Fatal("the test ledger gives a list fewer than two units")
	}

	dir := write(t, func(map[Role]*Published) {})
	got, err := LoadPublished(dir, cfg, byUnit)
	if err != nil || len(got) != 3 {
		t.Fatalf("the lists the release wrote were refused: %v", err)
	}
	if got, err := LoadPublished(t.TempDir(), cfg, byUnit); err != nil || len(got) != 0 {
		t.Errorf("a ledger with no published lists yet: %v %v", got, err)
	}

	for name, c := range map[string]struct {
		mut  func(map[Role]*Published)
		want string
	}{
		"another seed":           {func(m map[Role]*Published) { m[RoleAnchor].SeedSHA256 = strings.Repeat("0", 64) }, "is not the seed ledger.yaml records"},
		"another partition":      {func(m map[Role]*Published) { m[RoleFit].PartitionVersion = 2 }, "ledger.yaml is at 1"},
		"a role not published":   {func(m map[Role]*Published) { m[RoleReport].Role = RoleHeld }, `role "held" is not published`},
		"a role in another file": {func(m map[Role]*Published) { m[RoleReport].Role = RoleFit }, `role "fit" in the file for "report"`},
		"a unit no source has": {func(m map[Role]*Published) {
			m[RoleAnchor].Units = append(m[RoleAnchor].Units, strings.Repeat("f", 64))
		}, "is in no source's unit list"},
		"a unit on two lists": {func(m map[Role]*Published) {
			m[RoleReport].Units = append([]string{m[RoleFit].Units[0]}, m[RoleReport].Units...)
			sort.Strings(m[RoleReport].Units)
		}, "a unit has one role"},
		"an unsorted list": {func(m map[Role]*Published) {
			u := m[RoleFit].Units
			u[0], u[1] = u[1], u[0]
		}, "is out of order"},
		"a repeated unit": {func(m map[Role]*Published) {
			u := m[RoleFit].Units
			m[RoleFit].Units = append(u[:2:2], u[1:]...)
		}, "is listed twice"},
		"a name that is no unit id": {func(m map[Role]*Published) { m[RoleFit].Units[0] = "A" }, "is not a unit_id"},
		"only two of the three":     {func(m map[Role]*Published) { delete(m, RoleReport) }, "holds 2 of the three"},
	} {
		_, err := LoadPublished(write(t, c.mut), cfg, byUnit)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}

	// A list with the seed not yet recorded is refused: the lists are drawn from it.
	noSeed := testConfig()
	if _, err := LoadPublished(dir, noSeed, byUnit); err == nil || !strings.Contains(err.Error(), "records no seed") {
		t.Errorf("lists published with no recorded seed: %v", err)
	}
}
