package invariants

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"testing/fstest"
)

// designItem is one entry of testdata/design.json, which go run ./tools/invariants
// snapshot writes from the published design: every standing invariant, in design order.
type designItem struct {
	ID, Title, Statement string
}

func loadDesign(t *testing.T) []designItem {
	t.Helper()
	raw, err := os.ReadFile("testdata/design.json")
	if err != nil {
		t.Fatal(err)
	}
	var items []designItem
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatal(err)
	}
	return items
}

func mustEmbedded(t *testing.T) *Registry {
	t.Helper()
	reg, err := Embedded()
	if err != nil {
		t.Fatalf("the committed registry does not load:\n%v", err)
	}
	return reg
}

// Every identifier the design defines appears in the registry exactly once, in the
// design's order, with the design's statement and title; and the registry holds nothing
// the design does not define (design §13.1).
func TestEveryDesignInvariantAppearsExactlyOnce(t *testing.T) {
	reg := mustEmbedded(t)
	design := loadDesign(t)
	if len(design) == 0 {
		t.Fatal("the design snapshot is empty")
	}
	inDesign := map[string]bool{}
	for i, d := range design {
		if inDesign[d.ID] {
			t.Errorf("the design defines %s twice", d.ID)
		}
		inDesign[d.ID] = true
		inv, ok := reg.Get(d.ID)
		if !ok {
			t.Errorf("%s is defined in the design but has no registry record", d.ID)
			continue
		}
		if i < len(reg.Invariants) && reg.Invariants[i].ID != d.ID {
			t.Errorf("record %d is %s; the design's order has %s there", i, reg.Invariants[i].ID, d.ID)
		}
		if inv.Statement != d.Statement {
			t.Errorf("%s: the statement differs from the design's.\nregistry: %s\ndesign:   %s", d.ID, inv.Statement, d.Statement)
		}
		if inv.Title != d.Title {
			t.Errorf("%s: title %q; the design's is %q", d.ID, inv.Title, d.Title)
		}
	}
	for _, inv := range reg.Invariants {
		if !inDesign[inv.ID] {
			t.Errorf("%s has a registry record but the design does not define it", inv.ID)
		}
	}
}

// The release plan's §3.2 table is generated from the registry, so their statuses and
// milestones agree. testdata/plan-status.tsv is the table as published. While the M2 substrate
// stack is open, a layer holds only part of the registry's end state, so
// testdata/stack-differences.tsv lists the rows allowed to differ; the test fails on a
// difference not listed and on a listed row that no longer differs.
func TestRegistryAgreesWithThePlan(t *testing.T) {
	reg := mustEmbedded(t)
	allowed := stackDifferences(t)
	f, err := os.Open("testdata/plan-status.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	rows := 0
	differs := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		rows++
		cols := strings.Split(line, "\t")
		if len(cols) != 3 {
			t.Fatalf("plan-status.tsv: %q has %d columns, want 3", line, len(cols))
		}
		id, status, by := cols[0], cols[1], cols[2]
		inv, ok := reg.Get(id)
		if !ok {
			t.Errorf("the plan lists %s; the registry has no such record", id)
			continue
		}
		got := MilestoneLabel(inv.Target.Milestone)
		if inv.Status == status && got == by {
			continue
		}
		differs[id] = true
		a, listed := allowed[id]
		switch {
		case !listed:
			t.Errorf("%s: registry status %q and target %s; the plan has %q and %s (not in testdata/stack-differences.tsv)", id, inv.Status, got, status, by)
		case a.status != inv.Status || a.by != got:
			t.Errorf("%s: stack-differences.tsv says %q and %s; the registry has %q and %s", id, a.status, a.by, inv.Status, got)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	for id := range allowed {
		if !differs[id] {
			t.Errorf("%s is listed in testdata/stack-differences.tsv but no longer differs from the plan; delete the row", id)
		}
	}
	if rows != len(reg.Invariants) {
		t.Errorf("the plan lists %d invariants, the registry %d", rows, len(reg.Invariants))
	}
}

type stackDiff struct{ status, by string }

// stackDifferences reads the rows allowed to differ from the plan while the stack is open.
func stackDifferences(t *testing.T) map[string]stackDiff {
	t.Helper()
	raw, err := os.ReadFile("testdata/stack-differences.tsv")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]stackDiff{}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) != 4 {
			t.Fatalf("stack-differences.tsv: %q has %d columns, want 4", line, len(cols))
		}
		if _, dup := out[cols[0]]; dup {
			t.Fatalf("stack-differences.tsv lists %s twice", cols[0])
		}
		out[cols[0]] = stackDiff{cols[1], cols[2]}
	}
	return out
}

// Every record's fields hold together. This is what LoadFS checks; the committed registry
// must pass it, and each known-bad record below must fail it (X18).
func TestCommittedRegistryIsValid(t *testing.T) {
	reg := mustEmbedded(t)
	if n, want := len(reg.Invariants), len(loadDesign(t)); n != want {
		t.Errorf("the registry holds %d records; the design defines %d", n, want)
	}
	reviews := 0
	for _, inv := range reg.Invariants {
		reviews += len(inv.Review)
	}
	// X1 has two questions; X2 to X9, RTR-2, RTR-6, DEV-12, VAL-11, D1 and D7 one each.
	if reviews != 16 {
		t.Errorf("the registry holds %d review questions; the design asks 16", reviews)
	}
}

const goodRecord = `- id: X1
  section: §4
  statement: Users configure BLIS with the upstream's own configuration.
  scope: [blis-vllm, inference-sim]
  checks:
    - {repo: blis-vllm, check: fields are classified, runs: pull_request, status: ci, test: TestA}
    - {repo: inference-sim, check: router flags reach the router, runs: pull_request, status: new}
  review:
    - question: Does this new field describe something the real system observes?
      expertise: contracts owners
      when: a key is added to the environment list
      applies_when: {repos: [blis-substrate], paths: [contracts/spec/scenario/**]}
  status: partial
  target: {status: ci, milestone: 4}
`

// fsWith returns a registry directory whose x.yaml is the given text and whose other
// family files are empty lists.
func fsWith(x string) fstest.MapFS {
	m := fstest.MapFS{}
	for _, f := range Families {
		m[f.File] = &fstest.MapFile{Data: []byte("[]\n")}
	}
	m["x.yaml"] = &fstest.MapFile{Data: []byte(x)}
	return m
}

func TestLoadAcceptsAWellFormedRecord(t *testing.T) {
	reg, err := LoadFS(fsWith(goodRecord))
	if err != nil {
		t.Fatal(err)
	}
	if len(reg.Invariants) != 1 || reg.Invariants[0].ID != "X1" {
		t.Fatalf("loaded %+v", reg.Invariants)
	}
}

// Each malformed registry must be refused, with a message naming what is wrong. These are
// the loader's known-bad inputs (X18).
func TestLoadRefusesMalformedRecords(t *testing.T) {
	cases := []struct {
		name, old, new, want string
	}{
		{"unknown key", "  status: partial\n", "  status: partial\n  owner: me\n", "field owner not found"},
		{"duplicate key", "  status: partial\n", "  status: partial\n  status: new\n", `duplicate key "status"`},
		{"wrong family", "- id: X1", "- id: SUB-1", "does not belong in this file"},
		{"malformed id", "- id: X1", "- id: X01", "does not belong in this file"},
		{"unknown repository", "repo: inference-sim,", "repo: inference_sim,", `"inference_sim" is not a BLIS repository`},
		{"scope disagrees", "scope: [blis-vllm, inference-sim]", "scope: [blis-vllm]", "scope [blis-vllm] must equal"},
		{"ci check without a test", "status: ci, test: TestA}", "status: ci}", "has status ci but names no test"},
		{"bad runs", "runs: pull_request, status: new", "runs: weekly, status: new", `runs "weekly"`},
		{"the retired settings kind", "runs: pull_request, status: new", "runs: settings, status: new", `runs "settings"`},
		{"a practice with a test", "runs: pull_request, status: new}", "runs: practice, status: new, test: TestA}", "no program runs: it has no test"},
		{"a practice with a known-bad input", "runs: pull_request, status: new}", "runs: practice, status: new, known_bad: 'x: TestB'}", "no program runs: it has no test"},
		{"status disagrees with checks", "  status: partial\n", "  status: ci\n", `status "ci" disagrees with its checks`},
		{"status review", "  status: partial\n", "  status: review\n", "no invariant in the design is enforced by review alone"},
		{"no checks", "  checks:\n    - {repo: blis-vllm, check: fields are classified, runs: pull_request, status: ci, test: TestA}\n    - {repo: inference-sim, check: router flags reach the router, runs: pull_request, status: new}\n", "  checks: []\n", "has no check"},
		{"target already reached", "milestone: 4}", "milestone: null}", "target.milestone is null"},
		{"bad section", "section: §4", "section: 4", "is not a design section"},
		{"applies_when without paths", "paths: [contracts/spec/scenario/**]", "paths: []", "applies_when needs repos and paths"},
		{"glob with a partial **", "contracts/spec/scenario/**", "contracts/spec/scen**", "** must be a whole path segment"},
		{"absolute glob", "paths: [contracts/spec/scenario/**]", "paths: [/contracts]", "must be a relative path"},
		{"review without expertise", "      expertise: contracts owners\n", "", "needs question, expertise and when"},
		{"duplicate record", "", "", "is defined again"},
		{"carriage return", "  status: partial\n", "  status: partial\r\n", "carriage return"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			doc := goodRecord
			if c.name == "duplicate record" {
				doc = goodRecord + goodRecord
			} else {
				if !strings.Contains(doc, c.old) {
					t.Fatalf("test setup: %q not in the good record", c.old)
				}
				doc = strings.Replace(doc, c.old, c.new, 1)
			}
			_, err := LoadFS(fsWith(doc))
			if err == nil {
				t.Fatalf("loaded a registry with %s", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not say %q", err, c.want)
			}
		})
	}
}

func TestLoadRefusesAStrayFileAndAMissingFamily(t *testing.T) {
	m := fsWith(goodRecord)
	m["extra.yaml"] = &fstest.MapFile{Data: []byte("[]\n")}
	if _, err := LoadFS(m); err == nil || !strings.Contains(err.Error(), "not a registry family file") {
		t.Errorf("a stray extra.yaml: got %v", err)
	}
	m = fsWith(goodRecord)
	delete(m, "lif.yaml")
	if _, err := LoadFS(m); err == nil {
		t.Error("loaded a registry with no lif.yaml")
	}
}

func TestMatch(t *testing.T) {
	cases := []struct {
		glob, name string
		want       bool
	}{
		{"catalog/**", "catalog/models/x/model.yaml", true},
		{"catalog/**", "catalog", true},
		{"catalog/**", "catalogue/x", false},
		{"**/*_test.go", "kernel/a_test.go", true},
		{"**/*_test.go", "a_test.go", true},
		{"**/*_test.go", "kernel/a.go", false},
		{"sim/engine_driver*.go", "sim/engine_driver_group.go", true},
		{"sim/engine_driver*.go", "sim/x/engine_driver.go", false},
		{"oracle/mocks.json", "oracle/mocks.json", true},
		{"pkg/**/*.go", "pkg/a/b/c.go", true},
		{"pkg/**/*.go", "pkg/c.go", true},
	}
	for _, c := range cases {
		if got := Match(c.glob, c.name); got != c.want {
			t.Errorf("Match(%q, %q) = %v, want %v", c.glob, c.name, got, c.want)
		}
	}
}

func TestTriggerMatchesHonoursRepoAndExclude(t *testing.T) {
	tr := Trigger{Repos: []string{"blis-substrate"}, Paths: []string{"kernel/**"}, Exclude: []string{"**/*_test.go"}}
	if !tr.Matches("blis-substrate", "kernel/new.go") {
		t.Error("kernel/new.go should match")
	}
	if tr.Matches("blis-substrate", "kernel/new_test.go") {
		t.Error("a test file is excluded")
	}
	if tr.Matches("blis-vllm", "kernel/new.go") {
		t.Error("the trigger names only blis-substrate")
	}
	if (Trigger{Manual: true}).Matches("blis-substrate", "kernel/new.go") {
		t.Error("a manual trigger matches no path")
	}
}

// A maintainer practice (design §4.2) names its repository and states its checklist item in
// check, has no test and no known-bad input, and at status ci needs no test: no program runs
// it. It counts toward the invariant's status like any check, so a record whose checks are one
// automated check at ci and one practice at new is partial.
func TestAPracticeIsValidWithoutATest(t *testing.T) {
	doc := `- id: X1
  section: §4
  statement: A practice and a program keep this rule.
  scope: [blis-vllm]
  checks:
    - {repo: blis-vllm, check: the protected branch requires the named checks, runs: practice, status: ci}
    - {repo: blis-vllm, check: a program checks the form, runs: pull_request, status: ci, test: TestA}
  review: null
  status: ci
  target: {status: ci, milestone: null}
`
	reg, err := LoadFS(fsWith(doc))
	if err != nil {
		t.Fatalf("a practice at ci with no test was refused: %v", err)
	}
	if got := reg.Invariants[0].Checks[0]; got.Runs != RunsPractice || got.Test != "" {
		t.Errorf("loaded %+v", got)
	}
	// The same record with the practice still new is partial, not new.
	if _, err := LoadFS(fsWith(strings.Replace(strings.Replace(doc, "runs: practice, status: ci", "runs: practice, status: new", 1), "  status: ci\n  target: {status: ci, milestone: null}", "  status: partial\n  target: {status: ci, milestone: 2}", 1))); err != nil {
		t.Errorf("a practice at new beside an automated check at ci should make the record partial: %v", err)
	}
}

func TestMilestoneLabel(t *testing.T) {
	one, four, after := 1, 4, PostM4
	for _, c := range []struct {
		m    *int
		want string
	}{{nil, "now"}, {&one, "M1"}, {&four, "M4"}, {&after, "after M4"}} {
		if got := MilestoneLabel(c.m); got != c.want {
			t.Errorf("MilestoneLabel = %q, want %q", got, c.want)
		}
	}
}

// A target is a milestone of the plan, 1 to 4, or after M4; nothing else. The known-bad
// records go through the loader, which is where the rule lives.
func TestATargetMilestoneIsOneToFourOrAfterM4(t *testing.T) {
	with := func(m string) string {
		return strings.Replace(goodRecord, "milestone: 4", "milestone: "+m, 1)
	}
	if !strings.Contains(goodRecord, "milestone: 4") {
		t.Fatal("test setup: the good record's target is not milestone 4")
	}
	for _, m := range []string{"0", "-1", "6"} {
		if _, err := LoadFS(fsWith(with(m))); err == nil || !strings.Contains(err.Error(), "not a milestone of the release plan") {
			t.Errorf("milestone %s: %v", m, err)
		}
	}
	for _, m := range []string{"1", "4", "5"} {
		if _, err := LoadFS(fsWith(with(m))); err != nil {
			t.Errorf("milestone %s was refused: %v", m, err)
		}
	}
}
