package invariants

import (
	"bufio"
	"encoding/json"
	"fmt"
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
// milestones agree. testdata/plan-status.tsv is the table as published.
func TestRegistryAgreesWithThePlan(t *testing.T) {
	reg := mustEmbedded(t)
	f, err := os.Open("testdata/plan-status.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows := 0
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
		if inv.Status != status {
			t.Errorf("%s: registry status %q, plan status %q", id, inv.Status, status)
		}
		got := "now"
		if inv.Target.Milestone != nil {
			got = fmt.Sprintf("M%d", *inv.Target.Milestone)
		}
		if got != by {
			t.Errorf("%s: registry target milestone %s, plan In CI by %s", id, got, by)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if rows != len(reg.Invariants) {
		t.Errorf("the plan lists %d invariants, the registry %d", rows, len(reg.Invariants))
	}
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
      reviewer: contracts owners
      when: a key is added to the environment list
      trigger: {repos: [blis-substrate], paths: [contracts/spec/scenario/**]}
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
		{"status disagrees with checks", "  status: partial\n", "  status: ci\n", `status "ci" disagrees with its checks`},
		{"status review", "  status: partial\n", "  status: review\n", "no invariant in the design is enforced by review alone"},
		{"no checks", "  checks:\n    - {repo: blis-vllm, check: fields are classified, runs: pull_request, status: ci, test: TestA}\n    - {repo: inference-sim, check: router flags reach the router, runs: pull_request, status: new}\n", "  checks: []\n", "has no check"},
		{"target already reached", "milestone: 4}", "milestone: null}", "target.milestone is null"},
		{"bad section", "section: §4", "section: 4", "is not a design section"},
		{"trigger without paths", "paths: [contracts/spec/scenario/**]", "paths: []", "trigger needs repos and paths"},
		{"glob with a partial **", "contracts/spec/scenario/**", "contracts/spec/scen**", "** must be a whole path segment"},
		{"absolute glob", "paths: [contracts/spec/scenario/**]", "paths: [/contracts]", "must be a relative path"},
		{"review without a reviewer", "      reviewer: contracts owners\n", "", "needs question, reviewer and when"},
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
