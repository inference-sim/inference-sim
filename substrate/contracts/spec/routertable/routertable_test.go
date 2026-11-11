package routertable

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func ptr[T any](v T) *T { return &v }

// valid is an entry every rule accepts; each case below breaks one thing.
func valid() *Entry {
	cite := func(side Side, field Field, repo string) Citation {
		path := "src/router.go"
		if side == SideLLMD {
			path = "pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go"
		}
		return Citation{Side: side, Field: field, Repository: repo, Commit: strings.Repeat("a", 40),
			Path: path, Lines: "10-20", Claim: "the passage shows it"}
	}
	return &Entry{
		Kind: Kind, Name: "acme-round-robin", Router: "acme", Mode: "round-robin", Policy: PolicyRoundRobin,
		Revisions: []string{strings.Repeat("a", 40)},
		Reason:    "both rotate over the workers",
		Evidence: []Citation{
			cite(SideRouter, FieldMode, "acme/router"),
			cite(SideRouter, FieldPolicy, "acme/router"),
			cite(SideLLMD, FieldPolicy, "inference-sim/llm-d-router"),
		},
	}
}

func TestAValidEntryPasses(t *testing.T) {
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("the fixture must validate:\n%s", p.Error())
	}
	none := valid()
	none.Policy, none.Reason = PolicyNone, "its scoring has no counterpart in llm-d"
	none.Evidence[2].Path = "pkg/epp/anything.go" // a policy of none names no defining file
	if p := none.Validate(); !p.OK() {
		t.Fatalf("an explicit no close policy must validate:\n%s", p.Error())
	}
	aff := valid()
	aff.SessionAffinityTTLS = ptr(14400)
	aff.Evidence = append(aff.Evidence, Citation{Side: SideRouter, Field: FieldSessionAffinityTTLS, Repository: "acme/router",
		Commit: strings.Repeat("a", 40), Path: "docs/affinity.md", Lines: "40", Claim: "an idle timeout in seconds"})
	aff.Name = "acme-round-robin-affinity-14400"
	if p := aff.Validate(); !p.OK() {
		t.Fatalf("an entry with a session-affinity setting must validate:\n%s", p.Error())
	}
}

// A known-bad entry per refusal (SUB-2, design §11.4): no evidence, a policy llm-d's router
// does not offer, and each way a citation or a field can be wrong.
func TestEachFailureModeIsRefused(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(e *Entry)
		path   string
		msg    string
	}{
		// The revision key (design §11.4): the mapping is bound to the builds it is shown for.
		{"no revisions", func(e *Entry) { e.Revisions = nil }, "revisions", "required"},
		{"a revision that is not a full commit", func(e *Entry) { e.Revisions = []string{"abc1234"} }, "revisions[0]", "40-character"},
		{"a revision twice", func(e *Entry) { e.Revisions = []string{strings.Repeat("a", 40), strings.Repeat("a", 40)} }, "revisions[1]", "repeats"},
		{"revisions out of order", func(e *Entry) { e.Revisions = []string{strings.Repeat("b", 40), strings.Repeat("a", 40)} }, "revisions[1]", "sorted"},
		{"a router citation at a build the entry does not cover", func(e *Entry) { e.Evidence[1].Commit = strings.Repeat("c", 40) },
			"evidence[1].commit", "not among the entry's revisions"},
		{"a covered build with no citation of its behaviour", func(e *Entry) {
			e.Revisions = []string{strings.Repeat("a", 40), strings.Repeat("b", 40)}
		}, "evidence", "has no router-side citation for its mode and its behaviour"},
		{"wrong kind", func(e *Entry) { e.Kind = "Router" }, "kind", "must be"},
		{"no router", func(e *Entry) { e.Router = "" }, "router", "required"},
		{"router in upper case", func(e *Entry) { e.Router = "Dynamo" }, "router", "lower case"},
		{"llm-d itself", func(e *Entry) { e.Router = "llm-d" }, "router", "BLIS hosts llm-d's router"},
		{"no mode", func(e *Entry) { e.Mode = "" }, "mode", "required"},
		{"mode with a space", func(e *Entry) { e.Mode = "round robin" }, "mode", "lower case"},
		{"affinity of zero", func(e *Entry) { e.SessionAffinityTTLS = ptr(0) }, "session_affinity_ttl_s", "at least 1"},
		{"no policy", func(e *Entry) { e.Policy = "" }, "policy", "required"},
		{"a policy llm-d's router does not offer", func(e *Entry) { e.Policy = "kv-aware" }, "policy", "not a policy llm-d's router offers"},
		{"no reason", func(e *Entry) { e.Reason = "  " }, "reason", "required"},
		{"no reason for no policy", func(e *Entry) { e.Policy, e.Reason = PolicyNone, "" }, "reason", "why no policy"},
		{"no evidence", func(e *Entry) { e.Evidence = nil }, "evidence", "added only with evidence"},
		{"no citation of the router's mode", func(e *Entry) { e.Evidence = e.Evidence[1:] }, "evidence", "supports the mode"},
		{"no citation of the router's behaviour", func(e *Entry) { e.Evidence[1].Field = FieldMode }, "evidence", "behaviour that the policy is matched to"},
		{"no citation of llm-d", func(e *Entry) { e.Evidence = e.Evidence[:2] }, "evidence", "llm-d's router code or documentation supports the policy"},
		{"no citation of the affinity setting", func(e *Entry) { e.SessionAffinityTTLS = ptr(60) }, "evidence", "session-affinity setting"},
		{"a citation of an affinity the entry lacks", func(e *Entry) { e.Evidence[0].Field = FieldSessionAffinityTTLS }, "evidence", "does not state"},
		{"an unknown side", func(e *Entry) { e.Evidence[0].Side = "other" }, "evidence[0].side", "must be"},
		{"an unknown field", func(e *Entry) { e.Evidence[0].Field = "router" }, "evidence[0].field", "must be"},
		{"llm-d cited for the router's mode", func(e *Entry) { e.Evidence[2].Field = FieldMode }, "evidence[2].field", "policy only"},
		{"a repository without an owner", func(e *Entry) { e.Evidence[0].Repository = "router" }, "evidence[0].repository", "<owner>/<name>"},
		{"a short commit", func(e *Entry) { e.Evidence[0].Commit = "abc123" }, "evidence[0].commit", "40-character"},
		{"a branch instead of a commit", func(e *Entry) { e.Evidence[0].Commit = "main" }, "evidence[0].commit", "pinned revision"},
		{"an upper-case commit", func(e *Entry) { e.Evidence[0].Commit = strings.Repeat("A", 40) }, "evidence[0].commit", "40-character"},
		{"no path", func(e *Entry) { e.Evidence[0].Path = "" }, "evidence[0].path", "required"},
		{"an absolute path", func(e *Entry) { e.Evidence[0].Path = "/src/router.go" }, "evidence[0].path", "from the repository's root"},
		{"a path that climbs", func(e *Entry) { e.Evidence[0].Path = "../router.go" }, "evidence[0].path", "from the repository's root"},
		{"no lines", func(e *Entry) { e.Evidence[0].Lines = "" }, "evidence[0].lines", "line number"},
		{"line zero", func(e *Entry) { e.Evidence[0].Lines = "0" }, "evidence[0].lines", "line number"},
		{"a range that runs backward", func(e *Entry) { e.Evidence[0].Lines = "20-10" }, "evidence[0].lines", "forward"},
		{"no claim", func(e *Entry) { e.Evidence[0].Claim = "" }, "evidence[0].claim", "required"},
		{"llm-d cited at another repository", func(e *Entry) { e.Evidence[2].Repository = "acme/router" }, "evidence[2].repository", "one of"},
		{"the router cited at llm-d's repository", func(e *Entry) { e.Evidence[0].Repository = "inference-sim/llm-d-router" }, "evidence[0].repository", "cannot be of llm-d's router"},
		{"llm-d cited at a file that does not define the policy", func(e *Entry) { e.Evidence[2].Path = "pkg/epp/other.go" }, "evidence", "defines the policy"},
		{"a file name that lies about the key", func(e *Entry) { e.Name = "whatever-name" }, "name", "make its name"},
		{"a file name that omits the affinity", func(e *Entry) {
			e.SessionAffinityTTLS = ptr(60)
			e.Evidence = append(e.Evidence, Citation{Side: SideRouter, Field: FieldSessionAffinityTTLS, Repository: "acme/router",
				Commit: strings.Repeat("b", 40), Path: "d.md", Lines: "1", Claim: "x"})
		}, "name", "make its name"},
		{"a citation repeated", func(e *Entry) { e.Evidence = append(e.Evidence, e.Evidence[2]) }, "evidence[3]", "repeats"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e := valid()
			c.break_(e)
			p := e.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

// A key is one entry: the name is derived from the router, the mode and the affinity setting, so
// two files cannot state one key under two names, and a name cannot say another key.
func TestTheNameIsTheKey(t *testing.T) {
	e := valid()
	if e.WantName() != "acme-round-robin" || e.Key() != "acme/round-robin" {
		t.Errorf("%s %s", e.WantName(), e.Key())
	}
	e.SessionAffinityTTLS = ptr(300)
	if e.WantName() != "acme-round-robin-affinity-300" || e.Key() != "acme/round-robin/affinity-ttl-300" {
		t.Errorf("%s %s", e.WantName(), e.Key())
	}
}

func TestEveryPolicyIsDocumentedAndOffered(t *testing.T) {
	seen := map[Policy]bool{}
	for _, i := range Policies() {
		if i.Plugins == "" || len(i.Sources) == 0 {
			t.Errorf("%s: states no plugins or no source", i.Policy)
		}
		if !i.Policy.Offered() || seen[i.Policy] || i.Policy == PolicyNone {
			t.Errorf("%s is repeated, none, or not offered", i.Policy)
		}
		seen[i.Policy] = true
	}
	if Policy("bogus").Offered() || !PolicyNone.Offered() {
		t.Error("Offered must accept none and refuse a policy outside the list")
	}
}

// The committed entries are read the way the catalog reads them and satisfy every rule above.
func TestTheCommittedEntriesValidate(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "..", "catalog", "routers", "*.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no entries under catalog/routers: %v", err)
	}
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var e Entry
		if err := yaml.Unmarshal(b, &e); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		e.Name = strings.TrimSuffix(filepath.Base(path), ".yaml")
		if p := e.Validate(); !p.OK() {
			t.Errorf("%s:\n%s", path, p.Error())
		}
	}
}

// A mapping shown for one build of the router is not inherited by another with the same mode
// (design §11.4): the same router, mode and affinity setting at a revision no entry lists is not
// scored through the table, and neither is one at a listed revision whose entry says no policy
// is close enough.
func TestAnEntryIsNotInheritedByAnUncitedRevision(t *testing.T) {
	r1, r2 := strings.Repeat("a", 40), strings.Repeat("d", 40)
	mapped := valid()
	none := valid()
	none.Router, none.Policy = "other", PolicyNone
	entries := []*Entry{mapped, none}
	for _, c := range []struct {
		name         string
		router, mode string
		ttl          *int
		rev          string
		want         Outcome
		scored       bool
	}{
		{"the cited revision", "acme", "round-robin", nil, r1, OutcomeMapped, true},
		{"the same mode at an uncited revision", "acme", "round-robin", nil, r2, OutcomeRevisionNotCovered, false},
		{"no revision stated", "acme", "round-robin", nil, "", OutcomeRevisionNotCovered, false},
		{"a policy of none at its cited revision", "other", "round-robin", nil, r1, OutcomeNoPolicy, false},
		{"another mode", "acme", "kv", nil, r1, OutcomeNoEntry, false},
		{"an affinity setting the entry lacks", "acme", "round-robin", ptr(300), r1, OutcomeNoEntry, false},
	} {
		e, got := Resolve(entries, c.router, c.mode, c.ttl, c.rev)
		if got != c.want || got.Scored() != c.scored {
			t.Errorf("%s: %s (entry %v), want %s", c.name, got, e != nil, c.want)
		}
	}
}
