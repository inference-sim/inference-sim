package routertable

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// committed reads the entries of catalog/routers the way TestTheCommittedEntriesValidate does.
func committed(t *testing.T) []*Entry {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("..", "..", "..", "catalog", "routers", "*.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no entries under catalog/routers: %v", err)
	}
	var out []*Entry
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
		out = append(out, &e)
	}
	return out
}

// A recipe pins Dynamo by a wheel version. The version stands for the commits it could have been
// made from, and the table covers it only if the files that decide the mode are the same at all of
// them. Of the four versions the P/D recipes name, three are, and one is not: the nightly of 14
// June 2026 spans a commit (789736997696e0a874e07a70b507ffaffd1069b3) that rewrote the round-robin
// function's logging, so what the table shows at the other commits of that day is not shown for it.
func TestADynamoWheelVersionResolvesToItsCommits(t *testing.T) {
	entries := committed(t)
	for _, c := range []struct {
		build string
		want  Outcome
	}{
		{"1.2.0.dev20260426", OutcomeMapped},
		{"1.3.0.dev20260601", OutcomeMapped},
		{"1.2.1", OutcomeMapped},
		{"1.3.0.dev20260614", OutcomeMapped},
		{"", OutcomeRevisionNotCovered}, // 121 recipes state no version
		{"1.6.0.dev20261010", OutcomeRevisionNotCovered},
	} {
		e, got := ResolveBuild(entries, "dynamo", "round-robin", nil, c.build)
		if got != c.want {
			t.Errorf("build %q: %s, want %s", c.build, got, c.want)
		}
		if e == nil || e.Name != "dynamo-round-robin" {
			t.Errorf("build %q: the entry is returned whenever one has the key, got %v", c.build, e)
		}
	}
	if _, got := ResolveBuild(entries, "dynamo", "least-loaded", nil, "1.2.1"); got != OutcomeNoEntry {
		t.Errorf("a mode with no entry: %s, want %s", got, OutcomeNoEntry)
	}
}

// A commit that lies in a covered build's window is covered without being listed in revisions, and
// every commit of the 14 June window is, though one of them changed push_router.rs: it changed
// logging only (the normal form of both blobs is the same). This test uses only the lookup by
// commit.
func TestACommitOfACoveredWindowIsCoveredWithoutBeingListed(t *testing.T) {
	entries := committed(t)
	for name, c := range map[string]struct {
		commit string
		want   Outcome
	}{
		"the other commit of the 26 April window":                         {"55a639e2808ee86fb077fa993c2aded37eba6f0f", OutcomeMapped},
		"the head of the 1 June window":                                   {"bed9f269312151481cd67a8d21b70e0f52424c2b", OutcomeMapped},
		"the commit of tag v1.2.1":                                        {"919682da679aa699d5bca9c872f4c1d9a530bbc0", OutcomeMapped},
		"the 14 June commit that changed push_router.rs, in logging only": {"789736997696e0a874e07a70b507ffaffd1069b3", OutcomeMapped},
		"an earlier commit of that day":                                   {"9336dce0081207503b6435afcdcb4c02e4f57f7f", OutcomeMapped},
		"a commit of the next day, outside every window":                  {"0000000000000000000000000000000000000001", OutcomeRevisionNotCovered},
	} {
		if _, got := Resolve(entries, "dynamo", "round-robin", nil, c.commit); got != c.want {
			t.Errorf("%s (%s): %s, want %s", name, c.commit, got, c.want)
		}
	}
}

// The commit a point record names for a build is one the entry covers, and every commit of a
// window resolves.
func TestEveryCommitOfABuildResolvesAndTheRecordedOneIsCovered(t *testing.T) {
	entries := committed(t)
	e, _ := ResolveBuild(entries, "dynamo", "round-robin", nil, "1.3.0.dev20260601")
	b := e.BuildOf("1.3.0.dev20260601")
	if len(b.Commits) != 19 {
		t.Fatalf("the window of 1 June holds %d commits, want 19", len(b.Commits))
	}
	for _, c := range b.Commits {
		if _, got := Resolve(entries, "dynamo", "round-robin", nil, c); got != OutcomeMapped {
			t.Errorf("commit %s of build 1.3.0.dev20260601: %s, want %s", c, got, OutcomeMapped)
		}
	}
	if _, got := Resolve(entries, "dynamo", "round-robin", nil, b.Revision()); got != OutcomeMapped {
		t.Errorf("the commit a point record names for the build: %s", got)
	}
}

func withBuild() *Entry {
	e := valid()
	a, b := strings.Repeat("b", 40), strings.Repeat("c", 40)
	e.Builds = []Build{{Name: "1.0.0.dev20260101", Source: BuildNightly,
		Window:  "every first-parent commit of main on 2026-01-01 (UTC)",
		Commits: []string{a, b},
		Files:   []BuildFile{{Path: "src/router.go", Blob: strings.Repeat("e", 40)}}}}
	for _, f := range []Field{FieldMode, FieldPolicy} {
		e.Evidence = append(e.Evidence, Citation{Side: SideRouter, Field: f, Repository: "acme/router",
			Commit: a, Path: "src/router.go", Lines: "10-20", Claim: "the passage shows it"})
	}
	return e
}

// The build table's rules: a build is a claim that the deciding files are the same at all of its
// commits, so the entry names them, cites only them inside the window, and cites the build itself.
func TestABuildMustVouchForWhatItCites(t *testing.T) {
	if p := withBuild().Validate(); !p.OK() {
		t.Fatalf("the fixture must validate:\n%s", p.Error())
	}
	for name, c := range map[string]struct {
		edit func(*Entry)
		want string
	}{
		"a citation of a file the build does not vouch for": {func(e *Entry) { e.Evidence[len(e.Evidence)-1].Path = "src/other.go" }, "not among the files of build"},
		"no citation inside the window":                     {func(e *Entry) { e.Evidence = e.Evidence[:3] }, "has no router-side citation at one of its commits"},
		"no files":                                          {func(e *Entry) { e.Builds[0].Files = nil }, "files"},
		"a blob that is not an identifier":                  {func(e *Entry) { e.Builds[0].Files[0].Blob = "main" }, "blob"},
		"commits out of order":                              {func(e *Entry) { c := e.Builds[0].Commits; c[0], c[1] = c[1], c[0] }, "commits are sorted"},
		"no window statement":                               {func(e *Entry) { e.Builds[0].Window = " " }, "window"},
		"an unknown source":                                 {func(e *Entry) { e.Builds[0].Source = "weekly" }, "source"},
		"a release of two commits":                          {func(e *Entry) { e.Builds[0].Source = BuildRelease }, "one commit of its tag"},
		"two builds of one name":                            {func(e *Entry) { e.Builds = append(e.Builds, e.Builds[0]) }, "repeats the build"},
		"a variant whose normal form differs (the file differs in what it computes)": {func(e *Entry) {
			f := &e.Builds[0].Files[0]
			f.Rule, f.NormalForm = NormalFormLoggingOnly, strings.Repeat("a", 64)
			f.Variants = []BuildVariant{{Blob: strings.Repeat("b", 40), NormalForm: strings.Repeat("c", 64)}}
		}, "differs in more than"},
		"other blobs under an unknown rule": {func(e *Entry) {
			f := &e.Builds[0].Files[0]
			f.Rule, f.NormalForm = "whitespace", strings.Repeat("a", 64)
			f.Variants = []BuildVariant{{Blob: strings.Repeat("b", 40), NormalForm: strings.Repeat("a", 64)}}
		}, "must be \"rust-logging-only\""},
		"a variant that repeats the blob": {func(e *Entry) {
			f := &e.Builds[0].Files[0]
			f.Rule, f.NormalForm = NormalFormLoggingOnly, strings.Repeat("a", 64)
			f.Variants = []BuildVariant{{Blob: f.Blob, NormalForm: strings.Repeat("a", 64)}}
		}, "repeats blob"},
		"a Python file compared under the Rust rule": {func(e *Entry) {
			f := &e.Builds[0].Files[0]
			f.Path = "dynamo/frontend/main.py"
			f.Rule, f.NormalForm = NormalFormLoggingOnly, strings.Repeat("a", 64)
			f.Variants = []BuildVariant{{Blob: strings.Repeat("b", 40), NormalForm: strings.Repeat("a", 64)}}
		}, "must be \"python-routing-statements\""},
		"a rule with no variants": {func(e *Entry) { e.Builds[0].Files[0].Rule = NormalFormLoggingOnly }, "list them or drop both"},
		"a commit in two builds": {func(e *Entry) {
			o := e.Builds[0]
			o.Name = "1.0.0.dev20260102"
			e.Builds = append(e.Builds, o)
		}, "is in builds"},
	} {
		e := withBuild()
		c.edit(e)
		p := e.Validate()
		if p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: want a problem containing %q, got:\n%s", name, c.want, p.Error())
		}
	}
}

// An entry covers a build by its name and nothing else, and its window commits are covered by
// commit: a build the entry does not list, a different spelling, and no build at all are not.
func TestAnEntryCoversABuildByItsExactName(t *testing.T) {
	entries := []*Entry{withBuild()}
	for _, c := range []struct {
		build string
		want  Outcome
	}{
		{"1.0.0.dev20260101", OutcomeMapped},
		{"1.0.0.dev20260102", OutcomeRevisionNotCovered},
		{"1.0.0", OutcomeRevisionNotCovered},
		{"", OutcomeRevisionNotCovered},
	} {
		if _, got := ResolveBuild(entries, "acme", "round-robin", nil, c.build); got != c.want {
			t.Errorf("build %q: %s, want %s", c.build, got, c.want)
		}
	}
	if _, got := Resolve(entries, "acme", "round-robin", nil, strings.Repeat("c", 40)); got != OutcomeMapped {
		t.Errorf("a commit of the window that carries no citation: %s, want %s", got, OutcomeMapped)
	}
}

// A Python deciding file whose blobs share one normal form under the Python rule is accepted: the
// entry may vouch for a build whose frontend file differs only in code that names no router.
func TestAPythonFileWithOneNormalFormIsAccepted(t *testing.T) {
	e := withBuild()
	e.Builds[0].Files = append(e.Builds[0].Files, BuildFile{Path: "dynamo/frontend/main.py", Blob: strings.Repeat("a", 40),
		Rule: NormalFormPythonRouting, NormalForm: strings.Repeat("a", 64),
		Variants: []BuildVariant{{Blob: strings.Repeat("d", 40), NormalForm: strings.Repeat("a", 64)}}})
	if p := e.Validate(); !p.OK() {
		t.Fatalf("a Python file with one normal form under %s: %v", NormalFormPythonRouting, p)
	}
}
