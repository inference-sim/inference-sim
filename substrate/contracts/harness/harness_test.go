package harness

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestAWellFormedUpstreamHasNoProblems(t *testing.T) {
	u := Upstream{Version: "0.5.23", Tag: "v0.5.23", Commit: strings.Repeat("ab", 20)}
	if p := u.Problems(); len(p) != 0 {
		t.Errorf("Problems() = %q, want none", p)
	}
}

func TestAMalformedUpstreamNamesEachProblem(t *testing.T) {
	good := Upstream{Version: "0.5.23", Tag: "v0.5.23", Commit: strings.Repeat("ab", 20)}
	for name, c := range map[string]struct {
		u    Upstream
		want string
	}{
		"short commit":     {Upstream{good.Version, good.Tag, "297e54fd105e"}, "not a full 40-hex commit id"},
		"upper-case hex":   {Upstream{good.Version, good.Tag, strings.Repeat("AB", 20)}, "not a full 40-hex commit id"},
		"no tag":           {Upstream{good.Version, "", good.Commit}, "tag is empty"},
		"tag as version":   {Upstream{"v0.5.23", good.Tag, good.Commit}, "not a release number"},
		"version is empty": {Upstream{"", good.Tag, good.Commit}, "not a release number"},
	} {
		got := c.u.Problems()
		if len(got) != 1 || !strings.Contains(got[0], c.want) {
			t.Errorf("%s: Problems() = %q, want one problem containing %q", name, got, c.want)
		}
	}
}

func TestGoFilesSkipsTestsTestdataHiddenAndNamedDirectories(t *testing.T) {
	root := t.TempDir()
	for _, f := range []string{"a.go", "a_test.go", "core/b.go", "core/testdata/c.go", ".github/d.go", "cmd/tool/e.go", "core/cmd/f.go"} {
		write(t, root, f, "package x\n")
	}
	got, err := GoFiles(root, "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if want := "a.go core/b.go core/cmd/f.go"; strings.Join(got, " ") != want {
		t.Errorf("GoFiles = %q, want %q (only the top-level cmd is skipped)", got, want)
	}
}

func TestGoFilesFailsWhenItFindsNothing(t *testing.T) {
	root := t.TempDir()
	write(t, root, "only_test.go", "package x\n")
	if _, err := GoFiles(root); err == nil {
		t.Error("GoFiles found no file and returned no error; a check over it would pass on nothing")
	}
}

var rules = ProvenanceRules{
	Derived:  regexp.MustCompile(`^// Derived from upstream/\S+ at Upstream v\d+\.\d+\.\d+ \([0-9a-f]{40}\)`),
	Licence:  []string{"// SPDX-License-Identifier: Apache-2.0", "// SPDX-FileCopyrightText: Copyright the upstream authors"},
	Moved:    regexp.MustCompile(`^// Moved from other/\S+ at [0-9a-f]{40}`),
	Original: "// Original to blis-engine.",
}

func TestEveryWayOfStatingProvenancePasses(t *testing.T) {
	root := t.TempDir()
	commit := strings.Repeat("0a", 20)
	write(t, root, "derived.go", rules.Licence[0]+"\n"+rules.Licence[1]+"\n// Derived from upstream/sched.py at Upstream v1.2.3 ("+commit+").\n\npackage x\n")
	write(t, root, "moved.go", "// Moved from other/sim/a.go at "+commit+".\npackage x\n")
	write(t, root, "original.go", "// Original to blis-engine.\n\n// Package x does a thing.\npackage x\n")
	files, err := GoFiles(root)
	if err != nil {
		t.Fatal(err)
	}
	bad, err := ProvenanceProblems(root, files, rules)
	if err != nil {
		t.Fatal(err)
	}
	if len(bad) != 0 {
		t.Errorf("ProvenanceProblems = %q, want none", bad)
	}
}

func TestAFileThatStatesNoProvenanceOrDropsTheLicenceFails(t *testing.T) {
	commit := strings.Repeat("0a", 20)
	for name, src := range map[string]string{
		"nothing":                 "package x\n",
		"a plain comment":         "// Package x does a thing.\npackage x\n",
		"stated after the header": "// Package x.\n\npackage x\n\n// Original to blis-engine.\n",
		"derived, no licence":     "// Derived from upstream/sched.py at Upstream v1.2.3 (" + commit + ").\npackage x\n",
		"derived, licence swapped": rules.Licence[1] + "\n" + rules.Licence[0] +
			"\n// Derived from upstream/sched.py at Upstream v1.2.3 (" + commit + ").\npackage x\n",
		"derived, short commit": rules.Licence[0] + "\n" + rules.Licence[1] +
			"\n// Derived from upstream/sched.py at Upstream v1.2.3 (0a0a0a0).\npackage x\n",
	} {
		root := t.TempDir()
		write(t, root, "f.go", src)
		bad, err := ProvenanceProblems(root, []string{"f.go"}, rules)
		if err != nil {
			t.Fatal(err)
		}
		if len(bad) != 1 || !strings.HasPrefix(bad[0], "f.go ") {
			t.Errorf("%s: ProvenanceProblems = %q, want one problem naming f.go", name, bad)
		}
	}
}

func TestIncompleteProvenanceRulesAreRefused(t *testing.T) {
	if _, err := ProvenanceProblems(t.TempDir(), nil, ProvenanceRules{Original: "// Original."}); err == nil {
		t.Error("rules with no Derived pattern or licence were accepted; every file would be judged on Original alone")
	}
}

func TestClockReadsAndGoroutinesAreFound(t *testing.T) {
	root := t.TempDir()
	write(t, root, "a.go", "package x\nimport \"time\"\nfunc f() time.Duration { go f(); _ = time.Now(); return time.Second }\n")
	write(t, root, "b.go", "package x\nimport tm \"time\"\nfunc g() { tm.Sleep(1); <-tm.After(1) }\n")
	write(t, root, "c.go", "package x\nimport \"time\"\nfunc h(d time.Duration) time.Duration { return d * time.Millisecond }\n")
	write(t, root, "d.go", "package x\ntype clock struct{}\nfunc (clock) Now() int { return 0 }\nvar time clock\nfunc k() int { return time.Now() }\n")
	bad, err := ClockAndGoroutineUses(root, []string{"a.go", "b.go", "c.go", "d.go"})
	if err != nil {
		t.Fatal(err)
	}
	// Each finding names the file and line, and what was found there; c.go uses only durations, and
	// d.go's "time" is its own variable, not the package.
	want := [][2]string{
		{"a.go:3:", "starts a goroutine"},
		{"a.go:3:", "calls time.Now"},
		{"b.go:3:", "calls time.Sleep"},
		{"b.go:3:", "calls time.After"},
	}
	if len(bad) != len(want) {
		t.Fatalf("ClockAndGoroutineUses = %q, want %d findings", bad, len(want))
	}
	for i, w := range want {
		if !strings.HasPrefix(bad[i], w[0]) || !strings.Contains(bad[i], w[1]) {
			t.Errorf("finding %d = %q, want %s ... %s", i, bad[i], w[0], w[1])
		}
	}
}

func TestWrittenCopiesAreCurrentAndAnEditOrDeletionIsFound(t *testing.T) {
	root := t.TempDir()
	if err := WriteCopies(root); err != nil {
		t.Fatal(err)
	}
	if bad, err := StaleCopies(root); err != nil || len(bad) != 0 {
		t.Fatalf("StaleCopies after WriteCopies = %q, %v; want none", bad, err)
	}
	for _, c := range Copies {
		b, err := os.ReadFile(filepath.Join(root, c.Path))
		if err != nil {
			t.Fatal(err)
		}
		src, err := sources.ReadFile(c.Source)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(string(b), "# Generated from blis-substrate contracts/harness/"+c.Source) || !strings.HasSuffix(string(b), string(src)) {
			t.Errorf("%s is not the generated line followed by %s", c.Path, c.Source)
		}
	}
	write(t, root, "oracle/check_mocks.py", "# edited\n")
	if err := os.Remove(filepath.Join(root, ".golangci.yml")); err != nil {
		t.Fatal(err)
	}
	bad, err := StaleCopies(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(bad) != 2 || !strings.HasPrefix(bad[0], ".golangci.yml is missing") || !strings.HasPrefix(bad[1], "oracle/check_mocks.py differs") {
		t.Errorf("StaleCopies = %q, want the missing .golangci.yml and the edited check_mocks.py", bad)
	}
}

// The oracle's guard scripts are Python, as the oracles they guard are; their tests plant each
// violation in a scratch copy and assert the script reports it (X18). They need python3 and git;
// without them this fails rather than skips (X16).
func TestTheOracleChecksFailOnTheirKnownBadInputs(t *testing.T) {
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("python3 is not on PATH; the oracle checks' tests cannot run")
	}
	cmd := exec.Command(py, "-I", "oracle/test_checks.py")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("oracle/test_checks.py: %v\n%s", err, out)
	}
}
