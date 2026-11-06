package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// The quarantine list (X25): tests that fail on the main branch, each with an owning issue
// and an expiry date. A listed test does not block merges until it expires; on expiry CI
// fails. The list is .github/quarantine.yaml:
//
//	- test: TestName            # a Go test name, or a pytest node id
//	  package: ./kernel         # the Go package or the Python file
//	  issue: inference-sim/blis-substrate#12
//	  added: 2026-11-05         # the day the entry was added; not in the future
//	  expires: 2026-12-01       # at most 30 days after added
//	  reason: one line
//
// The required "main is green" check reads the same list: while a test fails on main, a
// pull request merges only if it names that test's issue, or the test is quarantined.

// QuarantineEntry is one entry of the quarantine list.
type QuarantineEntry struct {
	Test    string `yaml:"test"`
	Package string `yaml:"package"`
	Issue   string `yaml:"issue"`
	Added   string `yaml:"added"`
	Expires string `yaml:"expires"`
	Reason  string `yaml:"reason"`
}

const maxQuarantine = 30 * 24 * time.Hour

var issueName = regexp.MustCompile(`^[\w.-]+/[\w.-]+#\d+$`)

// CheckQuarantine parses the list and returns its violations on the given day. Every entry
// states the day it was added, and expires at most 30 days after that day, so a far expiry
// is refused whether or not git history is at hand. first maps a test name to the day it
// first appeared in the file (from git log), when history is available (see
// QuarantineHistory); an entry may not state an added day later than that, which would
// push its expiry out. A nil first checks the stated days alone.
func CheckQuarantine(src []byte, today time.Time, first map[string]time.Time) ([]QuarantineEntry, []string) {
	var list []QuarantineEntry
	if len(bytes.TrimSpace(src)) == 0 {
		return nil, nil
	}
	dec := yaml.NewDecoder(bytes.NewReader(src))
	dec.KnownFields(true)
	if err := dec.Decode(&list); err != nil {
		return nil, []string{fmt.Sprintf("quarantine.yaml: %v", err)}
	}
	var bad []string
	seen := map[string]bool{}
	for i, e := range list {
		at := fmt.Sprintf("quarantine.yaml entry %d (%s)", i+1, e.Test)
		key := e.Package + " " + e.Test
		if seen[key] {
			bad = append(bad, at+": listed twice")
		}
		seen[key] = true
		if e.Test == "" || e.Package == "" || strings.TrimSpace(e.Reason) == "" {
			bad = append(bad, at+": needs test, package and reason")
		}
		if !issueName.MatchString(e.Issue) {
			bad = append(bad, fmt.Sprintf("%s: issue %q is not <owner>/<repo>#<n>; every entry has an owning issue (X25)", at, e.Issue))
		}
		added, addedErr := time.Parse("2006-01-02", e.Added)
		switch {
		case e.Added == "":
			bad = append(bad, at+": added is required, the day the entry was added as YYYY-MM-DD; expires is counted from it (X25)")
		case addedErr != nil:
			bad = append(bad, fmt.Sprintf("%s: added %q is not a date YYYY-MM-DD (X25)", at, e.Added))
		case added.After(today):
			bad = append(bad, fmt.Sprintf("%s: added %s is in the future (X25)", at, e.Added))
		}
		if f, ok := first[e.Test]; ok && addedErr == nil && added.After(f) {
			bad = append(bad, fmt.Sprintf("%s: added %s is later than %s, when the entry first appeared in the file (X25)",
				at, e.Added, f.Format("2006-01-02")))
		}
		exp, err := time.Parse("2006-01-02", e.Expires)
		if err != nil {
			bad = append(bad, fmt.Sprintf("%s: expires %q is not a date YYYY-MM-DD (X25)", at, e.Expires))
			continue
		}
		if !today.Before(exp) {
			bad = append(bad, fmt.Sprintf("%s: expired on %s; fix the test or remove it, and the failure blocks merges again (X25)", at, e.Expires))
		}
		if addedErr == nil && exp.Sub(added) > maxQuarantine {
			bad = append(bad, fmt.Sprintf("%s: expires %s, more than 30 days after it was added on %s (X25)", at, e.Expires, e.Added))
		}
	}
	return list, bad
}

// QuarantineHistory returns, for each test named in the list file, the day it first
// appeared in that file in git history. It returns nil when history cannot be trusted:
// the repository is shallow (the first commit is not there), git is unavailable, or the
// file is not tracked. A test with no commit touching it is left out.
func QuarantineHistory(file string, src []byte) map[string]time.Time {
	dir := filepath.Dir(file)
	out, err := exec.Command("git", "-C", dir, "rev-parse", "--is-shallow-repository").Output()
	if err != nil || strings.TrimSpace(string(out)) != "false" {
		return nil
	}
	var list []QuarantineEntry
	if yaml.Unmarshal(src, &list) != nil {
		return nil
	}
	first := map[string]time.Time{}
	for _, e := range list {
		if e.Test == "" {
			continue
		}
		out, err := exec.Command("git", "-C", dir, "log", "--reverse", "--format=%cs", "-S"+e.Test, "--", filepath.Base(file)).Output()
		if err != nil {
			continue
		}
		if lines := strings.Fields(string(out)); len(lines) > 0 {
			if d, err := time.Parse("2006-01-02", lines[0]); err == nil {
				first[e.Test] = d
			}
		}
	}
	return first
}

// GoSkipPattern returns a go test -skip pattern for the quarantined tests of one package.
func GoSkipPattern(list []QuarantineEntry, pkg string) string {
	var names []string
	for _, e := range list {
		if e.Package == pkg {
			names = append(names, "^"+regexp.QuoteMeta(e.Test)+"$")
		}
	}
	return strings.Join(names, "|")
}

// SkipResult is one skipped test from a test run.
type SkipResult struct {
	Package, Test string
}

// CheckSkips returns a finding for each skipped test that the quarantine list does not
// cover (X16). An entry covers a skip when its package matches and its test equals the
// skipped test's name, or the name's part before the first "[" (a pytest parameter), or
// is a prefix ending in "*".
func CheckSkips(skips []SkipResult, list []QuarantineEntry) []string {
	var bad []string
	for _, s := range skips {
		covered := false
		base := s.Test
		if i := strings.Index(base, "["); i >= 0 {
			base = base[:i]
		}
		for _, e := range list {
			if e.Package != s.Package {
				continue
			}
			if e.Test == s.Test || e.Test == base || (strings.HasSuffix(e.Test, "*") && strings.HasPrefix(s.Test, strings.TrimSuffix(e.Test, "*"))) {
				covered = true
			}
		}
		if !covered {
			bad = append(bad, fmt.Sprintf("%s %s was skipped; a skipped test in a required job is a failure unless it is on the quarantine list (X16, X25)", s.Package, s.Test))
		}
	}
	return bad
}

var (
	pytestSkip = regexp.MustCompile(`^SKIPPED \[\d+\] ([^:]+):\d+:`)
	// pytest -v names each test: "validator/test_x.py::test_name[param] SKIPPED (reason)".
	// A test id may hold spaces (a parameter), so the id runs to the first " SKIPPED".
	pytestNodeSkip = regexp.MustCompile(`^(\S+?\.py)::(.+?) SKIPPED(?:\s|$)`)
)

// ParseSkips reads the skips from go test -json output (one JSON object per line) or
// from pytest output. With pytest -v, each skipped test is one result, its package the
// Python file under pyPackage and its test the test's name, so a quarantine entry covers
// one test, not every test in its file. The "SKIPPED [n] file:line: reason" summary lines
// of -rs repeat those skips; a summary line for a file with no skipped test named is a
// whole module skipped at import, reported as test "(module)" of that file. Without -v,
// only the summary lines name a skip, and they name a file: package pyPackage, test the
// file.
func ParseSkips(out []byte, pyPackage string) ([]SkipResult, int, error) {
	var skips, summary []SkipResult
	nodes := false
	ran := 0
	for _, l := range strings.Split(string(out), "\n") {
		l = strings.TrimSpace(l)
		if m := pytestNodeSkip.FindStringSubmatch(l); m != nil {
			skips = append(skips, SkipResult{path.Join(pyPackage, m[1]), m[2]})
			nodes = true
			continue
		}
		if m := pytestSkip.FindStringSubmatch(l); m != nil {
			summary = append(summary, SkipResult{pyPackage, m[1]})
			continue
		}
		if !strings.HasPrefix(l, "{") {
			continue
		}
		var e struct{ Action, Package, Test string }
		if err := json.Unmarshal([]byte(l), &e); err != nil {
			return nil, 0, err
		}
		if e.Test == "" {
			continue
		}
		switch e.Action {
		case "skip":
			skips = append(skips, SkipResult{e.Package, e.Test})
			ran++
		case "pass", "fail":
			ran++
		}
	}
	if !nodes {
		return append(skips, summary...), ran, nil
	}
	named := map[string]bool{}
	for _, sk := range skips {
		named[sk.Package] = true
	}
	for _, sm := range summary {
		if file := path.Join(pyPackage, sm.Test); !named[file] {
			skips = append(skips, SkipResult{file, "(module)"})
			named[file] = true
		}
	}
	return skips, ran, nil
}
