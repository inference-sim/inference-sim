package main

import (
	"bytes"
	"encoding/json"
	"fmt"
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
//	  expires: 2026-12-01       # at most 30 days after the entry was added
//	  reason: one line
//
// The required "main is green" check reads the same list: while a test fails on main, a
// pull request merges only if it names that test's issue, or the test is quarantined.

// QuarantineEntry is one entry of the quarantine list.
type QuarantineEntry struct {
	Test    string `yaml:"test"`
	Package string `yaml:"package"`
	Issue   string `yaml:"issue"`
	Expires string `yaml:"expires"`
	Reason  string `yaml:"reason"`
}

const maxQuarantine = 30 * 24 * time.Hour

var issueName = regexp.MustCompile(`^[\w.-]+/[\w.-]+#\d+$`)

// CheckQuarantine parses the list and returns its violations on the given day. added maps
// each entry's test name to the day it was first committed (from git log), when known.
func CheckQuarantine(src []byte, today time.Time, added map[string]time.Time) ([]QuarantineEntry, []string) {
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
		exp, err := time.Parse("2006-01-02", e.Expires)
		if err != nil {
			bad = append(bad, fmt.Sprintf("%s: expires %q is not a date YYYY-MM-DD (X25)", at, e.Expires))
			continue
		}
		if !today.Before(exp) {
			bad = append(bad, fmt.Sprintf("%s: expired on %s; fix the test or remove it, and the failure blocks merges again (X25)", at, e.Expires))
		}
		if a, ok := added[e.Test]; ok && exp.Sub(a) > maxQuarantine {
			bad = append(bad, fmt.Sprintf("%s: expires %s, more than 30 days after it was added on %s (X25)", at, e.Expires, a.Format("2006-01-02")))
		}
	}
	return list, bad
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

var pytestSkip = regexp.MustCompile(`^SKIPPED \[\d+\] ([^:]+):\d+:`)

// ParseSkips reads the skips from go test -json output (one JSON object per line) or
// from pytest -rs output ("SKIPPED [n] file:line: reason").
func ParseSkips(out []byte, pyPackage string) ([]SkipResult, int, error) {
	var skips []SkipResult
	ran := 0
	for _, l := range strings.Split(string(out), "\n") {
		l = strings.TrimSpace(l)
		if m := pytestSkip.FindStringSubmatch(l); m != nil {
			skips = append(skips, SkipResult{pyPackage, m[1]})
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
	return skips, ran, nil
}
