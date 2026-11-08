package main

import (
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// The known-bad sweep (X18): every check in the registry names a known-bad input that must
// make it fail. For a check run in this repository, known_bad names the Go test that feeds
// the check that input, "<package dir>: <TestName>", and passes only if the check refuses
// it. The nightly job runs every such test and fails if any does not pass, which is the
// case when a check stopped refusing its input. A check with status ci that names no
// known_bad is reported too, until X18's target milestone, after which it fails.

// KnownBadTests returns, per package directory, the tests the registry names for repo.
func KnownBadTests(reg *invariants.Registry, repo string) (map[string][]string, []string) {
	tests := map[string][]string{}
	var missing []string
	seen := map[string]bool{}
	for _, inv := range reg.Invariants {
		for _, c := range inv.Checks {
			if c.Repo != repo {
				continue
			}
			if c.Runs == invariants.RunsPractice {
				continue // a maintainer practice has no program to feed a bad input to
			}
			if c.KnownBad == "" {
				if c.Status == invariants.StatusCI {
					missing = append(missing, fmt.Sprintf("%s: %q names no known-bad input (X18)", inv.ID, c.Check))
				}
				continue
			}
			pkg, test, ok := strings.Cut(c.KnownBad, ": ")
			if !ok || !strings.HasPrefix(test, "Test") {
				missing = append(missing, fmt.Sprintf("%s: known_bad %q is not \"<package dir>: <TestName>\"", inv.ID, c.KnownBad))
				continue
			}
			if !seen[pkg+" "+test] {
				seen[pkg+" "+test] = true
				tests[pkg] = append(tests[pkg], test)
			}
		}
	}
	for _, t := range tests {
		sort.Strings(t)
	}
	sort.Strings(missing)
	return tests, missing
}

// runKnownBad runs each package's named tests with go test -run, requiring each to run.
func runKnownBad(tests map[string][]string) []string {
	var bad []string
	pkgs := make([]string, 0, len(tests))
	for p := range tests {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)
	for _, p := range pkgs {
		pattern := "^(" + strings.Join(tests[p], "|") + ")$"
		out, err := exec.Command("go", "test", "-count=1", "-v", "-run", pattern, "./"+p).CombinedOutput()
		if err != nil {
			bad = append(bad, fmt.Sprintf("%s: a known-bad test failed, so a check no longer refuses its input (X18):\n%s", p, out))
			continue
		}
		for _, t := range tests[p] {
			if !strings.Contains(string(out), "--- PASS: "+t+" ") {
				bad = append(bad, fmt.Sprintf("%s: %s did not run; the registry names a test that does not exist (X18)", p, t))
			}
		}
	}
	return bad
}
