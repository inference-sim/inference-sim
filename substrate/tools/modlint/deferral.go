package main

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// The pin-by-commit deferral. While the M1 pull requests are drafts, each pins the BLIS
// modules by commit, because tags come after review and merge (release plan §3.1). The lint
// reports that one finding, "resolves to <pseudo-version>, which is not a release tag", as
// deferred while .github/modlint-deferral.yaml holds. Every other finding still fails.
//
//	issue: inference-sim/blis-substrate#14   # the issue that owns the work
//	expires: 2026-12-05                       # at most 30 days after the file was added
//	reason: one line
//
// A malformed, unowned or expired file defers nothing and adds a finding of its own.

type deferral struct {
	Issue   string `yaml:"issue"`
	Expires string `yaml:"expires"`
	Reason  string `yaml:"reason"`
}

var issueName = regexp.MustCompile(`^[\w.-]+/[\w.-]+#\d+$`)

// deferrable reports whether a finding is a BLIS module pinned by commit.
func deferrable(finding string) bool {
	return strings.Contains(finding, "which is not a release tag (X11, LIF-2)")
}

// ApplyDeferral splits findings into those that fail and those the deferral covers on the
// given day.
func ApplyDeferral(src []byte, today time.Time, findings []string) (fail, deferred []string) {
	if len(bytes.TrimSpace(src)) == 0 {
		return findings, nil
	}
	var d deferral
	dec := yaml.NewDecoder(bytes.NewReader(src))
	dec.KnownFields(true)
	if err := dec.Decode(&d); err != nil {
		return append(append([]string{}, findings...), fmt.Sprintf("modlint-deferral.yaml: %v", err)), nil
	}
	var bad []string
	if !issueName.MatchString(d.Issue) {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: issue %q is not <owner>/<repo>#<n>; the deferral has an owning issue", d.Issue))
	}
	if strings.TrimSpace(d.Reason) == "" {
		bad = append(bad, "modlint-deferral.yaml: needs a reason")
	}
	exp, err := time.Parse("2006-01-02", d.Expires)
	if err != nil {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: expires %q is not a date YYYY-MM-DD", d.Expires))
	} else if !today.Before(exp) {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: expired on %s; pin the modules to release tags, or renew it with a reason (X11)", d.Expires))
	}
	if len(bad) > 0 {
		return append(append([]string{}, findings...), bad...), nil
	}
	for _, f := range findings {
		if deferrable(f) {
			deferred = append(deferred, f)
		} else {
			fail = append(fail, f)
		}
	}
	return fail, deferred
}
