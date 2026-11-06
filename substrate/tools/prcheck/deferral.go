package main

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// The X20 deferral. The release plan lets the commit-policy check defer two findings for a
// stated time (plan §3.3, "Deferring signing and sign-off"): that a commit is not signed,
// and that it has no Signed-off-by line. Both need a person, so agent-written commits carry
// them until the person acts. Every other X20 finding still fails: a Signed-off-by naming a
// tool, a Co-authored-by naming a tool, more than one Assisted-by line.
//
// The deferral is one file in blis-substrate, .github/x20-deferral.yaml, which each
// repository reads from the substrate commit its shared workflow calls:
//
//	issue: inference-sim/blis-substrate#12   # the issue that owns the work
//	added: 2026-11-05                         # the day the file was added; not in the future
//	expires: 2026-12-05                       # at most 30 days after added (plan §3.3)
//	reason: one line
//
// After the date the findings fail again, and so does the check itself, until the file is
// removed or renewed with a reason.

// Deferral is the deferral file.
type Deferral struct {
	Issue   string `yaml:"issue"`
	Added   string `yaml:"added"`
	Expires string `yaml:"expires"`
	Reason  string `yaml:"reason"`
}

// maxDeferral is the longest a deferral may last, counted from the day it was added.
const maxDeferral = 30 * 24 * time.Hour

// deferrable reports whether a commit-policy finding is one the deferral covers.
func deferrable(finding string) bool {
	return strings.HasSuffix(finding, "the commit is not signed (X20)") ||
		strings.Contains(finding, "no Signed-off-by line; the person who takes responsibility")
}

// ApplyDeferral splits findings into those that fail and those the deferral covers on the
// given day. An empty src means no deferral. A malformed or expired file defers nothing and
// adds a finding of its own.
func ApplyDeferral(src []byte, today time.Time, findings []string) (fail, deferred []string) {
	if len(bytes.TrimSpace(src)) == 0 {
		return findings, nil
	}
	var d Deferral
	dec := yaml.NewDecoder(bytes.NewReader(src))
	dec.KnownFields(true)
	if err := dec.Decode(&d); err != nil {
		return append(append([]string{}, findings...), fmt.Sprintf("x20-deferral.yaml: %v", err)), nil
	}
	var bad []string
	if !issueName.MatchString(d.Issue) {
		bad = append(bad, fmt.Sprintf("x20-deferral.yaml: issue %q is not <owner>/<repo>#<n>; the deferral has an owning issue (X20)", d.Issue))
	}
	if strings.TrimSpace(d.Reason) == "" {
		bad = append(bad, "x20-deferral.yaml: needs a reason")
	}
	added, aerr := time.Parse("2006-01-02", d.Added)
	if aerr != nil {
		bad = append(bad, fmt.Sprintf("x20-deferral.yaml: added %q is not a date YYYY-MM-DD", d.Added))
	} else if added.After(today) {
		bad = append(bad, fmt.Sprintf("x20-deferral.yaml: added %s is in the future", d.Added))
	}
	exp, err := time.Parse("2006-01-02", d.Expires)
	if err != nil {
		bad = append(bad, fmt.Sprintf("x20-deferral.yaml: expires %q is not a date YYYY-MM-DD", d.Expires))
	} else {
		if aerr == nil && exp.Sub(added) > maxDeferral {
			bad = append(bad, fmt.Sprintf("x20-deferral.yaml: expires %s is more than 30 days after added %s (plan §3.3)", d.Expires, d.Added))
		}
		if !today.Before(exp) {
			bad = append(bad, fmt.Sprintf("x20-deferral.yaml: expired on %s; sign and sign off the commits, or renew it with a reason (X20)", d.Expires))
		}
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
