package main

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"time"

	validate "github.com/inference-sim/inference-sim/substrate/contracts/check"
)

// The pin-by-commit deferral. While a pull request is a draft, the router fork may be pinned by
// commit, because its tag comes after review and merge (release plan §3.1). The lint reports that
// finding, the fork at a pseudo-version, as deferred while .github/modlint-deferral.yaml holds.
// Every other finding still fails.
//
//	issue: inference-sim/blis-substrate#14   # the issue that owns the work
//	added: 2026-11-05                         # the day the file was added; not in the future
//	expires: 2026-12-05                       # at most 30 days after added (plan §3.3)
//	modules: [llm-d-router]                   # the BLIS modules whose commit pin is deferred
//	reason: one line
//
// A malformed, unowned, over-long, future-dated or expired file defers nothing and adds a
// finding of its own. Only a commit pin of a listed module under github.com/inference-sim/
// is deferred.

type deferral struct {
	Issue   string   `yaml:"issue"`
	Added   string   `yaml:"added"`
	Expires string   `yaml:"expires"`
	Modules []string `yaml:"modules"`
	Reason  string   `yaml:"reason"`
}

var issueName = regexp.MustCompile(`^[\w.-]+/[\w.-]+#\d+$`)

// maxDeferral is the longest a deferral may last, counted from the day it was added.
const maxDeferral = 30 * 24 * time.Hour

// deferrableModules are the BLIS modules pinned by commit. Since M3.2 the substrate and the engines
// are directories of this module, so the router fork is the only module left to pin.
var deferrableModules = map[string]bool{"llm-d-router": true}

// deferrable reports whether a finding is a listed BLIS module pinned by commit.
func deferrable(finding string, modules []string) bool {
	// The router fork is pinned by a replace to a commit while drafts are open, and is covered
	// when the deferral lists llm-d-router.
	if strings.Contains(finding, "replace target version") && strings.HasSuffix(finding, "is not a -blis.N tag (X11, LIF-5)") {
		for _, m := range modules {
			if m == "llm-d-router" && isPseudoQuoted(finding) {
				return true
			}
		}
		return false
	}
	// The router through its one permitted replace, to a pseudo-version of the fork: LIF-2's
	// selected-version finding for the upstream router path. It is covered when the deferral
	// lists llm-d-router and the replace target is the fork itself at a commit (a pseudo-version).
	// A replace to any other module, or to a version that is no pseudo-version, still fails.
	const routerHead = upstreamRouter + " resolves to " + forkRouter + " "
	const routerTail = "; it must resolve to " + forkRouter + " at a -blis.N tag (LIF-2)"
	if strings.HasPrefix(finding, routerHead) && strings.HasSuffix(finding, routerTail) {
		v := strings.TrimSuffix(strings.TrimPrefix(finding, routerHead), routerTail)
		for _, m := range modules {
			if m == "llm-d-router" && isPseudo(v) {
				return true
			}
		}
		return false
	}
	return false
}

// ApplyDeferral splits findings into those that fail and those the deferral covers on the
// given day.
func ApplyDeferral(src []byte, today time.Time, findings []string) (fail, deferred []string) {
	if len(bytes.TrimSpace(src)) == 0 {
		return findings, nil
	}
	var d deferral
	if err := validate.DecodeStrict("modlint-deferral.yaml", src, &d); err != nil {
		return append(append([]string{}, findings...), fmt.Sprintf("modlint-deferral.yaml: %v", err)), nil
	}
	var bad []string
	if !issueName.MatchString(d.Issue) {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: issue %q is not <owner>/<repo>#<n>; the deferral has an owning issue", d.Issue))
	}
	if strings.TrimSpace(d.Reason) == "" {
		bad = append(bad, "modlint-deferral.yaml: needs a reason")
	}
	if len(d.Modules) == 0 {
		bad = append(bad, "modlint-deferral.yaml: modules lists no module; name the BLIS modules whose commit pin is deferred")
	}
	for _, m := range d.Modules {
		if !deferrableModules[m] {
			bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: modules names %q; only llm-d-router is pinned by commit", m))
		}
	}
	added, aerr := time.Parse("2006-01-02", d.Added)
	if aerr != nil {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: added %q is not a date YYYY-MM-DD", d.Added))
	} else if added.After(today) {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: added %s is in the future", d.Added))
	}
	exp, err := time.Parse("2006-01-02", d.Expires)
	if err == nil && aerr == nil && exp.Sub(added) > maxDeferral {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: expires %s is more than 30 days after added %s (release plan §3.3)", d.Expires, d.Added))
	}
	if err != nil {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: expires %q is not a date YYYY-MM-DD", d.Expires))
	} else if !today.Before(exp) {
		bad = append(bad, fmt.Sprintf("modlint-deferral.yaml: expired on %s; pin the modules to release tags, or renew it with a reason (X11)", d.Expires))
	}
	if len(bad) > 0 {
		return append(append([]string{}, findings...), bad...), nil
	}
	for _, f := range findings {
		if deferrable(f, d.Modules) {
			deferred = append(deferred, f)
		} else {
			fail = append(fail, f)
		}
	}
	return fail, deferred
}

// isPseudoQuoted reports whether the version a replace finding quotes is a pseudo-version.
func isPseudoQuoted(finding string) bool {
	i := strings.Index(finding, `version "`)
	if i < 0 {
		return false
	}
	rest := finding[i+len(`version "`):]
	j := strings.Index(rest, `"`)
	return j > 0 && isPseudo(rest[:j])
}
