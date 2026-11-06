package main

import (
	"fmt"
	"html"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// Markers delimit the generated part of a file that people also edit by hand. The text
// between them is rendered from the registry; everything outside them is left alone.
const (
	beginMarker = "<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->"
	endMarker   = "<!-- END GENERATED: invariants -->"
)

var idNumber = regexp.MustCompile(`[0-9]+$`)

// families groups the registry's records by family, in design order.
func families(reg *invariants.Registry) [][]invariants.Invariant {
	var out [][]invariants.Invariant
	for _, f := range invariants.Families {
		var g []invariants.Invariant
		for _, inv := range reg.Invariants {
			if strings.HasPrefix(inv.ID, f.Prefix) && idNumber.MatchString(strings.TrimPrefix(inv.ID, f.Prefix)) {
				g = append(g, inv)
			}
		}
		out = append(out, g)
	}
	return out
}

// renderAppendix writes the design's Appendix A, the invariant index: one row per family,
// with its identifiers, the repositories its checks run in, and the sections defining it.
func renderAppendix(reg *invariants.Registry) string {
	var b strings.Builder
	b.WriteString("<!-- Generated from blis-substrate/contracts/invariants by go run ./tools/invariants render. Do not edit. -->\n")
	b.WriteString("<div class=\"tbl\"><table>\n")
	b.WriteString("<thead><tr><th>Identifiers</th><th>Count</th><th>Checks run in</th><th>Defined in</th></tr></thead>\n<tbody>\n")
	for _, g := range families(reg) {
		if len(g) == 0 {
			continue
		}
		repos := map[string]bool{}
		secs := map[string]bool{}
		var secList []string
		for _, inv := range g {
			for _, r := range inv.Scope {
				repos[r] = true
			}
			if !secs[inv.Section] {
				secs[inv.Section] = true
				secList = append(secList, inv.Section)
			}
		}
		var rs []string
		for r := range repos {
			rs = append(rs, "<code>"+r+"</code>")
		}
		sort.Strings(rs)
		fmt.Fprintf(&b, "<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>\n",
			idRange(g), len(g), strings.Join(rs, ", "), strings.Join(secList, ", "))
	}
	b.WriteString("</tbody></table></div>\n")
	return b.String()
}

func idRange(g []invariants.Invariant) string {
	first, last := g[0].ID, g[len(g)-1].ID
	if first == last {
		return `<span class="inv">` + first + `</span>`
	}
	return `<span class="inv">` + first + `</span>–<span class="inv">` + idNumber.FindString(last) + `</span>`
}

// renderStatusTable writes the rows of the release plan's §3.2 table: identifier, the
// statement's first sentence, status, the milestone by which every check is in CI, and the
// note.
func renderStatusTable(reg *invariants.Registry) string {
	var b strings.Builder
	b.WriteString("<!-- Generated from blis-substrate/contracts/invariants by go run ./tools/invariants render. Do not edit. -->\n")
	b.WriteString("<div class=\"tbl\"><table>\n<thead><tr><th>Invariant</th><th>Statement, first sentence</th><th>Status</th><th>In CI by</th><th>Note</th></tr></thead>\n<tbody>\n")
	for _, inv := range reg.Invariants {
		by := "now"
		if inv.Target.Milestone != nil {
			by = fmt.Sprintf("M%d", *inv.Target.Milestone)
		}
		fmt.Fprintf(&b, "<tr><td><span class=\"inv\">%s</span></td><td>%s</td><td><code>%s</code></td><td>%s</td><td>%s</td></tr>\n",
			inv.ID, html.EscapeString(firstSentence(inv.Statement)), inv.Status, by, html.EscapeString(inv.Note))
	}
	b.WriteString("</tbody></table></div>\n")
	return b.String()
}

// firstSentence returns the statement up to the first full stop outside parentheses that
// ends a sentence, with a trailing colon turned into a full stop. It is the plan's rule.
func firstSentence(s string) string {
	depth := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			if depth > 0 {
				depth--
			}
		case '.':
			if depth == 0 && (i+1 == len(s) || s[i+1] == ' ') {
				return s[:i+1]
			}
		}
	}
	return strings.TrimSuffix(s, ":") + "."
}

// applies reports whether an invariant has a check in the repository.
func applies(inv invariants.Invariant, repo string) bool {
	for _, r := range inv.Scope {
		if r == repo {
			return true
		}
	}
	return false
}

// renderAgents writes the AGENTS.md section for one repository: every invariant whose
// checks run there, with each of its checks in that repository and their status, and the
// review questions that can be added to its pull requests.
func renderAgents(reg *invariants.Registry, repo string) string {
	var b strings.Builder
	b.WriteString(beginMarker + "\n")
	fmt.Fprintf(&b, "## Invariants checked in %s\n\n", repo)
	b.WriteString("Generated from the invariant registry, `blis-substrate/contracts/invariants`, at the\n")
	b.WriteString("substrate version this repository pins (design §13.1). Each line is an invariant this\n")
	b.WriteString("repository runs a check for: its identifier, the first sentence of its statement, its\n")
	b.WriteString("status, and each of its checks here. `ci` means the check runs in CI; `new` means it does not\n")
	b.WriteString("yet, and the target milestone says when it must. A pull request cites the identifiers it\n")
	b.WriteString("keeps or introduces; CI refuses an identifier that does not exist (X14).\n\n")
	for _, inv := range reg.Invariants {
		if !applies(inv, repo) {
			continue
		}
		target := "target reached"
		if inv.Target.Milestone != nil {
			target = fmt.Sprintf("target %s by M%d", inv.Target.Status, *inv.Target.Milestone)
		}
		fmt.Fprintf(&b, "- **%s** (%s; %s) %s\n", inv.ID, inv.Status, target, firstSentence(inv.Statement))
		for _, c := range inv.Checks {
			if c.Repo != repo {
				continue
			}
			line := fmt.Sprintf("  - `%s`, %s: %s", c.Status, c.Runs, c.Check)
			if c.Test != "" {
				line += fmt.Sprintf(" (`%s`)", c.Test)
			}
			b.WriteString(line + "\n")
		}
	}
	qs := reg.Questions(repo)
	if len(qs) > 0 {
		b.WriteString("\n### Review questions\n\n")
		b.WriteString("CI adds each question below to a pull request that changes one of its paths, as a\n")
		b.WriteString("required checklist item. A reviewer with write access answers it yes or no with a one-line reason; an agent\n")
		b.WriteString("may propose an answer but never gives one (design §13.7).\n\n")
		for _, q := range qs {
			fmt.Fprintf(&b, "- **%s** %s Reviewer: %s. When: %s. %s\n", q.ID, q.Question, q.Reviewer, q.When, triggerText(q.ID, q.Trigger))
		}
	}
	b.WriteString(endMarker + "\n")
	return b.String()
}

func triggerText(id string, t invariants.Trigger) string {
	if t.Manual {
		return "Added by a reviewer with the label `review:" + id + "`."
	}
	s := "Paths: `" + strings.Join(t.Paths, "`, `") + "`"
	if len(t.Exclude) > 0 {
		s += ", except `" + strings.Join(t.Exclude, "`, `") + "`"
	}
	return s + "."
}

// renderTemplate writes the pull-request template for one repository. The sections are
// those of the brief every BLIS pull request follows; the change-class field is DEV-12's.
func renderTemplate(reg *invariants.Registry, repo string) string {
	var b strings.Builder
	b.WriteString(beginMarker + "\n")
	b.WriteString(`## Summary

<!-- One to three sentences: what this pull request does and why. -->

## Change class

<!-- Exactly one, by what the change does (DEV-12, design §8.9). A change that is both is split. -->

- [ ] Verification: the model's behaviour must not change (a refactor, a speed-up, or a fix that
      makes the code match an oracle or a written rule).
- [ ] Validation: it changes what is modelled (a pricing form, a coefficient, an input model).

## Invariants

<!-- The identifiers this change keeps or introduces, one per line, such as "- SUB-6". CI fails
if an identifier does not exist (X14). The list of identifiers is in AGENTS.md. -->

## Closes

<!-- One "Closes <owner>/<repo>#<n>" line per issue. -->

## Tests

<!-- Commands run and their results. For each behaviour change, the test that fails without
it (X13). -->

## Evidence

<!-- Required for a validation change, and for a verification change that moves a published
number: the success criteria committed in advance, the held-out evidence, the price-change
report and the scoring harness's figures on published units (X13, VAL-6). Otherwise write
"Not needed: no published number moves." -->

## Depends on

<!-- Other pull requests, by URL and commit. "None" if none. -->

## Not in this pull request

<!-- What remains, and why. -->

## Review questions

<!-- CI adds a required checklist item here for each review question whose paths this pull
request changes (design §13.2). A reviewer with write access answers each one: tick it, and write "yes" or
"no" with a one-line reason. -->

## Checklist

`)
	for _, line := range checklist(reg, repo) {
		b.WriteString("- [ ] " + line + "\n")
	}
	b.WriteString(endMarker + "\n")
	return b.String()
}

// checklist returns the items every pull request in the repository confirms: the process
// invariants whose checks run there and that a contributor keeps by hand.
func checklist(reg *invariants.Registry, repo string) []string {
	items := []struct{ id, text string }{
		{"X12", "No third-party data committed; each derived value cites its source by URL and commit or hash; no private names."},
		{"X13", "Each behaviour change has a test that fails without it."},
		{"X14", "The commit hashes listed in this description are the branch heads."},
		{"X20", "Each commit written with an agent's help has one Assisted-by line and no Co-authored-by line naming a tool; the person responsible adds the DCO sign-off."},
		{"X24", "Every review finding is fixed, filed as a linked issue, or rejected with a reason."},
		{"X26", "Every path, symbol, issue and supported-feature statement in changed documentation is generated or checked."},
	}
	var out []string
	for _, it := range items {
		inv, ok := reg.Get(it.id)
		if ok && applies(inv, repo) {
			out = append(out, it.id+": "+it.text)
		}
	}
	return out
}

// splice replaces the generated block in an existing file, or appends one if the file has
// none. It refuses a file with unbalanced markers.
func splice(existing, block string) (string, error) {
	b := strings.Index(existing, beginMarker)
	e := strings.Index(existing, endMarker)
	switch {
	case b < 0 && e < 0:
		if existing != "" && !strings.HasSuffix(existing, "\n") {
			existing += "\n"
		}
		if existing != "" {
			existing += "\n"
		}
		return existing + block, nil
	case b < 0 || e < 0 || e < b || strings.Count(existing, beginMarker) != 1 || strings.Count(existing, endMarker) != 1:
		return "", fmt.Errorf("the generated block's markers are missing, repeated or out of order")
	}
	return existing[:b] + block + existing[e+len(endMarker)+1:], nil
}

// extract returns the generated block of a file, markers included.
func extract(s string) (string, bool) {
	b := strings.Index(s, beginMarker)
	e := strings.Index(s, endMarker)
	if b < 0 || e < b {
		return "", false
	}
	return s[b : e+len(endMarker)+1], true
}
