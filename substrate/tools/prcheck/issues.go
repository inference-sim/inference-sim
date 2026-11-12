package main

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The cited-issue check (X28). An issue in a BLIS repository that is cited as the reason
// for a refusal, for a refused or not-modelled item in fields.json or surface.json, or in a
// code comment that marks a gap or a divergence, stays open; it closes only in the change
// that removes or updates every citation of it.
//
// A citation is any of these, in a tracked file:
//   - in fields.json or surface.json, an "issue" value "<owner>/<repo>#<n>";
//   - on a line that contains a refusal or gap marker (refuse, unsupported, not modelled,
//     not_modelled, gap, divergence, diverges, TODO, FIXME), a reference
//     <owner>/<repo>#<n>, github.com/<owner>/<repo>/issues/<n>, or, with the repository
//     known, #<n>.
//
// The nightly job resolves each cited issue and fails on a closed one. The pull-request
// check fails a pull request whose body closes an issue still cited in the tree.

// Issue names one issue.
type Issue struct {
	Repo string // owner/name
	N    int
}

func (i Issue) String() string { return fmt.Sprintf("%s#%d", i.Repo, i.N) }

// Citation is one place an issue is cited.
type Citation struct {
	Issue Issue
	File  string
	Line  int
}

var (
	issueURL    = regexp.MustCompile(`github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)`)
	issueRef    = regexp.MustCompile(`\b([\w.-]+/[\w.-]+)#(\d+)\b`)
	bareRef     = regexp.MustCompile(`(?:^|[\s(\[,;])#(\d+)\b`)
	gapMarker   = regexp.MustCompile(`(?i)\b(refus\w*|unsupported|not[ _]modelled|gap|diverg\w*|TODO|FIXME)\b`)
	issueField  = regexp.MustCompile(`"issue"\s*:\s*"([\w.-]+/[\w.-]+)#(\d+)"`)
	blisRepoRef = regexp.MustCompile(`^inference-sim/`)
)

// Citations returns the X28 citations in one file. repo is the file's own repository
// (owner/name), used for bare #<n> references.
func Citations(repo, file string, src []byte) []Citation {
	var out []Citation
	add := func(r string, n string, line int) {
		var num int
		fmt.Sscan(n, &num)
		if !blisRepoRef.MatchString(r) {
			return // only issues in a BLIS repository are covered
		}
		out = append(out, Citation{Issue{r, num}, file, line})
	}
	list := strings.HasSuffix(file, "fields.json") || strings.HasSuffix(file, "surface.json")
	for i, line := range strings.Split(string(src), "\n") {
		ln := i + 1
		if list {
			for _, m := range issueField.FindAllStringSubmatch(line, -1) {
				add(m[1], m[2], ln)
			}
			continue
		}
		if !gapMarker.MatchString(line) {
			continue
		}
		for _, m := range issueURL.FindAllStringSubmatch(line, -1) {
			add(m[1], m[2], ln)
		}
		rest := issueURL.ReplaceAllString(line, "")
		for _, m := range issueRef.FindAllStringSubmatch(rest, -1) {
			add(m[1], m[2], ln)
		}
		rest = issueRef.ReplaceAllString(rest, "")
		for _, m := range bareRef.FindAllStringSubmatch(rest, -1) {
			add(repo, m[1], ln)
		}
	}
	return out
}

// IssueState answers whether an issue is open.
type IssueState func(Issue) (open bool, err error)

// CheckCitedOpen returns a finding for every citation of an issue that is not open.
func CheckCitedOpen(cites []Citation, state IssueState) []string {
	cache := map[Issue]bool{}
	var bad []string
	for _, c := range cites {
		open, ok := cache[c.Issue]
		if !ok {
			var err error
			open, err = state(c.Issue)
			if err != nil {
				bad = append(bad, fmt.Sprintf("%s:%d: %s: %v", c.File, c.Line, c.Issue, err))
				continue
			}
			cache[c.Issue] = open
		}
		if !open {
			bad = append(bad, fmt.Sprintf("%s:%d: cites %s, which is closed; a cited issue stays open until the change that removes every citation of it (X28)", c.File, c.Line, c.Issue))
		}
	}
	return bad
}

// closesLine matches the start of a line that closes issues: an optional list or quote
// prefix, one of GitHub's closing keywords, an optional colon, and the references after it.
var closesLine = regexp.MustCompile(`(?i)^\s*(?:(?:[-*+]|>)\s*)*(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+(.*)$`)

// closesRef is one reference, own-repository (#12) or cross-repository (owner/repo#12), at
// the start of s; a list continues after a comma, "and" or an ampersand.
var closesRef = regexp.MustCompile(`^(?:([\w.-]+/[\w.-]+))?#(\d+)\b(?:\s*(?:,|&|\band\b)\s*)?`)

// ClosedBy returns the issues a pull-request body closes, with repo the body's repository.
// Text in an HTML comment, in a fenced code block, or after a comment that is never closed
// does not count, as GitHub does not act on it.
func ClosedBy(repo, body string) []Issue {
	var out []Issue
	for _, line := range strings.Split(visibleText(body), "\n") {
		m := closesLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		rest := m[1]
		for {
			r := closesRef.FindStringSubmatch(rest)
			if r == nil {
				break
			}
			at := r[1]
			if at == "" {
				at = repo
			}
			n, err := strconv.Atoi(r[2])
			if err != nil {
				// The pattern matched digits only, so this is a number too large for an int.
				rest = rest[len(r[0]):]
				continue
			}
			out = append(out, Issue{at, n})
			rest = rest[len(r[0]):]
		}
	}
	return out
}

// visibleText removes the parts of a pull-request body that GitHub renders as nothing or as
// code: HTML comments (an unterminated one runs to the end) and fenced code blocks (an
// unterminated one likewise).
func visibleText(body string) string {
	body = htmlComment.ReplaceAllString(body, "")
	if i := strings.Index(body, "<!--"); i >= 0 {
		body = body[:i]
	}
	var out []string
	var fence string
	for _, line := range strings.Split(body, "\n") {
		t := strings.TrimLeft(line, " \t>")
		switch {
		case fence != "":
			if strings.HasPrefix(t, fence) {
				fence = ""
			}
		case strings.HasPrefix(t, "```"), strings.HasPrefix(t, "~~~"):
			fence = t[:3]
		default:
			out = append(out, line)
		}
	}
	return strings.Join(out, "\n")
}

// CheckClosesUncited returns a finding for each issue the pull request closes that the
// merge tree still cites.
func CheckClosesUncited(closes []Issue, cites []Citation) []string {
	at := map[Issue][]string{}
	for _, c := range cites {
		at[c.Issue] = append(at[c.Issue], fmt.Sprintf("%s:%d", c.File, c.Line))
	}
	var bad []string
	for _, i := range closes {
		if where := at[i]; len(where) > 0 {
			sort.Strings(where)
			bad = append(bad, fmt.Sprintf("this pull request closes %s, which is still cited at %s; remove or update every citation in the same change (X28)", i, strings.Join(where, ", ")))
		}
	}
	return bad
}
