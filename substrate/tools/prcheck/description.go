package main

import (
	"fmt"
	"regexp"
	"strings"
)

// The description check (X14): the commit hashes a pull request's description lists are
// the heads of the branches they name, the pull request merges cleanly, and every
// invariant it cites exists (the last part is tools/invariants cited).
//
// A listed hash is a full 40-hex commit. On a line that also names a pull request (a URL
// .../pull/<n>, or owner/repo#<n>), the hash must be that pull request's head. On any
// other line it must be the head of this pull request or of its base. A short hash cannot
// be checked and is refused, so a reader can always check what the description claims.

var (
	fullHash = regexp.MustCompile(`\b[0-9a-f]{40}\b`)
	hexRun   = regexp.MustCompile(`[0-9A-Za-z_]+`)
	// notCommit are the spans in which a hex token is not an abbreviated commit: a URL, a
	// sha256 digest, and a Go pseudo-version (v0.0.0-<14 digits>-<12 hex>).
	notCommit = regexp.MustCompile(`(?i)https?://\S+|sha256:[0-9a-f]+|v\d+\.\d+\.\d+-(?:[0-9a-z.]+\.)?\d{14}-[0-9a-f]{12}`)
	// urlRevision is the revision a GitHub commit, blob, tree or commits URL names. A URL is
	// not otherwise scanned, but an abbreviated revision there is as unverifiable as one in
	// prose.
	urlRevision = regexp.MustCompile(`(?i)https?://github\.com/[\w.-]+/[\w.-]+/(?:commit|commits|blob|tree)/([0-9a-f]{7,39})(?:[^0-9a-z]|$)`)
	prURL       = regexp.MustCompile(`https://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)`)
	prRef       = regexp.MustCompile(`\b([\w.-]+/[\w.-]+)#(\d+)\b`)
)

// PRHeads answers "what is the head commit of pull request n in repo".
type PRHeads func(repo string, n int) (string, error)

// DescriptionInput is what the description check reads.
type DescriptionInput struct {
	Body       string
	Repo       string // owner/name of this pull request's repository
	Head, Base string // this pull request's head and base commits
	Heads      PRHeads
	Merges     bool // git merge-tree of base and head reports no conflict
}

// CheckDescription returns the X14 violations in a pull-request description.
func CheckDescription(in DescriptionInput) []string {
	var bad []string
	if !in.Merges {
		bad = append(bad, "the pull request does not merge cleanly into its base (X14)")
	}
	for i, line := range strings.Split(stripComments(in.Body), "\n") {
		ln := i + 1
		for _, h := range shortHashes(line) {
			bad = append(bad, fmt.Sprintf("line %d: %s is a short hash; list full 40-character hashes so they can be checked (X14)", ln, h))
		}
		hashes := fullHash.FindAllString(line, -1)
		if len(hashes) == 0 {
			continue
		}
		type ref struct {
			repo string
			n    int
		}
		var refs []ref
		for _, m := range prURL.FindAllStringSubmatch(line, -1) {
			var n int
			fmt.Sscan(m[2], &n)
			refs = append(refs, ref{m[1], n})
		}
		if len(refs) == 0 {
			for _, m := range prRef.FindAllStringSubmatch(line, -1) {
				var n int
				fmt.Sscan(m[2], &n)
				refs = append(refs, ref{m[1], n})
			}
		}
		if len(refs) == 0 {
			for _, h := range hashes {
				if h != in.Head && h != in.Base {
					bad = append(bad, fmt.Sprintf("line %d: %s is neither this pull request's head %s nor its base %s; name the pull request it belongs to (X14)", ln, h, in.Head, in.Base))
				}
			}
			continue
		}
		if len(refs) != len(hashes) {
			bad = append(bad, fmt.Sprintf("line %d: %d pull requests and %d hashes; list one hash per pull request (X14)", ln, len(refs), len(hashes)))
			continue
		}
		for k, r := range refs {
			head, err := in.Heads(r.repo, r.n)
			if err != nil {
				bad = append(bad, fmt.Sprintf("line %d: %s#%d: %v", ln, r.repo, r.n, err))
				continue
			}
			if head != hashes[k] {
				bad = append(bad, fmt.Sprintf("line %d: %s#%d lists %s, but its head is %s (X14)", ln, r.repo, r.n, hashes[k], head))
			}
		}
	}
	return bad
}

// shortHashes returns the abbreviated commit-like tokens of a line, whatever the words
// around them: a maximal run of 7 to 39 lowercase hex characters holding at least one digit
// (so decade, facade and defaced are words) and at least one letter a-f (so a date or an
// issue number is a number). A run is maximal, so it is never part of a longer hex string,
// and 40 or more hex characters is a full hash or a longer digest. A token inside a URL, a
// sha256 digest or a pseudo-version, or next to a path separator or a file extension, is
// not a commit, except the revision in a GitHub commit, blob or tree URL, which is one.
func shortHashes(line string) []string {
	var out []string
	// In a commit, blob or tree URL the revision position is unambiguous, so every
	// abbreviated revision there is refused, including one of only digits or only letters.
	for _, m := range urlRevision.FindAllStringSubmatch(line, -1) {
		out = append(out, strings.ToLower(m[1]))
	}
	masked := []byte(line)
	for _, loc := range notCommit.FindAllStringIndex(line, -1) {
		for i := loc[0]; i < loc[1]; i++ {
			masked[i] = ' '
		}
	}
	text := string(masked)
	for _, loc := range hexRun.FindAllStringIndex(text, -1) {
		tok := text[loc[0]:loc[1]]
		if len(tok) < 7 || len(tok) > 39 || !isHex(tok) {
			continue
		}
		if !strings.ContainsAny(tok, "0123456789") || !strings.ContainsAny(tok, "abcdef") {
			continue
		}
		if loc[0] > 0 && (text[loc[0]-1] == '/' || (text[loc[0]-1] == '.' && loc[0] > 1 && isWordByte(text[loc[0]-2]))) {
			continue
		}
		if n := loc[1]; n < len(text) && (text[n] == '/' || (text[n] == '.' && n+1 < len(text) && isWordByte(text[n+1]))) {
			continue
		}
		out = append(out, tok)
	}
	return out
}

func isHex(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func isWordByte(c byte) bool {
	return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)

func stripComments(s string) string { return htmlComment.ReplaceAllString(s, "") }
