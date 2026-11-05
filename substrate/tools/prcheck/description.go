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
	fullHash  = regexp.MustCompile(`\b[0-9a-f]{40}\b`)
	shortHash = regexp.MustCompile("(?:commit|at|head)\\s+`?([0-9a-f]{7,39})`?\\b")
	prURL     = regexp.MustCompile(`https://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)`)
	prRef     = regexp.MustCompile(`\b([\w.-]+/[\w.-]+)#(\d+)\b`)
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
		for _, m := range shortHash.FindAllStringSubmatch(line, -1) {
			bad = append(bad, fmt.Sprintf("line %d: %s is a short hash; list full 40-character hashes so they can be checked (X14)", ln, m[1]))
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

var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)

func stripComments(s string) string { return htmlComment.ReplaceAllString(s, "") }
