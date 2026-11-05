package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// The merge check, run on the merge queue's tree (X22, X24).
//
// X22: a verdict names the git tree it judged. A CI verdict (a check run) is on the tree
// being merged. A review, or a person's answer to a review question, is on the head tree;
// it is requested again when merging changes a path it covers, which here means: when the
// merge tree differs from the head tree under any path the verdict covers. A review
// covers the paths the pull request changes, which the workflow lists; an answer covers
// the paths of its question's trigger. A verdict with no paths listed covers every path.
//
// A verdict names its tree with a line "Tree: <40-hex>" in its body (reviews and answers)
// or its output summary (check runs). The workflows that write verdicts add it.
//
// X24: every review thread ends in an outcome before merge: resolved with a reply that
// states "Outcome: fixed", "Outcome: filed <issue>" or "Outcome: rejected: <reason>".

var (
	treeLine = regexp.MustCompile(`(?m)^Tree:\s*([0-9a-f]{40})\s*$`)
	outcome  = regexp.MustCompile(`(?mi)^Outcome:\s*(fixed\b.*|filed\s+\S+.*|rejected:\s*\S.*)$`)
)

// Verdict is one CI result, review or answer.
type Verdict struct {
	Kind  string   // "check", "review" or "answer"
	Name  string   // the check's name, the reviewer, or the question's invariant
	Body  string   // where the "Tree:" line is
	Paths []string // the paths or globs it covers; empty means every path
}

// Thread is one review thread.
type Thread struct {
	URL      string
	Resolved bool
	Comments []string
}

// MergeInput is what the merge check reads.
type MergeInput struct {
	MergeTree, HeadTree string
	Required            []string // names of required checks
	Verdicts            []Verdict
	Threads             []Thread
	// ChangedByMerge lists the paths whose content differs between the head tree and the
	// merge tree.
	ChangedByMerge []string
}

func covers(globs []string, paths []string) bool {
	if len(globs) == 0 {
		return len(paths) > 0
	}
	for _, p := range paths {
		for _, g := range globs {
			if invariants.Match(g, p) {
				return true
			}
		}
	}
	return false
}

// CheckMerge returns the X22 and X24 violations.
func CheckMerge(in MergeInput) []string {
	var bad []string
	seen := map[string]bool{}
	for _, v := range in.Verdicts {
		m := treeLine.FindStringSubmatch(v.Body)
		if m == nil {
			bad = append(bad, fmt.Sprintf("%s %q names no tree; every verdict names the git tree it judged (X22)", v.Kind, v.Name))
			continue
		}
		tree := m[1]
		switch v.Kind {
		case "check":
			if tree != in.MergeTree {
				bad = append(bad, fmt.Sprintf("check %q judged tree %s, not the tree being merged, %s (X22)", v.Name, tree, in.MergeTree))
				continue
			}
			seen[v.Name] = true
		default:
			if tree == in.MergeTree {
				continue
			}
			if tree != in.HeadTree {
				bad = append(bad, fmt.Sprintf("%s %q judged tree %s, neither the head %s nor the merge tree; request it again (X22)", v.Kind, v.Name, tree, in.HeadTree))
				continue
			}
			if covers(v.Paths, in.ChangedByMerge) {
				bad = append(bad, fmt.Sprintf("%s %q judged the head tree, and merging changes paths it covers (%s); request it again (X22)", v.Kind, v.Name, strings.Join(in.ChangedByMerge, ", ")))
			}
		}
	}
	for _, r := range in.Required {
		if !seen[r] {
			bad = append(bad, fmt.Sprintf("required check %q has no successful run on the tree being merged, %s (X17, X22)", r, in.MergeTree))
		}
	}
	for _, t := range in.Threads {
		has := false
		for _, c := range t.Comments {
			has = has || outcome.MatchString(c)
		}
		switch {
		case !t.Resolved:
			bad = append(bad, fmt.Sprintf("review thread %s is open; resolve it with an outcome (X24)", t.URL))
		case !has:
			bad = append(bad, fmt.Sprintf("review thread %s is resolved with no \"Outcome: fixed | filed <issue> | rejected: <reason>\" line (X24)", t.URL))
		}
	}
	return bad
}
