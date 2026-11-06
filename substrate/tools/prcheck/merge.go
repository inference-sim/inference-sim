package main

import (
	"fmt"
	"regexp"
	"strings"
	"time"

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
	// At is when the verdict was given (RFC 3339): for an approving review, its submission;
	// for an answer, the last edit of the pull-request body that holds it. Empty when unknown.
	At string
	// Digest, on an answer, is AnswerDigest of the answer text in the body now and the head
	// tree.
	Digest string
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
	// Questions lists the review questions the pull request's changed paths trigger, as
	// "<invariant>: <question>". Each needs an answer verdict: an answer removed from the
	// body, or never given, leaves its question without one (X22).
	Questions []string
	// Recorded is what the pull-request check recorded for each answer (AnswerDigest) in force
	// when the latest approving review was submitted. An approval covers the answer text of
	// that record and no other: the check does not look up who wrote an answer, so what the
	// reviewer approved is the text, and the text is bound to the review by its digest.
	Recorded map[string]string
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
	answered := map[string]bool{}
	var approved time.Time
	for _, v := range in.Verdicts {
		at, err := time.Parse(time.RFC3339, v.At)
		if err != nil {
			continue
		}
		if v.Kind == "review" && at.After(approved) {
			approved = at
		}
	}
	for _, v := range in.Verdicts {
		if v.Kind != "answer" {
			continue
		}
		answered[v.Name] = true
		// An answer is the head tree's. Edited after the approving review, it was never
		// approved: the reviewer is the control on whether an answer is right (the project
		// has no code-owner lookup), so the review must come after the answer.
		if at, err := time.Parse(time.RFC3339, v.At); err == nil && !approved.IsZero() && at.After(approved) {
			bad = append(bad, fmt.Sprintf("answer %q was written or edited at %s, after the latest approving review at %s; request the review again (X22)", v.Name, v.At, approved.Format(time.RFC3339)))
		}
		if !approved.IsZero() {
			rec, ok := in.Recorded[v.Name]
			switch {
			case !ok:
				bad = append(bad, fmt.Sprintf("answer %q has no record from the pull-request check at the latest approving review (%s); it was not the text the reviewer saw. Request the review again (X22)", v.Name, approved.Format(time.RFC3339)))
			case rec != v.Digest:
				bad = append(bad, fmt.Sprintf("answer %q in the body differs from the answer recorded when the latest approving review was given (%s); request the review again (X22)", v.Name, approved.Format(time.RFC3339)))
			}
		}
	}
	for _, q := range in.Questions {
		id, _, _ := strings.Cut(q, ": ")
		if !answered[id] {
			bad = append(bad, fmt.Sprintf("review question %q has no answer in the pull-request body; it was never answered or the answer was removed (X22)", q))
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
