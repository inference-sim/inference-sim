package main

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// The merge audit (X17): a nightly job in blis-substrate lists the previous day's merges in
// every BLIS repository and fails if any merged head has no successful run of each of the
// repository's required checks on the tree it merged. The required checks per repository
// are the list a person sets in branch protection; .github/required-checks.yaml records the
// same list, so the audit knows what to look for:
//
//	blis-vllm: [blis checks, ci]
//
// A check that a path filter can leave without a run is not listed; its workflow's aggregator
// job (blis-substrate's `required`, see pathgates.go) is. Every listed name is one the merge
// queue produces on the tree it merges: blis-substrate's merge check requires exactly its list.
//
// Merge is one merged pull request.
type Merge struct {
	Repo      string
	PR        int
	MergeSHA  string
	MergedAt  time.Time
	Successes []string // names of the checks that succeeded on MergeSHA
}

// AuditMerges returns a finding for each merge missing a successful required check.
func AuditMerges(merges []Merge, required map[string][]string) []string {
	var bad []string
	for _, m := range merges {
		req, ok := required[m.Repo]
		if !ok {
			bad = append(bad, fmt.Sprintf("%s#%d: the repository names no required checks; a protected branch requires its named checks (X17)", m.Repo, m.PR))
			continue
		}
		have := map[string]bool{}
		for _, s := range m.Successes {
			have[s] = true
		}
		var missing []string
		for _, r := range req {
			if !have[r] {
				missing = append(missing, r)
			}
		}
		sort.Strings(missing)
		if len(missing) > 0 {
			bad = append(bad, fmt.Sprintf("%s#%d merged %s with no successful run of %s on that tree (X17, X22)", m.Repo, m.PR, short(m.MergeSHA), strings.Join(missing, ", ")))
		}
	}
	return bad
}

func ghMergesSince(repo string, since time.Time) ([]Merge, error) {
	var prs []struct {
		Number         int        `json:"number"`
		MergedAt       *time.Time `json:"merged_at"`
		MergeCommitSHA string     `json:"merge_commit_sha"`
	}
	if err := ghJSON(&prs, "api", fmt.Sprintf("repos/%s/pulls?state=closed&sort=updated&direction=desc&per_page=100", repo)); err != nil {
		return nil, err
	}
	var out []Merge
	for _, p := range prs {
		if p.MergedAt == nil || p.MergedAt.Before(since) {
			continue
		}
		var runs struct {
			CheckRuns []apiCheckRun `json:"check_runs"`
		}
		if err := ghJSON(&runs, "api", fmt.Sprintf("repos/%s/commits/%s/check-runs?per_page=100", repo, p.MergeCommitSHA)); err != nil {
			return nil, err
		}
		m := Merge{Repo: repo, PR: p.Number, MergeSHA: p.MergeCommitSHA, MergedAt: *p.MergedAt}
		for _, r := range runs.CheckRuns {
			if r.Conclusion == "success" {
				m.Successes = append(m.Successes, r.Name)
			}
		}
		out = append(out, m)
	}
	return out, nil
}
