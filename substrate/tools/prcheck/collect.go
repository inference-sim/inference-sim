package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

// Collecting the merge check's input from the GitHub API.
//
// A check run judged the commit it ran on, and so the tree of that commit; a review judged
// the commit it was submitted on (commit_id). The collector writes each verdict's tree as a
// "Tree:" line, from those commits, so the rule in merge.go reads one form for every kind.

type apiCheckRun struct {
	Name       string `json:"name"`
	HeadSHA    string `json:"head_sha"`
	Conclusion string `json:"conclusion"`
}

type apiReview struct {
	User     struct{ Login string } `json:"user"`
	State    string                 `json:"state"`
	CommitID string                 `json:"commit_id"`
}

type apiThread struct {
	IsResolved bool   `json:"isResolved"`
	Path       string `json:"path"`
	Comments   struct {
		Nodes []struct {
			Body string `json:"body"`
			URL  string `json:"url"`
		} `json:"nodes"`
	} `json:"comments"`
}

// BuildMergeInput turns API results into the merge check's input. treeOf maps a commit to
// its tree. Only successful check runs and approving or change-requesting reviews are
// verdicts; a dismissed or commented review judges nothing.
func BuildMergeInput(mergeSHA, headSHA string, required []string, runs []apiCheckRun, reviews []apiReview,
	threads []apiThread, changedByMerge, prPaths []string, treeOf func(string) (string, error)) (MergeInput, error) {
	mt, err := treeOf(mergeSHA)
	if err != nil {
		return MergeInput{}, err
	}
	ht, err := treeOf(headSHA)
	if err != nil {
		return MergeInput{}, err
	}
	in := MergeInput{MergeTree: mt, HeadTree: ht, Required: required, ChangedByMerge: changedByMerge}
	for _, r := range runs {
		if r.Conclusion != "success" {
			continue
		}
		t, err := treeOf(r.HeadSHA)
		if err != nil {
			return MergeInput{}, err
		}
		in.Verdicts = append(in.Verdicts, Verdict{Kind: "check", Name: r.Name, Body: "Tree: " + t})
	}
	for _, r := range reviews {
		if r.State != "APPROVED" && r.State != "CHANGES_REQUESTED" {
			continue
		}
		t, err := treeOf(r.CommitID)
		if err != nil {
			return MergeInput{}, err
		}
		in.Verdicts = append(in.Verdicts, Verdict{Kind: "review", Name: r.User.Login, Body: "Tree: " + t, Paths: prPaths})
	}
	for _, th := range threads {
		var t Thread
		t.Resolved = th.IsResolved
		for i, c := range th.Comments.Nodes {
			if i == 0 {
				t.URL = c.URL
			}
			t.Comments = append(t.Comments, c.Body)
		}
		in.Threads = append(in.Threads, t)
	}
	return in, nil
}

func gitTree(sha string) (string, error) {
	return git("rev-parse", sha+"^{tree}")
}

func ghJSON(v any, args ...string) error {
	out, err := exec.Command("gh", args...).Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		return fmt.Errorf("gh %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr))
	}
	return json.Unmarshal(out, v)
}

const threadsQuery = `query($owner:String!,$name:String!,$pr:Int!){repository(owner:$owner,name:$name){pullRequest(number:$pr){reviewThreads(first:100){nodes{isResolved path comments(first:50){nodes{body url}}}}}}}`

// collectMerge gathers the input for the merge check of one pull request in the merge queue.
func collectMerge(repo string, pr int, mergeSHA, headSHA, baseSHA string, required []string) (MergeInput, error) {
	var runs struct {
		CheckRuns []apiCheckRun `json:"check_runs"`
	}
	if err := ghJSON(&runs, "api", fmt.Sprintf("repos/%s/commits/%s/check-runs?per_page=100", repo, mergeSHA)); err != nil {
		return MergeInput{}, err
	}
	var reviews []apiReview
	if err := ghJSON(&reviews, "api", fmt.Sprintf("repos/%s/pulls/%d/reviews?per_page=100", repo, pr)); err != nil {
		return MergeInput{}, err
	}
	owner, name, _ := strings.Cut(repo, "/")
	var g struct {
		Data struct {
			Repository struct {
				PullRequest struct {
					ReviewThreads struct{ Nodes []apiThread } `json:"reviewThreads"`
				} `json:"pullRequest"`
			} `json:"repository"`
		} `json:"data"`
	}
	if err := ghJSON(&g, "api", "graphql", "-f", "query="+threadsQuery, "-F", "owner="+owner, "-F", "name="+name, "-F", fmt.Sprintf("pr=%d", pr)); err != nil {
		return MergeInput{}, err
	}
	changed, err := git("diff", "--name-only", headSHA, mergeSHA)
	if err != nil {
		return MergeInput{}, err
	}
	prPaths, err := git("diff", "--name-only", baseSHA+"..."+headSHA)
	if err != nil {
		return MergeInput{}, err
	}
	return BuildMergeInput(mergeSHA, headSHA, required, runs.CheckRuns, reviews,
		g.Data.Repository.PullRequest.ReviewThreads.Nodes, strings.Fields(changed), strings.Fields(prPaths), gitTree)
}
