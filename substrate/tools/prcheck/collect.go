package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
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
	User        struct{ Login string } `json:"user"`
	State       string                 `json:"state"`
	CommitID    string                 `json:"commit_id"`
	SubmittedAt string                 `json:"submitted_at"`
}

// answerSource is where the review-question answers come from: the pull-request body, the
// time it was last edited, and the questions the registry asks of the repository.
type answerSource struct {
	Body      string
	EditedAt  string
	Repo      string // the repository's short name, as the registry knows it
	Questions []invariants.Question
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
	threads []apiThread, changedByMerge, prPaths []string, ans answerSource, treeOf func(string) (string, error)) (MergeInput, error) {
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
		v := Verdict{Kind: "review", Name: r.User.Login, Body: "Tree: " + t, Paths: prPaths}
		if r.State == "APPROVED" {
			v.At = r.SubmittedAt
		}
		in.Verdicts = append(in.Verdicts, v)
	}
	// Answers: each question the changed paths trigger needs a ticked, yes-or-no answer in the
	// body. The answer is a verdict on the head tree, covering the paths that trigger it.
	_, section, _ := splitSection(ans.Body, questionsHeading)
	given := map[string]item{}
	for _, it := range parseItems(section) {
		given[it.id] = it
	}
	for _, q := range ans.Questions {
		var hit []string
		for _, p := range prPaths {
			if q.Trigger.Matches(ans.Repo, p) {
				hit = append(hit, p)
			}
		}
		if len(hit) == 0 {
			continue
		}
		in.Questions = append(in.Questions, q.ID+": "+q.Question)
		if it, ok := given[q.ID]; ok && it.ticked && answerRe.MatchString(it.answer) {
			in.Verdicts = append(in.Verdicts, Verdict{Kind: "answer", Name: q.ID, Body: "Tree: " + ht, Paths: q.Trigger.Paths, At: ans.EditedAt})
		}
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

const threadsQuery = `query($owner:String!,$name:String!,$pr:Int!){repository(owner:$owner,name:$name){pullRequest(number:$pr){body lastEditedAt createdAt reviewThreads(first:100){nodes{isResolved path comments(first:50){nodes{body url}}}}}}}`

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
					Body          string                      `json:"body"`
					LastEditedAt  string                      `json:"lastEditedAt"`
					CreatedAt     string                      `json:"createdAt"`
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
	reg, err := invariants.Embedded()
	if err != nil {
		return MergeInput{}, err
	}
	pr0 := g.Data.Repository.PullRequest
	edited := pr0.LastEditedAt
	if edited == "" {
		edited = pr0.CreatedAt
	}
	ans := answerSource{Body: pr0.Body, EditedAt: edited, Repo: name, Questions: reg.Questions(name)}
	return BuildMergeInput(mergeSHA, headSHA, required, runs.CheckRuns, reviews,
		pr0.ReviewThreads.Nodes, strings.Fields(changed), strings.Fields(prPaths), ans, gitTree)
}
