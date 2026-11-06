package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"time"

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
	// Recorded is what the pull-request check recorded at the latest approving review: each
	// answered question's digest (AnswerDigest), read from the "review answers" check run on
	// the reviewed commit that was last written at or before the review. nil when there is none.
	Recorded map[string]string
}

// AnswersCheck is the name of the check run the pull-request check writes on the head commit
// each time it runs: the digest of every answer in the body, with the head tree. The merge
// check compares the body's answers with the record in force when the latest approving review
// was submitted, so the approval covers exactly the text that was recorded then, whoever
// edits the body afterwards (X22). The API keeps no usable history of a body's text, so the
// record is the history.
const AnswersCheck = "review answers"

var answerRecordLine = regexp.MustCompile(`(?m)^Answer ([A-Z]+-?\d+): sha256:([0-9a-f]{64})$`)

// AnswerDigest is the SHA-256 of one answer's text bound to the question and the tree it was
// given for.
func AnswerDigest(id, answer, tree string) string {
	sum := sha256.Sum256([]byte("answer\x00" + id + "\x00" + strings.TrimSpace(answer) + "\x00" + tree))
	return hex.EncodeToString(sum[:])
}

// answers returns the questions the changed paths trigger and, for each one that has a ticked
// yes-or-no answer in the body, its digest on the head tree.
func answers(ans answerSource, prPaths []string, headTree string) (triggered []invariants.Question, digest map[string]string) {
	_, section, _ := splitSection(ans.Body, questionsHeading)
	given := map[string]item{}
	for _, it := range parseItems(section) {
		given[it.id] = it
	}
	digest = map[string]string{}
	for _, q := range ans.Questions {
		hit := false
		for _, p := range prPaths {
			hit = hit || q.Trigger.Matches(ans.Repo, p)
		}
		if !hit {
			continue
		}
		triggered = append(triggered, q)
		if it, ok := given[q.ID]; ok && it.ticked && answerRe.MatchString(it.answer) {
			digest[q.ID] = AnswerDigest(q.ID, it.answer, headTree)
		}
	}
	return triggered, digest
}

// FormatAnswerRecord is the summary of the "review answers" check run.
func FormatAnswerRecord(headTree string, digest map[string]string) string {
	ids := make([]string, 0, len(digest))
	for id := range digest {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var b strings.Builder
	b.WriteString("Tree: " + headTree + "\n")
	for _, id := range ids {
		b.WriteString("Answer " + id + ": sha256:" + digest[id] + "\n")
	}
	return b.String()
}

// ParseAnswerRecord reads the digests out of a record's summary.
func ParseAnswerRecord(summary string) map[string]string {
	out := map[string]string{}
	for _, m := range answerRecordLine.FindAllStringSubmatch(summary, -1) {
		out[m[1]] = m[2]
	}
	return out
}

type apiAnswerRecord struct {
	CompletedAt string `json:"completed_at"`
	Output      struct {
		Summary string `json:"summary"`
	} `json:"output"`
}

// recordAt picks, among the answer records on one commit, the last one completed at or before
// the review, and returns its digests. ok is false when there is none: nothing says what the
// body held when the review was given.
func recordAt(records []apiAnswerRecord, reviewedAt string) (map[string]string, bool) {
	rv, err := time.Parse(time.RFC3339, reviewedAt)
	if err != nil {
		return nil, false
	}
	var best time.Time
	var out map[string]string
	for _, r := range records {
		at, err := time.Parse(time.RFC3339, r.CompletedAt)
		if err != nil || at.After(rv) {
			continue
		}
		if out == nil || at.After(best) {
			best, out = at, ParseAnswerRecord(r.Output.Summary)
		}
	}
	return out, out != nil
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
	// body. The answer is a verdict on the head tree, covering the paths that trigger it, and
	// carries the digest of its text and that tree.
	triggered, digest := answers(ans, prPaths, ht)
	in.Recorded = ans.Recorded
	for _, q := range triggered {
		in.Questions = append(in.Questions, q.ID+": "+q.Question)
		if d, ok := digest[q.ID]; ok {
			in.Verdicts = append(in.Verdicts, Verdict{Kind: "answer", Name: q.ID, Body: "Tree: " + ht, Paths: q.Trigger.Paths, At: ans.EditedAt, Digest: d})
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
	// The record in force when the latest approving review was submitted, on the commit it
	// reviewed.
	var latest *apiReview
	for i := range reviews {
		r := &reviews[i]
		if r.State != "APPROVED" {
			continue
		}
		if latest == nil || r.SubmittedAt > latest.SubmittedAt {
			latest = r
		}
	}
	var recorded map[string]string
	if latest != nil {
		var recs struct {
			CheckRuns []apiAnswerRecord `json:"check_runs"`
		}
		if err := ghJSON(&recs, "api", fmt.Sprintf("repos/%s/commits/%s/check-runs?check_name=%s&per_page=100", repo, latest.CommitID, url.QueryEscape(AnswersCheck))); err != nil {
			return MergeInput{}, err
		}
		recorded, _ = recordAt(recs.CheckRuns, latest.SubmittedAt)
	}
	pr0 := g.Data.Repository.PullRequest
	edited := pr0.LastEditedAt
	if edited == "" {
		edited = pr0.CreatedAt
	}
	ans := answerSource{Body: pr0.Body, EditedAt: edited, Repo: name, Questions: reg.Questions(name), Recorded: recorded}
	return BuildMergeInput(mergeSHA, headSHA, required, runs.CheckRuns, reviews,
		pr0.ReviewThreads.Nodes, strings.Fields(changed), strings.Fields(prPaths), ans, gitTree)
}
