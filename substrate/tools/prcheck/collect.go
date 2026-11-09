package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// The record of review-question answers (X22).

// answerSource is where the review-question answers come from: the pull-request body, and
// the questions the invariant registry asks of the repository.
type answerSource struct {
	Body      string
	Repo      string // the repository's short name, as the invariant registry knows it
	Questions []invariants.Question
}

// AnswersCheck is the name of the check run the pull-request check writes on the head commit
// each time it runs: the digest of every answer in the body, with the head tree (X22). The
// record names the tree each answer was given for. A push that changes the tree dismisses an
// earlier approval through branch protection, so the next approver reads the answers again.
const AnswersCheck = "review answers"

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
