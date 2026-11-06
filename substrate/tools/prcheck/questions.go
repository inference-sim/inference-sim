package main

import (
	"regexp"
	"strings"
)

// The review-question automation (design §13.2). invariants questions lists, as Markdown
// checklist items, the questions whose triggers match the changed paths. MergeQuestions
// puts them in the pull-request body under "## Review questions", keeps every answer already
// given, and reports the questions still unanswered.
//
// An item is "- [ ] **<ID>** <question> ... Answer: <text>". A code owner answers by ticking
// it and writing yes or no with a reason after "Answer:". An item stays required while it is
// unticked or its answer does not start with yes or no. Branch protection makes this job
// required, so the merge waits for every answer. The check does not look up who ticked an
// item or wrote an answer: it requires each question to be ticked and answered, and the
// approving reviewer is the control on whether the answer is the right one.

const questionsHeading = "## Review questions"

var (
	itemRe   = regexp.MustCompile(`^- \[( |x|X)\] \*\*([A-Z]+-?\d+)\*\* (.*?)(?:\s*Answer:\s*(.*))?$`)
	answerRe = regexp.MustCompile(`(?i)^(yes|no)\b[:,.;]?\s*\S`)
)

type item struct {
	ticked   bool
	id, text string
	answer   string
}

func parseItems(block string) []item {
	var out []item
	for _, l := range strings.Split(block, "\n") {
		if m := itemRe.FindStringSubmatch(strings.TrimSpace(l)); m != nil {
			out = append(out, item{m[1] != " ", m[2], strings.TrimSpace(m[3]), strings.TrimSpace(m[4])})
		}
	}
	return out
}

// key identifies a question: its invariant and its text up to the reviewer, which the
// changed-path note after it does not change.
func (i item) key() string {
	t := i.text
	if j := strings.Index(t, " (changed:"); j >= 0 {
		t = t[:j]
	}
	return i.id + " " + t
}

// MergeQuestions returns the new body and the questions not yet answered.
func MergeQuestions(body, questions string) (string, []string) {
	want := parseItems(questions)
	before, section, after := splitSection(body, questionsHeading)
	have := map[string]item{}
	for _, it := range parseItems(section) {
		have[it.key()] = it
	}
	var b strings.Builder
	var open []string
	b.WriteString(questionsHeading + "\n\n")
	b.WriteString("<!-- Added by the review-question automation (design §13.2). A code owner ticks each item and writes yes or no with a one-line reason after \"Answer:\". -->\n\n")
	for _, w := range want {
		it := w
		if h, ok := have[w.key()]; ok {
			it.ticked, it.answer = h.ticked, h.answer
		}
		mark := " "
		if it.ticked {
			mark = "x"
		}
		b.WriteString("- [" + mark + "] **" + it.id + "** " + it.text + " Answer: " + it.answer + "\n")
		if !it.ticked || !answerRe.MatchString(it.answer) {
			open = append(open, it.id+": "+it.text)
		}
	}
	if len(want) == 0 {
		b.WriteString("None: no changed path matches a review question.\n")
	}
	return before + b.String() + after, open
}

// splitSection returns the body before a "## " heading, the section's text, and the rest
// from the next "## " heading on. A body without the heading gets the section at its end.
func splitSection(body, heading string) (string, string, string) {
	lines := strings.SplitAfter(body, "\n")
	start := -1
	for i, l := range lines {
		if strings.TrimRight(l, "\r\n") == heading {
			start = i
			break
		}
	}
	if start < 0 {
		sep := ""
		if body != "" && !strings.HasSuffix(body, "\n") {
			sep = "\n"
		}
		if body != "" {
			sep += "\n"
		}
		return body + sep, "", ""
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		if strings.HasPrefix(lines[i], "## ") {
			end = i
			break
		}
	}
	after := strings.Join(lines[end:], "")
	if after != "" {
		after = "\n" + after
	}
	return strings.Join(lines[:start], ""), strings.Join(lines[start+1:end], ""), after
}
