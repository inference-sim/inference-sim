package main

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// The thread-outcome report, run on every pull request (X24): every review thread ends in an
// outcome before merge. A thread is resolved with a reply that states "Outcome: fixed",
// "Outcome: filed <issue>" or "Outcome: rejected: <reason>".
//
// The step is informational. It lists the threads that are open and the resolved ones with no
// outcome line, as notices and in the job summary, and it never fails: no check waits on a
// person, or on an agent that has not yet answered (the maintainer's order of 2026-11-11).
// The duty sits with the author agent (AGENTS.md: after each fix, reply with the outcome and
// the commit and resolve the thread), and branch protection, which blocks the merge until
// every conversation is resolved, is the merge condition an agent satisfies.

var outcome = regexp.MustCompile(`(?mi)^Outcome:\s*(fixed\b.*|filed\s+\S+.*|rejected:\s*\S.*)$`)

// Thread is one review thread.
type Thread struct {
	URL      string
	Resolved bool
	Comments []string
}

// ThreadNotes returns the X24 notes: a thread still open, and a resolved thread with no
// outcome line. They are reported, never counted as failures.
func ThreadNotes(threads []Thread) []string {
	var bad []string
	for _, t := range threads {
		has := false
		for _, c := range t.Comments {
			has = has || outcome.MatchString(c)
		}
		switch {
		case !t.Resolved:
			bad = append(bad, fmt.Sprintf("review thread %s is open; the author agent replies with its outcome and the commit and resolves it (X24)", t.URL))
		case !has:
			bad = append(bad, fmt.Sprintf("review thread %s is resolved with no \"Outcome: fixed | filed <issue> | rejected: <reason>\" line; the author agent adds it (X24)", t.URL))
		}
	}
	return bad
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

// threadsQuery is the GraphQL query for one page of a pull request's review threads, with the
// first 50 comments of each.
const threadsQuery = `query($owner:String!,$name:String!,$pr:Int!,$after:String){repository(owner:$owner,name:$name){pullRequest(number:$pr){reviewThreads(first:100,after:$after){pageInfo{hasNextPage endCursor} nodes{isResolved path comments(first:50){nodes{body url}}}}}}}`

// threadsOf converts API threads to the form ThreadNotes reads.
func threadsOf(in []apiThread) []Thread {
	var out []Thread
	for _, th := range in {
		t := Thread{Resolved: th.IsResolved}
		for i, c := range th.Comments.Nodes {
			if i == 0 {
				t.URL = c.URL
			}
			t.Comments = append(t.Comments, c.Body)
		}
		out = append(out, t)
	}
	return out
}

// collectThreads reads every review thread of a pull request, following the pages.
func collectThreads(repo string, pr int) ([]Thread, error) {
	owner, name, ok := strings.Cut(repo, "/")
	if !ok {
		return nil, fmt.Errorf("-repo %q is not owner/name", repo)
	}
	var all []apiThread
	after := ""
	for {
		var g struct {
			Data struct {
				Repository struct {
					PullRequest struct {
						ReviewThreads struct {
							PageInfo struct {
								HasNextPage bool   `json:"hasNextPage"`
								EndCursor   string `json:"endCursor"`
							} `json:"pageInfo"`
							Nodes []apiThread `json:"nodes"`
						} `json:"reviewThreads"`
					} `json:"pullRequest"`
				} `json:"repository"`
			} `json:"data"`
		}
		args := []string{"api", "graphql", "-f", "query=" + threadsQuery, "-F", "owner=" + owner, "-F", "name=" + name, "-F", fmt.Sprintf("pr=%d", pr)}
		if after != "" {
			args = append(args, "-F", "after="+after)
		}
		if err := ghJSON(&g, args...); err != nil {
			return nil, err
		}
		rt := g.Data.Repository.PullRequest.ReviewThreads
		all = append(all, rt.Nodes...)
		if !rt.PageInfo.HasNextPage {
			return threadsOf(all), nil
		}
		after = rt.PageInfo.EndCursor
	}
}

// reportThreads prints the X24 notes as GitHub notices and, when summary names a file (the
// workflow's GITHUB_STEP_SUMMARY), appends them there as a list. It writes nothing that fails
// a job: a pull request with an open thread is not red because of this step.
func reportThreads(w io.Writer, summary string, n int, notes []string) {
	_, _ = fmt.Fprintf(w, "threads: %d checked, %d to answer (informational; no check waits on a thread)\n", n, len(notes))
	for _, note := range notes {
		_, _ = fmt.Fprintf(w, "::notice title=Review thread (X24)::%s\n", note)
	}
	if summary == "" {
		return
	}
	f, err := os.OpenFile(summary, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		_, _ = fmt.Fprintf(w, "::warning::the job summary could not be written: %v\n", err)
		return
	}
	defer func() { _ = f.Close() }()
	_, _ = fmt.Fprintf(f, "### Review threads (X24)\n\n%d thread(s) checked; %d to answer. This step never fails.\n\n", n, len(notes))
	for _, note := range notes {
		_, _ = fmt.Fprintf(f, "- %s\n", note)
	}
}
