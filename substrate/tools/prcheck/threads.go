package main

import (
	"fmt"
	"regexp"
	"strings"
)

// The thread-outcome check, run on every pull request (X24): every review thread ends in an
// outcome before merge. A thread is resolved with a reply that states "Outcome: fixed",
// "Outcome: filed <issue>" or "Outcome: rejected: <reason>".

var outcome = regexp.MustCompile(`(?mi)^Outcome:\s*(fixed\b.*|filed\s+\S+.*|rejected:\s*\S.*)$`)

// Thread is one review thread.
type Thread struct {
	URL      string
	Resolved bool
	Comments []string
}

// CheckThreads returns the X24 violations: a thread still open, and a resolved thread with
// no outcome line.
func CheckThreads(threads []Thread) []string {
	var bad []string
	for _, t := range threads {
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

// threadsPage is the GraphQL query for one page of a pull request's review threads, with the
// first 50 comments of each.
const threadsQuery = `query($owner:String!,$name:String!,$pr:Int!,$after:String){repository(owner:$owner,name:$name){pullRequest(number:$pr){reviewThreads(first:100,after:$after){pageInfo{hasNextPage endCursor} nodes{isResolved path comments(first:50){nodes{body url}}}}}}}`

// threadsOf converts API threads to the form CheckThreads reads.
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
