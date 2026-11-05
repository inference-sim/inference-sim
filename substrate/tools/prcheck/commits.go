package main

import (
	"fmt"
	"regexp"
	"strings"
)

// Commit is what the commit-policy check reads of one commit.
type Commit struct {
	Hash      string
	Message   string
	Signed    bool   // a signature GitHub verified, or git verify-commit accepted
	Author    string // "Name <email>"
	Committer string
}

// The release bot's identity. Its commits are signed and carry no sign-off (X20).
const releaseBotEmail = "blis-release-bot@users.noreply.github.com"

var (
	trailerLine   = regexp.MustCompile(`^([A-Za-z][A-Za-z-]*):\s*(.*)$`)
	toolNames     = regexp.MustCompile(`(?i)\b(claude|copilot|chatgpt|gpt-?\d|codex|cursor|gemini|aider|devin|anthropic|openai|bot)\b`)
	personSignOff = regexp.MustCompile(`^[^<>]+ <[^<>@\s]+@[^<>\s]+>$`)
)

// trailers returns the commit message's trailer block: the last paragraph, when every
// line of it is a "Key: value" line.
func trailers(msg string) [][2]string {
	paras := strings.Split(strings.TrimRight(strings.ReplaceAll(msg, "\r\n", "\n"), "\n"), "\n\n")
	if len(paras) < 2 {
		return nil
	}
	var out [][2]string
	for _, l := range strings.Split(paras[len(paras)-1], "\n") {
		m := trailerLine.FindStringSubmatch(l)
		if m == nil {
			return nil
		}
		out = append(out, [2]string{m[1], strings.TrimSpace(m[2])})
	}
	return out
}

// CommitPolicy returns the X20 violations of one commit:
//
//   - every commit is signed;
//   - a commit by the release bot carries no Signed-off-by;
//   - every other commit carries a Signed-off-by from a person (name and email), and none
//     naming a tool, since only a person can certify the DCO;
//   - at most one Assisted-by line, naming the tool;
//   - no Co-authored-by line naming a tool.
func CommitPolicy(c Commit) []string {
	var bad []string
	add := func(format string, args ...any) {
		bad = append(bad, fmt.Sprintf("%s: ", short(c.Hash))+fmt.Sprintf(format, args...))
	}
	if !c.Signed {
		add("the commit is not signed (X20)")
	}
	var signoffs, assisted []string
	for _, t := range trailers(c.Message) {
		switch strings.ToLower(t[0]) {
		case "signed-off-by":
			signoffs = append(signoffs, t[1])
		case "assisted-by":
			assisted = append(assisted, t[1])
		case "co-authored-by":
			if toolNames.MatchString(t[1]) {
				add("Co-authored-by: %s names a tool; write one Assisted-by line instead (X20)", t[1])
			}
		}
	}
	bot := strings.Contains(c.Author, "<"+releaseBotEmail+">")
	switch {
	case bot && len(signoffs) > 0:
		add("a release-bot commit carries a Signed-off-by line; the bot never signs off (X20)")
	case !bot && len(signoffs) == 0:
		add("no Signed-off-by line; the person who takes responsibility for the commit adds the DCO sign-off (X20)")
	}
	for _, s := range signoffs {
		if toolNames.MatchString(s) || !personSignOff.MatchString(s) {
			add("Signed-off-by: %s is not a person; an agent never signs off (X20)", s)
		}
	}
	if len(assisted) > 1 {
		add("%d Assisted-by lines; a commit has at most one (X20)", len(assisted))
	}
	for _, a := range assisted {
		if a == "" {
			add("an empty Assisted-by line; it names the tool (X20)")
		}
	}
	return bad
}

func short(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}
