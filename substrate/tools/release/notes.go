package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// Notes writes a release's notes from its changes (LIF-4, design §13.3). They are generated,
// never typed. This version lists what git and the registry hold:
//
//   - the tag, the previous tag and the commit range;
//   - each merged pull request (merge commits) and each other commit, by subject;
//   - the pins the release writes;
//   - the invariant identifiers the commits cite, with their status in the registry;
//   - the review answers recorded in merged pull-request bodies (the "## Review questions"
//     section, carried into the merge commit message by the merge queue);
//   - the changed files under the directories whose changes the design asks the notes to
//     list: coefficient files (registry/coefficients), recorded traces (oracle/traces),
//     contracts (contracts/), and the field and surface lists.
//
// The price-change report and the scoring harness's figures are produced by the jobs that
// compute them and attached by the release job; the notes name where they are.
func Notes(reg *invariants.Registry, st Step, dir string) (string, error) {
	rng := st.Head
	if st.Prev != "" {
		rng = st.Prev + ".." + st.Head
	}
	log, err := gitIn(dir, "log", "--reverse", "--format=%H%x00%s%x00%B%x01", rng)
	if err != nil {
		return "", err
	}
	files, err := changedFiles(dir, st.Prev, st.Head)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s\n\n", st.Repo, st.Tag)
	if st.Prev != "" {
		fmt.Fprintf(&b, "Changes since %s, %s..%s.\n", st.Prev, short(st.Prev), short(st.Head))
	} else {
		fmt.Fprintf(&b, "First release, at %s.\n", st.Head)
	}
	cited := map[string]bool{}
	var merges, commits, answers []string
	for _, entry := range strings.Split(log, "\x01") {
		entry = strings.TrimLeft(entry, "\n")
		if entry == "" {
			continue
		}
		parts := strings.SplitN(entry, "\x00", 3)
		if len(parts) < 3 {
			continue
		}
		hash, subj, body := parts[0], parts[1], parts[2]
		line := fmt.Sprintf("- %s %s", short(hash), subj)
		if strings.HasPrefix(subj, "Merge ") {
			merges = append(merges, line)
		} else {
			commits = append(commits, line)
		}
		for _, id := range citedIDs.FindAllString(body, -1) {
			cited[id] = true
		}
		answers = append(answers, reviewAnswers(body)...)
	}
	section := func(title string, lines []string) {
		if len(lines) == 0 {
			return
		}
		fmt.Fprintf(&b, "\n## %s\n\n%s\n", title, strings.Join(lines, "\n"))
	}
	section("Merged pull requests", merges)
	section("Commits", commits)
	var pins []string
	for k, v := range st.Pins {
		pins = append(pins, fmt.Sprintf("- %s %s", strings.TrimPrefix(k, "replace:"), v))
	}
	sort.Strings(pins)
	section("Pins written by this release", pins)
	var ids []string
	for id := range cited {
		if inv, ok := reg.Get(id); ok {
			ids = append(ids, fmt.Sprintf("- %s (%s)", id, inv.Status))
		}
	}
	sort.Strings(ids)
	section("Invariants the changes cite", ids)
	section("Review answers", answers)
	for _, g := range []struct{ title, prefix string }{
		{"Changed coefficient files", "registry/coefficients/"},
		{"Changed recorded traces", "oracle/traces/"},
		{"Changed contracts", "contracts/"},
		{"Changed field and surface lists", ""},
	} {
		var l []string
		for _, f := range files {
			if g.prefix == "" && (f == "fields.json" || f == "surface.json") || g.prefix != "" && strings.HasPrefix(f, g.prefix) {
				l = append(l, "- "+f)
			}
		}
		section(g.title, l)
	}
	b.WriteString("\n## Reports\n\nThe price-change report and the scoring harness's figures on published units are attached to the release by the release job (design §13.3).\n")
	return b.String(), nil
}

var citedIDs = regexp.MustCompile(`\b(?:X\d+|D\d+|[A-Z]{3}-\d+)\b`)

var answerItem = regexp.MustCompile(`(?m)^- \[[xX]\] \*\*([A-Z]+-?\d+)\*\* (.*?) Answer: (.+)$`)

func reviewAnswers(body string) []string {
	var out []string
	for _, m := range answerItem.FindAllStringSubmatch(body, -1) {
		q := m[2]
		if i := strings.Index(q, " (changed:"); i >= 0 {
			q = q[:i]
		}
		out = append(out, fmt.Sprintf("- %s: %s %s", m[1], q, strings.TrimSpace(m[3])))
	}
	return out
}

func changedFiles(dir, prev, head string) ([]string, error) {
	var out string
	var err error
	if prev == "" {
		out, err = gitIn(dir, "ls-tree", "-r", "--name-only", head)
	} else {
		out, err = gitIn(dir, "diff", "--name-only", prev, head)
	}
	if err != nil {
		return nil, err
	}
	return strings.Fields(out), nil
}
