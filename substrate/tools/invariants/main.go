// Command invariants renders and checks the copies of the invariant registry
// (contracts/invariants, design §13.1), and answers the questions the shared pull-request
// workflows ask of it.
//
//	go run ./tools/invariants render [-check]
//	    Writes this repository's generated files from the registry: the generated block
//	    of AGENTS.md, .github/pull_request_template.md, and docs/contracts/invariants/
//	    (the design's appendix index and the release plan's status table). With -check it
//	    writes nothing and fails if any file differs from what it would write.
//	go run ./tools/invariants agents -repo <name>
//	go run ./tools/invariants template -repo <name>
//	    Print another repository's AGENTS.md block or pull-request template.
//	go run ./tools/invariants check-copy -repo <name> -agents <file> -template <file>
//	    Fail if the given files' generated blocks differ from the registry's rendering.
//	go run ./tools/invariants cited -body <file>
//	    Print the invariant identifiers the pull-request body cites under its Invariants
//	    heading, and fail if any does not exist (X14).
//	go run ./tools/invariants questions -repo <name> -changed <file>
//	    Print, as a Markdown checklist, the review questions whose triggers match the
//	    changed paths listed one per line in <file> (design §13.2).
//	go run ./tools/invariants snapshot -design <design.html> -plan <plan.html> [-check]
//	    Write (or, with -check, compare) contracts/invariants/testdata/design.json and
//	    plan-status.tsv, the snapshots the registry's tests compare it with.
//
// Every subcommand exits 0 on success, 1 when a check fails, and 2 on a usage error.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// This repository's own name, for the files render writes.
const self = "blis-substrate"

var errCheck = errors.New("check failed")

func main() {
	err := run(os.Args[1:], os.Stdout, os.Stderr)
	switch {
	case err == nil:
	case errors.Is(err, errCheck):
		os.Exit(1)
	default:
		fmt.Fprintln(os.Stderr, "invariants:", err)
		os.Exit(2)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: invariants render|agents|template|check-copy|cited|questions|snapshot ...")
	}
	reg, err := invariants.Embedded()
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "repository name")
	check := fs.Bool("check", false, "compare instead of writing")
	root := fs.String("root", ".", "repository root")
	agents := fs.String("agents", "", "AGENTS.md path")
	template := fs.String("template", "", "pull-request template path")
	body := fs.String("body", "", "pull-request body file")
	changed := fs.String("changed", "", "file listing changed paths, one per line")
	design := fs.String("design", "", "design HTML")
	plan := fs.String("plan", "", "release plan HTML")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	needRepo := func() error {
		for _, r := range invariants.Repositories {
			if r == *repo {
				return nil
			}
		}
		return fmt.Errorf("-repo %q is not a BLIS repository; the repositories are %s", *repo, strings.Join(invariants.Repositories, ", "))
	}
	switch args[0] {
	case "render":
		return render(reg, *root, *check, stderr)
	case "agents":
		if err := needRepo(); err != nil {
			return err
		}
		_, err := io.WriteString(stdout, renderAgents(reg, *repo))
		return err
	case "template":
		if err := needRepo(); err != nil {
			return err
		}
		_, err := io.WriteString(stdout, renderTemplate(reg, *repo))
		return err
	case "check-copy":
		if err := needRepo(); err != nil {
			return err
		}
		return checkCopy(reg, *repo, *agents, *template, stderr)
	case "cited":
		return cited(reg, *body, stdout, stderr)
	case "questions":
		if err := needRepo(); err != nil {
			return err
		}
		return questions(reg, *repo, *changed, stdout)
	case "snapshot":
		return snapshot(*root, *design, *plan, *check, stderr)
	}
	return fmt.Errorf("unknown subcommand %q", args[0])
}

// outputs returns every file render writes in this repository, by path.
func outputs(reg *invariants.Registry, root string) (map[string]string, error) {
	agentsPath := filepath.Join(root, "AGENTS.md")
	existing, err := os.ReadFile(agentsPath)
	if err != nil {
		return nil, err
	}
	agents, err := splice(string(existing), renderAgents(reg, self))
	if err != nil {
		return nil, fmt.Errorf("AGENTS.md: %w", err)
	}
	return map[string]string{
		"AGENTS.md":                               agents,
		".github/pull_request_template.md":        renderTemplate(reg, self),
		"docs/contracts/invariants/appendix.html": renderAppendix(reg),
		"docs/contracts/invariants/status.html":   renderStatusTable(reg),
	}, nil
}

func render(reg *invariants.Registry, root string, check bool, stderr io.Writer) error {
	out, err := outputs(reg, root)
	if err != nil {
		return err
	}
	paths := make([]string, 0, len(out))
	for p := range out {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	stale := false
	for _, p := range paths {
		full := filepath.Join(root, p)
		if check {
			got, err := os.ReadFile(full)
			if err != nil || string(got) != out[p] {
				fmt.Fprintf(stderr, "%s does not match the registry; run go run ./tools/invariants render\n", p)
				stale = true
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(full, []byte(out[p]), 0o644); err != nil {
			return err
		}
	}
	if stale {
		return errCheck
	}
	return nil
}

func checkCopy(reg *invariants.Registry, repo, agentsPath, templatePath string, stderr io.Writer) error {
	if agentsPath == "" || templatePath == "" {
		return fmt.Errorf("check-copy needs -agents and -template")
	}
	bad := false
	for _, c := range []struct{ path, want string }{
		{agentsPath, renderAgents(reg, repo)},
		{templatePath, renderTemplate(reg, repo)},
	} {
		raw, err := os.ReadFile(c.path)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", c.path, err)
			bad = true
			continue
		}
		got, ok := extract(string(raw))
		if !ok || got != c.want {
			fmt.Fprintf(stderr, "%s: the generated block does not match the registry at this substrate version; regenerate it with go run github.com/inference-sim/blis-substrate/tools/invariants\n", c.path)
			bad = true
		}
	}
	if bad {
		return errCheck
	}
	return nil
}

// citedID matches an identifier-shaped token. Trigger outcomes (T-…) are accepted as
// names of design text but are not registry records, so they are reported separately.
var citedID = regexp.MustCompile(`\b(?:T-[A-Z]\d+|X\d+|D\d+|[A-Z]{3}-\d+)\b`)

// invariantsSection returns the text under the body's "## Invariants" heading.
func invariantsSection(body string) (string, bool) {
	var b strings.Builder
	in, found := false, false
	sc := bufio.NewScanner(strings.NewReader(body))
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "## ") {
			in = strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(line, "## ")), "Invariants")
			found = found || in
			continue
		}
		if in {
			b.WriteString(line + "\n")
		}
	}
	return stripComments(b.String()), found
}

var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)

func stripComments(s string) string { return htmlComment.ReplaceAllString(s, "") }

func cited(reg *invariants.Registry, bodyPath string, stdout, stderr io.Writer) error {
	if bodyPath == "" {
		return fmt.Errorf("cited needs -body")
	}
	raw, err := os.ReadFile(bodyPath)
	if err != nil {
		return err
	}
	sec, ok := invariantsSection(string(raw))
	if !ok {
		fmt.Fprintln(stderr, "the pull-request body has no \"## Invariants\" section (X14)")
		return errCheck
	}
	ids := citedID.FindAllString(sec, -1)
	seen := map[string]bool{}
	bad := false
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		if strings.HasPrefix(id, "T-") {
			fmt.Fprintf(stdout, "%s\ttrigger outcome\n", id)
			continue
		}
		inv, ok := reg.Get(id)
		if !ok {
			fmt.Fprintf(stderr, "%s is cited but is not in the invariant registry (X14)\n", id)
			bad = true
			continue
		}
		fmt.Fprintf(stdout, "%s\t%s\n", id, inv.Status)
	}
	if len(seen) == 0 {
		fmt.Fprintln(stderr, "the Invariants section cites no identifier (X14)")
		bad = true
	}
	if bad {
		return errCheck
	}
	return nil
}

func questions(reg *invariants.Registry, repo, changedPath string, stdout io.Writer) error {
	if changedPath == "" {
		return fmt.Errorf("questions needs -changed")
	}
	raw, err := os.ReadFile(changedPath)
	if err != nil {
		return err
	}
	var paths []string
	for _, l := range strings.Split(string(raw), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			paths = append(paths, l)
		}
	}
	for _, q := range reg.Questions(repo) {
		var hit []string
		for _, p := range paths {
			if q.AppliesWhen.Matches(repo, p) {
				hit = append(hit, p)
			}
		}
		if len(hit) == 0 {
			continue
		}
		more := ""
		if len(hit) > 3 {
			more = fmt.Sprintf(" and %d more", len(hit)-3)
			hit = hit[:3]
		}
		fmt.Fprintf(stdout, "- [ ] **%s** %s Expertise: %s. (changed: `%s`%s) Answer: \n", q.ID, q.Question, q.Expertise, strings.Join(hit, "`, `"), more)
	}
	return nil
}

// The snapshots of the design and the plan that the registry's tests read.
const (
	designSnapshot = "contracts/invariants/testdata/design.json"
	planSnapshot   = "contracts/invariants/testdata/plan-status.tsv"
)

func snapshot(root, designPath, planPath string, check bool, stderr io.Writer) error {
	if designPath == "" || planPath == "" {
		return fmt.Errorf("snapshot needs -design and -plan")
	}
	d, err := os.ReadFile(designPath)
	if err != nil {
		return err
	}
	p, err := os.ReadFile(planPath)
	if err != nil {
		return err
	}
	items, err := parseDesign(string(d))
	if err != nil {
		return err
	}
	rows, err := parsePlan(string(p))
	if err != nil {
		return err
	}
	var dj bytes.Buffer
	enc := json.NewEncoder(&dj)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", " ")
	if err := enc.Encode(items); err != nil {
		return err
	}
	var pt strings.Builder
	pt.WriteString("# The release plan's §3.2 status table: identifier, status, In CI by.\n")
	pt.WriteString("# Written by go run ./tools/invariants snapshot from the published plan; do not edit.\n")
	for _, r := range rows {
		fmt.Fprintf(&pt, "%s\t%s\t%s\n", r.ID, r.Status, r.By)
	}
	out := map[string]string{designSnapshot: dj.String(), planSnapshot: pt.String()}
	stale := false
	for _, rel := range []string{designSnapshot, planSnapshot} {
		full := filepath.Join(root, rel)
		if check {
			got, err := os.ReadFile(full)
			if err != nil || string(got) != out[rel] {
				fmt.Fprintf(stderr, "%s differs from the design or plan given\n", rel)
				stale = true
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(full, []byte(out[rel]), 0o644); err != nil {
			return err
		}
	}
	if stale {
		return errCheck
	}
	return nil
}
