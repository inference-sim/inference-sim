// Command prcheck holds the checks the shared workflows in blis-substrate/.github/workflows
// run in every BLIS repository (design §13.2). Each subcommand reads local files, git, and,
// where it must, the GitHub API through the gh command (GH_TOKEN set by the workflow).
//
//	prcheck commits -base <rev> -head <rev> [-signatures <file>]
//	    X20, over the commits base..head: at most one Assisted-by line, no Co-authored-by
//	    naming a tool, no Signed-off-by naming a tool or on a release-bot commit. It does not
//	    check for a signature or a sign-off: those are encouraged, not required.
//	prcheck workflows [-dir .github]
//	    X21, X22, X16: actions pinned by commit, pip --require-hashes, no ignored exit
//	    codes, no skipped tests in a job, gate trigger paths as declared.
//	prcheck description -body <file> -repo <owner/name> -head <rev> -base <rev>
//	    X14: listed hashes are branch heads; the pull request merges cleanly.
//	prcheck threads -repo <o/n> -pr <n>
//	    X24, informational: lists the review threads that are open or resolved with no outcome
//	    (fixed, filed <issue>, or rejected: <reason>) as notices and in the job summary. It
//	    never fails: no check waits on a person.
//	prcheck rerun -repo <o/n> -run <id>
//	    Reliability: reruns the failed jobs of a failed CI run once, and only when every failed
//	    job was ended by "The runner has received a shutdown signal" and this is attempt 1.
//	    Says what it did or why not in the job summary and, for a pull request, a comment.
//	prcheck issues -repo <owner/name> [-closes <body file>] [-resolve]
//	    X28: the issues cited as reasons for a refusal or gap. With -closes, fail if the
//	    body closes a cited issue; with -resolve, fail on any cited issue that is closed.
//	prcheck versions
//	    X23: version literals outside the sources of versions, and outside records under specs/.
//	prcheck docrefs
//	    X26: paths, line ranges and table rows in Markdown documents. Every finding fails; no
//	    list of known findings hides one.
//	prcheck skips -results <file> [-python-package <dir>] [-min <n>]
//	    X16, X22: a skipped test not on the quarantine list fails, and so does a go test
//	    run with fewer than -min results.
//	prcheck contracts-apidiff [-prev <tag>] [-accept-break <owner/repo#n>]
//	    SUB-1: the packages under contracts/ change only compatibly since the last release
//	    tag, by apidiff at a pinned version run through go run. -accept-break names the issue
//	    of a contract-change review set (design §12.8) and accepts, and prints, the changes.
//	prcheck tags [-lock .github/tags.lock]
//	    LIF-1: every tag resolves to the commit the lock records, and every tag is recorded.
//	prcheck status [-repo <o/n>] [-gates .github/gates.yaml] [-out page.md]
//	    X16: each gate's declared paths equal its workflow's triggers; with -repo, the
//	    status page, which shows a gate not run for seven days as failing.
//	prcheck path-gates -repo <o/n> -base <sha> -head <sha> [-runs-sha <sha>] [-dir .github] [-wait 45m]
//	    X16, X17: the path-filtered workflows (read from the workflow files) whose paths the
//	    change touched (git diff --name-only base head) must each have a successful check
//	    run on -runs-sha (default -head); one the change did not touch counts as passed.
//	prcheck knownbad [-repo-name blis-substrate] [-list-only]
//	    X18: run every known-bad test the registry names for this repository.
//	prcheck licence-files [-repo-name <name>]
//	    X21: LICENSE exists, and an engine repository's NOTICE names its upstream.
//	prcheck vocab
//	    Print the Go vocabularies as JSON, for the Python rule lint (X15).
//	prcheck quarantine [-list .github/quarantine.yaml] [-skip-pattern <pkg>]
//	    X25: every entry has an issue and an unexpired date.
//
// Exit status: 0 clean, 1 findings, 2 the check could not run.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

var errFindings = errors.New("findings")

func main() {
	err := run(os.Args[1:], os.Stdout)
	switch {
	case err == nil:
	case errors.Is(err, errFindings):
		os.Exit(1)
	default:
		fmt.Fprintln(os.Stderr, "prcheck:", err)
		os.Exit(2)
	}
}

func report(w io.Writer, name string, bad []string) error {
	for _, b := range bad {
		fmt.Fprintln(w, b)
	}
	if len(bad) > 0 {
		fmt.Fprintf(w, "%s: %d finding(s)\n", name, len(bad))
		return errFindings
	}
	fmt.Fprintf(w, "%s: clean\n", name)
	return nil
}

func run(args []string, w io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: prcheck commits|workflows|description|threads|issues|versions|docrefs|quarantine")
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	base := fs.String("base", "", "base revision")
	head := fs.String("head", "HEAD", "head revision")
	dir := fs.String("dir", ".github", "directory of workflows and actions")
	body := fs.String("body", "", "pull-request body file")
	repo := fs.String("repo", "", "owner/name")
	closes := fs.String("closes", "", "pull-request body file whose Closes lines are checked")
	resolve := fs.Bool("resolve", false, "resolve each cited issue through the GitHub API")
	list := fs.String("list", ".github/quarantine.yaml", "quarantine list")
	skipPkg := fs.String("skip-pattern", "", "print the go test -skip pattern for this package")
	repoName := fs.String("repo-name", "blis-substrate", "the repository's BLIS name")
	listOnly := fs.Bool("list-only", false, "list the known-bad tests without running them")
	gatesFile := fs.String("gates", ".github/gates.yaml", "the gate list")
	lockFile := fs.String("lock", ".github/tags.lock", "the recorded tags")
	prNum := fs.Int("pr", 0, "pull request number")
	runID := fs.Int64("run", 0, "rerun: the workflow run id")
	outFile := fs.String("out", "", "status: where to write the status page")
	prevTag := fs.String("prev", "", "contracts-apidiff: the tag to compare with; default the newest release tag reachable from HEAD")
	acceptBreak := fs.String("accept-break", "", "contracts-apidiff: the issue (owner/repo#n) of the contract-change review set that accepts an incompatible change")
	results := fs.String("results", "", "go test -json or pytest -rs output")
	pyPkg := fs.String("python-package", "", "the directory pytest ran in, for its results")
	minItems := fs.Int("min", 1, "the minimum number of go test results")
	runsSHA := fs.String("runs-sha", "", "the commit whose check runs the path gates are read from; default -head")
	wait := fs.Duration("wait", 45*time.Minute, "how long path-gates waits for a triggered gate to complete")
	today := fs.String("today", "", "the date to judge expiry on (YYYY-MM-DD); default today, UTC")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	switch args[0] {
	case "commits":
		if *base == "" {
			return fmt.Errorf("commits needs -base")
		}
		cs, err := gitCommits(*base, *head)
		if err != nil {
			return err
		}
		if len(cs) == 0 {
			return fmt.Errorf("no commits in %s..%s; the check processes at least one (X22)", *base, *head)
		}
		var bad []string
		for _, c := range cs {
			bad = append(bad, CommitPolicy(c)...)
		}
		fmt.Fprintf(w, "commits: %d checked\n", len(cs))
		return report(w, "commit-policy", bad)
	case "workflows":
		var files []string
		for _, pat := range []string{"workflows/*.yml", "workflows/*.yaml", "actions/*/action.yml", "actions/*/action.yaml"} {
			m, _ := filepath.Glob(filepath.Join(*dir, pat))
			files = append(files, m...)
		}
		if len(files) == 0 {
			return fmt.Errorf("no workflow files under %s; the check processes at least one (X22)", *dir)
		}
		sort.Strings(files)
		var bad []string
		for _, f := range files {
			src, err := os.ReadFile(f)
			if err != nil {
				return err
			}
			b, err := LintWorkflow(f, src)
			if err != nil {
				return err
			}
			bad = append(bad, b...)
		}
		fmt.Fprintf(w, "workflows: %d files checked\n", len(files))
		return report(w, "workflow-lint", bad)
	case "description":
		if *body == "" || *repo == "" || *base == "" {
			return fmt.Errorf("description needs -body, -repo, -base and -head")
		}
		raw, err := os.ReadFile(*body)
		if err != nil {
			return err
		}
		headSHA, err := git("rev-parse", *head)
		if err != nil {
			return err
		}
		baseSHA, err := git("rev-parse", *base)
		if err != nil {
			return err
		}
		_, mergeErr := git("merge-tree", "--write-tree", baseSHA, headSHA)
		return report(w, "description", CheckDescription(DescriptionInput{
			Body: string(raw), Repo: *repo, Head: headSHA, Base: baseSHA,
			Heads: ghPRHead, Tags: ghTagCommit, Merges: mergeErr == nil,
		}))
	case "threads":
		if *repo == "" || *prNum == 0 {
			return fmt.Errorf("threads needs -repo and -pr")
		}
		ts, err := collectThreads(*repo, *prNum)
		if err != nil {
			// An informational step does not turn a pull request red because the API did not
			// answer; it says so.
			_, _ = fmt.Fprintf(w, "::warning title=Review threads (X24)::the threads could not be read: %v\n", err)
			return nil
		}
		reportThreads(w, os.Getenv("GITHUB_STEP_SUMMARY"), len(ts), ThreadNotes(ts))
		return nil
	case "rerun":
		if *repo == "" || *runID == 0 {
			return fmt.Errorf("rerun needs -repo and -run")
		}
		return rerunRun(w, os.Getenv("GITHUB_STEP_SUMMARY"), *repo, *runID)
	case "issues":
		if *repo == "" {
			return fmt.Errorf("issues needs -repo")
		}
		cites, err := treeCitations(*repo)
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "issues: %d citations\n", len(cites))
		var bad []string
		if *closes != "" {
			raw, err := os.ReadFile(*closes)
			if err != nil {
				return err
			}
			bad = append(bad, CheckClosesUncited(ClosedBy(*repo, string(raw)), cites)...)
		}
		if *resolve {
			bad = append(bad, CheckCitedOpen(cites, ghIssueOpen)...)
		}
		return report(w, "cited-issues", bad)
	case "versions":
		files, err := trackedFiles()
		if err != nil {
			return err
		}
		var found []string
		for _, f := range files {
			src, err := os.ReadFile(f)
			if err != nil {
				continue // a submodule or a deleted path
			}
			found = append(found, VersionFindings(f, src)...)
		}
		return report(w, "version-literals", CheckVersions(found))
	case "docrefs":
		files, err := trackedFiles()
		if err != nil {
			return err
		}
		tree := diskTree(files)
		roots := map[string]bool{}
		for _, f := range files {
			roots[strings.SplitN(f, "/", 2)[0]] = true
		}
		var bad []string
		n := 0
		for _, f := range files {
			if !strings.HasSuffix(f, ".md") || strings.Contains(f, "/testdata/") {
				continue
			}
			src, err := os.ReadFile(f)
			if err != nil {
				return err
			}
			n++
			bad = append(bad, DocFindings(f, src, tree, roots)...)
		}
		fmt.Fprintf(w, "docrefs: %d documents checked\n", n)
		return report(w, "doc-references", bad)
	case "skips":
		// prcheck skips -results <file> [-python-package <dir>]: fail on any skipped test
		// the quarantine list does not cover (X16), and on fewer than -min results (X22).
		raw, err := os.ReadFile(*results)
		if err != nil {
			return err
		}
		src, err := os.ReadFile(*list)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		entries, bad := CheckQuarantine(src, time.Now().UTC(), QuarantineHistory(*list, src))
		skips, ran, err := ParseSkips(raw, *pyPkg)
		if err != nil {
			return err
		}
		if *pyPkg == "" && ran < *minItems {
			bad = append(bad, fmt.Sprintf("%d test results; the job states a minimum of %d (X22)", ran, *minItems))
		}
		bad = append(bad, CheckSkips(skips, entries)...)
		fmt.Fprintf(w, "skips: %d skipped\n", len(skips))
		return report(w, "skips", bad)
	case "contracts-apidiff":
		tags, err := git("tag", "--merged", "HEAD", "--list", "v[0-9]*.[0-9]*.[0-9]*")
		if err != nil {
			return err
		}
		prev := *prevTag
		if prev == "" {
			prev = LastReleaseTag(strings.Fields(tags))
		}
		if prev == "" {
			return fmt.Errorf("no release tag is reachable from HEAD; fetch the tags (the check compares against the last one)")
		}
		scratch, err := os.MkdirTemp("", "contracts-api-")
		if err != nil {
			return err
		}
		defer func() { _ = os.RemoveAll(scratch) }()
		old := filepath.Join(scratch, "prev")
		if _, err := git("worktree", "add", "--detach", "--quiet", old, prev); err != nil {
			return err
		}
		defer func() { _, _ = git("worktree", "remove", "--force", old) }()
		here, err := os.Getwd()
		if err != nil {
			return err
		}
		findings, err := CheckContractsAPI(goTool{}, old, here, scratch)
		if err != nil {
			return err
		}
		bad, note, err := JudgeContractsAPI(findings, prev, *acceptBreak)
		if err != nil {
			return err
		}
		if note != "" {
			_, _ = fmt.Fprintln(w, note)
		}
		return report(w, "contracts-api", bad)
	case "tags":
		raw, err := os.ReadFile(*lockFile)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		tags, err := gitTags()
		if err != nil {
			return err
		}
		return report(w, "tags", CheckTags(string(raw), tags))
	case "path-gates":
		if *repo == "" || *base == "" || *runsSHA == "" && *head == "" {
			return fmt.Errorf("path-gates needs -repo, -base and -head")
		}
		var files []string
		for _, pat := range []string{"workflows/*.yml", "workflows/*.yaml"} {
			m, _ := filepath.Glob(filepath.Join(*dir, pat))
			files = append(files, m...)
		}
		gates, err := LoadPathGates(os.ReadFile, files)
		if err != nil {
			return err
		}
		out, err := git("diff", "--name-only", *base, *head)
		if err != nil {
			return err
		}
		changed := strings.Fields(out)
		sha := *runsSHA
		if sha == "" {
			sha = *head
		}
		deadline := time.Now().Add(*wait)
		var runs map[string]GateRun
		for {
			if runs, err = ghGateRuns(*repo, sha); err != nil {
				return err
			}
			p := Pending(gates, changed, runs)
			if len(p) == 0 || !time.Now().Before(deadline) {
				break
			}
			time.Sleep(15 * time.Second)
		}
		_, _ = fmt.Fprintf(w, "path-gates: %d path-filtered workflow(s), %d changed file(s)\n", len(gates), len(changed))
		return report(w, "path-gates", CheckGates(gates, changed, runs))
	case "knownbad":
		// prcheck knownbad -repo-name blis-substrate [-list-only]
		reg, err := invariants.Embedded()
		if err != nil {
			return err
		}
		tests, missing := KnownBadTests(reg, *repoName)
		n := 0
		for _, t := range tests {
			n += len(t)
		}
		fmt.Fprintf(w, "knownbad: %d test(s) in %d package(s); %d ci check(s) name no known-bad input\n", n, len(tests), len(missing))
		for _, m := range missing {
			fmt.Fprintln(w, "note: "+m)
		}
		if n == 0 {
			return fmt.Errorf("the registry names no known-bad test for %s; the sweep processes at least one (X22)", *repoName)
		}
		if *listOnly {
			return nil
		}
		return report(w, "knownbad", runKnownBad(tests))
	case "licence-files":
		return report(w, "licence-files", CheckLicenceFiles(".", *repoName))
	case "vocab":
		// The closed vocabularies the Go validators enforce, as JSON, for the Python rule
		// lint (X15), so the lint compares Python literals with the Go sets, not a copy.
		enc := json.NewEncoder(w)
		enc.SetIndent("", " ")
		return enc.Encode(map[string][]string{
			"units": vocab.AllUnits(), "methods": vocab.AllMethods(), "scope_keys": vocab.AllScopeKeys(),
			"provenances": vocab.AllProvenances(), "source_kinds": vocab.AllSourceKinds(),
			"source_roles": vocab.AllSourceRoles(), "licences": vocab.RedistributableLicences(),
		})
	case "status":
		// prcheck status -repo o/n -gates .github/gates.yaml -out page.md
		reg, err := invariants.Embedded()
		if err != nil {
			return err
		}
		raw, err := os.ReadFile(*gatesFile)
		if err != nil {
			return err
		}
		gs, err := ParseGates(raw, reg)
		if err != nil {
			return err
		}
		wp, err := workflowPRPaths(".github/workflows")
		if err != nil {
			return err
		}
		bad := CheckGatePaths(gs, wp)
		last := map[string]*Run{}
		if *repo != "" {
			for _, g := range gs {
				r, err := ghLastRealRun(*repo, g.Workflow)
				if err != nil {
					return err
				}
				last[g.Name] = r
			}
		}
		name := *repo
		if i := strings.LastIndex(name, "/"); i >= 0 {
			name = name[i+1:]
		}
		page, failing := StatusPage(name, time.Now(), gs, last, reg)
		if *outFile != "" {
			if err := os.WriteFile(*outFile, []byte(page), 0o644); err != nil {
				return err
			}
		}
		if *repo != "" {
			bad = append(bad, failing...)
		}
		return report(w, "status", bad)
	case "quarantine":
		src, err := os.ReadFile(*list)
		if errors.Is(err, os.ErrNotExist) {
			src = nil
		} else if err != nil {
			return err
		}
		day := time.Now().UTC()
		if *today != "" {
			if day, err = time.Parse("2006-01-02", *today); err != nil {
				return err
			}
		}
		entries, bad := CheckQuarantine(src, day, QuarantineHistory(*list, src))
		if *skipPkg != "" && len(bad) == 0 {
			fmt.Fprintln(w, GoSkipPattern(entries, *skipPkg))
			return nil
		}
		return report(w, "quarantine", bad)
	}
	return fmt.Errorf("unknown subcommand %q", args[0])
}

func git(args ...string) (string, error) {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return "", fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(ee.Stderr)))
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func gitCommits(base, head string) ([]Commit, error) {
	out, err := git("rev-list", "--reverse", base+".."+head)
	if err != nil {
		return nil, err
	}
	var cs []Commit
	for _, h := range strings.Fields(out) {
		msg, err := git("log", "-1", "--format=%B", h)
		if err != nil {
			return nil, err
		}
		author, _ := git("log", "-1", "--format=%an <%ae>", h)
		committer, _ := git("log", "-1", "--format=%cn <%ce>", h)
		cs = append(cs, Commit{Hash: h, Message: msg, Author: author, Committer: committer})
	}
	return cs, nil
}

func trackedFiles() ([]string, error) {
	out, err := git("ls-files", "-z")
	if err != nil {
		return nil, err
	}
	var fs []string
	for _, f := range strings.Split(out, "\x00") {
		if f != "" {
			fs = append(fs, f)
		}
	}
	return fs, nil
}

type mapTree map[string]int

func (t mapTree) Lines(p string) (int, bool) { n, ok := t[p]; return n, ok }

// diskTree records every tracked file's line count and every directory above it.
func diskTree(files []string) mapTree {
	t := mapTree{}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err == nil {
			t[f] = strings.Count(string(raw), "\n")
			if len(raw) > 0 && raw[len(raw)-1] != '\n' {
				t[f]++
			}
		}
		for d := filepath.ToSlash(filepath.Dir(f)); d != "." && d != "/"; d = filepath.ToSlash(filepath.Dir(d)) {
			if _, ok := t[d]; !ok {
				t[d] = 0
			}
		}
	}
	return t
}

func treeCitations(repo string) ([]Citation, error) {
	files, err := trackedFiles()
	if err != nil {
		return nil, err
	}
	var out []Citation
	for _, f := range files {
		if strings.Contains(f, "/testdata/") || strings.HasPrefix(f, "contracts/invariants/") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil || isBinary(src) {
			continue
		}
		out = append(out, Citations(repo, f, src)...)
	}
	return out, nil
}

func isBinary(b []byte) bool {
	n := len(b)
	if n > 8000 {
		n = 8000
	}
	for _, c := range b[:n] {
		if c == 0 {
			return true
		}
	}
	return false
}

func ghPRHead(repo string, n int) (string, error) {
	out, err := exec.Command("gh", "api", fmt.Sprintf("repos/%s/pulls/%d", repo, n), "--jq", ".head.sha").Output()
	if err != nil {
		return "", fmt.Errorf("gh api: %v", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// ghTagCommit is the commit a release tag of an owner/repo points at: the tag ref, and when it
// names an annotated tag object, the commit that object tags.
func ghTagCommit(repo, tag string) (string, error) {
	out, err := exec.Command("gh", "api", fmt.Sprintf("repos/%s/commits/%s", repo, tag), "--jq", ".sha").Output()
	if err != nil {
		return "", fmt.Errorf("release tag %s of %s does not resolve: %v", tag, repo, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func ghIssueOpen(i Issue) (bool, error) {
	out, err := exec.Command("gh", "api", fmt.Sprintf("repos/%s/issues/%d", i.Repo, i.N), "--jq", ".state").Output()
	if err != nil {
		return false, fmt.Errorf("gh api: %v", err)
	}
	return strings.TrimSpace(string(out)) == "open", nil
}
