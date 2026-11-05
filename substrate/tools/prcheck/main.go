// Command prcheck holds the checks the shared workflows in blis-substrate/.github/workflows
// run in every BLIS repository (design §13.2). Each subcommand reads local files, git, and,
// where it must, the GitHub API through the gh command (GH_TOKEN set by the workflow).
//
//	prcheck commits -base <rev> -head <rev> [-signatures <file>]
//	    X20, over the commits base..head: signed, a person's DCO sign-off (none on
//	    release-bot commits), at most one Assisted-by line, no Co-authored-by naming a
//	    tool. -signatures is a file of "<hash> <G|N>" lines from the GitHub API, used in
//	    place of git verify-commit when given.
//	prcheck workflows [-dir .github]
//	    X21, X22, X16: actions pinned by commit, pip --require-hashes, no ignored exit
//	    codes, no skipped tests in a job, gate trigger paths as declared.
//	prcheck description -body <file> -repo <owner/name> -head <rev> -base <rev>
//	    X14: listed hashes are branch heads; the pull request merges cleanly.
//	prcheck merge-collect -repo <o/n> -pr <n> -merge <sha> -head <sha> -base <sha> -required <a,b>
//	    Print the merge check's input, from the GitHub API, as JSON.
//	prcheck merge -input <file.json>
//	    X22 and X24, over the verdicts and review threads the workflow collected.
//	prcheck issues -repo <owner/name> [-closes <body file>] [-resolve]
//	    X28: the issues cited as reasons for a refusal or gap. With -closes, fail if the
//	    body closes a cited issue; with -resolve, fail on any cited issue that is closed.
//	prcheck versions [-baseline .github/version-literals.txt]
//	    X23: version literals outside the sources of versions.
//	prcheck docrefs [-doc-baseline .github/doc-references.txt]
//	    X26: paths, line ranges and table rows in Markdown documents. Findings listed in
//	    the baseline (without their line numbers) are reported as known; the baseline only
//	    shrinks.
//	prcheck skips -results <file> [-python-package <dir>] [-min <n>]
//	    X16, X22: a skipped test not on the quarantine list fails, and so does a go test
//	    run with fewer than -min results.
//	prcheck questions -body <file> -questions <file> -out <file> -unanswered <file>
//	    Design §13.2: put the matching review questions in the body's "## Review
//	    questions" section, keep given answers, and list those still unanswered.
//	prcheck tags [-lock .github/tags.lock]
//	    LIF-1: every tag resolves to the commit the lock records, and every tag is recorded.
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
		return fmt.Errorf("usage: prcheck commits|workflows|description|merge|issues|versions|docrefs|quarantine ...")
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	base := fs.String("base", "", "base revision")
	head := fs.String("head", "HEAD", "head revision")
	sigs := fs.String("signatures", "", "file of '<hash> G|N' signature results")
	dir := fs.String("dir", ".github", "directory of workflows and actions")
	body := fs.String("body", "", "pull-request body file")
	repo := fs.String("repo", "", "owner/name")
	input := fs.String("input", "", "merge-check input JSON")
	closes := fs.String("closes", "", "pull-request body file whose Closes lines are checked")
	resolve := fs.Bool("resolve", false, "resolve each cited issue through the GitHub API")
	baseline := fs.String("baseline", ".github/version-literals.txt", "version-literal baseline")
	docBaseline := fs.String("doc-baseline", ".github/doc-references.txt", "documentation-reference baseline")
	list := fs.String("list", ".github/quarantine.yaml", "quarantine list")
	skipPkg := fs.String("skip-pattern", "", "print the go test -skip pattern for this package")
	lockFile := fs.String("lock", ".github/tags.lock", "the recorded tags")
	prNum := fs.Int("pr", 0, "pull request number")
	mergeSHA := fs.String("merge", "", "the merge-queue commit")
	requiredList := fs.String("required", "", "comma-separated names of required checks")
	questionsFile := fs.String("questions", "", "the questions invariants questions printed")
	outFile := fs.String("out", "", "where to write the new body")
	unanswered := fs.String("unanswered", "", "where to list the unanswered questions")
	results := fs.String("results", "", "go test -json or pytest -rs output")
	pyPkg := fs.String("python-package", "", "the directory pytest ran in, for its results")
	minItems := fs.Int("min", 1, "the minimum number of go test results")
	today := fs.String("today", "", "the date to judge expiry on (YYYY-MM-DD); default today, UTC")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	switch args[0] {
	case "commits":
		if *base == "" {
			return fmt.Errorf("commits needs -base")
		}
		cs, err := gitCommits(*base, *head, *sigs)
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
			Heads: ghPRHead, Merges: mergeErr == nil,
		}))
	case "merge-collect":
		// prcheck merge-collect -repo o/n -pr N -merge <sha> -head <sha> -base <sha> -required a,b
		if *repo == "" || *prNum == 0 || *mergeSHA == "" || *base == "" {
			return fmt.Errorf("merge-collect needs -repo, -pr, -merge, -head and -base")
		}
		var req []string
		for _, r := range strings.Split(*requiredList, ",") {
			if r = strings.TrimSpace(r); r != "" {
				req = append(req, r)
			}
		}
		in, err := collectMerge(*repo, *prNum, *mergeSHA, *head, *base, req)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(w)
		enc.SetIndent("", " ")
		return enc.Encode(in)
	case "merge":
		raw, err := os.ReadFile(*input)
		if err != nil {
			return err
		}
		var in MergeInput
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&in); err != nil {
			return fmt.Errorf("%s: %w", *input, err)
		}
		return report(w, "merge-check", CheckMerge(in))
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
		bl := map[string]bool{}
		if raw, err := os.ReadFile(*baseline); err == nil {
			bl = ParseBaseline(string(raw))
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return report(w, "version-literals", CheckVersions(found, bl))
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
		bl := map[string]bool{}
		if raw, err := os.ReadFile(*docBaseline); err == nil {
			bl = ParseBaseline(string(raw))
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return report(w, "doc-references", CheckBaseline(bad, bl, "X26"))
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
		entries, bad := CheckQuarantine(src, time.Now().UTC(), nil)
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
	case "questions":
		raw, err := os.ReadFile(*body)
		if err != nil {
			return err
		}
		qs, err := os.ReadFile(*questionsFile)
		if err != nil {
			return err
		}
		nb, open := MergeQuestions(string(raw), string(qs))
		if err := os.WriteFile(*outFile, []byte(nb), 0o644); err != nil {
			return err
		}
		return os.WriteFile(*unanswered, []byte(strings.Join(open, "\n")), 0o644)
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
		entries, bad := CheckQuarantine(src, day, nil)
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

func gitCommits(base, head, sigFile string) ([]Commit, error) {
	out, err := git("rev-list", "--reverse", base+".."+head)
	if err != nil {
		return nil, err
	}
	sig := map[string]bool{}
	if sigFile != "" {
		raw, err := os.ReadFile(sigFile)
		if err != nil {
			return nil, err
		}
		for _, l := range strings.Split(string(raw), "\n") {
			if f := strings.Fields(l); len(f) == 2 {
				sig[f[0]] = f[1] == "G"
			}
		}
	}
	var cs []Commit
	for _, h := range strings.Fields(out) {
		msg, err := git("log", "-1", "--format=%B", h)
		if err != nil {
			return nil, err
		}
		author, _ := git("log", "-1", "--format=%an <%ae>", h)
		committer, _ := git("log", "-1", "--format=%cn <%ce>", h)
		signed, ok := sig[h]
		if !ok {
			// %G? is G for a good signature, U good but untrusted key; both are signed.
			st, _ := git("log", "-1", "--format=%G?", h)
			signed = st == "G" || st == "U"
		}
		cs = append(cs, Commit{Hash: h, Message: msg, Signed: signed, Author: author, Committer: committer})
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

func ghIssueOpen(i Issue) (bool, error) {
	out, err := exec.Command("gh", "api", fmt.Sprintf("repos/%s/issues/%d", i.Repo, i.N), "--jq", ".state").Output()
	if err != nil {
		return false, fmt.Errorf("gh api: %v", err)
	}
	return strings.TrimSpace(string(out)) == "open", nil
}
