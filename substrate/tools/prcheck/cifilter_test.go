package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"gopkg.in/yaml.v3"
)

// changesFilterScript is the shell of the step `filter` of job `changes` in ci.yml, the script
// that decides which jobs a pull request runs.
func changesFilterScript(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	var w struct {
		Jobs map[string]struct {
			Steps []struct {
				ID  string `yaml:"id"`
				Run string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &w); err != nil {
		t.Fatal(err)
	}
	for _, s := range w.Jobs["changes"].Steps {
		if s.ID == "filter" {
			return s.Run
		}
	}
	t.Fatal("ci.yml has no step `filter` in job `changes`")
	return ""
}

// hermeticGo is the environment every go command the filter runs gets: the module root is the
// repository's, no workspace and no flags of the developer's shell apply, and the toolchain is the
// installed one, never downloaded.
var hermeticGo = []string{"GOFLAGS=", "GOWORK=off", "GOTOOLCHAIN=local", "GO111MODULE=on"}

var (
	changedJobsOnce sync.Once
	changedJobsBin  string
	changedJobsErr  error
)

// changedJobsBinary builds tools/changedjobs once. The filter runs it as `go run ./tools/changedjobs .`
// from the checkout, but the test runs the filter in a scratch repository that has no module, where
// go would find no go.mod (or read a file the test wrote as one), so the test runs the built
// binary on the real module root instead.
func changedJobsBinary(t *testing.T) (bin, root string) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	changedJobsOnce.Do(func() {
		dir, err := os.MkdirTemp("", "changedjobs")
		if err != nil {
			changedJobsErr = err
			return
		}
		changedJobsBin = filepath.Join(dir, "changedjobs")
		cmd := exec.Command("go", "build", "-o", changedJobsBin, "./tools/changedjobs")
		cmd.Dir = root
		cmd.Env = append(os.Environ(), hermeticGo...)
		if out, err := cmd.CombinedOutput(); err != nil {
			changedJobsErr = fmt.Errorf("go build ./tools/changedjobs: %v\n%s", err, out)
		}
	})
	if changedJobsErr != nil {
		t.Fatal(changedJobsErr)
	}
	return changedJobsBin, root
}

// runFilter commits the files into a scratch repository on top of a base commit, runs the
// workflow's own filter script against that base, and returns the outputs that came out true.
func runFilter(t *testing.T, script string, files []string) []string {
	t.Helper()
	dir := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"}, args...)...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init", "-q")
	write := func(name, text string) {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("base.txt", "base\n")
	git("add", "-A")
	git("commit", "-q", "-m", "base")
	base := strings.TrimSpace(gitOut(t, dir, "rev-parse", "HEAD"))
	for _, f := range files {
		write(f, "changed\n")
	}
	git("add", "-A")
	git("commit", "-q", "-m", "change")
	outFile := filepath.Join(dir, "..", filepath.Base(dir)+".out")
	// The filter asks tools/changedjobs which jobs the files reach; see changedJobsBinary.
	const invocation = "go run ./tools/changedjobs ."
	if strings.Contains(script, invocation) {
		bin, root := changedJobsBinary(t)
		script = strings.Replace(script, invocation, "'"+bin+"' '"+root+"'", 1)
	}
	cmd := exec.Command("bash", "-c", script)
	cmd.Dir = dir
	cmd.Env = append(append(os.Environ(), hermeticGo...), "BASE="+base, "GITHUB_OUTPUT="+outFile)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("filter: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatal(err)
	}
	var on []string
	for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if k, ok := strings.CutSuffix(l, "=true"); ok {
			on = append(on, k)
		}
	}
	sort.Strings(on)
	return on
}

func gitOut(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

// Which jobs a change runs: the workflow's own filter, executed on sample change lists. The
// first case is the one that motivated narrowing (a tag lock line ran the cost model); the
// others state what each directory's change must and must not run.
func TestChangesFilterRunsEachJobForTheFilesItReads(t *testing.T) {
	script := changesFilterScript(t)
	all := []string{"apidiff", "catalog", "contracts", "golang", "kernel", "ledger", "registry"}
	cases := []struct {
		name  string
		files []string
		want  []string
	}{
		{"a cost model source file", []string{"kernel/step.go"}, []string{"golang", "kernel"}},
		{"a registry script", []string{"registry/scripts/fit_fabric.py"}, []string{"contracts", "registry"}},
		{"documentation only", []string{"README.md"}, nil},
		{"the CI workflow", []string{".github/workflows/ci.yml"}, all},
		{"a contracts source file", []string{"contracts/spec/model/model.go"}, []string{"apidiff", "catalog", "contracts", "golang", "kernel", "ledger", "registry"}},
		{"a coefficient set, which the cost model embeds", []string{"registry/coefficients/fabric.yaml"}, []string{"contracts", "kernel", "registry"}},
		{"a catalog model graph, which the cost model embeds", []string{"catalog/models/m/graph.yaml"}, []string{"catalog", "contracts", "kernel"}},
		{"the shard weights", []string{"tools/shard/weights-kernel.txt"}, []string{"contracts", "kernel"}},
		{"go.mod", []string{"go.mod"}, all},
		{"a ledger source file", []string{"ledger/check.go"}, []string{"contracts", "golang", "ledger"}},
		// ledger imports load: a change to the loader reaches the ledger job although ci.yml's
		// path patterns do not name load/ for it (the hole blis-substrate#353 closed).
		{"the loader, which the ledger imports", []string{"load/load.go"}, []string{"contracts", "golang", "kernel", "ledger"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := runFilter(t, script, c.files)
			if strings.Join(got, ",") != strings.Join(c.want, ",") {
				t.Errorf("%v runs %v, want %v", c.files, got, c.want)
			}
		})
	}
	// With no usable base (a manual run, a new branch) every job runs.
	t.Run("no base", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "out")
		cmd := exec.Command("bash", "-c", script)
		cmd.Dir = t.TempDir()
		cmd.Env = append(os.Environ(), "BASE=", "GITHUB_OUTPUT="+out)
		if b, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%v\n%s", err, b)
		}
		raw, _ := os.ReadFile(out)
		for _, k := range all {
			if !strings.Contains(string(raw), k+"=true") {
				t.Errorf("a run with no base did not select %s", k)
			}
		}
	})
}
