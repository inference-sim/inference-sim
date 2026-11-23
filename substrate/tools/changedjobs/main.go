// Command changedjobs says which CI jobs a change reaches through the Go packages they build.
//
// ci.yml selects the jobs of a pull request from the files it changes. A hand-written list of
// directories per job goes stale: a change under load/ reached the ledger job's code (ledger
// imports load) and the list did not say so, so the job was skipped. This tool derives the
// reach from the import graph instead. For each job it lists the packages the job's commands
// build and test, with `go list -deps -test`, and the job is selected when a changed file is
//
//   - a file in the directory of one of those packages (its sources and tests),
//   - under that directory's testdata/, which its tests read, or
//   - a file the package embeds (go:embed), wherever it lies under the package directory.
//
// Standard input is the changed paths, one per line, relative to the module root. Standard
// output is one `job=true` or `job=false` line per job, ready to append to $GITHUB_OUTPUT.
// ci.yml also keeps its path patterns for data that no package imports; the two are or-ed.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// Jobs maps a CI job to the package patterns its commands build and test. A job whose commands
// read no Go package (docs) is not listed: its patterns in ci.yml are all it needs.
var Jobs = map[string][]string{
	"contracts": {"./contracts/...", "./validate/...", "./tools/..."},
	"kernel":    {"./kernel/..."},
	"catalog":   {"./validate/cmd/validate-catalog"},
	"ledger":    {"./ledger/..."},
	"registry":  {"./validate/cmd/validate-registry"},
	"apidiff":   {"./tools/prcheck"},
}

// Pkg is a package of this module that a job depends on: its directory relative to the module
// root ("." for the root package) and the files it embeds, relative to the module root.
type Pkg struct {
	Dir      string
	Embedded []string
}

// Loader returns the module's packages that the patterns build and test, with their
// dependencies.
type Loader func(patterns []string) ([]Pkg, error)

// Select returns, for every job in jobs, whether a changed file reaches one of its packages.
func Select(changed []string, jobs map[string][]string, load Loader) (map[string]bool, error) {
	out := map[string]bool{}
	for job, patterns := range jobs {
		pkgs, err := load(patterns)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", job, err)
		}
		out[job] = false
		for _, f := range changed {
			if reaches(f, pkgs) {
				out[job] = true
				break
			}
		}
	}
	return out, nil
}

func reaches(file string, pkgs []Pkg) bool {
	file = path.Clean(file)
	for _, p := range pkgs {
		// A Markdown file beside a package's sources is not part of its build, unless the package
		// embeds it (checked below).
		if path.Dir(file) == p.Dir && !strings.HasSuffix(file, ".md") {
			return true
		}
		testdata := p.Dir + "/testdata/"
		if p.Dir == "." {
			testdata = "testdata/"
		}
		if strings.HasPrefix(file, testdata) {
			return true
		}
		for _, e := range p.Embedded {
			if file == e {
				return true
			}
		}
	}
	return false
}

// goList runs `go list -deps -test` in root and keeps the packages of the main module.
func goList(root string) Loader {
	return func(patterns []string) ([]Pkg, error) {
		rootAbs, err := filepath.Abs(root)
		if err != nil {
			return nil, err
		}
		if r, err := filepath.EvalSymlinks(rootAbs); err == nil {
			rootAbs = r
		}
		args := append([]string{"list", "-deps", "-test", "-json=Dir,Module,EmbedFiles,TestEmbedFiles,XTestEmbedFiles"}, patterns...)
		cmd := exec.Command("go", args...)
		cmd.Dir = rootAbs
		cmd.Stderr = os.Stderr
		stdout, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("go list: %w", err)
		}
		var pkgs []Pkg
		dec := json.NewDecoder(strings.NewReader(string(stdout)))
		for {
			var p struct {
				Dir             string
				Module          *struct{ Main bool }
				EmbedFiles      []string
				TestEmbedFiles  []string
				XTestEmbedFiles []string
			}
			if err := dec.Decode(&p); err == io.EOF {
				break
			} else if err != nil {
				return nil, err
			}
			if p.Module == nil || !p.Module.Main {
				continue
			}
			dir := p.Dir
			if r, err := filepath.EvalSymlinks(dir); err == nil {
				dir = r
			}
			rel, err := filepath.Rel(rootAbs, dir)
			if err != nil || strings.HasPrefix(rel, "..") {
				return nil, fmt.Errorf("package directory %s is outside %s", p.Dir, rootAbs)
			}
			rel = filepath.ToSlash(rel)
			q := Pkg{Dir: rel}
			for _, group := range [][]string{p.EmbedFiles, p.TestEmbedFiles, p.XTestEmbedFiles} {
				for _, e := range group {
					q.Embedded = append(q.Embedded, path.Join(rel, e))
				}
			}
			pkgs = append(pkgs, q)
		}
		return pkgs, nil
	}
}

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	var changed []string
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		if l := strings.TrimSpace(sc.Text()); l != "" {
			changed = append(changed, l)
		}
	}
	got, err := Select(changed, Jobs, goList(root))
	if err != nil {
		fmt.Fprintln(os.Stderr, "changedjobs:", err)
		os.Exit(1)
	}
	var names []string
	for j := range got {
		names = append(names, j)
	}
	sort.Strings(names)
	for _, j := range names {
		fmt.Printf("%s=%v\n", j, got[j])
	}
}
