package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// The path-filtered gates (X16, X17). A workflow whose pull_request trigger lists paths
// starts no run on a change that touches none of them, so branch protection cannot require its
// check run by name: a required check that never starts blocks the merge. One always-run
// aggregator job in ci.yml reads the gates from the workflow files themselves (never from a
// copy of their path lists), decides from the change's file list whether each was triggered,
// and passes only if every triggered gate ran and passed. A gate whose paths the change did
// not touch counts as passed.

// PathGate is one path-filtered workflow: its file, its pull_request paths, and the check
// runs its jobs make.
type PathGate struct {
	File  string
	Paths []string
	Jobs  []string // check-run names
}

// ParsePathGate reads a workflow. It returns ok false when the workflow has no
// pull_request path filter, which is a workflow that runs on every pull request or on none.
func ParsePathGate(file string, src []byte) (g PathGate, ok bool, err error) {
	var wf struct {
		On   map[string]yaml.Node `yaml:"on"`
		Jobs yaml.Node            `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(src, &wf); err != nil {
		return g, false, fmt.Errorf("%s: %w", file, err)
	}
	pr, has := wf.On["pull_request"]
	if !has {
		return g, false, nil
	}
	var trig struct {
		Paths       []string `yaml:"paths"`
		PathsIgnore []string `yaml:"paths-ignore"`
	}
	if err := pr.Decode(&trig); err != nil {
		return g, false, fmt.Errorf("%s: pull_request: %w", file, err)
	}
	if len(trig.PathsIgnore) > 0 {
		return g, false, fmt.Errorf("%s: paths-ignore is not supported by the gate aggregator; list the paths (X16)", file)
	}
	if len(trig.Paths) == 0 {
		return g, false, nil
	}
	for _, p := range trig.Paths {
		if strings.HasPrefix(p, "!") {
			return g, false, fmt.Errorf("%s: a negated path %q is not supported by the gate aggregator (X16)", file, p)
		}
	}
	g = PathGate{File: file, Paths: trig.Paths}
	for i := 0; i+1 < len(wf.Jobs.Content); i += 2 {
		id := wf.Jobs.Content[i].Value
		var job struct {
			Name string `yaml:"name"`
		}
		if err := wf.Jobs.Content[i+1].Decode(&job); err != nil {
			return g, false, fmt.Errorf("%s: job %s: %w", file, id, err)
		}
		if job.Name == "" {
			job.Name = id
		}
		g.Jobs = append(g.Jobs, job.Name)
	}
	if len(g.Jobs) == 0 {
		return g, false, fmt.Errorf("%s: a path-filtered workflow has no jobs", file)
	}
	return g, true, nil
}

// LoadPathGates reads every workflow under dir/workflows and returns the path-filtered ones.
func LoadPathGates(read func(string) ([]byte, error), files []string) ([]PathGate, error) {
	var out []PathGate
	sort.Strings(files)
	for _, f := range files {
		src, err := read(f)
		if err != nil {
			return nil, err
		}
		g, ok, err := ParsePathGate(f, src)
		if err != nil {
			return nil, err
		}
		if ok {
			out = append(out, g)
		}
	}
	return out, nil
}

// pathPattern turns a GitHub path filter into a regular expression: "**" matches across
// directories, "*" within one, "?" one character other than "/".
func pathPattern(p string) *regexp.Regexp {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(p); i++ {
		switch c := p[i]; {
		case c == '*' && i+1 < len(p) && p[i+1] == '*':
			i++
			if i+1 < len(p) && p[i+1] == '/' {
				// "**/" is zero or more directories: "a/**/b" matches "a/b".
				b.WriteString("(?:.*/)?")
				i++
			} else {
				b.WriteString(".*")
			}
		case c == '*':
			b.WriteString("[^/]*")
		case c == '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	return regexp.MustCompile(b.String())
}

// Triggered reports whether any changed path matches one of the gate's paths.
func (g PathGate) Triggered(changed []string) bool {
	for _, p := range g.Paths {
		re := pathPattern(p)
		for _, c := range changed {
			if re.MatchString(c) {
				return true
			}
		}
	}
	return false
}

// GateRun is a check run's state: Done when completed, with its Conclusion.
type GateRun struct {
	Done       bool
	Conclusion string
}

// Pending lists the check runs of the triggered gates that have not completed, so the caller
// can wait for them.
func Pending(gates []PathGate, changed []string, runs map[string]GateRun) []string {
	var out []string
	for _, g := range gates {
		if !g.Triggered(changed) {
			continue
		}
		for _, j := range g.Jobs {
			if r, ok := runs[j]; !ok || !r.Done {
				out = append(out, j)
			}
		}
	}
	return out
}

// CheckGates returns a finding for each triggered gate whose check run is missing, not
// finished or not successful. A gate none of whose paths the change touched is passed.
func CheckGates(gates []PathGate, changed []string, runs map[string]GateRun) []string {
	var bad []string
	for _, g := range gates {
		if !g.Triggered(changed) {
			continue
		}
		for _, j := range g.Jobs {
			r, ok := runs[j]
			switch {
			case !ok:
				bad = append(bad, fmt.Sprintf("%s: the change touches %s's paths, but no check run %q exists on the head (X16, X17)", filepath.Base(g.File), filepath.Base(g.File), j))
			case !r.Done:
				bad = append(bad, fmt.Sprintf("%s: check run %q has not completed (X16, X17)", filepath.Base(g.File), j))
			case r.Conclusion != "success":
				bad = append(bad, fmt.Sprintf("%s: check run %q concluded %s (X16, X17)", filepath.Base(g.File), j, r.Conclusion))
			}
		}
	}
	return bad
}

// ChangesSuffix ends the check-run name of the cheap first job every gate workflow has
// ("rederive changes"; a bare "changes" would collide with ci.yml's job of that name). On a merge_group
// event GitHub does not apply a trigger's paths, so the workflow starts for every queue entry;
// the changes job compares the merge group's base_sha..head_sha with the workflow's own
// pull_request paths (prcheck gate-changed), and the heavy jobs are skipped when none changed.
const ChangesSuffix = " changes"

func isChangesJob(name string) bool { return strings.HasSuffix(name, ChangesSuffix) }

// QueuePending lists the check runs of every gate that have not completed, on the
// merge-queue commit, where each gate workflow runs for every entry.
func QueuePending(gates []PathGate, runs map[string]GateRun) []string {
	var out []string
	for _, g := range gates {
		for _, j := range g.Jobs {
			if r, ok := runs[j]; !ok || !r.Done {
				out = append(out, j)
			}
		}
	}
	return out
}

// CheckQueueGates judges the gates on the merge-queue commit (X17, X22), where CI must have run
// on the tree being merged. Every gate workflow must have run there, always: the changes job
// must have succeeded. When the change touches the gate's paths (changed is the diff of the
// queue entry against its base), every job must have succeeded; when it does not, the heavy
// jobs must have succeeded or been skipped by the changes job. A job that failed, was
// cancelled or never started is a finding either way.
func CheckQueueGates(gates []PathGate, changed []string, runs map[string]GateRun) []string {
	var bad []string
	for _, g := range gates {
		name := filepath.Base(g.File)
		touched := g.Triggered(changed)
		if !slices.ContainsFunc(g.Jobs, isChangesJob) {
			bad = append(bad, fmt.Sprintf("%s: a gate workflow has no job named \"<workflow> changes\", so it cannot run on the merge queue (X17)", name))
			continue
		}
		for _, j := range g.Jobs {
			r, ok := runs[j]
			switch {
			case !ok:
				bad = append(bad, fmt.Sprintf("%s: no check run %q exists on the merge-queue commit (X17, X22)", name, j))
			case !r.Done:
				bad = append(bad, fmt.Sprintf("%s: check run %q has not completed (X17, X22)", name, j))
			case r.Conclusion == "success":
			case r.Conclusion == "skipped" && !touched && !isChangesJob(j):
			default:
				bad = append(bad, fmt.Sprintf("%s: check run %q concluded %s on the merge-queue commit (X17, X22)", name, j, r.Conclusion))
			}
		}
	}
	return bad
}

// ghGateRuns lists a commit's check runs by name. When a name has several runs, one that has
// not completed, or has not succeeded, wins over a success, so a re-run cannot hide a failure.
func ghGateRuns(repo, sha string) (map[string]GateRun, error) {
	var res struct {
		CheckRuns []struct {
			Name       string `json:"name"`
			Status     string `json:"status"`
			Conclusion string `json:"conclusion"`
		} `json:"check_runs"`
	}
	if err := ghJSON(&res, "api", fmt.Sprintf("repos/%s/commits/%s/check-runs?per_page=100", repo, sha)); err != nil {
		return nil, err
	}
	runs := map[string]GateRun{}
	for _, r := range res.CheckRuns {
		g := GateRun{Done: r.Status == "completed", Conclusion: r.Conclusion}
		if old, ok := runs[r.Name]; ok && (!old.Done || old.Conclusion != "success") {
			continue
		}
		runs[r.Name] = g
	}
	return runs, nil
}
