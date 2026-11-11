package main

import (
	"fmt"
	"path/filepath"
	"regexp"
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

// gateRunRecord is one check run as the API returns it. ID is given when the run is created and
// grows with time, so it orders the attempts of a name; completed_at is empty until a run ends.
type gateRunRecord struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
}

// foldGateRuns keeps, for each name, its newest attempt (the highest id). GitHub keeps the runs
// of an earlier attempt when a workflow is re-run on the same head. A gate protects the current
// result for the head, so an earlier failure that a later run replaced must not hold it back,
// and an earlier pass must not hide a later failure. A newest attempt that has not completed
// stays not done, and a name with no run is absent: CheckGates refuses both. A newest attempt
// that was cancelled (a workflow's concurrency group replaces a run with a newer one) is done and
// did not succeed, so CheckGates refuses it as well: a cancelled run is never a pass, and the
// newer run that replaced it is the one this reads.
func foldGateRuns(recs []gateRunRecord) map[string]GateRun {
	newest := map[string]gateRunRecord{}
	for _, r := range recs {
		if old, ok := newest[r.Name]; !ok || r.ID > old.ID {
			newest[r.Name] = r
		}
	}
	runs := make(map[string]GateRun, len(newest))
	for n, r := range newest {
		runs[n] = GateRun{Done: r.Status == "completed", Conclusion: r.Conclusion}
	}
	return runs
}

// ghGateRuns lists a commit's check runs by name, each at its newest attempt (foldGateRuns).
func ghGateRuns(repo, sha string) (map[string]GateRun, error) {
	var res struct {
		CheckRuns []gateRunRecord `json:"check_runs"`
	}
	if err := ghJSON(&res, "api", fmt.Sprintf("repos/%s/commits/%s/check-runs?per_page=100", repo, sha)); err != nil {
		return nil, err
	}
	return foldGateRuns(res.CheckRuns), nil
}
