package invariants

import (
	"bytes"
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed *.yaml
var embedded embed.FS

// Repositories are the seven BLIS repositories of design §2.1, by repository name. A
// check, a scope entry and a review trigger may name only these.
var Repositories = []string{
	"accuracy-dashboard",
	"blis-config-search",
	"blis-sglang",
	"blis-substrate",
	"blis-vllm",
	"inference-sim",
	"llm-d-router",
}

// Families maps each family file to the identifier prefix of the invariants it holds,
// in the order the design defines them.
var Families = []Family{
	{File: "x.yaml", Prefix: "X"},
	{File: "sub.yaml", Prefix: "SUB-"},
	{File: "vlm.yaml", Prefix: "VLM-"},
	{File: "sgl.yaml", Prefix: "SGL-"},
	{File: "rtr.yaml", Prefix: "RTR-"},
	{File: "sim.yaml", Prefix: "SIM-"},
	{File: "cfg.yaml", Prefix: "CFG-"},
	{File: "fid.yaml", Prefix: "FID-"},
	{File: "dev.yaml", Prefix: "DEV-"},
	{File: "val.yaml", Prefix: "VAL-"},
	{File: "d.yaml", Prefix: "D"},
	{File: "cov.yaml", Prefix: "COV-"},
	{File: "lif.yaml", Prefix: "LIF-"},
}

// A Family is one registry file and the identifier prefix its records carry.
type Family struct {
	File   string
	Prefix string
}

// Status values of an invariant and of a check (design §13.1).
const (
	StatusCI      = "ci"
	StatusPartial = "partial"
	StatusNew     = "new"
	StatusReview  = "review"
)

// When a check runs.
const (
	RunsPullRequest = "pull_request"
	RunsNightly     = "nightly"
	RunsRelease     = "release"
	RunsSettings    = "settings"
)

// Invariant is one record of the registry.
type Invariant struct {
	ID        string   `yaml:"id"`
	Section   string   `yaml:"section"`
	Title     string   `yaml:"title,omitempty"`
	Statement string   `yaml:"statement"`
	Scope     []string `yaml:"scope"`
	Checks    []Check  `yaml:"checks"`
	Review    []Review `yaml:"review"`
	Status    string   `yaml:"status"`
	Target    Target   `yaml:"target"`
	Note      string   `yaml:"note,omitempty"`
	family    string   // the file the record came from
	line      int      // the record's line in that file
}

// Check is one check of an invariant, run in one repository.
type Check struct {
	Repo     string `yaml:"repo"`
	Check    string `yaml:"check"`
	Runs     string `yaml:"runs"`
	Status   string `yaml:"status"`
	Test     string `yaml:"test,omitempty"`
	KnownBad string `yaml:"known_bad,omitempty"`
}

// Review is a yes/no question a reviewer with write access answers on a pull request (design §13.2).
type Review struct {
	Question string  `yaml:"question"`
	Reviewer string  `yaml:"reviewer"`
	When     string  `yaml:"when"`
	Trigger  Trigger `yaml:"trigger"`
}

// Trigger says when CI adds a review question to a pull request: when the pull request,
// in one of Repos, changes a path that matches a glob in Paths and none in Exclude. A
// manual trigger has no paths; a reviewer adds the question by label.
type Trigger struct {
	Manual  bool     `yaml:"manual,omitempty"`
	Repos   []string `yaml:"repos,omitempty"`
	Paths   []string `yaml:"paths,omitempty"`
	Exclude []string `yaml:"exclude,omitempty"`
}

// Target is the status an invariant must reach, and the release plan milestone by which
// it must reach it. Milestone is nil when the status already equals the target.
type Target struct {
	Status    string `yaml:"status"`
	Milestone *int   `yaml:"milestone"`
}

// Registry is the loaded, validated set of invariants, in design order.
type Registry struct {
	Invariants []Invariant
	byID       map[string]int
}

// Get returns the invariant with the given identifier.
func (r *Registry) Get(id string) (Invariant, bool) {
	i, ok := r.byID[id]
	if !ok {
		return Invariant{}, false
	}
	return r.Invariants[i], true
}

// Embedded loads the registry compiled into this package.
func Embedded() (*Registry, error) {
	return LoadFS(embedded)
}

// Load reads the registry from a directory with the layout of contracts/invariants/.
func Load(dir string) (*Registry, error) {
	return LoadFS(os.DirFS(dir))
}

// LoadFS reads the registry from the root of fsys. Every family file must exist, and no
// other .yaml file may: a stray file would hold records no reader sees.
func LoadFS(fsys fs.FS) (*Registry, error) {
	known := map[string]bool{}
	for _, f := range Families {
		known[f.File] = true
	}
	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".yaml") || strings.HasSuffix(e.Name(), ".yml") {
			if !known[e.Name()] {
				return nil, fmt.Errorf("%s: not a registry family file; the families are %s", e.Name(), familyFiles())
			}
		}
	}
	reg := &Registry{byID: map[string]int{}}
	var errs []error
	for _, f := range Families {
		raw, err := fs.ReadFile(fsys, f.File)
		if err != nil {
			return nil, err
		}
		recs, err := decode(f, raw)
		if err != nil {
			return nil, err
		}
		for _, inv := range recs {
			if prev, dup := reg.byID[inv.ID]; dup {
				p := reg.Invariants[prev]
				errs = append(errs, fmt.Errorf("%s:%d: %s is defined again; first at %s:%d", inv.family, inv.line, inv.ID, p.family, p.line))
				continue
			}
			reg.byID[inv.ID] = len(reg.Invariants)
			reg.Invariants = append(reg.Invariants, inv)
		}
	}
	for _, inv := range reg.Invariants {
		for _, e := range inv.validate() {
			errs = append(errs, fmt.Errorf("%s:%d: %s: %w", inv.family, inv.line, inv.ID, e))
		}
	}
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return reg, nil
}

func familyFiles() string {
	var s []string
	for _, f := range Families {
		s = append(s, f.File)
	}
	return strings.Join(s, ", ")
}

// decode parses one family file strictly: an unknown key, a duplicate key or a second
// document is refused (SUB-12).
func decode(f Family, raw []byte) ([]Invariant, error) {
	if bytes.ContainsRune(raw, '\r') {
		return nil, fmt.Errorf("%s: contains a carriage return", f.File)
	}
	var root yaml.Node
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&root); err != nil {
		return nil, fmt.Errorf("%s: %w", f.File, err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%s: holds more than one YAML document", f.File)
	}
	if len(root.Content) != 1 || root.Content[0].Kind != yaml.SequenceNode {
		return nil, fmt.Errorf("%s: the top level must be a list of records", f.File)
	}
	if err := refuseDuplicateKeys(root.Content[0], f.File); err != nil {
		return nil, err
	}
	strict := yaml.NewDecoder(bytes.NewReader(raw))
	strict.KnownFields(true)
	var recs []Invariant
	if err := strict.Decode(&recs); err != nil {
		return nil, fmt.Errorf("%s: %w", f.File, err)
	}
	for i := range recs {
		recs[i].family = f.File
		recs[i].line = root.Content[0].Content[i].Line
		if !hasFamilyPrefix(recs[i].ID, f.Prefix) {
			return nil, fmt.Errorf("%s:%d: %q does not belong in this file, which holds %s<n>", f.File, recs[i].line, recs[i].ID, f.Prefix)
		}
	}
	return recs, nil
}

func refuseDuplicateKeys(n *yaml.Node, file string) error {
	if n.Kind == yaml.MappingNode {
		seen := map[string]bool{}
		for i := 0; i+1 < len(n.Content); i += 2 {
			k := n.Content[i].Value
			if seen[k] {
				return fmt.Errorf("%s:%d: duplicate key %q", file, n.Content[i].Line, k)
			}
			seen[k] = true
		}
	}
	if n.Kind == yaml.AliasNode {
		return fmt.Errorf("%s:%d: aliases are not allowed", file, n.Line)
	}
	for _, c := range n.Content {
		if err := refuseDuplicateKeys(c, file); err != nil {
			return err
		}
	}
	return nil
}

var idPattern = regexp.MustCompile(`^(X|D|[A-Z]{3}-)([1-9][0-9]*)$`)

func hasFamilyPrefix(id, prefix string) bool {
	m := idPattern.FindStringSubmatch(id)
	return m != nil && m[1] == prefix
}

var sectionPattern = regexp.MustCompile(`^§[0-9]+(\.[0-9]+)?$`)

func (inv Invariant) validate() []error {
	var errs []error
	bad := func(format string, args ...any) { errs = append(errs, fmt.Errorf(format, args...)) }
	if !sectionPattern.MatchString(inv.Section) {
		bad("section %q is not a design section such as §5 or §4.1", inv.Section)
	}
	if strings.TrimSpace(inv.Statement) == "" {
		bad("statement is empty")
	}
	if len(inv.Checks) == 0 {
		bad("has no check; a rule with no check is not enforced (design §13.1)")
	}
	nci := 0
	scope := map[string]bool{}
	for i, c := range inv.Checks {
		at := fmt.Sprintf("checks[%d]", i)
		if !isRepo(c.Repo) {
			bad("%s.repo %q is not a BLIS repository; the repositories are %s", at, c.Repo, strings.Join(Repositories, ", "))
		}
		scope[c.Repo] = true
		if strings.TrimSpace(c.Check) == "" {
			bad("%s.check is empty", at)
		}
		switch c.Runs {
		case RunsPullRequest, RunsNightly, RunsRelease, RunsSettings:
		default:
			bad("%s.runs %q is not one of pull_request, nightly, release, settings", at, c.Runs)
		}
		switch c.Status {
		case StatusCI:
			nci++
			if strings.TrimSpace(c.Test) == "" {
				bad("%s has status ci but names no test", at)
			}
		case StatusNew:
		default:
			bad("%s.status %q is not ci or new", at, c.Status)
		}
	}
	want := make([]string, 0, len(scope))
	for r := range scope {
		want = append(want, r)
	}
	sort.Strings(want)
	if strings.Join(want, ",") != strings.Join(inv.Scope, ",") {
		bad("scope %v must equal the sorted repositories its checks name, %v", inv.Scope, want)
	}
	switch derived := statusOf(nci, len(inv.Checks)); {
	case inv.Status == StatusReview:
		bad("status review: no invariant in the design is enforced by review alone (design §13.1)")
	case inv.Status != derived:
		bad("status %q disagrees with its checks: %d of %d are ci, which is %q", inv.Status, nci, len(inv.Checks), derived)
	}
	for i, r := range inv.Review {
		at := fmt.Sprintf("review[%d]", i)
		if strings.TrimSpace(r.Question) == "" || strings.TrimSpace(r.Reviewer) == "" || strings.TrimSpace(r.When) == "" {
			bad("%s needs question, reviewer and when", at)
		}
		errs = append(errs, r.Trigger.validate(at)...)
	}
	if inv.Target.Status != StatusCI {
		bad("target.status %q: every invariant's target is ci", inv.Target.Status)
	}
	switch {
	case inv.Target.Milestone == nil && inv.Status != inv.Target.Status:
		bad("target.milestone is null, but status %q has not reached target %q", inv.Status, inv.Target.Status)
	case inv.Target.Milestone != nil && inv.Status == inv.Target.Status:
		bad("target.milestone is M%d, but status already equals the target; set it to null", *inv.Target.Milestone)
	case inv.Target.Milestone != nil && *inv.Target.Milestone < 1:
		bad("target.milestone %d is not a milestone of the release plan", *inv.Target.Milestone)
	}
	return errs
}

func statusOf(nci, n int) string {
	switch {
	case nci == 0:
		return StatusNew
	case nci == n:
		return StatusCI
	default:
		return StatusPartial
	}
}

func (t Trigger) validate(at string) []error {
	var errs []error
	if t.Manual {
		if len(t.Repos)+len(t.Paths)+len(t.Exclude) > 0 {
			errs = append(errs, fmt.Errorf("%s.trigger is manual and must name no repos or paths", at))
		}
		return errs
	}
	if len(t.Repos) == 0 || len(t.Paths) == 0 {
		errs = append(errs, fmt.Errorf("%s.trigger needs repos and paths, or manual: true", at))
	}
	for _, r := range t.Repos {
		if !isRepo(r) {
			errs = append(errs, fmt.Errorf("%s.trigger.repos: %q is not a BLIS repository", at, r))
		}
	}
	for _, g := range append(append([]string{}, t.Paths...), t.Exclude...) {
		if err := validGlob(g); err != nil {
			errs = append(errs, fmt.Errorf("%s.trigger: %w", at, err))
		}
	}
	return errs
}

func isRepo(r string) bool {
	i := sort.SearchStrings(Repositories, r)
	return i < len(Repositories) && Repositories[i] == r
}

func validGlob(g string) error {
	if g == "" || strings.HasPrefix(g, "/") || strings.Contains(g, `\`) {
		return fmt.Errorf("glob %q must be a relative path with forward slashes", g)
	}
	for _, seg := range strings.Split(g, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return fmt.Errorf("glob %q has an empty, . or .. segment", g)
		}
		if strings.Contains(seg, "**") && seg != "**" {
			return fmt.Errorf("glob %q: ** must be a whole path segment", g)
		}
		if seg != "**" {
			if _, err := path.Match(seg, ""); err != nil {
				return fmt.Errorf("glob %q: %w", g, err)
			}
		}
	}
	return nil
}

// Match reports whether a slash-separated path matches a glob of the form doc.go
// describes.
func Match(glob, name string) bool {
	return matchSegs(strings.Split(glob, "/"), strings.Split(name, "/"))
}

func matchSegs(g, n []string) bool {
	for len(g) > 0 {
		if g[0] == "**" {
			for i := 0; i <= len(n); i++ {
				if matchSegs(g[1:], n[i:]) {
					return true
				}
			}
			return false
		}
		if len(n) == 0 {
			return false
		}
		if ok, _ := path.Match(g[0], n[0]); !ok {
			return false
		}
		g, n = g[1:], n[1:]
	}
	return len(n) == 0
}

// Matches reports whether a change to the given path, in the given repository, adds
// this trigger's question. A manual trigger matches no path.
func (t Trigger) Matches(repo, name string) bool {
	if t.Manual {
		return false
	}
	inRepo := false
	for _, r := range t.Repos {
		inRepo = inRepo || r == repo
	}
	if !inRepo {
		return false
	}
	for _, g := range t.Exclude {
		if Match(g, name) {
			return false
		}
	}
	for _, g := range t.Paths {
		if Match(g, name) {
			return true
		}
	}
	return false
}

// Question is one review question, with the invariant it belongs to.
type Question struct {
	ID string
	Review
}

// Questions returns the review questions that can be added to a pull request in repo, in
// registry order: those whose trigger names repo, and the manual ones of every invariant
// with a check in repo.
func (r *Registry) Questions(repo string) []Question {
	var out []Question
	for _, inv := range r.Invariants {
		inScope := false
		for _, s := range inv.Scope {
			inScope = inScope || s == repo
		}
		for _, q := range inv.Review {
			in := q.Trigger.Manual && inScope
			for _, tr := range q.Trigger.Repos {
				in = in || tr == repo
			}
			if in {
				out = append(out, Question{inv.ID, q})
			}
		}
	}
	return out
}
