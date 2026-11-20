// Package codepaths describes an engine release's code-path table: code_paths.json, which names
// the functions of the engine's own code that its host, scheduler and graph-launch costs depend
// on, with a hash of each and the CodeID that follows from them (design §9.2, §12.4, Table DC).
//
// A host, scheduler or graph-launch coefficient prices work an engine's code does. Its value is
// valid for that code and for no other, whatever the engine's version string says: a release that
// leaves the code alone keeps its coefficients and one that changes a function in the closure
// does not. The engine publishes the table, so a tag of the release carries it; the lowering
// states the CodeID of each path in the plan (plan.HostPlan.CodeID, plan.GraphPolicy.CodeID), and
// the cost model's lookup matches an entry that lists CodeIDs only for the one that runs
// (coefficient.Entry.MatchCode). This package is the table's one definition: its Go type, its
// validator, its strict loader and the CodeID rule every engine uses.
//
// The function hashes are written by the engine's generator from the release's source, which a
// Go reader does not have; the rule that turns them into a CodeID is here, and the loader
// recomputes it, so the generator's restatement of the rule cannot drift from this one.
//
// The yaml tags repeat the json tags: Parse decodes through the strict YAML decoder every BLIS
// document uses (JSON is YAML).
package codepaths

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	check "github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
)

// Kind is the value of Table.Kind.
const Kind = "CodePathTable"

// Paths are the code paths a table lists, in the order it lists them: the paths
// coefficient.CodePathOf binds a coefficient name to.
var Paths = []string{coefficient.CodePathHost, coefficient.CodePathGraphs}

// Table is one engine release's code-path table.
type Table struct {
	Kind string `json:"kind" yaml:"kind"`
	// Engine names the engine release whose code the table identifies.
	Engine Engine `json:"engine" yaml:"engine"`
	// Paths has one entry per code path, host first and then graphs.
	Paths []Path `json:"paths" yaml:"paths"`
}

// Engine names an engine release.
type Engine struct {
	// Kind is the engine kind the repository registers.
	Kind string `json:"kind" yaml:"kind"`
	// Version is the upstream release.
	Version string `json:"version" yaml:"version"`
	// Commit is the upstream release's commit, 40 lower-case hexadecimal digits.
	Commit string `json:"commit" yaml:"commit"`
}

// Path is the closure of engine code that one kind of cost depends on.
type Path struct {
	// Path is "host" (admission, per-token and completion work, the scheduler's per-request
	// step cost) or "graphs" (eager, per-kernel and graph-replay launch).
	Path string `json:"path" yaml:"path"`
	// CodeID is CodeID(Path, Functions): the identity a coefficient is keyed by.
	CodeID string `json:"code_id" yaml:"code_id"`
	// Functions are the functions the path's cost depends on, sorted by ID, each once.
	Functions []Function `json:"functions" yaml:"functions"`
}

// Function is one function of the engine's code.
type Function struct {
	// ID names the function: <source path>::<qualified name>, as the engine writes it.
	ID string `json:"id" yaml:"id"`
	// Source cites where the function is defined, path:line@commit8 at the release's commit.
	// It locates the function for a reader and is not part of the CodeID: moving a function
	// within its file does not change the code.
	Source string `json:"source" yaml:"source"`
	// Hash is the SHA-256, in lower-case hexadecimal, of the function's code with comments,
	// docstrings, annotations and logging removed. The engine's generator states how.
	Hash string `json:"hash" yaml:"hash"`
}

// hashVersion opens the text CodeID digests. A change to the rule is a new version, and every
// CodeID changes with it.
const hashVersion = "blis code path hash v1"

// CodeID is the rule every engine uses to identify the code behind a path: the SHA-256, in
// lower-case hexadecimal, of the lines
//
//	blis code path hash v1
//	path <path>
//	function <id> <hash>        (one line per function, in byte order of the id)
//
// each ending in a newline. The path is hashed, so the host and graph paths never share a CodeID;
// the citation is not, so a function that moves within its file keeps it. No id may hold a line
// break, which would make two tables hash the same text; the validator refuses one.
func CodeID(path string, functions []Function) string {
	fs := append([]Function(nil), functions...)
	sort.Slice(fs, func(i, j int) bool { return fs[i].ID < fs[j].ID })
	var b strings.Builder
	b.WriteString(hashVersion + "\n")
	b.WriteString("path " + path + "\n")
	for _, f := range fs {
		b.WriteString("function " + f.ID + " " + f.Hash + "\n")
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:])
}

// CodeIDOf returns the CodeID the table states for a path, and whether the table lists the path.
func (t *Table) CodeIDOf(path string) (string, bool) {
	for _, p := range t.Paths {
		if p.Path == path {
			return p.CodeID, true
		}
	}
	return "", false
}

// Load reads and validates the table at path.
func Load(path string) (*Table, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(path, data)
}

// Parse reads a table strictly and validates it: an unknown key, a key that appears twice, a
// second document or a number that does not convert is refused. name is the file or source, for
// messages.
func Parse(name string, data []byte) (*Table, error) {
	var t Table
	if err := check.DecodeStrict(name, data, &t); err != nil {
		return nil, err
	}
	if err := t.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return &t, nil
}

// Marshal writes the table's canonical bytes: JSON indented by two spaces, keys in the declared
// order, no HTML escaping, and a final newline. An engine checks its committed file by comparing
// it with Marshal of what Parse read.
func (t *Table) Marshal() ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(t); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

var (
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
	hashRE   = regexp.MustCompile(`^[0-9a-f]{64}$`)
	citeRE   = regexp.MustCompile(`^[^\s:@]+:[1-9][0-9]*@([0-9a-f]{8})$`)
)

// Validate checks the table and returns every problem found, or nil.
func (t *Table) Validate() error {
	var p check.Problems
	if t.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, t.Kind)
	}
	if strings.TrimSpace(t.Engine.Kind) == "" {
		p.Field("engine.kind", "required")
	}
	if strings.TrimSpace(t.Engine.Version) == "" {
		p.Field("engine.version", "required")
	}
	if !commitRE.MatchString(t.Engine.Commit) {
		p.Field("engine.commit", "must be the release's commit, 40 lower-case hexadecimal digits, got %q", t.Engine.Commit)
	}
	if len(t.Paths) != len(Paths) {
		p.Field("paths", "must list the paths %s, in that order, got %d", strings.Join(Paths, ", "), len(t.Paths))
	}
	for i, path := range t.Paths {
		at := fmt.Sprintf("paths[%d]", i)
		if i < len(Paths) && path.Path != Paths[i] {
			p.Field(at+".path", "must be %q (the paths are %s, in that order), got %q", Paths[i], strings.Join(Paths, ", "), path.Path)
		}
		t.validatePath(&p, at, path)
	}
	if !p.OK() {
		return errors.New(p.Error())
	}
	return nil
}

func (t *Table) validatePath(p *check.Problems, at string, path Path) {
	if len(path.Functions) == 0 {
		p.Field(at+".functions", "required: a path lists the functions its cost depends on")
	}
	for i, f := range path.Functions {
		fat := fmt.Sprintf("%s.functions[%d]", at, i)
		switch {
		case strings.TrimSpace(f.ID) == "":
			p.Field(fat+".id", "required")
		case strings.ContainsAny(f.ID, "\n\r"):
			p.Field(fat+".id", "holds a line break")
		}
		if i > 0 {
			switch prev := path.Functions[i-1].ID; {
			case f.ID == prev:
				p.Field(fat+".id", "%q listed twice", f.ID)
			case f.ID < prev:
				p.Field(fat+".id", "functions must be sorted by id in byte order: %q follows %q", f.ID, prev)
			}
		}
		if !hashRE.MatchString(f.Hash) {
			p.Field(fat+".hash", "must be a SHA-256, 64 lower-case hexadecimal digits, got %q", f.Hash)
		}
		m := citeRE.FindStringSubmatch(f.Source)
		switch {
		case m == nil:
			p.Field(fat+".source", "%q must cite path:line@commit8", f.Source)
		case len(t.Engine.Commit) >= 8 && m[1] != t.Engine.Commit[:8]:
			p.Field(fat+".source", "%q cites commit %s, not the release's %s", f.Source, m[1], t.Engine.Commit[:8])
		}
	}
	if want := CodeID(path.Path, path.Functions); path.CodeID != want {
		p.Field(at+".code_id", "is %q, but the functions give %s (codepaths.CodeID)", path.CodeID, want)
	}
}
