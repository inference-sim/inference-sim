// Package stack builds stack.json: the record of every BLIS module and upstream release a
// build of inference-sim selects, its Go toolchain, and the llm-d guides commit (design SIM-3,
// LIF-2, LIF-6). stack.json is one of the few files that states a version (X23); the
// documents that mention one are rendered from it.
//
// Build computes the record from the module graph (go list -m -json all), the toolchain the
// go command runs (go env GOVERSION), the router fork's own record of the llm-d release it is
// based on, and guides.lock. The committed stack.json must equal what Build returns; the test
// in this package checks it, and cmd/stackgen rewrites it.
package stack

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/mod/module"
)

// The module paths the record names.
const (
	SimulatorModule = "github.com/inference-sim/inference-sim"
	SubstrateModule = "github.com/inference-sim/blis-substrate"
	VLLMModule      = "github.com/inference-sim/blis-vllm"
	// RouterModule is llm-d's module path, which the fork keeps (X11).
	RouterModule = "github.com/llm-d/llm-d-router"
	// RouterFork is the module the one replace directive substitutes for RouterModule.
	RouterFork = "github.com/inference-sim/llm-d-router"
)

// Kind is the value of Stack.Kind.
const Kind = "Stack"

// Stack is stack.json.
type Stack struct {
	Kind string `json:"kind"`
	// Toolchain is the Go toolchain the build uses, as go1.N.M. It equals the substrate's
	// contracts/toolchain (LIF-2).
	Toolchain string `json:"toolchain"`
	// Modules are the BLIS modules the build selects, sorted by path.
	Modules []Module `json:"modules"`
	// Engines are the upstream engine releases this build runs, one per kind (LIF-6).
	Engines []Upstream `json:"engines"`
	// Router is the llm-d release the router fork is based on.
	Router Upstream `json:"router"`
	// GuidesCommit is the llm-d/llm-d commit whose guides this build is tested with
	// (guides.lock).
	GuidesCommit string `json:"guides_commit"`
}

// Module is one BLIS module at the version the build selects.
type Module struct {
	Path string `json:"path"`
	// Version is a release tag, or a pseudo-version while a review set is in draft.
	Version string `json:"version"`
	// Commit is the full commit a pseudo-version names. It is empty for a tag, which names
	// its commit itself (LIF-1). A draft names a full SHA because a pseudo-version carries
	// only twelve hex digits.
	Commit string `json:"commit,omitempty"`
	// Replaces names the module path this one stands in for through go.mod's replace
	// directive; only the router fork has one (X11).
	Replaces string `json:"replaces,omitempty"`
}

// Upstream is one upstream release.
type Upstream struct {
	Kind    string `json:"kind"`
	Version string `json:"version"`
	Commit  string `json:"commit"`
}

// Inputs are what Build reads besides the module graph. Engines are passed in, not
// discovered, because each engine module states its own upstream record in Go (for vLLM,
// blisvllm.Upstream), and this package must not import an engine (X3: one build site).
type Inputs struct {
	// Root is the inference-sim repository root: where go.mod and guides.lock are, and the
	// directory the go command runs in.
	Root    string
	Engines []Upstream
}

// listed is the part of go list -m -json the record reads.
type listed struct {
	Path      string
	Version   string
	Main      bool
	Dir       string
	GoVersion string
	Replace   *listed
}

// Build computes the record for the build rooted at in.Root.
func Build(in Inputs) (*Stack, error) {
	mods, err := listModules(in.Root)
	if err != nil {
		return nil, err
	}
	tool, err := goVersion(in.Root)
	if err != nil {
		return nil, err
	}
	guides, err := ReadGuidesLock(filepath.Join(in.Root, "guides.lock"))
	if err != nil {
		return nil, err
	}
	s := &Stack{Kind: Kind, Toolchain: tool, GuidesCommit: guides, Engines: append([]Upstream(nil), in.Engines...)}
	sort.Slice(s.Engines, func(i, j int) bool { return s.Engines[i].Kind < s.Engines[j].Kind })

	var routerDir string
	for _, m := range mods {
		if m.Main || !isBLIS(m.Path) {
			continue
		}
		rec := Module{Path: m.Path, Version: m.Version}
		if m.Replace != nil {
			if m.Path != RouterModule || m.Replace.Path != RouterFork {
				return nil, fmt.Errorf("%s is replaced by %s %s; the only replace is %s => %s (X11)",
					m.Path, m.Replace.Path, m.Replace.Version, RouterModule, RouterFork)
			}
			rec = Module{Path: RouterFork, Version: m.Replace.Version, Replaces: RouterModule}
			routerDir = m.Replace.Dir
		} else if m.Path == RouterModule {
			return nil, fmt.Errorf("%s is not replaced by %s; the build would fetch upstream llm-d", RouterModule, RouterFork)
		}
		if module.IsPseudoVersion(rec.Version) {
			c, err := originCommit(in.Root, rec.Path, rec.Version)
			if err != nil {
				return nil, err
			}
			rev, _ := module.PseudoVersionRev(rec.Version)
			if !strings.HasPrefix(c, rev) {
				return nil, fmt.Errorf("%s %s names commit %s, not %s", rec.Path, rec.Version, rev, c)
			}
			rec.Commit = c
		}
		s.Modules = append(s.Modules, rec)
	}
	sort.Slice(s.Modules, func(i, j int) bool { return s.Modules[i].Path < s.Modules[j].Path })
	if routerDir == "" {
		return nil, fmt.Errorf("the build selects no %s through a replace of %s", RouterFork, RouterModule)
	}
	if s.Router, err = routerUpstream(routerDir); err != nil {
		return nil, err
	}
	if err := s.setRouterVersion(); err != nil {
		return nil, err
	}
	return s, nil
}

// isBLIS reports whether a module path is a BLIS module or the llm-d router BLIS forks.
func isBLIS(path string) bool {
	return strings.HasPrefix(path, "github.com/inference-sim/") || path == RouterModule
}

// Marshal writes the record as stack.json is committed: indented, with a final newline.
func (s *Stack) Marshal() ([]byte, error) {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// Read decodes a committed stack.json, refusing unknown keys.
func Read(path string) (*Stack, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var s Stack
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if s.Kind != Kind {
		return nil, fmt.Errorf("%s: kind %q, want %q", path, s.Kind, Kind)
	}
	return &s, nil
}

// Module returns the record of a module path, or false.
func (s *Stack) Module(path string) (Module, bool) {
	for _, m := range s.Modules {
		if m.Path == path {
			return m, true
		}
	}
	return Module{}, false
}

// ReadGuidesLock reads guides.lock: one line, the full llm-d/llm-d commit the guides are read
// at. Comment lines start with #.
func ReadGuidesLock(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var commit string
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if commit != "" {
			return "", fmt.Errorf("%s: more than one commit", path)
		}
		commit = line
	}
	if !isFullSHA(commit) {
		return "", fmt.Errorf("%s: %q is not a full 40-digit commit", path, commit)
	}
	return commit, nil
}

func isFullSHA(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return false
		}
	}
	return true
}

// routerUpstream reads the llm-d release the fork is based on from the fork's own record,
// hack/hosting-lint/budget.txt's upstream_base, which the fork's merge check keeps equal to
// an upstream release tag it contains (RTR-1). The version is the tag's, taken from the
// fork's -blis.N version: v1.0.0-blis.2 and its pseudo-versions are based on v1.0.0.
func routerUpstream(dir string) (Upstream, error) {
	path := filepath.Join(dir, "hack", "hosting-lint", "budget.txt")
	b, err := os.ReadFile(path)
	if err != nil {
		return Upstream{}, fmt.Errorf("the router fork's upstream record: %w", err)
	}
	var commit string
	for _, line := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(k) == "upstream_base" {
			commit = strings.TrimSpace(v)
		}
	}
	if !isFullSHA(commit) {
		return Upstream{}, fmt.Errorf("%s: no upstream_base commit", path)
	}
	return Upstream{Kind: "llm-d", Commit: commit}, nil
}

// setRouterVersion fills the router's upstream version from the fork's module version:
// v<upstream>-blis.N, or a pseudo-version built on such a tag (design §14.1).
func (s *Stack) setRouterVersion() error {
	m, ok := s.Module(RouterFork)
	if !ok {
		return fmt.Errorf("no %s module in the record", RouterFork)
	}
	v, _, ok := strings.Cut(m.Version, "-blis")
	if !ok || v == "" {
		return fmt.Errorf("%s %s is not a -blis.N version or built on one", RouterFork, m.Version)
	}
	s.Router.Version = strings.TrimPrefix(v, "v")
	return nil
}

// originCommit is the full commit a module version was fetched at, as go mod download
// records it (Origin.Hash).
func originCommit(root, path, version string) (string, error) {
	out, err := run(root, "go", "mod", "download", "-json", path+"@"+version)
	if err != nil {
		return "", err
	}
	var d struct{ Origin *struct{ Hash string } }
	if err := json.Unmarshal(out, &d); err != nil {
		return "", fmt.Errorf("go mod download -json %s@%s: %w", path, version, err)
	}
	if d.Origin == nil || !isFullSHA(d.Origin.Hash) {
		return "", fmt.Errorf("go mod download -json %s@%s reports no origin commit", path, version)
	}
	return d.Origin.Hash, nil
}

func listModules(root string) ([]listed, error) {
	out, err := run(root, "go", "list", "-m", "-json", "all")
	if err != nil {
		return nil, err
	}
	var mods []listed
	dec := json.NewDecoder(bytes.NewReader(out))
	for dec.More() {
		var m listed
		if err := dec.Decode(&m); err != nil {
			return nil, fmt.Errorf("go list -m -json all: %w", err)
		}
		mods = append(mods, m)
	}
	return mods, nil
}

func goVersion(root string) (string, error) {
	out, err := run(root, "go", "env", "GOVERSION")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func run(dir, name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %v: %s", name, strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}
