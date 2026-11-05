package stack

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/mod/modfile"

	"github.com/inference-sim/blis-substrate/contracts"
)

// repoRoot is the inference-sim repository root; this file is <root>/internal/stack/.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller information")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// The committed go.mod passes the selected-version lint (X11).
func TestGoModHasOnlyTheRouterReplaceAndEngineExcludes(t *testing.T) {
	findings, err := CheckGoMod(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range findings {
		t.Error(f)
	}
}

// Each kind of bad go.mod is caught (X18: a check names a known-bad input that makes it fail).
func TestGoModLintCatchesEachBadForm(t *testing.T) {
	const head = "module github.com/inference-sim/inference-sim\n\ngo 1.27.2\n\n"
	const good = "replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.3\n"
	for name, body := range map[string]string{
		"no replace":                 "",
		"a second replace":           good + "replace github.com/inference-sim/blis-vllm => ../blis-vllm\n",
		"a directory replace":        "replace github.com/llm-d/llm-d-router => ../llm-d-router\n",
		"a fork at a plain tag":      "replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0\n",
		"another module":             "replace github.com/inference-sim/blis-substrate => github.com/x/y v1.0.0-blis.1\n",
		"an exclude of a -blis tag":  good + "exclude github.com/inference-sim/blis-vllm v0.32.0-blis.1\n",
		"an exclude of a non-engine": good + "exclude github.com/inference-sim/blis-substrate v0.11.0\n",
	} {
		f, err := modfile.Parse("go.mod", []byte(head+body), nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := checkModFile(t.TempDir(), f); len(got) == 0 {
			t.Errorf("%s: not caught", name)
		}
	}
	f, err := modfile.Parse("go.mod", []byte(head+good+"exclude github.com/inference-sim/blis-vllm v0.32.0\n"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := checkModFile(t.TempDir(), f); len(got) != 0 {
		t.Errorf("the allowed form is refused: %v", got)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.work"), []byte("go 1.27.2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := checkModFile(dir, f); len(got) == 0 {
		t.Error("a go.work beside go.mod is not caught")
	}
}

// The toolchain the build runs is the substrate's contracts/toolchain, and no module in the
// build declares a go version above it (LIF-2).
func TestTheToolchainIsTheSubstrates(t *testing.T) {
	want := contracts.Toolchain()
	if got := runtime.Version(); got != want {
		t.Errorf("go version is %s; contracts/toolchain at the selected substrate says %s (set GOTOOLCHAIN=%s)", got, want, want)
	}
	findings, err := CheckGoVersions(repoRoot(t), want)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range findings {
		t.Error(f)
	}
	if got, err := CheckGoVersions(repoRoot(t), "go1.20.0"); err != nil || len(got) == 0 {
		t.Errorf("a toolchain below the modules' go lines is not caught: %v %v", got, err)
	}
}

// inference-sim's own substrate requirement is the version its build selects, so no engine or
// router release can change the simulator's data (design §3.2, LIF-2).
func TestTheSubstrateRequirementIsTheSelectedVersion(t *testing.T) {
	root := repoRoot(t)
	b, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	f, err := modfile.Parse("go.mod", b, nil)
	if err != nil {
		t.Fatal(err)
	}
	var required string
	for _, r := range f.Require {
		if r.Mod.Path == SubstrateModule {
			required = r.Mod.Version
			if r.Indirect {
				t.Errorf("go.mod requires %s as indirect; inference-sim imports it directly", SubstrateModule)
			}
		}
	}
	mods, err := listModules(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range mods {
		if m.Path == SubstrateModule && m.Version != required {
			t.Errorf("go.mod requires %s %s, the build selects %s; re-pin the simulator first", SubstrateModule, required, m.Version)
		}
	}
	if required == "" {
		t.Errorf("go.mod does not require %s", SubstrateModule)
	}
}

// The committed stack.json is the build's record (SIM-3). The engines come from the build site;
// this package must not import an engine, so the committed engines stand in here and
// cmd's TestStackJSONNamesTheEnginesThisBinaryRuns holds them to the build site.
func TestStackJSONMatchesTheBuild(t *testing.T) {
	root := repoRoot(t)
	committed, err := Read(filepath.Join(root, "stack.json"))
	if err != nil {
		t.Fatal(err)
	}
	built, err := Build(Inputs{Root: root, Engines: committed.Engines})
	if err != nil {
		t.Fatal(err)
	}
	if d := Diff(committed, built); len(d) > 0 {
		t.Errorf("stack.json does not match the build; run go run ./internal/stack/cmd/stackgen:\n  %s", strings.Join(d, "\n  "))
	}
	if len(built.Modules) < 3 {
		t.Errorf("the record names %d BLIS modules; want the substrate, the engine and the router fork", len(built.Modules))
	}
}

// Diff sees each kind of drift.
func TestDiffSeesEachDrift(t *testing.T) {
	base := Stack{Kind: Kind, Toolchain: "go1.27.2", GuidesCommit: strings.Repeat("a", 40),
		Modules: []Module{{Path: SubstrateModule, Version: "v0.11.0"}},
		Engines: []Upstream{{Kind: "vllm", Version: "0.32.0", Commit: "c"}},
		Router:  Upstream{Kind: "llm-d", Version: "1.0.0", Commit: "r"}}
	clone := func(f func(*Stack)) *Stack {
		s := base
		s.Modules = append([]Module(nil), base.Modules...)
		s.Engines = append([]Upstream(nil), base.Engines...)
		f(&s)
		return &s
	}
	if d := Diff(&base, clone(func(*Stack) {})); len(d) != 0 {
		t.Fatalf("equal records differ: %v", d)
	}
	for name, f := range map[string]func(*Stack){
		"toolchain":      func(s *Stack) { s.Toolchain = "go1.27.3" },
		"guides":         func(s *Stack) { s.GuidesCommit = strings.Repeat("b", 40) },
		"module version": func(s *Stack) { s.Modules[0].Version = "v0.11.1" },
		"extra module":   func(s *Stack) { s.Modules = append(s.Modules, Module{Path: VLLMModule, Version: "v0.32.0-blis.2"}) },
		"engine":         func(s *Stack) { s.Engines[0].Version = "0.31.0" },
		"extra engine":   func(s *Stack) { s.Engines = append(s.Engines, Upstream{Kind: "sglang"}) },
		"router":         func(s *Stack) { s.Router.Version = "1.0.1" },
	} {
		if d := Diff(&base, clone(f)); len(d) == 0 {
			t.Errorf("%s: drift not seen", name)
		}
	}
}

func TestGuidesLockIsOneFullCommit(t *testing.T) {
	dir := t.TempDir()
	for body, ok := range map[string]bool{
		"# comment\n" + strings.Repeat("a", 40) + "\n":                  true,
		strings.Repeat("a", 12) + "\n":                                  false,
		strings.Repeat("a", 40) + "\n" + strings.Repeat("b", 40) + "\n": false,
		"":       false,
		"main\n": false,
	} {
		p := filepath.Join(dir, "guides.lock")
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadGuidesLock(p); (err == nil) != ok {
			t.Errorf("%q: err = %v, want ok %v", body, err, ok)
		}
	}
}
