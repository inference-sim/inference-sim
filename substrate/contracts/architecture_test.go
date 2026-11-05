package contracts

import (
	"os/exec"
	"strings"
	"testing"
)

// These tests guard the repository's structure rather than its contents. The two
// validation layers stay separate only if the dependency direction enforces it, and
// a dependency added in a hurry is exactly the kind of change a reviewer misses.

// TestSpecDoesNotDependOnRules is the layering invariant. If a schema package could
// import a rules pack, a field-level check could quietly become version-specific,
// and the separation this repository is organized around would exist only in the
// documentation.
func TestSpecDoesNotDependOnRules(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	for _, pkg := range []string{
		"./spec/model", "./spec/hardware", "./spec/coefficient",
		"./spec/scenario", "./spec/deployment", "./spec/workload",
		"./spec/evaluation", "./spec/acceptance", "./spec/routing", "./spec/measurement", "./spec/router",
		"./spec/connector",
		"./spec/result", "./spec/fields",
		"./vocab", "./check",
	} {
		out, err := exec.Command("go", "list", "-deps", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Split(string(out), "\n") {
			if !strings.HasPrefix(dep, mod) {
				continue
			}
			rel := strings.TrimPrefix(dep, mod)
			if strings.HasPrefix(rel, "rules") {
				t.Errorf("%s depends on %s; field validation must not know about engine versions",
					pkg, rel)
			}
		}
	}
}

// TestKernelImportsOnlySpec checks the interface's DIRECT imports. It describes what
// a cost model computes, so it needs the document types and must not reach for a
// rules pack or the top-level validator. Transitive dependencies are not checked
// here: spec/ legitimately imports the problem list, and forbidding that
// transitively would forbid spec/ itself.
func TestKernelImportsOnlySpec(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	out, err := exec.Command("go", "list", "-f", "{{join .Imports \"\\n\"}}", "./kernel").Output()
	if err != nil {
		t.Fatalf("go list ./kernel: %v", err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		if !strings.HasPrefix(dep, mod) {
			continue
		}
		rel := strings.TrimPrefix(dep, mod)
		if !strings.HasPrefix(rel, "spec/") && !strings.HasPrefix(rel, "vocab") {
			t.Errorf("kernel imports %s directly; it should need only document types", rel)
		}
	}
}

// TestRulesPackagesAreVersioned: a pack lives under a version-named directory, so
// adding a release cannot accidentally edit an existing one.
func TestRulesPackagesAreVersioned(t *testing.T) {
	out, err := exec.Command("go", "list", "./rules/...").Output()
	if err != nil {
		t.Fatalf("go list ./rules/...: %v", err)
	}
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	for _, pkg := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		rel := strings.TrimPrefix(pkg, mod)
		if rel == "rules" {
			continue // the mechanism itself
		}
		if !strings.HasPrefix(rel, "rules/v") {
			t.Errorf("%s is under rules/ but is not version-named; a pack must be scoped to a release", rel)
		}
	}
}

// TestEngineIsEngineNeutral: the engine contract names no engine, so it imports nothing
// from this module. A document type or a rules pack reaching into it would make a second
// engine edit this module to plug in, which is what the contract exists to rule out.
func TestEngineIsEngineNeutral(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	for _, pkg := range []string{"./engine", "./conformance/enginetest"} {
		out, err := exec.Command("go", "list", "-deps", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Split(string(out), "\n") {
			rel, ok := strings.CutPrefix(dep, mod)
			if ok && !strings.HasPrefix(rel, "engine") && rel != "conformance/enginetest" {
				t.Errorf("%s depends on %s; the engine contract must not know any document type or rules pack", pkg, rel)
			}
		}
	}
}

// TestEngineSettingsImportOnlyTheStandardLibrary: a latency kernel prices from these settings
// and an engine model fills them from its own engine block, so they import nothing either side
// would then share.
func TestEngineSettingsImportOnlyTheStandardLibrary(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}",
		"./kernel/enginesettings").Output()
	if err != nil {
		t.Fatalf("go list -deps ./kernel/enginesettings: %v", err)
	}
	for _, dep := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if dep != "" && dep != "github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings" {
			t.Errorf("kernel/enginesettings depends on %s; it must need only the standard library", dep)
		}
	}
}

// TestEmbedImportsOnlyTheStandardLibrary: the router contract is implemented in a router's
// repository and called from a simulator's, so importing it must cost either side nothing but
// this module. A dependency here would become a dependency of every router that implements it.
func TestEmbedImportsOnlyTheStandardLibrary(t *testing.T) {
	for _, pkg := range []string{"./embed", "./conformance/embedtest"} {
		out, err := exec.Command("go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Fields(string(out)) {
			if dep == "github.com/inference-sim/blis-substrate/contracts/embed" || dep == "github.com/inference-sim/blis-substrate/contracts/conformance/embedtest" {
				continue
			}
			t.Errorf("%s depends on %s; the router contract uses the standard library only", pkg, dep)
		}
	}
}

// TestInvariantsImportOnlyTheStandardLibraryAndYAML: every BLIS repository reads the
// invariant registry at the substrate version it pins, the engine repositories and the
// router fork included, so the package may import only what SUB-1 allows a contracts/
// package: the standard library and the YAML parser.
func TestInvariantsImportOnlyTheStandardLibraryAndYAML(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}", "./invariants").Output()
	if err != nil {
		t.Fatalf("go list -deps ./invariants: %v", err)
	}
	for _, dep := range strings.Fields(string(out)) {
		if dep == "github.com/inference-sim/blis-substrate/contracts/invariants" || dep == "gopkg.in/yaml.v3" {
			continue
		}
		t.Errorf("contracts/invariants depends on %s; it may use only the standard library and gopkg.in/yaml.v3 (SUB-1)", dep)
	}
}
