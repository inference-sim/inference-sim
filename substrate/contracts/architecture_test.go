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
		"./spec/result", "./spec/trace", "./spec/pointrecord", "./metrics", "./spec/fields",
		"./vocab", "./check", "./units", "./engine/plan", "./engine/env",
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

// TestEngineIsEngineNeutral: the engine contract names no engine, so it imports nothing from
// this module beyond the one document type its Fields method returns. A document type or a
// rules pack reaching into it would make a second engine edit this module to plug in, which is
// what the contract exists to rule out.
//
// The exception is the field list (spec/fields) and the strict-decoding helpers it is built
// on (check). The list is a document about engines in general: it holds no engine's fields, and
// each engine repository states its own (design VLM-3, SGL-3). It is allowed because the
// design makes it a method of engine.Engine, and it is the only one: every other package under
// contracts/ stays out of engine's dependencies.
func TestEngineIsEngineNeutral(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	allowed := map[string]bool{"spec/fields": true, "check": true}
	for _, pkg := range []string{"./engine", "./conformance/enginetest"} {
		out, err := exec.Command("go", "list", "-deps", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Split(string(out), "\n") {
			rel, ok := strings.CutPrefix(dep, mod)
			if ok && !strings.HasPrefix(rel, "engine") && rel != "conformance/enginetest" && !allowed[rel] {
				t.Errorf("%s depends on %s; the engine contract must not know any document type or rules pack beyond spec/fields", pkg, rel)
			}
		}
	}
}

// X18 for the engine-neutrality rule: spec/fields does not itself reach a rules pack, a
// scenario, a model or any other document, so allowing it does not let those in behind it.
func TestTheFieldListImportsNoEngineKnowledge(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	out, err := exec.Command("go", "list", "-deps", "./spec/fields").Output()
	if err != nil {
		t.Fatal(err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		rel, ok := strings.CutPrefix(dep, mod)
		if ok && rel != "spec/fields" && rel != "check" {
			t.Errorf("spec/fields depends on %s; it may import only check, or the exception in TestEngineIsEngineNeutral would let it in", rel)
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

// The boundary types import only the units package, so a plan names no document type, no rules
// pack and no engine (X2, SUB-16); the Environment interface adds only the catalog and cost
// model types it returns, and the reference implementation adds nothing else.
func TestPlanPackagesStayInTheirLane(t *testing.T) {
	const mod = "github.com/inference-sim/blis-substrate/contracts/"
	for pkg, allowed := range map[string][]string{
		"./units":       {"units"},
		"./engine/plan": {"units", "engine/plan"},
		"./engine/env": {"units", "engine/plan", "engine/env", "kernel", "check", "vocab",
			"spec/deployment", "spec/model", "spec/hardware", "spec/acceptance"},
	} {
		out, err := exec.Command("go", "list", "-deps", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		ok := map[string]bool{}
		for _, a := range allowed {
			ok[a] = true
		}
		for _, dep := range strings.Split(string(out), "\n") {
			if rel, in := strings.CutPrefix(dep, mod); in && !ok[rel] {
				t.Errorf("%s depends on %s; it may import only %v", pkg, rel, allowed)
			}
		}
	}
}
