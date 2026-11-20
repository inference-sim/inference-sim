package cmd

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The engine that cmd/engine_build.go builds reads a release's rules only through blis-schemas'
// enginefacts.EngineRules (#2021 B4; PR BC-3). The test type-checks the pinned module's own source --
// the versions go.mod names, not a copy -- so the claim is checked rather than taken on trust.
//
// The concrete pack type (the vLLM engine's rules/pack.Pack since https://github.com/inference-sim/blis-schemas/issues/102 2c38342) and
// the vLLM engine's pack registry may appear in exactly one place: the adaptation that hands
// engines/vllm/rules' lookup to the kernel's RulesFrom, which turns it into a function returning
// enginefacts.EngineRules, with the list of releases it answers for (allowedPackUses). Anywhere else,
// any use of *pack.Pack is a finding: a
// type assertion or type-switch case, a conversion, a parameter, result, field or variable of
// that type, a field read or write on a pack, or an alias of the type. So is any call to the
// global registry (engines/vllm/rules' Lookup, Packs and Versions), which looks a pack up instead of
// taking the caller's.
//
// contracts.Validate takes the caller's engine checker (Bundle.Engine); with none it refuses the
// bundle. A call to Validate in scope is still a finding unless the Bundle it is handed states
// Engine (code review round 2, N1), so a validation that would only refuse is not mistaken for
// one that checks.
func TestPinnedEngineReadsRulesOnlyThroughEngineRules(t *testing.T) {
	findings := packFindings(t, engineRulesScope)
	if len(findings) > 0 {
		t.Errorf("the pinned engine reads rules other than through enginefacts.EngineRules:\n  %s",
			strings.Join(findings, "\n  "))
	}
}

// engineRulesScope is the code a blis run calls that sees engine rules: the vLLM engine's build and
// kernel-adapter packages, and the latency kernel they hand the rules to.
var engineRulesScope = []string{
	"github.com/inference-sim/inference-sim/engines/vllm/core/build",
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel",
	"github.com/inference-sim/inference-sim/substrate/kernel",
	"github.com/inference-sim/inference-sim/substrate/kernel/internal/resolve",
	"github.com/inference-sim/inference-sim/substrate/kernel/internal/price",
}

// allowedPackUses are the expressions that may name the concrete pack type or the vLLM engine's pack
// registry: the one adaptation of engines/vllm/rules' lookup into the interface (withEngineRules),
// which is how the engine model's own rules reach the kernel, and the kernel's adapter itself.
// Keyed by package and the expressions' source text.
var allowedPackUses = map[string][]string{
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel": {"latencykernel.RulesFrom(vllmrules.Lookup)", "vllmrules.Versions"},
	"github.com/inference-sim/inference-sim/substrate/kernel":              {"RulesFrom"},
}

const (
	vllmPackPath    = "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
	vllmRulesPath   = "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack"
	schemasRootPath = "github.com/inference-sim/inference-sim/substrate/contracts"
)

// validatorsReadingRules are blis-schemas' root functions that apply a release's rules, which they
// take from the Bundle (Engine) and refuse without.
var validatorsReadingRules = map[string]bool{"Validate": true}

// globalRegistry are the functions that look a pack up by version instead of taking the
// caller's rules, in the vLLM engine's packs module.
var globalRegistry = map[string]bool{"Lookup": true, "Apply": true, "Register": true, "Versions": true, "Packs": true}

// packFindings type-checks each package and lists every use of the concrete pack type and every
// call to a global pack registry, outside allowedPackUses.
func packFindings(t *testing.T, pkgs []string) []string {
	t.Helper()
	var findings []string
	fset := token.NewFileSet()
	imp := importer.ForCompiler(fset, "source", nil)
	for _, path := range pkgs {
		dir, files := listedGoFiles(t, path)
		var parsed []*ast.File
		for _, f := range files {
			af, err := parser.ParseFile(fset, filepath.Join(dir, f), nil, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			parsed = append(parsed, af)
		}
		findings = append(findings, checkPackUse(t, path, fset, imp, parsed)...)
	}
	return findings
}

// listedGoFiles is a package's directory and non-test Go files for this build, from go list, so
// the files checked are the ones the pinned build compiles.
func listedGoFiles(t *testing.T, path string) (string, []string) {
	t.Helper()
	out, err := exec.Command("go", "list", "-f", "{{.Dir}}|{{join .GoFiles \",\"}}", path).Output()
	if err != nil {
		t.Fatalf("go list %s: %v", path, err)
	}
	parts := strings.SplitN(strings.TrimSpace(string(out)), "|", 2)
	if len(parts) != 2 || parts[1] == "" {
		t.Fatalf("go list %s: no Go files (%q)", path, out)
	}
	return parts[0], strings.Split(parts[1], ",")
}

// checkPackUse type-checks one package's files and reports, by position, each expression of the
// concrete pack type or naming it, and each call into a global registry.
func checkPackUse(t *testing.T, path string, fset *token.FileSet, imp types.Importer, files []*ast.File) []string {
	t.Helper()
	conf := types.Config{Importer: imp}
	info := &types.Info{
		Types: map[ast.Expr]types.TypeAndValue{},
		Uses:  map[*ast.Ident]types.Object{},
		Defs:  map[*ast.Ident]types.Object{},
	}
	if _, err := conf.Check(path, fset, files, info); err != nil {
		t.Fatalf("type-checking %s: %v", path, err)
	}
	allowed := map[token.Pos]bool{}
	src := func(n ast.Node) string {
		p, e := fset.Position(n.Pos()), fset.Position(n.End())
		raw, err := os.ReadFile(p.Filename)
		if err != nil || e.Offset > len(raw) {
			return ""
		}
		return string(raw[p.Offset:e.Offset])
	}
	isAllowed := func(e ast.Expr) bool {
		for _, a := range allowedPackUses[path] {
			if src(e) == a {
				return true
			}
		}
		return false
	}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if e, ok := n.(ast.Expr); ok && isAllowed(e) {
				ast.Inspect(e, func(m ast.Node) bool {
					if m != nil {
						allowed[m.Pos()] = true
					}
					return true
				})
				return false
			}
			return true
		})
	}
	seen := map[string]bool{}
	var out []string
	report := func(n ast.Node, what string) {
		if allowed[n.Pos()] {
			return
		}
		at := fset.Position(n.Pos())
		msg := fmt.Sprintf("%s:%d: %s", filepath.Base(at.Filename), at.Line, what)
		if !seen[msg] {
			seen[msg] = true
			out = append(out, path+": "+msg)
		}
	}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.Ident:
				obj := info.Uses[x]
				if obj == nil {
					obj = info.Defs[x]
				}
				if obj == nil {
					return true
				}
				if fn, ok := obj.(*types.Func); ok && fn.Pkg() != nil && globalRegistry[fn.Name()] &&
					fn.Pkg().Path() == vllmRulesPath &&
					fn.Type().(*types.Signature).Recv() == nil {
					report(x, "uses the global registry "+fn.Pkg().Name()+"."+fn.Name())
				}
				if tn, ok := obj.(*types.TypeName); ok && isPackType(tn.Type()) {
					report(x, "names the concrete pack type")
				}
				if v, ok := obj.(*types.Var); ok && mentionsPack(v.Type()) {
					report(x, "declares or uses a "+v.Type().String())
				}
			case *ast.CallExpr:
				if fn := calledFunc(info, x.Fun); fn != nil && fn.Pkg() != nil && fn.Pkg().Path() == schemasRootPath &&
					validatorsReadingRules[fn.Name()] && !bundleStatesRules(x) {
					report(x, "calls blis-schemas' "+fn.Name()+" with no Bundle.Engine, so it validates nothing (it refuses the bundle)")
				}
				if tv, ok := info.Types[x]; ok && tv.IsValue() && mentionsPack(tv.Type) {
					report(x, "has the concrete type "+tv.Type.String())
				}
			case ast.Expr:
				if tv, ok := info.Types[x]; ok && tv.IsValue() && mentionsPack(tv.Type) {
					report(x, "has the concrete type "+tv.Type.String())
				}
			}
			return true
		})
	}
	sort.Strings(out)
	return out
}

// calledFunc is the function a call's callee resolves to, or nil.
func calledFunc(info *types.Info, fun ast.Expr) *types.Func {
	switch f := fun.(type) {
	case *ast.Ident:
		fn, _ := info.Uses[f].(*types.Func)
		return fn
	case *ast.SelectorExpr:
		fn, _ := info.Uses[f.Sel].(*types.Func)
		return fn
	}
	return nil
}

// bundleStatesRules reports whether a call's argument is a Bundle literal that sets Engine. A
// bundle built elsewhere and passed by name is not credited: the check cannot see that it
// states Engine, so it reports the call.
func bundleStatesRules(call *ast.CallExpr) bool {
	if len(call.Args) != 1 {
		return false
	}
	lit, ok := call.Args[0].(*ast.CompositeLit)
	if !ok {
		return false
	}
	for _, el := range lit.Elts {
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			if k, ok := kv.Key.(*ast.Ident); ok && k.Name == "Engine" {
				return true
			}
		}
	}
	return false
}

// isPackType reports whether typ is the vLLM engine's pack.Pack (behind any aliases).
func isPackType(typ types.Type) bool {
	n, ok := types.Unalias(typ).(*types.Named)
	return ok && n.Obj().Pkg() != nil && n.Obj().Pkg().Path() == vllmPackPath && n.Obj().Name() == "Pack"
}

// mentionsPack reports whether typ is, points to, or is built from pack.Pack: a pointer, slice,
// array, map, channel or function signature that carries a pack.
func mentionsPack(typ types.Type) bool {
	switch u := types.Unalias(typ).(type) {
	case *types.Named:
		return isPackType(u)
	case *types.Pointer:
		return mentionsPack(u.Elem())
	case *types.Slice:
		return mentionsPack(u.Elem())
	case *types.Array:
		return mentionsPack(u.Elem())
	case *types.Map:
		return mentionsPack(u.Key()) || mentionsPack(u.Elem())
	case *types.Chan:
		return mentionsPack(u.Elem())
	case *types.Signature:
		for _, tup := range []*types.Tuple{u.Params(), u.Results()} {
			for i := 0; i < tup.Len(); i++ {
				if mentionsPack(tup.At(i).Type()) {
					return true
				}
			}
		}
	}
	return false
}

// The check catches each way code can reach a concrete pack, which a name-based scan misses: an
// assertion, a type-switch case, a conversion, a parameter or result, a field, a field read or
// write, an alias, and the global registry. Each planted form, in a package that imports the
// real pack and rules packages, is a finding; the interface-only form is none.
func TestPackUseCheckCatchesEachForm(t *testing.T) {
	// The registry forms import engines/vllm/rules, whose every export is the registry; the others
	// do not, so the clean probe imports nothing that is itself a finding.
	header := func(registry bool) string {
		imp := ""
		if registry {
			imp = "\tvllmrules \"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack\"\n"
		}
		return `package probe

import (
	blisschemas "github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
` + imp + `	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
)

var (
	_ = pack.NormalizeMoEBackend
	_ = blisschemas.Validate
	_ enginefacts.EngineRules
)
`
	}
	clean := `func ok(r enginefacts.EngineRules) string { return r.EngineVersion() }

func okValidate(r enginefacts.EngineRules) bool {
	return blisschemas.Validate(blisschemas.Bundle{Engine: func(string) blisschemas.EngineChecker { return r }}).OK()
}`
	forms := map[string]string{
		"type assertion":                      `func f(r enginefacts.EngineRules) bool { _, ok := r.(*pack.Pack); return ok }`,
		"type switch case":                    `func f(r enginefacts.EngineRules) int { switch r.(type) { case *pack.Pack: return 1 }; return 0 }`,
		"parameter":                           `func f(p *pack.Pack) {}`,
		"result":                              `func f() *pack.Pack { return nil }`,
		"struct field":                        `type s struct{ p *pack.Pack }`,
		"field write":                         `func f(r enginefacts.EngineRules) { if p, ok := r.(*pack.Pack); ok { p.Version = "x" } }`,
		"alias":                               `type myPack = pack.Pack`,
		"conversion":                          `func f(r any) any { return (*pack.Pack)(r.(*pack.Pack)) }`,
		"global lookup":                       `func f() bool { return vllmrules.Lookup("0.31.0") != nil }`,
		"global packs":                        `func f() int { return len(vllmrules.Packs()) }`,
		"validate without the caller's rules": `func f() bool { return blisschemas.Validate(blisschemas.Bundle{}).OK() }`,
		"validate a bundle built elsewhere":   `func f(b blisschemas.Bundle) bool { return blisschemas.Validate(b).OK() }`,
	}
	fset := token.NewFileSet()
	imp := importer.ForCompiler(fset, "source", nil)
	run := func(body string) []string {
		path := filepath.Join(t.TempDir(), "probe.go")
		if err := os.WriteFile(path, []byte(header(strings.Contains(body, "vllmrules."))+body+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		return checkPackUse(t, "probe", fset, imp, []*ast.File{f})
	}
	if got := run(clean); len(got) != 0 {
		t.Errorf("reading rules through enginefacts.EngineRules is allowed, but got %v", got)
	}
	names := make([]string, 0, len(forms))
	for name := range forms {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if got := run(forms[name]); len(got) == 0 {
			t.Errorf("%s: not caught", name)
		}
	}
}

// The blis binary reaches the vLLM engine through the build site alone (GOAL.md phase 2c): no other
// package it links imports a the vLLM engine package, and no other non-test file under sim/ or cmd/
// does, linked or not. Every engine, the compatibility engine included, is built there and driven
// through blis-schemas' engine.Engine contract; the host's types are its own. The offline
// scoring tools, which open kernels through the vLLM engine's adapter themselves, live under tools/.
func TestTheBinaryReachesBlisVLLMOnlyThroughTheBuildSite(t *testing.T) {
	onlyThroughBuildSite(t, "github.com/inference-sim/inference-sim/engines/vllm", "cmd/engine_build.go")
}

// The same rule for the SGLang engine: the SGLang engine is built in cmd/sglang_engine.go, the file
// newContractEngine hands an sglang spec to, and no other package or file under sim/ or cmd/
// imports a the SGLang engine package.
func TestTheBinaryReachesBlisSGLangOnlyThroughTheBuildSite(t *testing.T) {
	onlyThroughBuildSite(t, "github.com/inference-sim/inference-sim/engines/sglang", "cmd/sglang_engine.go")
}

// onlyThroughBuildSite checks that no package the binary links, and no non-test file under sim/
// or cmd/, imports a package of module except cmd's file site.
func onlyThroughBuildSite(t *testing.T, module, site string) {
	t.Helper()
	allowed := map[string]bool{
		"github.com/inference-sim/inference-sim/cmd": true, // checked per file below
	}
	out, err := exec.Command("go", "list", "-deps", "-f", "{{.ImportPath}}|{{join .Imports \",\"}}", "..").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	linked := 0
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		pkg, imports, _ := strings.Cut(line, "|")
		if !strings.HasPrefix(pkg, "github.com/inference-sim/inference-sim") {
			continue
		}
		linked++
		if pkg == module || strings.HasPrefix(pkg, module+"/") {
			continue // the engine's own packages import each other
		}
		for _, imp := range strings.Split(imports, ",") {
			if strings.HasPrefix(imp, module) && !allowed[pkg] {
				t.Errorf("%s imports %s; the binary reaches %s only through %s", pkg, imp, module, site)
			}
		}
	}
	if linked < 3 {
		t.Fatalf("go list -deps saw %d inference-sim packages; the check is vacuous", linked)
	}

	// Under sim/ and cmd/, in every package and not only the binary's, only the build site.
	sites := map[string]bool{site: true}
	seen := map[string]bool{}
	for _, tree := range []string{"cmd", "sim"} {
		err := filepath.WalkDir(filepath.Join("..", tree), func(f string, d fs.DirEntry, err error) error {
			switch {
			case err != nil:
				return err
			case d.IsDir() && d.Name() == "testdata":
				return filepath.SkipDir
			case d.IsDir() || !strings.HasSuffix(f, ".go") || strings.HasSuffix(f, "_test.go"):
				return nil
			}
			parsed, err := parser.ParseFile(token.NewFileSet(), f, nil, parser.ImportsOnly)
			if err != nil {
				return err
			}
			rel, err := filepath.Rel("..", f)
			if err != nil {
				return err
			}
			rel = filepath.ToSlash(rel)
			for _, imp := range parsed.Imports {
				if strings.HasPrefix(strings.Trim(imp.Path.Value, `"`), module) {
					seen[rel] = true
					if !sites[rel] {
						t.Errorf("%s imports %s; only %s may", rel, imp.Path.Value, site)
					}
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for site := range sites {
		if !seen[site] {
			t.Errorf("non-vacuity: %s imports no %s package, so the per-file check sees nothing", site, module)
		}
	}
}

// inference-sim reaches the vLLM engine's v0.33.0 engine only through blis-schemas' engine.Engine
// contract (GOAL.md): no inference-sim package imports engines/vllm/core/... directly, the
// build site's registration package included only through its facade, and the host's driver
// (sim/engine_driver.go) imports the contract's engine package and no engine implementation.
func TestTheHostReachesTheV032EngineOnlyThroughTheContract(t *testing.T) {
	// The engine's host-facing packages (what the copy called compat/*): the build entry, the
	// kernel adapter and the settings types the host fills. Every other core/ package is the
	// engine's own.
	hostFacade := map[string]bool{
		"github.com/inference-sim/inference-sim/engines/vllm/core/build":       true,
		"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel": true,
		"github.com/inference-sim/inference-sim/engines/vllm/core/runconfig":   true,
	}
	out, err := exec.Command("go", "list", "-deps", "-f", "{{.ImportPath}}|{{join .Imports \",\"}}", "..").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	checked := 0
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		pkg, imports, _ := strings.Cut(line, "|")
		if !strings.HasPrefix(pkg, "github.com/inference-sim/inference-sim") {
			continue
		}
		checked++
		if strings.HasPrefix(pkg, "github.com/inference-sim/inference-sim/engines/vllm/") {
			continue // the engine's own packages import each other
		}
		for _, imp := range strings.Split(imports, ",") {
			if strings.HasPrefix(imp, "github.com/inference-sim/inference-sim/engines/vllm/core") && !hostFacade[imp] {
				t.Errorf("%s imports %s; inference-sim reaches the v0.33.0 engine only through engine.Engine", pkg, imp)
			}
		}
	}
	if checked < 3 {
		t.Fatalf("go list -deps saw %d inference-sim packages; the check is vacuous", checked)
	}
	parsed, err := parser.ParseFile(token.NewFileSet(), "../sim/engine_driver.go", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatal(err)
	}
	contract := false
	for _, imp := range parsed.Imports {
		p := strings.Trim(imp.Path.Value, `"`)
		if p == "github.com/inference-sim/inference-sim/substrate/contracts/engine" {
			contract = true
		}
		if strings.HasPrefix(p, "github.com/inference-sim/inference-sim/engines/vllm") {
			t.Errorf("sim/engine_driver.go imports %s; the driver knows only the contract", p)
		}
	}
	if !contract {
		t.Error("sim/engine_driver.go does not import blis-schemas' engine package: it must drive the engine through the contract")
	}
}
