package cmd

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// cmd builds the engine's configuration in one place, the build site (#2021 B2): no
// production file in package cmd but cmd/engine_build.go calls the engine's configuration
// constructors or validators (blis-vllm compat/engine, which cmd reaches as sim's names for
// them, sim/engine_aliases.go). The build site hands the values to blis-vllm compat/build,
// which parses, validates and assembles them, so cmd parses and validates no engine
// configuration itself.
func TestOnlyTheBuildSiteBuildsTheEngineConfiguration(t *testing.T) {
	findings, scanned := engineConfigCallsIn(t, ".")
	if scanned < 10 {
		t.Fatalf("non-vacuity: only %d production files scanned", scanned)
	}
	var outside []string
	inSite := 0
	for _, f := range findings {
		if strings.HasPrefix(f, "engine_build.go:") {
			inSite++
			continue
		}
		outside = append(outside, f)
	}
	if len(outside) > 0 {
		t.Errorf("engine configuration built or validated in cmd outside cmd/engine_build.go:\n  %s",
			strings.Join(outside, "\n  "))
	}
	if inSite != 0 {
		t.Errorf("non-vacuity aside, cmd/engine_build.go itself calls %d engine configuration constructors; "+
			"they belong in blis-vllm compat/build, which the build site calls", inSite)
	}
}

// engineConfigNames are the engine's configuration constructors and validators, as sim
// re-exports them (sim/engine_aliases.go, sim/bundle.go) and as compat/engine declares them.
var engineConfigNames = map[string]bool{
	"NewKVCacheConfig": true, "NewBatchConfig": true, "NewPolicyConfig": true,
	"NewModelHardwareConfig": true, "NewSpeculativeConfig": true,
	"WithKVOffload": true, "WithKVTransferTicksPerBlock": true, "WithSequenceState": true,
	"WithPrefixHit": true, "WithPrefixCachingDisabled": true,
	"IsValidScheduler": true, "IsValidPreemptionPolicy": true, "ValidSchedulerNames": true,
	"ValidPreemptionPolicyNames": true, "SchedulerNameTable": true, "PreemptionPolicyNameTable": true,
	"BuildAdapterCost": true, "AdapterRegistryConstructor": true,
}

// engineConfigPackages are the import paths whose engineConfigNames the check refuses.
var engineConfigPackages = map[string]bool{
	"github.com/inference-sim/inference-sim/sim":       true,
	"github.com/inference-sim/blis-vllm/compat/engine": true,
}

// engineConfigCallsIn lists, for the non-test Go files of dir, each selector that names an
// engineConfigNames member of an engineConfigPackages import, whatever the import is named, as
// "file:line: pkg.Name". It also returns how many files it scanned.
func engineConfigCallsIn(t *testing.T, dir string) ([]string, int) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	scanned := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		scanned++
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		local := map[string]bool{}
		for _, imp := range file.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			if !engineConfigPackages[path] {
				continue
			}
			name := filepath.Base(path)
			if imp.Name != nil {
				name = imp.Name.Name
			}
			local[name] = true
		}
		ast.Inspect(file, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if id, ok := sel.X.(*ast.Ident); ok && local[id.Name] && engineConfigNames[sel.Sel.Name] {
				at := fset.Position(sel.Pos())
				out = append(out, fmt.Sprintf("%s:%d: %s.%s", filepath.Base(at.Filename), at.Line, id.Name, sel.Sel.Name))
			}
			return true
		})
	}
	sort.Strings(out)
	return out, scanned
}

// The check sees a configuration call however the seam is imported: through sim, through a
// renamed sim import, as a function value, and through compat/engine itself. A file with none,
// and another package that happens to be named sim, are clean.
func TestEngineConfigCheckCatchesEachForm(t *testing.T) {
	cases := map[string]string{
		"a constructor through sim": `package probe
import "github.com/inference-sim/inference-sim/sim"
func f() sim.BatchConfig { return sim.NewBatchConfig(1, 1, 0) }`,
		"an option through a renamed import": `package probe
import s "github.com/inference-sim/inference-sim/sim"
func f() s.BatchOption { return s.WithPrefixCachingDisabled(true) }`,
		"a function value": `package probe
import "github.com/inference-sim/inference-sim/sim"
var g = sim.NewKVCacheConfig`,
		"a name check": `package probe
import "github.com/inference-sim/inference-sim/sim"
func f() bool { return sim.IsValidPreemptionPolicy("fcfs") }`,
		"through compat/engine": `package probe
import "github.com/inference-sim/blis-vllm/compat/engine"
func f() any { return engine.AdapterRegistryConstructor() }`,
	}
	run := func(src string) []string {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "probe.go"), []byte(src+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		got, _ := engineConfigCallsIn(t, dir)
		return got
	}
	names := make([]string, 0, len(cases))
	for name := range cases {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if got := run(cases[name]); len(got) != 1 {
			t.Errorf("%s: %v, want one finding", name, got)
		}
	}
	if got := run(`package probe
import "github.com/inference-sim/inference-sim/sim"
func clean(c sim.LoRAConfig) bool { return c.HasAdapters() }`); len(got) != 0 {
		t.Errorf("a file with no configuration call: %v, want none", got)
	}
	if got := run(`package probe
import "other/sim"
func f() any { return sim.NewBatchConfig }`); len(got) != 0 {
		t.Errorf("another package named sim: %v, want none", got)
	}
}

// Outside the build site, package cmd neither builds a value of the engine's configuration types
// as a composite literal, nor writes one of its fields, nor validates or checks one: such a value
// comes from blis-vllm compat/build, which cmd/engine_build.go calls. Nor does it ask the kernel
// model to resolve the engine or reach its kernel, or call the kernel's pricing. The check is
// type-checked, so it sees the types under any name: sim's aliases (sim/engine_aliases.go),
// compat/engine's own, or a local alias. The engine's types are the struct types compat/engine
// declares whose names end in Config or Tier.
func TestNoEngineConfigValuesOutsideTheBuildSite(t *testing.T) {
	findings, scanned := engineConfigValuesIn(t, ".", "github.com/inference-sim/inference-sim/cmd")
	if scanned < 10 {
		t.Fatalf("non-vacuity: only %d production files type-checked", scanned)
	}
	var outside []string
	inSite := 0
	for _, f := range findings {
		if strings.HasPrefix(f, "engine_build.go:") {
			inSite++
			continue
		}
		outside = append(outside, f)
	}
	if len(outside) > 0 {
		t.Errorf("engine configuration built or validated in cmd outside cmd/engine_build.go:\n  %s",
			strings.Join(outside, "\n  "))
	}
	if inSite == 0 {
		t.Error("non-vacuity: cmd/engine_build.go builds the instance configuration, so the check must see it there")
	}
}

const (
	compatEnginePath = "github.com/inference-sim/blis-vllm/compat/engine"
	clusterPath      = "github.com/inference-sim/inference-sim/sim/cluster"
)

// isEngineConfigType reports whether typ is a struct type compat/engine declares whose name ends
// in Config or Tier (KVOffloadConfig, KVOffloadTier, FrontendConfig, SimConfig, ...), or the
// cluster's per-pool engine settings (cluster.PoolOverrides), which the build site fills from
// blis-vllm's resolution of each pool (build.PoolSettingsOf).
func isEngineConfigType(typ types.Type) bool {
	n, ok := types.Unalias(typ).(*types.Named)
	if ok && n.Obj().Pkg() != nil && n.Obj().Pkg().Path() == clusterPath && n.Obj().Name() == "PoolOverrides" {
		return true
	}
	if !ok || n.Obj().Pkg() == nil || n.Obj().Pkg().Path() != compatEnginePath {
		return false
	}
	if _, isStruct := n.Underlying().(*types.Struct); !isStruct {
		return false
	}
	name := n.Obj().Name()
	return strings.HasSuffix(name, "Config") || strings.HasSuffix(name, "Tier")
}

// kernelModelPath is blis-vllm's kernel adapter, whose Model resolves a pool's engine.
const kernelModelPath = "github.com/inference-sim/blis-vllm/compat/kernelmodel"

// kernelModelResolvers are the kernel model's methods that resolve a pool's engine settings or
// reach the kernel that prices them (Kernel): the build site calls them, through blis-vllm
// compat/build, and nothing else in cmd does.
var kernelModelResolvers = map[string]bool{"Settings": true, "SettingsReserving": true, "PrefixHit": true, "Engine": true, "Kernel": true}

// isNamed reports whether typ is, or points to, the named type pkg.name.
func isNamed(typ types.Type, pkg, name string) bool {
	if p, ok := types.Unalias(typ).(*types.Pointer); ok {
		typ = p.Elem()
	}
	n, ok := types.Unalias(typ).(*types.Named)
	return ok && n.Obj().Pkg() != nil && n.Obj().Pkg().Path() == pkg && n.Obj().Name() == name
}

// derefType is typ, or the type it points to.
func derefType(typ types.Type) types.Type {
	if p, ok := types.Unalias(typ).(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// returnsError reports whether the selected method's last result is an error: on an engine
// configuration type, such a method is one of vLLM's checks of it (Validate, CheckFitsOneChunk).
func returnsError(sel *types.Selection) bool {
	sig, ok := sel.Type().(*types.Signature)
	if !ok || sig.Results().Len() == 0 {
		return false
	}
	return types.Identical(sig.Results().At(sig.Results().Len()-1).Type(), types.Universe.Lookup("error").Type())
}

// isKernelModel reports whether typ is, or points to, the kernel adapter's Model.
func isKernelModel(typ types.Type) bool { return isNamed(typ, kernelModelPath, "Model") }

// kernelPath is blis-schemas' kernel contract, whose Kernel prices a pool's batches, blocks and
// transfers.
const kernelPath = "github.com/inference-sim/blis-schemas/kernel"

// engineConfigValuesIn type-checks the non-test Go files of dir as package path (dependencies from
// the build's export data) and lists, as "file:line: what", each composite literal of an engine
// configuration type, each write to a field of one, each check of one (a method returning an
// error: Validate, CheckFitsOneChunk), each call of a kernel model resolver
// (kernelModelResolvers), each read of a kernelmodel.Settings field, and each method call on the
// kernel. An empty literal, a zero value passed through, is not counted. It also returns how many
// files it checked.
func engineConfigValuesIn(t *testing.T, dir, path string) ([]string, int) {
	t.Helper()
	return engineConfigValuesInWith(t, dir, path, dir)
}

// engineConfigValuesInWith is engineConfigValuesIn with the export data taken from the build of
// depsDir (a package whose dependencies cover dir's imports).
func engineConfigValuesInWith(t *testing.T, dir, path, depsDir string) ([]string, int) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var parsed []*ast.File
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		parsed = append(parsed, af)
	}
	info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}, Selections: map[*ast.SelectorExpr]*types.Selection{}}
	conf := types.Config{Importer: exportImporter(t, fset, depsDir)}
	if _, err := conf.Check(path, fset, parsed, info); err != nil {
		t.Fatalf("type-checking %s: %v", dir, err)
	}
	var out []string
	for _, f := range parsed {
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.AssignStmt:
				// A write to a field of an engine configuration value (cfg.PerBlockBytes = ...):
				// the value is the engine's, set by blis-vllm compat/build through the build site.
				for _, lhs := range x.Lhs {
					sel, ok := lhs.(*ast.SelectorExpr)
					if !ok {
						continue
					}
					if s, ok := info.Selections[sel]; ok && s.Kind() == types.FieldVal && isEngineConfigType(derefType(s.Recv())) {
						at := fset.Position(sel.Pos())
						out = append(out, fmt.Sprintf("%s:%d: writes %s.%s", filepath.Base(at.Filename), at.Line,
							types.Unalias(derefType(s.Recv())).(*types.Named).Obj().Name(), sel.Sel.Name))
					}
				}
			case *ast.CompositeLit:
				if tv, ok := info.Types[x]; ok && isEngineConfigType(tv.Type) && len(x.Elts) > 0 {
					at := fset.Position(x.Pos())
					out = append(out, fmt.Sprintf("%s:%d: a %s literal", filepath.Base(at.Filename), at.Line, types.Unalias(tv.Type).(*types.Named).Obj().Name()))
				}
			case *ast.SelectorExpr:
				sel, ok := info.Selections[x]
				if !ok {
					return true
				}
				at := fset.Position(x.Pos())
				switch {
				case sel.Kind() == types.MethodVal && isEngineConfigType(derefType(sel.Recv())) && returnsError(sel):
					out = append(out, fmt.Sprintf("%s:%d: %s.%s", filepath.Base(at.Filename), at.Line,
						types.Unalias(derefType(sel.Recv())).(*types.Named).Obj().Name(), x.Sel.Name))
				case sel.Kind() == types.MethodVal && isKernelModel(sel.Recv()) && kernelModelResolvers[x.Sel.Name]:
					out = append(out, fmt.Sprintf("%s:%d: Model.%s", filepath.Base(at.Filename), at.Line, x.Sel.Name))
				case sel.Kind() == types.MethodVal && isNamed(sel.Recv(), kernelPath, "Kernel"):
					out = append(out, fmt.Sprintf("%s:%d: Kernel.%s", filepath.Base(at.Filename), at.Line, x.Sel.Name))
				case sel.Kind() == types.FieldVal && isNamed(sel.Recv(), kernelModelPath, "Settings"):
					out = append(out, fmt.Sprintf("%s:%d: Settings.%s", filepath.Base(at.Filename), at.Line, x.Sel.Name))
				}
			}
			return true
		})
	}
	sort.Strings(out)
	return out, len(parsed)
}

// exportImporter imports the packages dir's build depends on from the build's own export data
// (go list -export), so the check type-checks against the pinned modules exactly.
func exportImporter(t *testing.T, fset *token.FileSet, dir string) types.Importer {
	t.Helper()
	cmd := exec.Command("go", "list", "-export", "-deps", "-json=ImportPath,Export", ".")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list -export: %v", err)
	}
	exports := map[string]string{}
	dec := json.NewDecoder(strings.NewReader(string(out)))
	for dec.More() {
		var p struct{ ImportPath, Export string }
		if err := dec.Decode(&p); err != nil {
			t.Fatal(err)
		}
		exports[p.ImportPath] = p.Export
	}
	return importer.ForCompiler(fset, "gc", func(path string) (io.ReadCloser, error) {
		f, ok := exports[path]
		if !ok || f == "" {
			return nil, fmt.Errorf("no export data for %s", path)
		}
		return os.Open(f)
	})
}

// The value check sees each form: a literal through sim's alias, through compat/engine itself, a
// nested tier literal, a pointer literal, a Validate call through a local alias, a chunk check
// through a pointer, a field write, the kernel model's resolvers and its kernel, a kernel pricing
// call, and a Settings field read. A literal of another package's type with the same name, an
// empty literal, a method that checks nothing and a kernel model's role are not findings.
func TestEngineConfigValueCheckCatchesEachForm(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"a literal through sim":           `func f() sim.FrontendConfig { return sim.FrontendConfig{RendererNumWorkers: 1} }`,
		"a literal through compat/engine": `func f() engine.KVOffloadConfig { return engine.KVOffloadConfig{Enabled: true} }`,
		"a nested tier literal":           `func f() []sim.KVOffloadTier { return []sim.KVOffloadTier{{Type: "fs"}} }`,
		"a pointer literal":               `func f() *sim.FrontendConfig { return &sim.FrontendConfig{APIServerCount: 2} }`,
		"a Validate through a local alias": `type fc = sim.FrontendConfig
func f(c fc) error { return c.Validate() }`,
		"a pool overrides literal":        `func f() cluster.PoolOverrides { return cluster.PoolOverrides{GPU: "h100"} }`,
		"a kernel model's settings":       `func f(m *kernelmodel.Model) error { _, err := m.Settings(); return err }`,
		"a kernel model's prefix hit":     `func f(m *kernelmodel.Model) bool { return m.PrefixHit(nil).Note == "" }`,
		"a settings field read":           `func f(s kernelmodel.Settings) int64 { return s.KVBlocks }`,
		"a kernel model's kernel":         `func f(m *kernelmodel.Model) any { return m.Kernel() }`,
		"a kernel pricing call":           `func f(k kernel.Kernel) int64 { return k.SequenceFixedBytes() }`,
		"a config field write":            `func f(c *sim.KVOffloadConfig) { c.PerBlockBytes = 64 }`,
		"a chunk check through a pointer": `func f(c *sim.KVOffloadConfig) error { return c.CheckFitsOneChunk() }`,
	}
	clean := map[string]string{
		"another package's FrontendConfig": `type FrontendConfig struct{ N int }
func f() FrontendConfig { return FrontendConfig{N: 1} }`,
		"an empty literal":      `func f() sim.KVOffloadConfig { return sim.KVOffloadConfig{} }`,
		"another method":        `func f(c sim.FrontendConfig) bool { return c.Enabled() }`,
		"a kernel model's role": `func f(m *kernelmodel.Model) string { return string(m.Role()) }`,
	}
	run := func(body string) []string {
		pkgDir := filepath.Join(dir, fmt.Sprintf("p%d", len(body)))
		if err := os.MkdirAll(pkgDir, 0o755); err != nil {
			t.Fatal(err)
		}
		src := "package probe\n\nimport (\n\t\"github.com/inference-sim/blis-schemas/kernel\"\n\t\"github.com/inference-sim/blis-vllm/compat/engine\"\n\t\"github.com/inference-sim/blis-vllm/compat/kernelmodel\"\n\t\"github.com/inference-sim/inference-sim/sim\"\n\t\"github.com/inference-sim/inference-sim/sim/cluster\"\n)\n\nvar (\n\t_ = kernel.Placement{}\n\t_ = engine.ReplicaAPIServers\n\t_ = kernelmodel.Open\n\t_ = sim.ReplicaAPIServers\n\t_ = cluster.ResolvePoolConfig\n)\n\n" + body + "\n"
		if err := os.WriteFile(filepath.Join(pkgDir, "probe.go"), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		// The probe's imports are cmd's own dependencies, so cmd's export data serves it.
		got, _ := engineConfigValuesInWith(t, pkgDir, "probe", ".")
		return got
	}
	for _, name := range sortedKeysOf(cases) {
		if got := run(cases[name]); len(got) != 1 {
			t.Errorf("%s: %v, want one finding", name, got)
		}
	}
	for _, name := range sortedKeysOf(clean) {
		if got := run(clean[name]); len(got) != 0 {
			t.Errorf("%s: %v, want none", name, got)
		}
	}
}

func sortedKeysOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
