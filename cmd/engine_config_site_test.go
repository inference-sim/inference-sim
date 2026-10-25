package cmd

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
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
