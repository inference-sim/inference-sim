package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// NS-6 (#1733): a model runs if and only if it is in the catalog.
//
// This file holds the contracts for the two run-time behaviours #1733 removes:
//   - BC-3: no run-time path can create or modify a catalog file (there is no fetch left);
//   - BC-4: the deployment is never inferred -- it comes from the kernel scenario, and a
//     flag restating the model, hardware or TP is refused BY NAME.
//
// BC-1/BC-2 (the refusal itself and its message) are properties of blis-latency-kernel, which
// reads the scenario's model graph; catalog_layout_test.go pins that they reach the CLI.

// ---------------------------------------------------------------------------
// BC-3: no run-time path creates or modifies a catalog file
// ---------------------------------------------------------------------------

// TestNS6_NoRuntimeFetch_StaticGuard asserts the fetch machinery is GONE from the cmd
// package's production sources, not merely unreachable. A behavioral test can only show
// that one input does not fetch; this shows there is no code left that could.
//
// Two checks, both on the real mechanisms that made the catalog grow:
//
//  1. Package-wide: no production source may name a removed fetch symbol, nor carry a
//     huggingface.co STRING LITERAL — the outbound host. (Prose mentioning HuggingFace is
//     fine; only a literal could become a URL. `net/http` itself cannot be banned
//     package-wide because `blis observe` legitimately dispatches to a real server.)
//  2. At the resolution boundary (cmd/catalog_root.go, the file that locates the catalog):
//     no network import and no file-creating call, so that file cannot write a catalog entry
//     however it is edited.
func TestNS6_NoRuntimeFetch_StaticGuard(t *testing.T) {
	bannedIdents := map[string]string{
		"fetchHFConfig":        "the HuggingFace config fetch was removed by #1733",
		"fetchHFConfigFunc":    "the HuggingFace config fetch indirection was removed by #1733",
		"fetchHFConfigFromURL": "the HuggingFace config fetch was removed by #1733",
		"GetDefaultSpecs":      "per-model --hardware/--tp inference was removed by #1733",
	}
	// The resolution boundary must be incapable of writing or fetching.
	resolverFile := "catalog_root.go"
	resolverBannedImports := map[string]string{
		`"net/http"`: "the catalog resolver must not reach the network (NS-6)",
		`"io"`:       "the catalog resolver reads one file via os.ReadFile; streaming I/O implies a fetch",
	}
	resolverBannedCalls := map[string]string{
		"os.WriteFile": "the catalog resolver must never write a catalog entry (NS-6)",
		"os.MkdirAll":  "the catalog resolver must never create a catalog directory (NS-6)",
		"os.Create":    "the catalog resolver must never create a catalog entry (NS-6)",
		"os.OpenFile":  "the catalog resolver must never open a catalog entry for writing (NS-6)",
		"os.Remove":    "the catalog resolver must never delete a catalog entry (it may be operator-authored)",
	}
	if len(bannedIdents) == 0 || len(resolverBannedImports) == 0 || len(resolverBannedCalls) == 0 {
		t.Fatal("non-vacuity: the guard has nothing to check")
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}

	scanned, sawResolver := 0, false
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		scanned++
		isResolver := file == resolverFile
		sawResolver = sawResolver || isResolver

		src, readErr := os.ReadFile(file)
		if readErr != nil {
			t.Fatalf("read %s: %v", file, readErr)
		}
		fset := token.NewFileSet()
		f, parseErr := parser.ParseFile(fset, file, src, parser.SkipObjectResolution)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
		}
		line := func(n ast.Node) int { return fset.Position(n.Pos()).Line }

		if isResolver {
			for _, imp := range f.Imports {
				if why, banned := resolverBannedImports[imp.Path.Value]; banned {
					t.Errorf("%s imports %s — %s", file, imp.Path.Value, why)
				}
			}
		}

		ast.Inspect(f, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.Ident:
				if why, banned := bannedIdents[node.Name]; banned {
					t.Errorf("%s:%d references %s — %s", file, line(node), node.Name, why)
				}
			case *ast.BasicLit:
				if node.Kind == token.STRING && strings.Contains(strings.ToLower(node.Value), "huggingface.co") {
					t.Errorf("%s:%d carries a huggingface.co string literal — no run-time path may "+
						"reach HuggingFace (NS-6, #1733)", file, line(node))
				}
			case *ast.SelectorExpr:
				if !isResolver {
					return true
				}
				pkg, ok := node.X.(*ast.Ident)
				if !ok {
					return true
				}
				if why, banned := resolverBannedCalls[pkg.Name+"."+node.Sel.Name]; banned {
					t.Errorf("%s:%d calls %s.%s — %s", file, line(node), pkg.Name, node.Sel.Name, why)
				}
			}
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("non-vacuity: scanned no production sources")
	}
	if !sawResolver {
		t.Fatalf("non-vacuity: %s was not scanned — the resolver checks did not run", resolverFile)
	}
}

// TestNS6_ObserveTakesNoDeploymentFlags pins the boundary that makes BC-3 hold for
// `blis observe` by construction: observe is a black-box dispatcher against a real
// server, so it places no instances and takes no deployment -- no --scenario, and none of the
// retired --hardware/--tp. Were someone to give it registerSimConfigFlags, it would inherit
// the scenario requirement silently; this test says that has to be a deliberate decision.
//
// #1769 narrowed this: observe DOES now declare --catalog, because the named workload
// presets moved into the catalog and observe resolves one for --workload. It still resolves
// no MODEL config, which is what the deployment flags are about, so the deployment flags stay
// banned here, and TestCatalogPresets_CatalogFlagOnEveryConsumer owns the positive --catalog
// claim.
func TestNS6_ObserveTakesNoDeploymentFlags(t *testing.T) {
	for _, flag := range []string{"scenario", "hardware", "tp"} {
		if f := observeCmd.Flags().Lookup(flag); f != nil {
			t.Errorf("`blis observe` must not declare --%s: it resolves no model config and "+
				"places no instances, so a deployment flag there would be inert", flag)
		}
	}
	// Non-vacuity: observe does declare its own flags, so Lookup works.
	if observeCmd.Flags().Lookup("server-url") == nil {
		t.Fatal("non-vacuity: expected `blis observe` to declare --server-url")
	}
}

// ---------------------------------------------------------------------------
// BC-4: the deployment has one source -- the scenario
// ---------------------------------------------------------------------------

// TestNS6_DeploymentComesOnlyFromTheScenario is BC-4 on the kernel: BLIS never infers a
// deployment, and since the scenario states the model, the hardware and the tensor-parallel
// width, no flag can restate them. Each such flag is refused by name on BOTH commands
// (INV-13: neither can accept a deployment input the other refuses), and a run that names
// no scenario is refused naming the missing flag rather than run on a default.
func TestNS6_DeploymentComesOnlyFromTheScenario(t *testing.T) {
	for _, command := range []string{"run", "replay"} {
		for _, flag := range []string{"--model", "--hardware", "--tp"} {
			t.Run(command+" "+flag, func(t *testing.T) {
				_, stderr, err := runKernelCLI(t, command, "--scenario", kernelTestScenario, flag, "1")
				if err == nil {
					t.Fatalf("%s %s was accepted; the scenario is the only deployment source", command, flag)
				}
				if !strings.Contains(stderr, "unknown flag: "+flag) {
					t.Errorf("refusal must name %s; stderr:\n%s", flag, lastLines(stderr, 3))
				}
			})
		}
	}
	t.Run("run without a scenario", func(t *testing.T) {
		_, stderr, err := runKernelCLI(t, "run", "--num-requests", "1")
		if err == nil || !strings.Contains(stderr, "--scenario") {
			t.Errorf("a run naming no scenario must be refused naming --scenario: err=%v\n%s",
				err, lastLines(stderr, 3))
		}
	})
}
