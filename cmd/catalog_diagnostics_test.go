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

// #1776 (R1 cleanup of tracker #1767): CLI diagnostics in the catalog path told the operator
// the wrong thing. This file holds the contract that survives the move of catalog reading into
// blis-latency-kernel:
//
//	BC-4  resolveCatalogRoot's dispositions match what it actually checks (exists + is a
//	      directory), and it does NOT reject a catalog root it can traverse but not list

// ---------------------------------------------------------------------------
// BC-4: resolveCatalogRoot claims only what it checks
// ---------------------------------------------------------------------------

// TestResolveCatalogRoot_DispositionsMatchTheCause is the first half of BC-4: the two
// unusable-root cases an operator actually hits get distinct, accurate messages. Before
// #1776 both went through one "%q is not readable: %w" branch, so the overwhelmingly
// common typo (a path that does not exist) was described as a permission problem.
func TestResolveCatalogRoot_DispositionsMatchTheCause(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	t.Setenv(catalogEnvVar, "")

	fileNotDir := filepath.Join(t.TempDir(), "catalog-is-a-file")
	if err := os.WriteFile(fileNotDir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		root    string
		wantAll []string
		wantNot []string
	}{
		{
			name:    "absent root says it does not exist",
			root:    filepath.Join(t.TempDir(), "no-such-catalog"),
			wantAll: []string{"does not exist"},
			wantNot: []string{"not a directory"},
		},
		{
			name:    "a file where the root belongs says it is not a directory",
			root:    fileNotDir,
			wantAll: []string{"not a directory"},
			wantNot: []string{"does not exist"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			catalogPath = tt.root
			got, err := resolveCatalogRoot()
			if err == nil {
				t.Fatalf("expected refusal, got %q", got)
			}
			for _, want := range append(tt.wantAll, tt.root, "--catalog", catalogEnvVar) {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("refusal must mention %q, got: %v", want, err)
				}
			}
			for _, notWant := range tt.wantNot {
				if strings.Contains(err.Error(), notWant) {
					t.Errorf("refusal must not mention %q, got: %v", notWant, err)
				}
			}
		})
	}
}

// TestResolveCatalogRoot_AcceptsSearchOnlyRoot is the second half of BC-4 and the reason
// #1776's "add a real accessibility check" option was NOT taken: BLIS never LISTS the
// catalog — it opens the files a scenario names, at paths it derives — so a root with
// search-only permission (mode --x) is usable, and a read/list probe here would refuse a
// catalog that works. This test fails if such a probe is ever added, which is what keeps
// the reworded claim ("exists and is a directory") honest rather than merely softer.
func TestResolveCatalogRoot_AcceptsSearchOnlyRoot(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: mode 0111 does not deny listing")
	}
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	t.Setenv(catalogEnvVar, "")

	root := filepath.Join(t.TempDir(), "search-only-catalog")
	graph := filepath.Join(root, catalogModelsSubdir, "test-model", catalogModelGraphFile)
	if err := os.MkdirAll(filepath.Dir(graph), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(graph, []byte("kind: ModelGraph\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(root, 0o111); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(root, 0o755) })

	catalogPath = root
	got, err := resolveCatalogRoot()
	if err != nil {
		t.Fatalf("a traversable-but-unlistable catalog root must be accepted (BLIS opens a "+
			"derived path and never lists the catalog): %v", err)
	}
	if got != root {
		t.Errorf("resolveCatalogRoot() = %q, want %q", got, root)
	}
	// Non-vacuity: the mode really does deny listing, so the acceptance above is meaningful.
	if _, readErr := os.ReadDir(root); readErr == nil {
		t.Skip("this filesystem does not enforce mode 0111 (cannot demonstrate unlistability)")
	}
	// And a derived path inside it is still readable, which is what makes rejecting it wrong.
	if _, readErr := os.ReadFile(graph); readErr != nil {
		t.Errorf("a file at a derived path in a search-only catalog must still be readable: %v", readErr)
	}
}

// TestResolveCatalogRoot_DoesNotProbeListability is the root-independent companion to
// TestResolveCatalogRoot_AcceptsSearchOnlyRoot, which can only demonstrate its point when
// the test process is unprivileged (mode bits do not constrain root). The claim in
// resolveCatalogRoot's doc comment — "exists and is a directory", deliberately not an
// accessibility probe — is structural, so it can be checked structurally: the function must
// not list or open the catalog root, because either call needs read permission BLIS does not
// need and would refuse a working search-only catalog (#1776).
func TestResolveCatalogRoot_DoesNotProbeListability(t *testing.T) {
	banned := map[string]string{
		"os.ReadDir":     "listing the catalog root needs read permission BLIS never needs",
		"os.Open":        "opening the root as a directory needs read permission BLIS never needs",
		"os.OpenFile":    "opening the root needs permission BLIS never needs",
		"filepath.Walk":  "walking the catalog root needs read permission BLIS never needs",
		"filepath.Glob":  "globbing the catalog root needs read permission BLIS never needs",
		"ioutil.ReadDir": "listing the catalog root needs read permission BLIS never needs",
	}

	fn := findFuncDecl(t, "catalog_root.go", "resolveCatalogRoot")
	sawStat := false
	ast.Inspect(fn, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok {
			return true
		}
		call := pkg.Name + "." + sel.Sel.Name
		if call == "os.Stat" {
			sawStat = true
		}
		if why, isBanned := banned[call]; isBanned {
			t.Errorf("resolveCatalogRoot calls %s — %s. It checks that the root exists and is a "+
				"directory; a listability probe would refuse a catalog that works (#1776)", call, why)
		}
		return true
	})
	// Non-vacuity: the function really does perform the existence check it claims, so the
	// absence of the banned calls above is a deliberate scope and not an empty body.
	if !sawStat {
		t.Error("non-vacuity: expected resolveCatalogRoot to call os.Stat for its existence check")
	}
}

// findFuncDecl parses a production source in this package and returns the named top-level
// function declaration, failing the test if it is absent.
func findFuncDecl(t *testing.T, file, name string) *ast.FuncDecl {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, file, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", file, err)
	}
	for _, decl := range parsed.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Recv == nil && fn.Name.Name == name {
			return fn
		}
	}
	t.Fatalf("%s declares no top-level func %s", file, name)
	return nil
}
