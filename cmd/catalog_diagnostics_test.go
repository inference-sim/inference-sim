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

// #1776 (R1 cleanup of tracker #1767): three CLI diagnostics in the catalog /
// deployment-flag path told the operator the wrong thing. This file holds their contracts.
//
//	BC-3  a catalog read failure that is not absence is not reported as "not in the catalog"
//	BC-4  resolveCatalogRoot's dispositions match what it actually checks (exists + is a
//	      directory), and it does NOT reject a catalog root it can traverse but not list

// ---------------------------------------------------------------------------
// BC-3: a read failure is not "the model is not catalogued"
// ---------------------------------------------------------------------------

// notCataloguedPhrase is the wording reserved for a model that genuinely has no entry.
// Asserting on it is what makes BC-3 a real classification test rather than "some error".
const notCataloguedPhrase = "is not in the catalog"

// TestReadCatalogEntry_AbsenceIsTheOnlyNotCataloguedVerdict is BC-3. Every catalog read
// failure used to be reported as "model is not in the catalog", so a permission error, a
// directory where a file belongs, or an I/O error all blamed the model for being absent —
// and the operator's fix (add a catalog entry) was the wrong fix. Absence must be the only
// input that earns that verdict.
func TestReadCatalogEntry_AbsenceIsTheOnlyNotCataloguedVerdict(t *testing.T) {
	// The permission-based subtests below skip individually when running as root; the
	// absence, non-file and symlink-loop cases run everywhere, so the classification
	// contract is never left entirely unverified by the test environment.
	skipIfRoot := func(t *testing.T) {
		t.Helper()
		if os.Geteuid() == 0 {
			t.Skip("running as root: mode 000 does not deny access")
		}
	}

	t.Run("genuinely absent says not in the catalog", func(t *testing.T) {
		root := t.TempDir()
		_, err := resolveModelConfigInCatalog("test-org/absent-model", root)
		if err == nil {
			t.Fatal("an absent model must be refused")
		}
		if !strings.Contains(err.Error(), notCataloguedPhrase) {
			t.Errorf("an absent model is exactly the %q case, got: %v", notCataloguedPhrase, err)
		}
	})

	t.Run("unreadable entry does not say not in the catalog", func(t *testing.T) {
		skipIfRoot(t)
		root := t.TempDir()
		entryDir := filepath.Join(root, catalogModelsSubdir, "test-model")
		path := writeCatalogEntry(t, entryDir, minimalHFConfig)
		if err := os.Chmod(path, 0o000); err != nil {
			t.Fatal(err)
		}

		_, err := resolveModelConfigInCatalog("test-org/test-model", root)
		if err == nil {
			t.Fatal("an unreadable entry must be refused")
		}
		if strings.Contains(err.Error(), notCataloguedPhrase) {
			t.Errorf("an entry that IS catalogued but unreadable must not be reported as absent "+
				"(the fix is a permission change, not a new entry), got: %v", err)
		}
		if !strings.Contains(err.Error(), path) {
			t.Errorf("refusal must name the offending file %s, got: %v", path, err)
		}
	})

	t.Run("a directory where config.json belongs does not say not readable", func(t *testing.T) {
		// A directory named config.json is not a plausible catalogued config, so it is
		// grouped with absence — and must therefore earn the absent verdict, not a
		// permission-flavoured one.
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "test-model", hfConfigFile), 0o755); err != nil {
			t.Fatal(err)
		}
		_, err := resolveModelConfigInCatalog("test-org/test-model", root)
		if err == nil {
			t.Fatal("a directory where config.json belongs must be refused")
		}
		if !strings.Contains(err.Error(), notCataloguedPhrase) {
			t.Errorf("a non-file at the config.json path is absence, got: %v", err)
		}
	})

	// A symlink loop is the root-independent form of "present at the path, but the OS
	// cannot tell me anything about it": os.Stat follows the link and returns ELOOP, which
	// is neither ENOENT nor ENOTDIR. It exercises the same classification branch the
	// permission cases do, without depending on the test process being unprivileged.
	t.Run("unresolvable entry does not say not in the catalog", func(t *testing.T) {
		root := t.TempDir()
		entryDir := filepath.Join(root, catalogModelsSubdir, "test-model")
		if err := os.MkdirAll(entryDir, 0o755); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(entryDir, hfConfigFile)
		if err := os.Symlink(path, path); err != nil {
			t.Skipf("this filesystem cannot create a self-referential symlink: %v", err)
		}
		// Non-vacuity: the loop must really be unstatable and NOT report absence,
		// otherwise the assertion below would pass for the wrong reason.
		if _, statErr := os.Stat(path); statErr == nil || os.IsNotExist(statErr) {
			t.Skipf("symlink loop did not produce a non-absence stat error: %v", statErr)
		}

		_, err := resolveModelConfigInCatalog("test-org/test-model", root)
		if err == nil {
			t.Fatal("an unresolvable entry must be refused")
		}
		if strings.Contains(err.Error(), notCataloguedPhrase) {
			t.Errorf("an entry that is present but unresolvable must not be reported as absent, got: %v", err)
		}
		if !strings.Contains(err.Error(), path) {
			t.Errorf("refusal must name the offending path %s, got: %v", path, err)
		}
	})

	t.Run("untraversable entry directory reports the real cause", func(t *testing.T) {
		skipIfRoot(t)
		root := t.TempDir()
		entryDir := filepath.Join(root, catalogModelsSubdir, "test-model")
		path := writeCatalogEntry(t, entryDir, minimalHFConfig)
		if err := os.Chmod(entryDir, 0o000); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(entryDir, 0o755) })

		_, err := resolveModelConfigInCatalog("test-org/test-model", root)
		if err == nil {
			t.Fatal("an untraversable entry directory must be refused")
		}
		if strings.Contains(err.Error(), notCataloguedPhrase) {
			t.Errorf("an inaccessible entry must not be reported as absent, got: %v", err)
		}
		if !strings.Contains(err.Error(), path) {
			t.Errorf("refusal must name the path it could not reach (%s), got: %v", path, err)
		}
	})
}

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
// catalog — it opens one config.json per model at a path it derives — so a root with
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
	entryDir := filepath.Join(root, catalogModelsSubdir, "test-model")
	writeCatalogEntry(t, entryDir, minimalHFConfig)
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
	// And the model still resolves through it, which is what makes rejecting it wrong.
	if _, resolveErr := resolveModelConfigInCatalog("test-org/test-model", root); resolveErr != nil {
		t.Errorf("a model in a search-only catalog must still resolve: %v", resolveErr)
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

	fn := findFuncDecl(t, "hfconfig.go", "resolveCatalogRoot")
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
