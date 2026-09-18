package cmd

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// #1776 (R1 cleanup of tracker #1767): three CLI diagnostics in the catalog /
// deployment-flag path told the operator the wrong thing. This file holds their contracts.
//
//	BC-1  an explicitly-supplied unusable --tp/--hardware is refused as an INVALID VALUE,
//	      not as a missing flag (and vice versa for an omitted one)
//	BC-2  which deployments are REFUSED is unchanged — only the wording differs (INV-6)
//	BC-3  a catalog read failure that is not absence is not reported as "not in the catalog"
//	BC-4  resolveCatalogRoot's dispositions match what it actually checks (exists + is a
//	      directory), and it does NOT reject a catalog root it can traverse but not list
//
// The end-to-end half of BC-1 (the message reaching the operator through cobra's
// Flags().Changed on both `blis run` and `blis replay`) is the subprocess test at the
// bottom; the pure-function tests here cover the branch matrix.

// ---------------------------------------------------------------------------
// BC-1: omitted vs explicitly-invalid deployment flags
// ---------------------------------------------------------------------------

// TestDeploymentFlagRefusal_OmittedVsExplicitlyInvalid is BC-1. Before #1776 the guard
// tested `resolvedTP <= 0` with no knowledge of whether the flag had been supplied, so
// `--tp 0` — a value the operator can type, and also --tp's unset sentinel — was reported
// as a MISSING flag. The two need different fixes: a missing flag has to be added, an
// invalid one corrected, and sending an operator to look for a flag that is already on
// their command line is the diagnostic defect this closes.
func TestDeploymentFlagRefusal_OmittedVsExplicitlyInvalid(t *testing.T) {
	tests := []struct {
		name    string
		in      deploymentFlagValues
		wantAll []string
		wantNot []string
	}{
		{
			name:    "tp omitted is reported as missing",
			in:      deploymentFlagValues{GPU: "H100", GPUSupplied: true},
			wantAll: []string{"missing required flag", "--tp"},
			wantNot: []string{"invalid", "--hardware"},
		},
		{
			name:    "tp zero is reported as invalid, naming the value",
			in:      deploymentFlagValues{GPU: "H100", GPUSupplied: true, TP: 0, TPSupplied: true},
			wantAll: []string{"invalid", "--tp 0", "must be > 0"},
			wantNot: []string{"missing required flag"},
		},
		{
			name:    "tp negative is reported as invalid, naming the value",
			in:      deploymentFlagValues{GPU: "H100", GPUSupplied: true, TP: -1, TPSupplied: true},
			wantAll: []string{"invalid", "--tp -1", "must be > 0"},
			wantNot: []string{"missing required flag"},
		},
		{
			name:    "hardware omitted is reported as missing",
			in:      deploymentFlagValues{TP: 1, TPSupplied: true},
			wantAll: []string{"missing required flag", "--hardware"},
			wantNot: []string{"invalid", "--tp"},
		},
		{
			name:    "explicitly empty hardware is reported as invalid",
			in:      deploymentFlagValues{GPU: "", GPUSupplied: true, TP: 1, TPSupplied: true},
			wantAll: []string{"invalid", "--hardware"},
			wantNot: []string{"missing required flag"},
		},
		{
			// A mixed refusal must not collapse into one disposition: the operator has to
			// add one flag AND correct the other, so both clauses are needed.
			name:    "one omitted and one invalid reports both dispositions",
			in:      deploymentFlagValues{TP: 0, TPSupplied: true},
			wantAll: []string{"missing required flag", "--hardware", "invalid", "--tp 0"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deploymentFlagRefusal(tt.in)
			if got == "" {
				t.Fatalf("deploymentFlagRefusal(%+v) accepted an unusable deployment", tt.in)
			}
			for _, want := range tt.wantAll {
				if !strings.Contains(got, want) {
					t.Errorf("refusal must mention %q, got: %s", want, got)
				}
			}
			for _, notWant := range tt.wantNot {
				if strings.Contains(got, notWant) {
					t.Errorf("refusal must NOT mention %q (that flag is fine), got: %s", notWant, got)
				}
			}
			// Every refusal keeps the operator-actionable tail: BLIS never infers the
			// deployment, on either command.
			if !strings.Contains(got, "does not infer the deployment") {
				t.Errorf("refusal must state that the deployment is never inferred, got: %s", got)
			}
		})
	}
}

// TestDeploymentFlagRefusal_AcceptConditionUnchanged is BC-2, the INV-6 guard on #1776: the
// change is diagnostic-only, so the set of deployments BLIS RUNS must be exactly what it was
// — accepted iff the GPU is non-empty and TP is positive, regardless of whether the value
// arrived from the flag or (for TP) a positive default. The "supplied" bits may only change
// the wording of a refusal, never turn a refusal into an acceptance or back.
func TestDeploymentFlagRefusal_AcceptConditionUnchanged(t *testing.T) {
	for _, gpu := range []string{"", "H100"} {
		for _, tp := range []int{-1, 0, 1, 8} {
			for _, gpuSupplied := range []bool{false, true} {
				for _, tpSupplied := range []bool{false, true} {
					in := deploymentFlagValues{GPU: gpu, GPUSupplied: gpuSupplied, TP: tp, TPSupplied: tpSupplied}
					accepted := deploymentFlagRefusal(in) == ""
					wantAccepted := gpu != "" && tp > 0
					if accepted != wantAccepted {
						t.Errorf("deploymentFlagRefusal(%+v): accepted=%v, want %v — #1776 may only "+
							"change refusal WORDING, never which deployments run (INV-6)",
							in, accepted, wantAccepted)
					}
				}
			}
		}
	}
}

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

// ---------------------------------------------------------------------------
// BC-1, end to end on both commands (INV-13)
// ---------------------------------------------------------------------------

// invalidDeploymentFatalSubprocess drives resolveLatencyConfig in a subprocess with one
// deployment flag EXPLICITLY set to an unusable value. Everything else is fully specified,
// so the only reason it can fail is the flag under test. Going through cobra's ParseFlags
// is the point: the omitted/supplied distinction is Flags().Changed, which a direct call to
// the pure refusal function cannot exercise.
func invalidDeploymentFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	dir := t.TempDir()
	catalogDir, hwPath, err := writeMoEConfigFixture(dir)
	if err != nil {
		os.Exit(2)
	}

	args := []string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--catalog", catalogDir, "--hardware-config", hwPath,
		"--total-kv-blocks", "1000", "--defaults-filepath", "../defaults.yaml",
	}
	switch os.Getenv("BLIS_1776_SCENARIO") {
	case "tp-zero":
		args = append(args, "--hardware", "H100", "--tp", "0")
	case "tp-negative":
		args = append(args, "--hardware", "H100", "--tp", "-1")
	case "hardware-empty":
		args = append(args, "--hardware", "", "--tp", "1")
	default:
		os.Exit(2)
	}

	defaultsFilePath = "../defaults.yaml"
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags(args); err != nil {
		os.Exit(2)
	}
	resolveLatencyConfig(testCmd) // must Fatalf before returning
	os.Exit(0)
}

// TestInvalidDeploymentFlagIsRefusedAsInvalid is BC-1 at the CLI boundary. Both `blis run`
// and `blis replay` resolve through the same resolveLatencyConfig, so one subprocess
// exercise covers both (INV-13) — the same argument TestNS6_DeploymentFlagsRequiredOnRunAndReplay
// makes structurally for the required-flag rule.
func TestInvalidDeploymentFlagIsRefusedAsInvalid(t *testing.T) {
	invalidDeploymentFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}

	tests := []struct {
		scenario string
		wantAll  []string
		wantNot  []string
	}{
		{
			scenario: "tp-zero",
			wantAll:  []string{"invalid", "--tp 0", "must be > 0"},
			wantNot:  []string{"missing required flag"},
		},
		{
			scenario: "tp-negative",
			wantAll:  []string{"invalid", "--tp -1"},
			wantNot:  []string{"missing required flag"},
		},
		{
			scenario: "hardware-empty",
			wantAll:  []string{"invalid", "--hardware"},
			wantNot:  []string{"missing required flag"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.scenario, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=TestInvalidDeploymentFlagIsRefusedAsInvalid", "-test.v")
			cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1", "BLIS_1776_SCENARIO="+tt.scenario)
			out, err := cmd.CombinedOutput()

			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) {
				t.Fatalf("expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", err, out)
			}
			if exitErr.ExitCode() != 1 {
				t.Fatalf("expected exit 1, got %d; output:\n%s", exitErr.ExitCode(), out)
			}
			for _, want := range tt.wantAll {
				if !strings.Contains(string(out), want) {
					t.Errorf("refusal must mention %q; output:\n%s", want, out)
				}
			}
			for _, notWant := range tt.wantNot {
				if strings.Contains(string(out), notWant) {
					t.Errorf("an explicitly-supplied unusable value must not be reported as %q; output:\n%s", notWant, out)
				}
			}
		})
	}
}
