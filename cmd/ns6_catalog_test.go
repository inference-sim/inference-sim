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

// NS-6 (#1733): a model runs if and only if it is in the catalog.
//
// This file holds the contracts for the two run-time behaviours #1733 removes:
//   - BC-3: no run-time path can create or modify a catalog file (there is no fetch left);
//   - BC-4: --hardware and --tp are required, refused BY NAME rather than inferred from
//     defaults.yaml and warned-about.
//
// BC-1/BC-2 (the refusal itself and its message) live with the resolver in
// hfconfig_test.go, since they are properties of resolveModelConfig.

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
//  2. At the resolution boundary (cmd/hfconfig.go, the only file that maps a model to a
//     catalog directory): no network import and no file-creating call, so that file cannot
//     write a catalog entry however it is edited.
func TestNS6_NoRuntimeFetch_StaticGuard(t *testing.T) {
	bannedIdents := map[string]string{
		"fetchHFConfig":        "the HuggingFace config fetch was removed by #1733",
		"fetchHFConfigFunc":    "the HuggingFace config fetch indirection was removed by #1733",
		"fetchHFConfigFromURL": "the HuggingFace config fetch was removed by #1733",
		"GetDefaultSpecs":      "per-model --hardware/--tp inference was removed by #1733",
	}
	// The resolution boundary must be incapable of writing or fetching.
	resolverFile := "hfconfig.go"
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
// server, so it resolves no model config (nothing to fetch, nothing to write) and takes
// no --hardware/--tp. Were someone to give it registerSimConfigFlags, it would inherit
// the required-flag rule silently; this test says that has to be a deliberate decision.
func TestNS6_ObserveTakesNoDeploymentFlags(t *testing.T) {
	for _, flag := range []string{"hardware", "tp", "catalog"} {
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
// BC-4: --hardware and --tp are required, refused by name
// ---------------------------------------------------------------------------

// TestNS6_RequireDeploymentFlags_AcceptsFullySpecified is the negative control for the
// refusal tests below: a fully-specified deployment must pass the guard (if it fatally
// exited here the subprocess tests would prove nothing about the missing-flag case).
func TestNS6_RequireDeploymentFlags_AcceptsFullySpecified(t *testing.T) {
	// requireDeploymentFlags terminates the process on refusal, so reaching the next
	// statement IS the assertion that a complete deployment is accepted.
	requireDeploymentFlags(deploymentFlagValues{GPU: "H100", GPUSupplied: true, TP: 1, TPSupplied: true})
}

// ns6DeploymentFatalSubprocess drives resolveLatencyConfig in a subprocess with one of
// --hardware/--tp withheld. Everything else is fully specified (explicit --catalog and
// --hardware-config), so the ONLY reason it can fail is the missing flag.
func ns6DeploymentFatalSubprocess(t *testing.T) {
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
	switch os.Getenv("BLIS_NS6_SCENARIO") {
	case "missing-hardware":
		args = append(args, "--tp", "1")
	case "missing-tp":
		args = append(args, "--hardware", "H100")
	case "missing-both":
		// neither flag supplied
	default:
		os.Exit(2)
	}

	// Mirror the package-var state the CLI would have; registerSimConfigFlags below
	// resets the bound vars to their flag defaults ("" / 0), which is exactly the
	// "flag absent" state under test.
	defaultsFilePath = "../defaults.yaml"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags(args); err != nil {
		os.Exit(2)
	}
	resolveLatencyConfig(testCmd) // must Fatalf before returning
	os.Exit(0)
}

// TestNS6_MissingDeploymentFlagIsRefusedByName is BC-4. Before #1733 each of these ran to
// completion on a defaults.yaml-supplied deployment after a `logrus.Warnf`; now each is a
// hard refusal that names the flag the operator omitted.
func TestNS6_MissingDeploymentFlagIsRefusedByName(t *testing.T) {
	ns6DeploymentFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}

	tests := []struct {
		scenario string
		wantAll  []string
		wantNot  []string
	}{
		{
			scenario: "missing-hardware",
			wantAll:  []string{"missing required flag", "--hardware"},
			wantNot:  []string{"--tp ("},
		},
		{
			scenario: "missing-tp",
			wantAll:  []string{"missing required flag", "--tp"},
			wantNot:  []string{"--hardware ("},
		},
		{
			scenario: "missing-both",
			wantAll:  []string{"missing required flag", "--hardware", "--tp"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.scenario, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=TestNS6_MissingDeploymentFlagIsRefusedByName", "-test.v")
			cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1", "BLIS_NS6_SCENARIO="+tt.scenario)
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
			// The refusal must name only the flag that is actually missing — a message
			// listing both would not tell the operator what to fix.
			for _, notWant := range tt.wantNot {
				if strings.Contains(string(out), notWant) {
					t.Errorf("refusal must not mention %q when that flag was supplied; output:\n%s", notWant, out)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BC-5: byte-identity on the fully-specified path (INV-6)
// ---------------------------------------------------------------------------

// #1768 removed TestNS6_ByteIdentityAnchor_GoldenModelDeployment, which lived here. It read
// defaults.yaml's per-model GPU/tensor_parallelism keys to machine-verify that the explicit
// --hardware H100 --tp 1 passed by TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline equalled
// what defaults.yaml would have inferred pre-#1733. Those keys are now deleted, so the
// cross-check has no data left to read. The INV-6 evidence itself is undiminished and lives in
// that byte-identity test: its golden (specs/007-lora-control-plane/testdata/baseline_noop.json)
// was captured from a run that supplied NEITHER flag, and it still matches.

// TestNS6_DeploymentFlagsRequiredOnRunAndReplay is the INV-13 half of BC-4: both commands
// register the flags and both resolve through the same requireDeploymentFlags call inside
// resolveLatencyConfig, so neither can drift into inferring a deployment the other refuses.
func TestNS6_DeploymentFlagsRequiredOnRunAndReplay(t *testing.T) {
	for _, name := range []string{"run", "replay"} {
		c := &cobra.Command{}
		registerSimConfigFlags(c)
		for flag, wantDefault := range map[string]string{"hardware": "", "tp": "0"} {
			f := c.Flags().Lookup(flag)
			if f == nil {
				t.Fatalf("%s: --%s must be registered (registerSimConfigFlags)", name, flag)
			}
			// The "absent" sentinel must stay a value requireDeploymentFlags rejects,
			// otherwise the requirement could be satisfied by the default itself.
			if f.DefValue != wantDefault {
				t.Errorf("%s: --%s default must remain the unset sentinel %q so an omitted "+
					"flag is refused, got %q", name, flag, wantDefault, f.DefValue)
			}
		}
	}
}
