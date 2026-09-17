package cmd

import (
	"bytes"
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

// Contract tests for #1731 (R1 task S4): the model catalog is located by --catalog or
// BLIS_CATALOG and nothing else, and --model-config-folder is retired.
//
//	AC-1 — both forms locate the catalog; --catalog wins when both are set.
//	AC-2 — a run with NEITHER is refused, naming both forms. No working-directory default.
//	AC-3 — --model-config-folder is gone from cmd/'s production sources and every command.
//	AC-5 — locating a catalogued model via --catalog changes no number (INV-6).
//
// The precedence law itself and the unusable-root refusal are unit-tested on
// catalogRootFrom / resolveCatalogRoot in hfconfig_test.go; the tests here prove the
// requirement is actually WIRED INTO the commands rather than merely available as a helper.

// catalogLegEnv selects which leg of TestRunCmd_CatalogLocation the re-exec subprocess
// executes.
const catalogLegEnv = "BLIS_CATALOG_LEG"

// runCatalogLeg re-execs this test binary as `blis run` with the catalog supplied the way
// leg names, and returns its stdout plus the process error. The subprocess environment
// always sets BLIS_CATALOG explicitly (to "" where the leg wants it absent), so an ambient
// value in a developer's or CI's environment cannot make a leg pass vacuously.
func runCatalogLeg(t *testing.T, leg, env string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_CatalogLocation$")
	cmd.Env = append(os.Environ(), catalogLegEnv+"="+leg, catalogEnvVar+"="+env)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// TestRunCmd_CatalogLocation is AC-1 and AC-2 at the CLI boundary: `--catalog` and
// `BLIS_CATALOG` each locate the catalog for a real `blis run`, and a run with neither is
// refused naming both forms rather than falling back to a working-directory default.
//
// Each leg runs in a re-exec subprocess so the real cobra tree executes and a
// logrus.Fatalf surfaces as exit status 1.
func TestRunCmd_CatalogLocation(t *testing.T) {
	if leg := os.Getenv(catalogLegEnv); leg != "" {
		args := []string{
			"run", "--model", "qwen/qwen3-14b",
			"--hardware", "H100", "--tp", "1",
			"--seed", "42", "--num-requests", "5",
			"--defaults-filepath", "../defaults.yaml",
		}
		if leg == "flag" {
			args = append(args, "--catalog", "../model_configs")
		}
		rootCmd.SetArgs(args)
		if execErr := rootCmd.Execute(); execErr != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	tests := []struct {
		name      string
		leg       string
		env       string // BLIS_CATALOG value for the subprocess ("" = unset-equivalent)
		wantFatal bool
	}{
		{name: "--catalog locates the catalog", leg: "flag", wantFatal: false},
		{name: "BLIS_CATALOG locates the catalog", leg: "env", env: "../model_configs", wantFatal: false},
		{name: "neither is refused", leg: "none", wantFatal: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stdout, stderr, err := runCatalogLeg(t, tt.leg, tt.env)

			if !tt.wantFatal {
				if err != nil {
					t.Fatalf("expected the run to succeed, got %v\nstderr:\n%s", err, stderr)
				}
				// Non-vacuity: the leg must actually have simulated something, otherwise a
				// silently-empty run would satisfy "no error".
				if !strings.Contains(stdout, "completed_requests") {
					t.Errorf("expected simulation metrics on stdout, got:\n%s", stdout)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected the run to be refused when neither --catalog nor %s is set\nstdout:\n%s",
					catalogEnvVar, stdout)
			}
			// The refusal must name BOTH forms, so the operator learns either works.
			if !strings.Contains(stderr, "--catalog") {
				t.Errorf("refusal must name --catalog; stderr:\n%s", stderr)
			}
			if !strings.Contains(stderr, catalogEnvVar) {
				t.Errorf("refusal must name %s; stderr:\n%s", catalogEnvVar, stderr)
			}
			// And it must NOT silently fall back to a working-directory location.
			if strings.Contains(stderr, "using config from") {
				t.Errorf("refusal leg resolved a config from a working-directory default; stderr:\n%s", stderr)
			}
		})
	}
}

// TestRunCmd_CatalogLocation_ByteIdenticalAcrossForms is AC-5 / INV-6: HOW the catalog is
// named is not an input to the simulation. The same catalogued model located by --catalog
// and by BLIS_CATALOG must produce byte-identical stdout.
//
// This is the transitive form of the retired-flag comparison AC-5 asks for: the
// --model-config-folder run it names no longer exists to compare against, but
// TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline still matches a golden captured
// before this change from a run that named no catalog at all — so the pre-feature value is
// pinned there, and this test pins that both new spellings land on it.
func TestRunCmd_CatalogLocation_ByteIdenticalAcrossForms(t *testing.T) {
	if os.Getenv(catalogLegEnv) != "" {
		t.Skip("subprocess leg driven by TestRunCmd_CatalogLocation")
	}
	viaFlag, stderrFlag, err := runCatalogLeg(t, "flag", "")
	if err != nil {
		t.Fatalf("--catalog leg failed: %v\nstderr:\n%s", err, stderrFlag)
	}
	viaEnv, stderrEnv, err := runCatalogLeg(t, "env", "../model_configs")
	if err != nil {
		t.Fatalf("%s leg failed: %v\nstderr:\n%s", catalogEnvVar, err, stderrEnv)
	}
	// Non-vacuity: an empty stdout would make any comparison trivially pass.
	if !strings.Contains(viaFlag, "completed_requests") {
		t.Fatalf("non-vacuity: --catalog leg produced no metrics:\n%s", viaFlag)
	}
	if viaFlag != viaEnv {
		t.Errorf("stdout must be byte-identical whichever form locates the catalog (INV-6)\n--catalog:\n%s\n%s:\n%s",
			viaFlag, catalogEnvVar, viaEnv)
	}
}

// TestS4_ModelConfigFolderRetired_StaticGuard is AC-3: the retired flag must be GONE, not
// merely unused. It fails if `model-config-folder` or `ModelConfigFolder` reappears
// anywhere in cmd/'s production sources — the grep the acceptance criterion names, made
// executable so a future PR cannot quietly reintroduce a second way to locate a model
// config. Test files are excluded (they may narrate the retirement in comments), as are
// cmd/*_test.go generally.
func TestS4_ModelConfigFolderRetired_StaticGuard(t *testing.T) {
	banned := []string{"model-config-folder", "ModelConfigFolder", "modelConfigFolder"}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read cmd/: %v", err)
	}
	scanned := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, readErr := os.ReadFile(name)
		if readErr != nil {
			t.Fatalf("read %s: %v", name, readErr)
		}
		scanned++
		for _, b := range banned {
			if strings.Contains(string(data), b) {
				t.Errorf("%s still mentions %q — the retired --model-config-folder input must be "+
					"gone (#1731 AC-3); --catalog / %s is the only way to locate a model config",
					name, b, catalogEnvVar)
			}
		}
	}
	if scanned == 0 {
		t.Fatal("non-vacuity: scanned no production sources in cmd/")
	}

	// Non-vacuity for the pattern itself: the replacement identifiers ARE present, so the
	// scan is looking at the file contents it thinks it is.
	root, err := os.ReadFile("root.go")
	if err != nil {
		t.Fatalf("read root.go: %v", err)
	}
	if !strings.Contains(string(root), "catalogPath") {
		t.Fatal("non-vacuity: cmd/root.go does not mention catalogPath, so the guard is not " +
			"scanning the flag-registration source")
	}
}

// TestS4_ModelConfigFolderFlag_NotRegisteredOnAnyCommand is the runtime half of AC-3: no
// command may still ACCEPT --model-config-folder, and --catalog must be registered on both
// `run` and `replay` (INV-13 — the two commands resolve a model config through the same
// shared helper, so a flag on one and not the other would break parity).
func TestS4_ModelConfigFolderFlag_NotRegisteredOnAnyCommand(t *testing.T) {
	var walk func(c *cobra.Command)
	seen := 0
	walk = func(c *cobra.Command) {
		seen++
		if f := c.Flags().Lookup("model-config-folder"); f != nil {
			t.Errorf("command %q still registers --model-config-folder; it is retired (#1731 AC-3)", c.Name())
		}
		for _, child := range c.Commands() {
			walk(child)
		}
	}
	walk(rootCmd)
	if seen < 2 {
		t.Fatalf("non-vacuity: walked %d commands, expected the root plus subcommands", seen)
	}

	// INV-13: both commands take --catalog, via the shared registerSimConfigFlags.
	for _, name := range []string{"run", "replay"} {
		c := &cobra.Command{}
		registerSimConfigFlags(c)
		f := c.Flags().Lookup("catalog")
		if f == nil {
			t.Fatalf("%s: --catalog must be registered (registerSimConfigFlags)", name)
		}
		// The default must stay the empty sentinel resolveCatalogRoot refuses, otherwise
		// the requirement could be satisfied by the default itself — reintroducing exactly
		// the working-directory default #1731 removed.
		if f.DefValue != "" {
			t.Errorf("%s: --catalog default must remain empty so an omitted catalog is refused, got %q",
				name, f.DefValue)
		}
	}
}

// TestS4_CatalogEnvVarReadOnlyInResolver pins WHERE the environment is consulted: exactly
// one production site reads BLIS_CATALOG (resolveCatalogRoot in hfconfig.go), mirroring the
// single HF_TOKEN read that was cmd/'s only os.Getenv before this change. A second reader
// elsewhere would be a second, unaudited way for the environment to steer a run.
func TestS4_CatalogEnvVarReadOnlyInResolver(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read cmd/: %v", err)
	}
	var readers []string
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, parseErr := parser.ParseFile(fset, name, nil, 0)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", name, parseErr)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			lit, ok := n.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return true
			}
			if lit.Value == `"`+catalogEnvVar+`"` {
				readers = append(readers, filepath.Base(name))
			}
			return true
		})
	}
	// hfconfig.go names it twice: the catalogEnvVar const, and nothing else.
	if len(readers) != 1 || readers[0] != "hfconfig.go" {
		t.Errorf("the %s literal must appear in exactly one production file (cmd/hfconfig.go, "+
			"as the catalogEnvVar const); found it in %v — every other site must use the const "+
			"and go through resolveCatalogRoot", catalogEnvVar, readers)
	}
}
