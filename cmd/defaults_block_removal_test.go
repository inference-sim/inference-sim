package cmd

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// #1768: the per-model `defaults:` block (GPU / tensor_parallelism / hf_repo) and the Go
// surface that declared it (Config.Defaults, the DefaultConfig struct, GetHFRepo) are gone.
//
// NS-6 (#1733) already made that policy unreachable — the deployment is a required operator
// input (--hardware/--tp) and the model config is read from the catalog (#1731) — so this is
// the retirement of inert policy, not a data migration. Nothing consumed it, so nothing is
// lost, and no number moves (INV-6).
//
// Three contracts live here:
//
//	BC-1 the trimmed defaults.yaml still parses under KnownFields(true), with every
//	     surviving section intact;
//	BC-2 a defaults file that still carries a `defaults:` block is REFUSED at load rather
//	     than silently ignored — strict parsing (R10) is what enforces the removal;
//	BC-3 the deleted surface cannot come back unnoticed.
//
// BC-4 (byte-identity) is TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline, which compares
// a run against a golden captured before #1733 and is unaffected by this change.

// ---------------------------------------------------------------------------
// BC-1: the trimmed file still parses, and nothing but the block was removed
// ---------------------------------------------------------------------------

// TestDefaultsBlockRemoved_BundledFileStillParses is BC-1. The load itself is the assertion
// that KnownFields(true) accepts the trimmed file (loadDefaultsConfig Fatalf's on a parse
// error, which would take the test binary down). The section checks are the non-vacuity half:
// they fail if the trim reached past the `defaults:` block into a section that is still read.
func TestDefaultsBlockRemoved_BundledFileStillParses(t *testing.T) {
	cfg := loadDefaultsConfig("../defaults.yaml")

	if cfg.Version == "" {
		t.Error("version must survive the defaults: trim")
	}
	// The `workloads:` block that used to be checked here is gone too (#1769) — the presets
	// live in the catalog now, and TestCatalogPresets_BundledCatalogMatchesRetiredDefaults
	// owns their values.
	if cfg.TrainedPhysicsDefaults == nil {
		t.Fatal("trained_physics_coefficients must survive the defaults: trim — it is the " +
			"default latency backend's coefficient source")
	}
	if len(cfg.TrainedPhysicsDefaults.AlphaCoeffs) == 0 || len(cfg.TrainedPhysicsDefaults.BetaCoeffs) == 0 {
		t.Errorf("trained_physics_coefficients parsed empty: alpha=%d beta=%d",
			len(cfg.TrainedPhysicsDefaults.AlphaCoeffs), len(cfg.TrainedPhysicsDefaults.BetaCoeffs))
	}
	if cfg.LoRADefaults == nil {
		t.Error("lora block must survive the defaults: trim")
	}
	// NOTE (#1770): kv_offload_devices: used to be asserted here as a section that must
	// survive #1768's trim. It has since been removed from defaults.yaml altogether — the
	// KV-offload storage-device table now lives in the catalog
	// (<catalog>/devices/storage.yaml, cmd/catalog_devices.go), which is the single source
	// of truth. Its "the bundled file still resolves" coverage moved to
	// TestKVOffloadDevices_CommittedCatalogTableParses, and the "a stale block is refused"
	// coverage to TestKVOffloadDevicesBlockRemoved_StaleBlockIsRefused.
}

// ---------------------------------------------------------------------------
// BC-2: a surviving `defaults:` block is refused, not ignored
// ---------------------------------------------------------------------------

// TestDefaultsBlockRemoved_StaleBlockIsRefused is BC-2, and it is the one observable behavior
// change in #1768: an operator's hand-maintained defaults.yaml that still carries a
// `defaults:` block no longer loads. That is the intended outcome — the alternative (keep the
// field declared so the block parses) is a config key with no consumer, which is precisely the
// silent-acceptance antipattern R10's strict parsing exists to prevent. The refusal must name
// the offending field so the fix is obvious.
//
// loadDefaultsConfig reports a parse error via logrus.Fatalf, so this runs in a subprocess.
func TestDefaultsBlockRemoved_StaleBlockIsRefused(t *testing.T) {
	if os.Getenv("BLIS_STALE_DEFAULTS_SUBPROCESS") == "1" {
		// The block is the ONLY difference from a file that loads (`version` is still
		// declared), so a failure here can only be the `defaults` key.
		content := `defaults:
  test-org/test-model:
    GPU: H100
    tensor_parallelism: 2
    hf_repo: TestOrg/Test-Model
version: "0.0.1"
`
		path := filepath.Join(t.TempDir(), "defaults.yaml")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			os.Exit(2)
		}
		loadDefaultsConfig(path) // must Fatalf before returning
		os.Exit(0)               // reached only if the stale block was accepted
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestDefaultsBlockRemoved_StaleBlockIsRefused", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_STALE_DEFAULTS_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()

	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("a defaults.yaml carrying a stale `defaults:` block must be refused at load, "+
			"not silently ignored; got err=%v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected logrus.Fatalf (exit 1), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	// The diagnostic must name the field, so the operator knows which block to delete.
	if !strings.Contains(string(out), "defaults") {
		t.Errorf("the refusal must name the offending `defaults` field; output:\n%s", out)
	}
}

// TestDefaultsBlockRemoved_SameFileWithoutBlockLoads is the negative control for BC-2: the
// fixture above differs from a loadable file ONLY in the `defaults:` block, so the rejection
// is attributable to that key and not to anything else about the fixture.
func TestDefaultsBlockRemoved_SameFileWithoutBlockLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "defaults.yaml")
	content := "version: \"0.0.1\"\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if cfg := loadDefaultsConfig(path); cfg.Version != "0.0.1" {
		t.Errorf("control fixture must load: got version %q, want \"0.0.1\"", cfg.Version)
	}
}

// ---------------------------------------------------------------------------
// BC-3: the deleted surface cannot come back unnoticed
// ---------------------------------------------------------------------------

// TestDefaultsBlockRemoved_StaticGuard is BC-3. It asserts the removal is structural rather
// than merely unused: no production source in cmd/ names the deleted symbols, and the bundled
// defaults.yaml declares no per-model deployment key. A behavioral test can only show that
// today's inputs do not read the policy; this shows there is nothing left to read.
func TestDefaultsBlockRemoved_StaticGuard(t *testing.T) {
	bannedIdents := map[string]string{
		"GetHFRepo":     "the caller-less hf_repo accessor was removed by #1768",
		"DefaultConfig": "the per-model deployment entry was removed by #1768",
	}
	if len(bannedIdents) == 0 {
		t.Fatal("non-vacuity: the guard has nothing to check")
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	scanned := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		scanned++
		fset := token.NewFileSet()
		parsed, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		ast.Inspect(parsed, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			if why, banned := bannedIdents[id.Name]; banned {
				t.Errorf("%s:%d: %s must not appear in cmd/'s production sources — %s",
					file, fset.Position(id.Pos()).Line, id.Name, why)
			}
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("non-vacuity: no production sources were scanned")
	}

	// The YAML half: no top-level `defaults:` key, and no per-model deployment key anywhere.
	data, err := os.ReadFile("../defaults.yaml")
	if err != nil {
		t.Fatalf("read defaults.yaml: %v", err)
	}
	// Match declarations only, so the explanatory comments in the file (which name the
	// retired keys on purpose) do not trip the guard.
	for _, banned := range []struct {
		pattern *regexp.Regexp
		what    string
	}{
		{regexp.MustCompile(`(?m)^defaults:`), "a top-level `defaults:` block"},
		{regexp.MustCompile(`(?m)^\s+GPU:`), "a per-model `GPU:` key"},
		{regexp.MustCompile(`(?m)^\s+tensor_parallelism:`), "a per-model `tensor_parallelism:` key"},
		{regexp.MustCompile(`(?m)^\s+hf_repo:`), "a per-model `hf_repo:` key"},
	} {
		if banned.pattern.Match(data) {
			t.Errorf("defaults.yaml must not declare %s: the deployment is a required "+
				"--hardware/--tp input (NS-6, #1733) and the model config comes from the "+
				"catalog (#1731), so such a key has no consumer (#1768)", banned.what)
		}
	}
}
