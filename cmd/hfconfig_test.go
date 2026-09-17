package cmd

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestCatalogRootFrom_Precedence is AC-1 and AC-2 of #1731 (R1/S4) as a pure law:
// --catalog and BLIS_CATALOG both locate the catalog, --catalog wins when both are set,
// and NEITHER is refused naming both forms. catalogRootFrom takes both inputs as
// arguments, so the law is table-testable without touching globals or the environment.
func TestCatalogRootFrom_Precedence(t *testing.T) {
	tests := []struct {
		name    string
		flag    string
		env     string
		want    string
		wantErr bool
	}{
		{"flag only", "/from/flag", "", "/from/flag", false},
		{"env only", "", "/from/env", "/from/env", false},
		{"both set: flag wins", "/from/flag", "/from/env", "/from/flag", false},
		{"both set to the same path", "/same", "/same", "/same", false},
		{"neither set: refused", "", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := catalogRootFrom(tt.flag, tt.env)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected a refusal, got %q", got)
				}
				// AC-2: the refusal must name BOTH forms so the operator knows either works.
				if !strings.Contains(err.Error(), "--catalog") {
					t.Errorf("refusal must name --catalog, got: %v", err)
				}
				if !strings.Contains(err.Error(), catalogEnvVar) {
					t.Errorf("refusal must name %s, got: %v", catalogEnvVar, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("catalogRootFrom(%q, %q) = %q, want %q", tt.flag, tt.env, got, tt.want)
			}
		})
	}
}

// TestResolveCatalogRoot_ReadsFlagAndEnv is AC-1 through the production wrapper, which
// is the only place the --catalog package var and os.Getenv(BLIS_CATALOG) are read.
func TestResolveCatalogRoot_ReadsFlagAndEnv(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })

	flagDir := t.TempDir()
	envDir := t.TempDir()

	// Env only.
	catalogPath = ""
	t.Setenv(catalogEnvVar, envDir)
	got, err := resolveCatalogRoot()
	if err != nil {
		t.Fatalf("env-only: unexpected error: %v", err)
	}
	if got != envDir {
		t.Errorf("env-only: got %q, want %q", got, envDir)
	}

	// Flag wins over env.
	catalogPath = flagDir
	got, err = resolveCatalogRoot()
	if err != nil {
		t.Fatalf("flag+env: unexpected error: %v", err)
	}
	if got != flagDir {
		t.Errorf("flag+env: got %q, want %q (--catalog must win)", got, flagDir)
	}

	// Neither: refused. There is no working-directory default.
	catalogPath = ""
	t.Setenv(catalogEnvVar, "")
	if got, err = resolveCatalogRoot(); err == nil {
		t.Errorf("neither --catalog nor %s set must be refused, got %q", catalogEnvVar, got)
	}
}

// TestResolveCatalogRoot_RejectsUnusableRoot: a mistyped catalog path is refused as a
// CATALOG error naming both forms, not deferred into a per-model "not in the catalog"
// message that would blame the model (R1).
func TestResolveCatalogRoot_RejectsUnusableRoot(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	t.Setenv(catalogEnvVar, "")

	fileNotDir := filepath.Join(t.TempDir(), "catalog-is-a-file")
	if err := os.WriteFile(fileNotDir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct{ name, root string }{
		{"missing directory", filepath.Join(t.TempDir(), "no-such-catalog")},
		{"not a directory", fileNotDir},
	} {
		t.Run(tt.name, func(t *testing.T) {
			catalogPath = tt.root
			got, err := resolveCatalogRoot()
			if err == nil {
				t.Fatalf("expected refusal for %s, got %q", tt.name, got)
			}
			if !strings.Contains(err.Error(), tt.root) {
				t.Errorf("refusal must name the offending root %q, got: %v", tt.root, err)
			}
			if !strings.Contains(err.Error(), "--catalog") || !strings.Contains(err.Error(), catalogEnvVar) {
				t.Errorf("refusal must name both --catalog and %s, got: %v", catalogEnvVar, err)
			}
		})
	}
}

// TestResolveModelConfig_CatalogHit is the happy path: a catalogued model resolves to
// its catalog directory.
func TestResolveModelConfig_CatalogHit(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, "test-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"num_hidden_layers": 32, "hidden_size": 4096}`), 0o644); err != nil {
		t.Fatal(err)
	}

	dir, err := resolveModelConfigInCatalog("test-org/test-model", tmpDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := filepath.Join(tmpDir, "test-model")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

// TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath is BC-1 (NS-6, #1733): a
// model with no catalog entry is REFUSED, and the refusal names the path the entry
// belongs at so an operator can act on it. Before #1733 this case silently downloaded
// config.json from HuggingFace and wrote it into the catalog.
func TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath(t *testing.T) {
	tmpDir := t.TempDir()
	dir, err := resolveModelConfigInCatalog("test-org/uncatalogued-model", tmpDir)
	if err == nil {
		t.Fatalf("expected an uncatalogued model to be refused, got dir=%q", dir)
	}

	wantPath := filepath.Join(tmpDir, "uncatalogued-model", hfConfigFile)
	if !strings.Contains(err.Error(), wantPath) {
		t.Errorf("refusal must name the catalog path an entry belongs at (%s), got: %v", wantPath, err)
	}
	if !strings.Contains(err.Error(), "test-org/uncatalogued-model") {
		t.Errorf("refusal must name the model, got: %v", err)
	}
}

// TestResolveModelConfig_AbsentFromCatalog_CreatesNothing is the other half of BC-1 and
// the core of BC-3 at the resolution boundary: refusing an uncatalogued model must not
// create a directory or a file anywhere under the catalog root. This is the observable
// form of "no run adds a catalog entry as a side effect".
func TestResolveModelConfig_AbsentFromCatalog_CreatesNothing(t *testing.T) {
	tmpDir := t.TempDir()
	before := listTree(t, tmpDir)

	if _, err := resolveModelConfigInCatalog("test-org/uncatalogued-model", tmpDir); err == nil {
		t.Fatal("expected refusal for an uncatalogued model")
	}

	after := listTree(t, tmpDir)
	if len(after) != len(before) {
		t.Errorf("resolution must not create anything under the catalog root; before=%v after=%v", before, after)
	}
	entryDir := filepath.Join(tmpDir, "uncatalogued-model")
	if _, statErr := os.Stat(entryDir); statErr == nil {
		t.Errorf("resolution must not create the catalog entry directory %s", entryDir)
	}
}

// listTree returns every path under root, sorted — used to assert nothing was written.
func listTree(t *testing.T, root string) []string {
	t.Helper()
	var paths []string
	err := filepath.WalkDir(root, func(path string, _ fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	sort.Strings(paths)
	return paths
}

// TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved is BC-2: an existing
// catalog entry that is not a HuggingFace config.json is refused naming the file, and the
// file is left byte-for-byte unchanged (never overwritten by a fetch, never deleted — it
// may be an operator's hand-written entry with a fixable typo).
func TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{"not JSON at all", `<html>not json</html>`},
		{"valid JSON but not an HF config", `{"error": "not found"}`},
		{"empty JSON object", `{}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			localDir := filepath.Join(tmpDir, "test-model")
			if err := os.MkdirAll(localDir, 0o755); err != nil {
				t.Fatal(err)
			}
			entryPath := filepath.Join(localDir, hfConfigFile)
			if err := os.WriteFile(entryPath, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}

			if _, err := resolveModelConfigInCatalog("test-org/test-model", tmpDir); err == nil {
				t.Fatal("expected refusal for a malformed catalog entry")
			} else if !strings.Contains(err.Error(), entryPath) {
				t.Errorf("refusal must name the offending catalog file (%s), got: %v", entryPath, err)
			}

			// The entry must survive untouched.
			got, readErr := os.ReadFile(entryPath)
			if readErr != nil {
				t.Fatalf("catalog entry must be preserved, not deleted: %v", readErr)
			}
			if string(got) != tt.content {
				t.Errorf("catalog entry must be preserved byte-for-byte; got %q want %q", string(got), tt.content)
			}
		})
	}
}

func TestResolveModelConfig_MultimodalConfig(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, "llama4-test")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write a multimodal config (text_config structure)
	multimodalConfig := `{
		"architectures": ["Llama4ForConditionalGeneration"],
		"model_type": "llama4",
		"text_config": {
			"num_hidden_layers": 48,
			"hidden_size": 5120,
			"num_attention_heads": 40,
			"num_key_value_heads": 8
		},
		"vision_config": {
			"num_hidden_layers": 34,
			"hidden_size": 1408
		}
	}`
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(multimodalConfig), 0o644); err != nil {
		t.Fatal(err)
	}

	dir, err := resolveModelConfigInCatalog("test-org/llama4-test", tmpDir)
	if err != nil {
		t.Fatalf("multimodal config should be recognized: %v", err)
	}
	expected := filepath.Join(tmpDir, "llama4-test")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

func TestResolveHardwareConfig_ExplicitOverride(t *testing.T) {
	path, err := resolveHardwareConfig("/explicit/hw.json", "defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != "/explicit/hw.json" {
		t.Errorf("expected /explicit/hw.json, got %s", path)
	}
}

func TestResolveHardwareConfig_BundledDefault(t *testing.T) {
	tmpDir := t.TempDir()
	hwPath := filepath.Join(tmpDir, "hardware_config.json")
	if err := os.WriteFile(hwPath, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	path, err := resolveHardwareConfig("", defaultsFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != hwPath {
		t.Errorf("expected %s, got %s", hwPath, path)
	}
}

func TestResolveHardwareConfig_Missing_ReturnsError(t *testing.T) {
	_, err := resolveHardwareConfig("", "/nonexistent/dir/defaults.yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

// TestCatalogModelDir maps a model name onto its catalog entry directory. Since #1731
// the catalog ROOT is a required input (there is no working-directory default), so an
// empty root is an error rather than a relative path.
func TestCatalogModelDir(t *testing.T) {
	tests := []struct {
		model    string
		catalog  string
		expected string
		wantErr  bool
	}{
		{"meta-llama/llama-3.1-8b-instruct", "/base", filepath.Join("/base", "llama-3.1-8b-instruct"), false},
		{"codellama/codellama-34b-instruct-hf", "/base", filepath.Join("/base", "codellama-34b-instruct-hf"), false},
		{"simple-model", "/base", filepath.Join("/base", "simple-model"), false},
		{"meta-llama/llama-3.1-8b-instruct", "relative/catalog", filepath.Join("relative/catalog", "llama-3.1-8b-instruct"), false},
		// An empty root must NOT silently resolve to a working-directory-relative path:
		// that is the retired default #1731 removed.
		{"meta-llama/llama-3.1-8b-instruct", "", "", true},
		{"evil/../../../etc/passwd", "/base", "", true},
		{"org/../../etc/shadow", "/base", "", true},
	}

	for _, tt := range tests {
		got, err := catalogModelDir(tt.model, tt.catalog)
		if tt.wantErr {
			if err == nil {
				t.Errorf("catalogModelDir(%q, %q) expected error, got nil", tt.model, tt.catalog)
			}
			continue
		}
		if err != nil {
			t.Errorf("catalogModelDir(%q, %q) unexpected error: %v", tt.model, tt.catalog, err)
			continue
		}
		if got != tt.expected {
			t.Errorf("catalogModelDir(%q, %q) = %q, want %q", tt.model, tt.catalog, got, tt.expected)
		}
	}
}

// TestResolveModelConfig_ResolutionInvariant verifies the documented resolution order,
// which after #1733 (NS-6) and #1731 (S4) has exactly ONE step and no fallback: the
// entry <catalog>/<short-name>/config.json inside the catalog located by --catalog /
// BLIS_CATALOG. Removing the entry does not open a second path — it makes resolution
// fail. Two DIFFERENT catalogs holding the same model resolve independently, which is
// what makes "point --catalog at a scratch clone" the replacement for the retired
// --model-config-folder.
func TestResolveModelConfig_ResolutionInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	const cfg = `{"num_hidden_layers": 32, "hidden_size": 4096}`

	catalogA := filepath.Join(tmpDir, "catalog-a")
	catalogB := filepath.Join(tmpDir, "catalog-b")
	for _, root := range []string{catalogA, catalogB} {
		entry := filepath.Join(root, "precedence-model")
		if err := os.MkdirAll(entry, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(entry, hfConfigFile), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// The resolved entry follows the catalog root it was asked for — nothing else.
	for _, root := range []string{catalogA, catalogB} {
		dir, err := resolveModelConfigInCatalog("test-org/precedence-model", root)
		if err != nil {
			t.Fatalf("catalog %s: resolution failed: %v", root, err)
		}
		if want := filepath.Join(root, "precedence-model"); dir != want {
			t.Errorf("catalog %s: expected %s, got %s", root, want, dir)
		}
	}

	// There is no second step: with catalog A's entry removed, A is refused while B
	// still resolves (no cross-catalog or working-directory fallback).
	if err := os.Remove(filepath.Join(catalogA, "precedence-model", hfConfigFile)); err != nil {
		t.Fatal(err)
	}
	if dir, err := resolveModelConfigInCatalog("test-org/precedence-model", catalogA); err == nil {
		t.Errorf("expected refusal once catalog A's entry is gone, got dir=%q", dir)
	}
	if _, err := resolveModelConfigInCatalog("test-org/precedence-model", catalogB); err != nil {
		t.Errorf("catalog B must be unaffected by catalog A's missing entry: %v", err)
	}
}

// TestResolveModelConfig_CompletenessInvariant verifies the resolution chain's
// completeness law: resolution never returns ("", nil). It must always return either a
// non-empty directory path or a non-nil error (R7: invariant test).
func TestResolveModelConfig_CompletenessInvariant(t *testing.T) {
	emptyCatalog := t.TempDir()

	// Table of inputs covering edge cases
	tests := []struct {
		name    string
		model   string
		catalog string
	}{
		{"empty model", "", emptyCatalog},
		{"org/model, empty catalog", "test-org/test-model", emptyCatalog},
		{"simple model, empty catalog", "simple-model", emptyCatalog},
		{"nonexistent catalog root", "meta-llama/llama-3.1-8b", "/no/such/catalog"},
		{"empty catalog root", "meta-llama/llama-3.1-8b", ""},
		{"path traversal model name", "evil/../../../etc/passwd", emptyCatalog},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir, err := resolveModelConfigInCatalog(tt.model, tt.catalog)
			// Completeness invariant: never ("", nil)
			if dir == "" && err == nil {
				t.Errorf("resolveModelConfigInCatalog(%q, %q) returned (\"\", nil) — "+
					"must return either a non-empty path or a non-nil error",
					tt.model, tt.catalog)
			}
		})
	}
}

// TestIsHFConfig verifies semantic validation of HuggingFace config JSON.
func TestIsHFConfig(t *testing.T) {
	tests := []struct {
		name string
		json string
		want bool
	}{
		{"valid with num_hidden_layers", `{"num_hidden_layers": 32, "hidden_size": 4096}`, true},
		{"valid with hidden_size only", `{"hidden_size": 4096}`, true},
		{"valid with num_hidden_layers only", `{"num_hidden_layers": 32}`, true},
		{"empty object", `{}`, false},
		{"error response", `{"error": "not found"}`, false},
		{"array", `[1, 2, 3]`, false},
		{"string", `"hello"`, false},
		{"invalid JSON", `not json`, false},
		{"multimodal with text_config num_hidden_layers", `{"text_config": {"num_hidden_layers": 48}}`, true},
		{"multimodal with text_config hidden_size", `{"text_config": {"hidden_size": 5120}}`, true},
		{"multimodal with both text_config fields", `{"text_config": {"num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40}}`, true},
		{"multimodal without expected fields", `{"text_config": {"other_field": 123}, "vision_config": {"hidden_size": 1408}}`, false},
		{"vision_config only (no text_config)", `{"vision_config": {"num_hidden_layers": 34, "hidden_size": 1408}}`, false},
		{"text_config is not an object (string)", `{"text_config": "not_an_object"}`, false},
		{"text_config is not an object (null)", `{"text_config": null}`, false},
		{"deeply nested text_config", `{"text_config": {"text_config": {"num_hidden_layers": 48}}}`, false},
		{"zero-value num_hidden_layers at top level", `{"num_hidden_layers": 0}`, true},
		{"zero-value hidden_size at top level", `{"hidden_size": 0}`, true},
		{"zero-value num_hidden_layers in text_config", `{"text_config": {"num_hidden_layers": 0}}`, true},
		{"zero-value hidden_size in text_config", `{"text_config": {"hidden_size": 0}}`, true},
		{"mixed top-level and text_config fields", `{"num_hidden_layers": 48, "text_config": {"hidden_size": 5120}}`, true},
		// #1729 (NS-4): a config whose only layer-count evidence is the block-type array
		// is one the parser can now read, so the presence detector must accept it too —
		// otherwise the located config would be judged unusable and a fetch attempted.
		{"block-type array only at top level", `{"layers_block_type": ["attention", "mamba"]}`, true},
		{"block-type array only in text_config", `{"text_config": {"layers_block_type": ["attention", "mamba"]}}`, true},
		{"block-type array alongside other fields", `{"layers_block_type": ["attention"], "vocab_size": 32000}`, true},
		// Still rejected: the key names a value the parser cannot count.
		{"block-type array empty", `{"layers_block_type": []}`, false},
		{"block-type array wrong type", `{"layers_block_type": 52}`, false},
		{"vision_config block-type array only", `{"vision_config": {"layers_block_type": ["attention"]}}`, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isHFConfig([]byte(tt.json))
			if got != tt.want {
				t.Errorf("isHFConfig(%s) = %v, want %v", tt.json, got, tt.want)
			}
		})
	}
}
