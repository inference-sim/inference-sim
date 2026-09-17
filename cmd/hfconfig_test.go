package cmd

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestResolveModelConfig_ExplicitOverrideTakesPrecedence(t *testing.T) {
	dir, err := resolveModelConfig("any-model", "/explicit/path", "defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dir != "/explicit/path" {
		t.Errorf("expected /explicit/path, got %s", dir)
	}
}

// TestResolveModelConfig_CatalogHit is the happy path: a catalogued model resolves to
// its catalog directory.
func TestResolveModelConfig_CatalogHit(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"num_hidden_layers": 32, "hidden_size": 4096}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Use a defaultsFile inside tmpDir so paths resolve relative to it
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	dir, err := resolveModelConfig("test-org/test-model", "", defaultsFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := filepath.Join(tmpDir, modelConfigsDir, "test-model")
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
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	dir, err := resolveModelConfig("test-org/uncatalogued-model", "", defaultsFile)
	if err == nil {
		t.Fatalf("expected an uncatalogued model to be refused, got dir=%q", dir)
	}

	wantPath := filepath.Join(tmpDir, modelConfigsDir, "uncatalogued-model", hfConfigFile)
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
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	before := listTree(t, tmpDir)

	if _, err := resolveModelConfig("test-org/uncatalogued-model", "", defaultsFile); err == nil {
		t.Fatal("expected refusal for an uncatalogued model")
	}

	after := listTree(t, tmpDir)
	if len(after) != len(before) {
		t.Errorf("resolution must not create anything under the catalog root; before=%v after=%v", before, after)
	}
	catalogDir := filepath.Join(tmpDir, modelConfigsDir)
	if _, statErr := os.Stat(catalogDir); statErr == nil {
		t.Errorf("resolution must not create the catalog directory %s", catalogDir)
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
			localDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
			if err := os.MkdirAll(localDir, 0o755); err != nil {
				t.Fatal(err)
			}
			entryPath := filepath.Join(localDir, hfConfigFile)
			if err := os.WriteFile(entryPath, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}

			defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
			if _, err := resolveModelConfig("test-org/test-model", "", defaultsFile); err == nil {
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
	localDir := filepath.Join(tmpDir, modelConfigsDir, "llama4-test")
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

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	dir, err := resolveModelConfig("test-org/llama4-test", "", defaultsFile)
	if err != nil {
		t.Fatalf("multimodal config should be recognized: %v", err)
	}
	expected := filepath.Join(tmpDir, modelConfigsDir, "llama4-test")
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

func TestBundledModelConfigDir(t *testing.T) {
	tests := []struct {
		model    string
		baseDir  string
		expected string
		wantErr  bool
	}{
		{"meta-llama/llama-3.1-8b-instruct", "", filepath.Join(modelConfigsDir, "llama-3.1-8b-instruct"), false},
		{"codellama/codellama-34b-instruct-hf", "", filepath.Join(modelConfigsDir, "codellama-34b-instruct-hf"), false},
		{"simple-model", "", filepath.Join(modelConfigsDir, "simple-model"), false},
		{"meta-llama/llama-3.1-8b-instruct", "/base", filepath.Join("/base", modelConfigsDir, "llama-3.1-8b-instruct"), false},
		{"evil/../../../etc/passwd", "", "", true},
		{"org/../../etc/shadow", "", "", true},
	}

	for _, tt := range tests {
		got, err := bundledModelConfigDir(tt.model, tt.baseDir)
		if tt.wantErr {
			if err == nil {
				t.Errorf("bundledModelConfigDir(%q, %q) expected error, got nil", tt.model, tt.baseDir)
			}
			continue
		}
		if err != nil {
			t.Errorf("bundledModelConfigDir(%q, %q) unexpected error: %v", tt.model, tt.baseDir, err)
			continue
		}
		if got != tt.expected {
			t.Errorf("bundledModelConfigDir(%q, %q) = %q, want %q", tt.model, tt.baseDir, got, tt.expected)
		}
	}
}

func TestGetHFRepo_ValidModel(t *testing.T) {
	// Create a minimal defaults.yaml with hf_repo
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  test-org/test-model:
    GPU: H100
    tensor_parallelism: 2
    hf_repo: TestOrg/Test-Model
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("test-org/test-model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "TestOrg/Test-Model" {
		t.Errorf("expected TestOrg/Test-Model, got %q", repo)
	}
}

func TestGetHFRepo_ModelWithoutHFRepo(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  test-org/test-model:
    GPU: H100
    tensor_parallelism: 2
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("test-org/test-model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" {
		t.Errorf("expected empty string for model without hf_repo, got %q", repo)
	}
}

func TestGetHFRepo_ModelNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	content := `defaults:
  other-model:
    GPU: H100
    tensor_parallelism: 2
workloads: {}
version: "0.0.1"
`
	if err := os.WriteFile(defaultsPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	repo, err := GetHFRepo("nonexistent/model", defaultsPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" {
		t.Errorf("expected empty string for nonexistent model, got %q", repo)
	}
}

func TestGetHFRepo_NonexistentFile(t *testing.T) {
	_, err := GetHFRepo("any-model", "/nonexistent/defaults.yaml")
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestGetHFRepo_MalformedYAML(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsPath := filepath.Join(tmpDir, "defaults.yaml")
	if err := os.WriteFile(defaultsPath, []byte(`{invalid yaml: [`), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := GetHFRepo("any-model", defaultsPath)
	if err == nil {
		t.Fatal("expected error for malformed YAML, got nil")
	}
}

// TestResolveModelConfig_PrecedenceInvariant verifies the documented resolution order,
// which after #1733 has exactly TWO steps and no fallback: explicit --model-config-folder
// > the catalog entry. Removing the catalog entry no longer opens a third path — it makes
// resolution fail (NS-6).
func TestResolveModelConfig_PrecedenceInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	// Set up both resolution sources
	explicitDir := filepath.Join(tmpDir, "explicit")
	if err := os.MkdirAll(explicitDir, 0o755); err != nil {
		t.Fatal(err)
	}

	localDir := filepath.Join(tmpDir, modelConfigsDir, "precedence-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"num_hidden_layers": 32, "hidden_size": 4096}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Precedence 1: Explicit override wins over the catalog
	dir, err := resolveModelConfig("test-org/precedence-model", explicitDir, defaultsFile)
	if err != nil {
		t.Fatalf("explicit override failed: %v", err)
	}
	if dir != explicitDir {
		t.Errorf("explicit override: expected %s, got %s", explicitDir, dir)
	}

	// Precedence 2: the catalog entry, when no explicit folder is given
	expectedLocal := filepath.Join(tmpDir, modelConfigsDir, "precedence-model")
	dir, err = resolveModelConfig("test-org/precedence-model", "", defaultsFile)
	if err != nil {
		t.Fatalf("catalog hit failed: %v", err)
	}
	if dir != expectedLocal {
		t.Errorf("catalog precedence: expected %s, got %s", expectedLocal, dir)
	}

	// There is no third step: with the entry removed, resolution is refused.
	if err := os.Remove(filepath.Join(localDir, hfConfigFile)); err != nil {
		t.Fatal(err)
	}
	if dir, err = resolveModelConfig("test-org/precedence-model", "", defaultsFile); err == nil {
		t.Errorf("expected refusal once the catalog entry is gone, got dir=%q", dir)
	}
}

// TestResolveModelConfig_CompletenessInvariant verifies the resolution chain's
// completeness law: resolveModelConfig never returns ("", nil). It must always
// return either a non-empty directory path or a non-nil error (R7: invariant test).
func TestResolveModelConfig_CompletenessInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	// Table of inputs covering edge cases
	tests := []struct {
		name           string
		model          string
		explicitFolder string
		defaultsFile   string
	}{
		{"empty model", "", "", defaultsFile},
		{"org/model no sources", "test-org/test-model", "", defaultsFile},
		{"simple model no sources", "simple-model", "", defaultsFile},
		{"explicit override", "any-model", "/explicit/path", defaultsFile},
		{"nonexistent defaults", "meta-llama/llama-3.1-8b", "", "/no/such/file.yaml"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir, err := resolveModelConfig(tt.model, tt.explicitFolder, tt.defaultsFile)
			// Completeness invariant: never ("", nil)
			if dir == "" && err == nil {
				t.Errorf("resolveModelConfig(%q, %q, %q) returned (\"\", nil) — "+
					"must return either a non-empty path or a non-nil error",
					tt.model, tt.explicitFolder, tt.defaultsFile)
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
