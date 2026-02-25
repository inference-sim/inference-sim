package cmd

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

func TestResolveModelConfig_CacheHit(t *testing.T) {
	// Create a temporary cache directory with valid JSON config.json
	tmpDir := t.TempDir()
	cacheModelID := "test-org-test-model"
	cacheDir := filepath.Join(tmpDir, blisCacheDir, modelConfigsDir, cacheModelID)
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, hfConfigFile), []byte(`{"valid": true}`), 0o600); err != nil {
		t.Fatal(err)
	}

	// Override home dir for the test
	t.Setenv("HOME", tmpDir)

	dir, err := resolveModelConfig("test-org/test-model", "", "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dir != cacheDir {
		t.Errorf("expected cache dir %s, got %s", cacheDir, dir)
	}
}

func TestResolveModelConfig_CacheCorrupted_FallsThrough(t *testing.T) {
	// Note: Cannot use t.Parallel() — mutates package-level fetchHFConfigFunc.
	// Create a cache directory with invalid JSON — should be removed and fall through
	tmpDir := t.TempDir()
	cacheModelID := "test-org-test-model"
	cacheDir := filepath.Join(tmpDir, blisCacheDir, modelConfigsDir, cacheModelID)
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, hfConfigFile), []byte(`<html>not json</html>`), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HOME", tmpDir)

	// Mock HF fetch to fail so we fall all the way through to error
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	_, err := resolveModelConfig("test-org/test-model", "", "nonexistent-defaults.yaml")
	if err == nil {
		t.Fatal("expected error when cache is corrupted and no fallbacks exist")
	}

	// Verify the corrupted cache file was removed
	cachePath := filepath.Join(cacheDir, hfConfigFile)
	if _, err := os.Stat(cachePath); err == nil {
		t.Error("corrupted cache file should have been removed")
	}
}

func TestResolveModelConfig_BundledFallback(t *testing.T) {
	// Create a temporary directory with bundled model_configs
	tmpDir := t.TempDir()
	bundledDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	if err := os.MkdirAll(bundledDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundledDir, hfConfigFile), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Use nonexistent HOME so cache misses
	t.Setenv("HOME", filepath.Join(tmpDir, "no-home"))

	// Mock HF fetch to fail so we reach bundled fallback without real HTTP
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	// Temporarily override bundledModelConfigDir's base to tmpDir
	// by constructing the expected path and verifying resolution succeeds.
	// Note: resolveModelConfig calls bundledModelConfigDir with empty baseDir,
	// which produces relative paths. We test this by changing to tmpDir.
	// Since bundledModelConfigDir now accepts baseDir, we can test it directly
	// in TestBundledModelConfigDir. Here we verify the full resolution chain.
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(origDir); err != nil {
			t.Logf("warning: could not restore working directory: %v", err)
		}
	})

	dir, err := resolveModelConfig("org/test-model", "", "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedRelative := filepath.Join(modelConfigsDir, "test-model")
	if dir != expectedRelative {
		t.Errorf("expected bundled dir %s, got %s", expectedRelative, dir)
	}
}

func TestResolveModelConfig_AllMiss_ReturnsError(t *testing.T) {
	// Set HOME to a temp dir with no cache
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	// Mock HF fetch to fail without real HTTP
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	_, err := resolveModelConfig("nonexistent/model", "", "nonexistent-defaults.yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestResolveModelConfig_AllMiss_IncludesDefaultsError(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	// Mock HF fetch to fail
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	_, err := resolveModelConfig("nonexistent/model", "", "nonexistent-defaults.yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Error should mention defaults.yaml read failure
	errStr := err.Error()
	if !strings.Contains(errStr, "defaults.yaml") {
		t.Errorf("expected error to mention defaults.yaml, got: %s", errStr)
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

func TestFetchHFConfig_Success(t *testing.T) {
	// Set up a test HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/test-org/test-model/resolve/main/config.json" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"num_hidden_layers": 32}`))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	dir, err := fetchHFConfigFromURL(server.URL+"/test-org/test-model/resolve/main/config.json", "test-org-test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify cached file exists
	cachedPath := filepath.Join(dir, hfConfigFile)
	data, err := os.ReadFile(cachedPath)
	if err != nil {
		t.Fatalf("cache file not found: %v", err)
	}
	if string(data) != `{"num_hidden_layers": 32}` {
		t.Errorf("unexpected cached content: %s", string(data))
	}
}

func TestFetchHFConfig_404(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	_, err := fetchHFConfigFromURL(server.URL+"/nonexistent/model/resolve/main/config.json", "nonexistent-model")
	if err == nil {
		t.Fatal("expected error for 404, got nil")
	}
}

func TestFetchHFConfig_401(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	_, err := fetchHFConfigFromURL(server.URL+"/gated/model/resolve/main/config.json", "gated-model")
	if err == nil {
		t.Fatal("expected error for 401, got nil")
	}
}

func TestFetchHFConfig_HFTokenHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"num_hidden_layers":32,"hidden_size":4096}`))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	t.Setenv("HF_TOKEN", "test-token-123")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", "test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer test-token-123" {
		t.Errorf("expected Bearer auth header, got %q", gotAuth)
	}
}

func TestFetchHFConfig_NoAuthHeaderWithoutToken(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"num_hidden_layers":32,"hidden_size":4096}`))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	t.Setenv("HF_TOKEN", "")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", "test-model-noauth")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "" {
		t.Errorf("expected no Authorization header when HF_TOKEN is empty, got %q", gotAuth)
	}
}

func TestFetchHFConfig_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`<html>Error page</html>`))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", "test-model")
	if err == nil {
		t.Fatal("expected error for invalid JSON response, got nil")
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

func TestHfCacheDir_NoHome(t *testing.T) {
	// When HOME is unset, hfCacheDir should either return an error or return
	// an absolute path (i.e., it must NOT fall back to a CWD-relative path).
	t.Setenv("HOME", "")
	t.Setenv("USERPROFILE", "")

	dir, err := hfCacheDir("test-model")
	if err != nil {
		// Error is the expected path when HOME is truly unresolvable — test passes
		return
	}
	// If it succeeded (some OS may resolve home via other means), verify the path
	// is absolute and not a CWD-relative fallback
	if !filepath.IsAbs(dir) {
		t.Errorf("hfCacheDir returned relative path %q when HOME is unset — must be absolute or error", dir)
	}
	if !strings.Contains(dir, "test-model") {
		t.Errorf("hfCacheDir result %q does not contain the model ID", dir)
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
    vllm_version: vllm/vllm-openai:v0.8.4
    hf_repo: TestOrg/Test-Model
workloads: {}
models: []
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
    vllm_version: vllm/vllm-openai:v0.8.4
workloads: {}
models: []
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
    vllm_version: vllm/vllm-openai:v0.8.4
workloads: {}
models: []
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

// TestResolveModelConfig_PrecedenceInvariant verifies the documented resolution
// order: explicit flag > cache > HF fetch > bundled fallback.
func TestResolveModelConfig_PrecedenceInvariant(t *testing.T) {
	// Note: Cannot use t.Parallel() — mutates package-level fetchHFConfigFunc.
	tmpDir := t.TempDir()

	// Set up all resolution sources
	explicitDir := filepath.Join(tmpDir, "explicit")
	if err := os.MkdirAll(explicitDir, 0o755); err != nil {
		t.Fatal(err)
	}

	cacheModelID := "test-org-precedence-model"
	cacheDir := filepath.Join(tmpDir, blisCacheDir, modelConfigsDir, cacheModelID)
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, hfConfigFile), []byte(`{"source":"cache"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HOME", tmpDir)

	// Mock HF fetch to succeed
	old := fetchHFConfigFunc
	hfDir := filepath.Join(tmpDir, "hf-fetched")
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		if err := os.MkdirAll(hfDir, 0o755); err != nil {
			return "", err
		}
		return hfDir, nil
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	// Precedence 1: Explicit override wins over everything
	dir, err := resolveModelConfig("test-org/precedence-model", explicitDir, "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("explicit override failed: %v", err)
	}
	if dir != explicitDir {
		t.Errorf("explicit override: expected %s, got %s", explicitDir, dir)
	}

	// Precedence 2: Cache wins over HF fetch and bundled
	dir, err = resolveModelConfig("test-org/precedence-model", "", "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("cache hit failed: %v", err)
	}
	if dir != cacheDir {
		t.Errorf("cache precedence: expected %s, got %s", cacheDir, dir)
	}

	// Precedence 3: HF fetch wins over bundled (remove cache first)
	if err := os.Remove(filepath.Join(cacheDir, hfConfigFile)); err != nil {
		t.Fatal(err)
	}
	dir, err = resolveModelConfig("test-org/precedence-model", "", "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("HF fetch failed: %v", err)
	}
	if dir != hfDir {
		t.Errorf("HF fetch precedence: expected %s, got %s", hfDir, dir)
	}
}

// TestResolveModelConfig_CompletenessInvariant verifies the resolution chain's
// completeness law: resolveModelConfig never returns ("", nil). It must always
// return either a non-empty directory path or a non-nil error (R7: invariant test).
func TestResolveModelConfig_CompletenessInvariant(t *testing.T) {
	// Note: Cannot use t.Parallel() — mutates package-level fetchHFConfigFunc.
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	// Table of inputs covering edge cases: empty model, org/model, simple name,
	// nonexistent defaults, explicit override.
	tests := []struct {
		name            string
		model           string
		explicitFolder  string
		defaultsFile    string
	}{
		{"empty model", "", "", "nonexistent.yaml"},
		{"org/model no sources", "test-org/test-model", "", "nonexistent.yaml"},
		{"simple model no sources", "simple-model", "", "nonexistent.yaml"},
		{"explicit override", "any-model", "/explicit/path", "nonexistent.yaml"},
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
