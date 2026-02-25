package cmd

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	// Create a temporary cache directory with config.json
	tmpDir := t.TempDir()
	cacheModelID := "test-org-test-model"
	cacheDir := filepath.Join(tmpDir, blisCacheDir, modelConfigsDir, cacheModelID)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, hfConfigFile), []byte(`{}`), 0o644); err != nil {
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

func TestResolveModelConfig_BundledFallback(t *testing.T) {
	// Create a temporary bundled model_configs directory
	tmpDir := t.TempDir()
	bundledDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	if err := os.MkdirAll(bundledDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundledDir, hfConfigFile), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Change to tmpDir so bundled path resolves correctly
	origDir, _ := os.Getwd()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(origDir); err != nil {
			t.Logf("warning: could not restore working directory: %v", err)
		}
	}()

	// Use a nonexistent HOME so cache misses, and no real HF fetch
	t.Setenv("HOME", filepath.Join(tmpDir, "no-home"))

	dir, err := resolveModelConfig("org/test-model", "", "nonexistent-defaults.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// bundledModelConfigDir returns a relative path (model_configs/test-model)
	expectedRelative := filepath.Join(modelConfigsDir, "test-model")
	if dir != expectedRelative {
		t.Errorf("expected bundled dir %s, got %s", expectedRelative, dir)
	}
}

func TestResolveModelConfig_AllMiss_ReturnsError(t *testing.T) {
	// Set HOME to a temp dir with no cache
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	_, err := resolveModelConfig("nonexistent/model", "", "nonexistent-defaults.yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
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

	// Temporarily override hfBaseURL by using the server URL directly
	// Since hfBaseURL is a const, we test via the full function indirectly
	// Instead, test with a mock server and call fetchHFConfigWithURL
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
		_, _ = w.Write([]byte(`{}`))
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

func TestBundledModelConfigDir(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		{"meta-llama/llama-3.1-8b-instruct", filepath.Join(modelConfigsDir, "llama-3.1-8b-instruct")},
		{"codellama/codellama-34b-instruct-hf", filepath.Join(modelConfigsDir, "codellama-34b-instruct-hf")},
		{"simple-model", filepath.Join(modelConfigsDir, "simple-model")},
	}

	for _, tt := range tests {
		got := bundledModelConfigDir(tt.model)
		if got != tt.expected {
			t.Errorf("bundledModelConfigDir(%q) = %q, want %q", tt.model, got, tt.expected)
		}
	}
}
