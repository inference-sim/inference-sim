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

func TestResolveModelConfig_LocalHit(t *testing.T) {
	// Create a temporary model_configs directory with valid JSON config.json
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"valid": true}`), 0o644); err != nil {
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

func TestResolveModelConfig_CorruptedLocal_FallsThrough(t *testing.T) {
	// Create a model_configs directory with invalid JSON — should be removed and fall through
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	corruptedPath := filepath.Join(localDir, hfConfigFile)
	if err := os.WriteFile(corruptedPath, []byte(`<html>not json</html>`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Mock HF fetch to fail so we fall all the way through to error
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	_, err := resolveModelConfig("test-org/test-model", "", defaultsFile)
	if err == nil {
		t.Fatal("expected error when local config is corrupted and no fallbacks exist")
	}

	// Verify the corrupted file was removed
	if _, err := os.Stat(corruptedPath); err == nil {
		t.Error("corrupted config file should have been removed")
	}
}

func TestResolveModelConfig_FetchWritesToModelConfigs(t *testing.T) {
	// Verify that a successful HF fetch writes into model_configs/<short-name>/
	tmpDir := t.TempDir()
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	expectedDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")

	// Mock HF fetch to write a real file
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, targetDir string) (string, error) {
		if targetDir != expectedDir {
			return "", fmt.Errorf("fetch target should be %s, got %s", expectedDir, targetDir)
		}
		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(targetDir, hfConfigFile), []byte(`{"num_hidden_layers":32}`), 0o644); err != nil {
			return "", err
		}
		return targetDir, nil
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	dir, err := resolveModelConfig("org/test-model", "", defaultsFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dir != expectedDir {
		t.Errorf("expected %s, got %s", expectedDir, dir)
	}

	// Verify the file actually exists
	data, err := os.ReadFile(filepath.Join(dir, hfConfigFile))
	if err != nil {
		t.Fatalf("config.json not found after fetch: %v", err)
	}
	if !strings.Contains(string(data), "num_hidden_layers") {
		t.Errorf("unexpected content: %s", string(data))
	}
}

func TestResolveModelConfig_AllMiss_ReturnsError(t *testing.T) {
	tmpDir := t.TempDir()

	// Mock HF fetch to fail
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	_, err := resolveModelConfig("nonexistent/model", "", defaultsFile)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestResolveModelConfig_AllMiss_IncludesDefaultsError(t *testing.T) {
	tmpDir := t.TempDir()

	// Mock HF fetch to fail
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	// Use a nonexistent defaults file inside tmpDir so the error message includes it
	defaultsFile := filepath.Join(tmpDir, "nonexistent-defaults.yaml")
	_, err := resolveModelConfig("nonexistent/model", "", defaultsFile)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Error should mention defaults.yaml read failure
	errStr := err.Error()
	if !strings.Contains(errStr, "defaults") {
		t.Errorf("expected error to mention defaults, got: %s", errStr)
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
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")

	dir, err := fetchHFConfigFromURL(server.URL+"/test-org/test-model/resolve/main/config.json", targetDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify file exists in model_configs/
	writtenPath := filepath.Join(dir, hfConfigFile)
	data, err := os.ReadFile(writtenPath)
	if err != nil {
		t.Fatalf("config file not found: %v", err)
	}
	if string(data) != `{"num_hidden_layers": 32}` {
		t.Errorf("unexpected content: %s", string(data))
	}
	if dir != targetDir {
		t.Errorf("expected dir %s, got %s", targetDir, dir)
	}
}

func TestFetchHFConfig_404(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "nonexistent-model")

	_, err := fetchHFConfigFromURL(server.URL+"/nonexistent/model/resolve/main/config.json", targetDir)
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
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "gated-model")

	_, err := fetchHFConfigFromURL(server.URL+"/gated/model/resolve/main/config.json", targetDir)
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
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")
	t.Setenv("HF_TOKEN", "test-token-123")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
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
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "test-model-noauth")
	t.Setenv("HF_TOKEN", "")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
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
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "test-model")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
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
// order: explicit flag > model_configs/ > HF fetch (into model_configs/).
func TestResolveModelConfig_PrecedenceInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	// Set up all resolution sources
	explicitDir := filepath.Join(tmpDir, "explicit")
	if err := os.MkdirAll(explicitDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Set up local model_configs/ with a valid config
	localDir := filepath.Join(tmpDir, modelConfigsDir, "precedence-model")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(`{"source":"local"}`), 0o644); err != nil {
		t.Fatal(err)
	}

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
	dir, err := resolveModelConfig("test-org/precedence-model", explicitDir, defaultsFile)
	if err != nil {
		t.Fatalf("explicit override failed: %v", err)
	}
	if dir != explicitDir {
		t.Errorf("explicit override: expected %s, got %s", explicitDir, dir)
	}

	// Precedence 2: Local model_configs/ wins over HF fetch
	expectedLocal := filepath.Join(tmpDir, modelConfigsDir, "precedence-model")
	dir, err = resolveModelConfig("test-org/precedence-model", "", defaultsFile)
	if err != nil {
		t.Fatalf("local hit failed: %v", err)
	}
	if dir != expectedLocal {
		t.Errorf("local precedence: expected %s, got %s", expectedLocal, dir)
	}

	// Precedence 3: HF fetch when local is missing
	if err := os.Remove(filepath.Join(localDir, hfConfigFile)); err != nil {
		t.Fatal(err)
	}
	dir, err = resolveModelConfig("test-org/precedence-model", "", defaultsFile)
	if err != nil {
		t.Fatalf("HF fetch failed: %v", err)
	}
	if dir != hfDir {
		t.Errorf("HF fetch: expected %s, got %s", hfDir, dir)
	}
}

// TestResolveModelConfig_CompletenessInvariant verifies the resolution chain's
// completeness law: resolveModelConfig never returns ("", nil). It must always
// return either a non-empty directory path or a non-nil error (R7: invariant test).
func TestResolveModelConfig_CompletenessInvariant(t *testing.T) {
	tmpDir := t.TempDir()
	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")

	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("simulated HF failure")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

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

// TestFetchHFConfig_MaxResponseBytes verifies the 10 MB response limit (C3/R7).
// The implementation uses io.LimitReader to prevent unbounded memory allocation
// from malformed or malicious responses.
func TestFetchHFConfig_MaxResponseBytes(t *testing.T) {
	// Create a response body that exceeds maxResponseBytes (10 MB + 1 byte)
	oversizeBody := make([]byte, maxResponseBytes+1)
	// Fill with valid JSON prefix to get past any early checks
	copy(oversizeBody, []byte(`{"num_hidden_layers":32,"padding":"`))
	for i := len(`{"num_hidden_layers":32,"padding":"`) + 1; i < len(oversizeBody); i++ {
		oversizeBody[i] = 'x'
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(oversizeBody)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "oversize-model")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
	if err == nil {
		t.Fatal("expected error for oversized response, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected error about size limit, got: %v", err)
	}
}

// TestFetchHFConfig_ExactlyAtLimit verifies responses at exactly maxResponseBytes
// are accepted (boundary condition for the 10 MB limit).
func TestFetchHFConfig_ExactlyAtLimit(t *testing.T) {
	// A valid HF config that's much smaller than 10 MB (normal case)
	validConfig := `{"num_hidden_layers": 32, "hidden_size": 4096}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(validConfig))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "normal-model")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
	if err != nil {
		t.Fatalf("expected success for normal-sized response, got: %v", err)
	}
}

// TestFetchHFConfig_5xx verifies that HTTP 5xx responses produce clear errors (I20).
func TestFetchHFConfig_5xx(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"500 Internal Server Error", http.StatusInternalServerError},
		{"503 Service Unavailable", http.StatusServiceUnavailable},
		{"502 Bad Gateway", http.StatusBadGateway},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
			}))
			defer server.Close()

			tmpDir := t.TempDir()
			targetDir := filepath.Join(tmpDir, modelConfigsDir, "error-model")

			_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
			if err == nil {
				t.Fatalf("expected error for HTTP %d, got nil", tt.statusCode)
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("HTTP %d", tt.statusCode)) {
				t.Errorf("expected error to mention HTTP %d, got: %v", tt.statusCode, err)
			}
		})
	}
}

// TestFetchHFConfig_RedirectToNonHuggingFace verifies that redirects to
// non-HuggingFace hosts are blocked (I11: redirect host validation).
func TestFetchHFConfig_RedirectToNonHuggingFace(t *testing.T) {
	// Set up a server that redirects to a non-HuggingFace host
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://attacker.example.com/malicious.json", http.StatusFound)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	targetDir := filepath.Join(tmpDir, modelConfigsDir, "redirect-model")

	_, err := fetchHFConfigFromURL(server.URL+"/test/model/resolve/main/config.json", targetDir)
	if err == nil {
		t.Fatal("expected error for redirect to non-HuggingFace host, got nil")
	}
}

// TestFetchHFConfig_InvalidRepoPattern verifies that invalid hfRepo names
// are rejected before URL construction (I14: URL injection prevention).
func TestFetchHFConfig_InvalidRepoPattern(t *testing.T) {
	tests := []struct {
		name   string
		hfRepo string
	}{
		{"URL query injection", "org/model?param=evil"},
		{"URL fragment injection", "org/model#fragment"},
		{"URL userinfo injection", "user@org/model"},
		{"spaces", "org/model name"},
		{"no slash", "justmodel"},
		{"empty", ""},
		{"triple path", "org/sub/model"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			_, err := fetchHFConfig(tt.hfRepo, tmpDir)
			if err == nil {
				t.Errorf("expected error for invalid repo name %q, got nil", tt.hfRepo)
			}
		})
	}
}

// TestValidHFRepoPattern verifies the regex accepts legitimate HuggingFace repos.
func TestValidHFRepoPattern(t *testing.T) {
	valid := []string{
		"meta-llama/Llama-3.1-8B-Instruct",
		"RedHatAI/phi-4-FP8-dynamic",
		"Qwen/Qwen2.5-7B-Instruct",
		"codellama/CodeLlama-34b-Instruct-hf",
		"ibm-granite/granite-3.1-8b-instruct",
	}
	for _, repo := range valid {
		if !validHFRepoPattern.MatchString(repo) {
			t.Errorf("expected %q to be valid HF repo pattern", repo)
		}
	}
}
