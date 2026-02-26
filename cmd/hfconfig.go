package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// validHFRepoPattern matches valid HuggingFace repo paths (e.g., "meta-llama/Llama-3.1-8B-Instruct").
// Rejects URL-special characters (?, #, @, spaces) that could alter URL semantics (I14).
var validHFRepoPattern = regexp.MustCompile(`^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$`)

const (
	hfBaseURL       = "https://huggingface.co"
	hfConfigFile    = "config.json"
	modelConfigsDir = "model_configs"
	httpTimeout     = 30 * time.Second
	// maxResponseBytes caps HF config.json reads to 10 MB — real config.json files
	// are typically <100 KB. This prevents unbounded memory allocation from
	// malformed or malicious responses.
	maxResponseBytes = 10 << 20 // 10 MB
)

// resolveModelConfig finds a HuggingFace config.json for the given model.
// Resolution order: explicit flag > model_configs/ > HF fetch (into model_configs/).
// Returns the path to a directory containing config.json.
// Paths are resolved relative to defaultsFile's directory (consistent with resolveHardwareConfig).
func resolveModelConfig(model, explicitFolder, defaultsFile string) (string, error) {
	// 1. Explicit override takes precedence
	if explicitFolder != "" {
		return explicitFolder, nil
	}

	// Derive the local model_configs/<short-name>/ path relative to defaults.yaml location
	// (consistent with resolveHardwareConfig using filepath.Dir(defaultsFile))
	baseDir := filepath.Dir(defaultsFile)
	localDir, err := bundledModelConfigDir(model, baseDir)
	if err != nil {
		return "", fmt.Errorf("--roofline: invalid model name %q: %w", model, err)
	}

	// 2. Check model_configs/ for an existing config.json (bundled or previously fetched)
	localPath := filepath.Join(localDir, hfConfigFile)
	if data, err := os.ReadFile(localPath); err == nil {
		if json.Valid(data) && isHFConfig(data) {
			logrus.Infof("--roofline: using config from %s", localDir)
			return localDir, nil
		}
		// Don't delete — the file may be a user-provided config with non-standard
		// field names. Fall through to HF fetch, which will overwrite if successful.
		logrus.Warnf("--roofline: config at %s exists but lacks expected HuggingFace fields (num_hidden_layers, hidden_size); trying HuggingFace fetch", localPath)
	}

	// 3. Fetch from HuggingFace and write into model_configs/<short-name>/
	var defaultsErr error
	hfRepo, err := GetHFRepo(model, defaultsFile)
	if err != nil {
		defaultsErr = err
		logrus.Warnf("--roofline: could not read hf_repo from defaults: %v (HuggingFace fetch may fail due to case-sensitivity)", err)
	}
	if hfRepo == "" {
		hfRepo = model
	}

	fetchedDir, err := fetchHFConfigFunc(hfRepo, localDir)
	if err == nil {
		logrus.Infof("--roofline: fetched config for %s into %s", model, fetchedDir)
		return fetchedDir, nil
	}
	logrus.Warnf("--roofline: HF fetch failed for %s: %v", model, err)

	errMsg := fmt.Sprintf(
		"--roofline: could not find config.json for model %q.\n"+
			"  Tried: %s, HuggingFace (%s/%s).\n"+
			"  Provide --model-config-folder explicitly",
		model, localDir, hfBaseURL, hfRepo,
	)
	if defaultsErr != nil {
		errMsg += fmt.Sprintf("\n  Note: defaults.yaml read failed: %v", defaultsErr)
	}
	return "", fmt.Errorf("%s", errMsg)
}

// resolveHardwareConfig finds the hardware config JSON file.
// Returns the explicit path if provided, or the bundled default.
func resolveHardwareConfig(explicitPath, defaultsFile string) (string, error) {
	if explicitPath != "" {
		return explicitPath, nil
	}

	// Derive bundled path from defaults.yaml location
	defaultsDir := filepath.Dir(defaultsFile)
	bundledPath := filepath.Join(defaultsDir, "hardware_config.json")
	if _, err := os.Stat(bundledPath); err == nil {
		logrus.Infof("--roofline: using bundled hardware config at %s", bundledPath)
		return bundledPath, nil
	}

	return "", fmt.Errorf(
		"--roofline: bundled hardware config not found at %q. Provide --hardware-config explicitly",
		bundledPath,
	)
}

// resolveBenchDataPath finds the benchmark data directory for MFU lookups.
// Returns the explicit path if provided, or the bundled default (bench_data/ next to defaults.yaml).
func resolveBenchDataPath(explicitPath, defaultsFile string) (string, error) {
	if explicitPath != "" {
		return explicitPath, nil
	}

	defaultsDir := filepath.Dir(defaultsFile)
	bundledPath := filepath.Join(defaultsDir, "bench_data")
	if info, err := os.Stat(bundledPath); err == nil && info.IsDir() {
		logrus.Infof("--roofline: using bundled bench data at %s", bundledPath)
		return bundledPath, nil
	}

	return "", fmt.Errorf(
		"--roofline: bundled bench data not found at %q. Provide --bench-data-path explicitly",
		bundledPath,
	)
}

// fetchHFConfigFunc is the function used to fetch HF configs. Package-level
// variable allows tests to inject a mock without hitting real HuggingFace.
// Second parameter is the target directory to write config.json into.
//
// WARNING: NOT safe for t.Parallel() — tests that swap this variable must
// run sequentially within the cmd package. See t.Cleanup() restore pattern.
var fetchHFConfigFunc = fetchHFConfig

// fetchHFConfig downloads config.json from HuggingFace and writes it to targetDir.
// Supports HF_TOKEN env var for gated models.
// Validates hfRepo format to prevent URL injection (I14).
func fetchHFConfig(hfRepo, targetDir string) (string, error) {
	if !validHFRepoPattern.MatchString(hfRepo) {
		return "", fmt.Errorf("invalid HuggingFace repo name %q: must match org/model pattern with alphanumeric, '.', '-', '_' characters", hfRepo)
	}
	fetchURL := fmt.Sprintf("%s/%s/resolve/main/%s", hfBaseURL, hfRepo, hfConfigFile)
	return fetchHFConfigFromURL(fetchURL, targetDir)
}

// fetchHFConfigFromURL fetches config.json from the given URL and writes it to targetDir.
// Extracted for testability (allows injecting test server URLs).
func fetchHFConfigFromURL(url, targetDir string) (string, error) {

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	// Support gated models via HF_TOKEN
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{
		Timeout: httpTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects (max 3)")
			}
			// I11: Validate redirect targets stay on HuggingFace domains.
			// HF uses CDN redirects (e.g., cdn-lfs.huggingface.co) which are legitimate.
			host := req.URL.Hostname()
			if host != "huggingface.co" && !strings.HasSuffix(host, ".huggingface.co") {
				return fmt.Errorf("redirect to non-HuggingFace host %q blocked", host)
			}
			// Strip Authorization header on subdomain redirects to avoid leaking
			// HF_TOKEN to CDN or other HuggingFace subdomains (defense-in-depth).
			if host != "huggingface.co" {
				req.Header.Del("Authorization")
			}
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusOK:
		// success, continue
	case http.StatusNotFound:
		return "", fmt.Errorf("not found on HuggingFace (HTTP 404). Check --model spelling. URL: %s", url)
	case http.StatusUnauthorized:
		return "", fmt.Errorf("authentication required (HTTP 401). Set HF_TOKEN env var. URL: %s", url)
	default:
		return "", fmt.Errorf("unexpected HTTP %d from HuggingFace for %s", resp.StatusCode, url)
	}

	// Limit response body to maxResponseBytes to prevent unbounded memory allocation
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}
	if int64(len(body)) > maxResponseBytes {
		return "", fmt.Errorf("response body exceeds %d bytes limit — likely not a config.json", maxResponseBytes)
	}

	// Validate that the response is valid JSON before writing — prevents writing
	// HTML error pages or other non-JSON responses
	if !json.Valid(body) {
		return "", fmt.Errorf("response from %s is not valid JSON", url)
	}

	// Semantic validation: verify the JSON contains at least one expected
	// HuggingFace config field. Catches empty objects {}, HF error responses
	// like {"error": "..."}, and non-config JSON that passes json.Valid.
	if !isHFConfig(body) {
		return "", fmt.Errorf("response from %s is valid JSON but does not contain expected "+
			"HuggingFace config fields (num_hidden_layers, hidden_size). "+
			"The model may not exist or the response is an error page", url)
	}

	// Write to target directory
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return "", fmt.Errorf("create directory %s: %w", targetDir, err)
	}

	targetPath := filepath.Join(targetDir, hfConfigFile)
	if err := os.WriteFile(targetPath, body, 0o644); err != nil {
		return "", fmt.Errorf("write config file %s: %w", targetPath, err)
	}

	return targetDir, nil
}

// isHFConfig checks whether JSON bytes contain at least one expected
// HuggingFace transformer config field. This prevents caching empty JSON {},
// error responses, or non-config JSON.
func isHFConfig(data []byte) bool {
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}
	// Check for fields present in every HuggingFace transformer config.json
	_, hasLayers := m["num_hidden_layers"]
	_, hasHidden := m["hidden_size"]
	return hasLayers || hasHidden
}

// bundledModelConfigDir returns the expected path for bundled model configs.
// Model names like "meta-llama/llama-3.1-8b-instruct" map to "<baseDir>/model_configs/llama-3.1-8b-instruct/".
// When baseDir is empty, returns a relative path (resolved relative to CWD).
// Returns an error if the model name contains path traversal sequences.
//
// Note (I12): The org prefix is stripped, so different orgs with identical model names
// (e.g., "org-a/llama" and "org-b/llama") would share the same directory. This matches
// the existing model_configs/ convention and is acceptable because HuggingFace model
// names are unique within orgs, and BLIS uses hf_repo for case-sensitive HF API calls.
func bundledModelConfigDir(model, baseDir string) (string, error) {
	// Use the part after the org prefix (after the /)
	parts := strings.SplitN(model, "/", 2)
	shortName := model
	if len(parts) == 2 {
		shortName = parts[1]
	}

	// Reject path traversal attempts (Clean first to normalize sequences like "a/./b")
	shortName = filepath.Clean(shortName)
	if strings.Contains(shortName, "..") || filepath.IsAbs(shortName) {
		return "", fmt.Errorf("model name %q contains invalid path components", model)
	}

	if baseDir != "" {
		return filepath.Join(baseDir, modelConfigsDir, shortName), nil
	}
	return filepath.Join(modelConfigsDir, shortName), nil
}
