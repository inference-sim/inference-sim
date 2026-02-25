package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	hfBaseURL       = "https://huggingface.co"
	hfConfigFile    = "config.json"
	blisCacheDir    = ".blis"
	modelConfigsDir = "model_configs"
	httpTimeout     = 30 * time.Second
	// maxResponseBytes caps HF config.json reads to 10 MB — real config.json files
	// are typically <100 KB. This prevents unbounded memory allocation from
	// malformed or malicious responses.
	maxResponseBytes = 10 << 20 // 10 MB
)

// resolveModelConfig finds a HuggingFace config.json for the given model.
// Resolution order: explicit flag > cache > HF fetch > bundled fallback.
// Returns the path to a directory containing config.json.
func resolveModelConfig(model, explicitFolder, defaultsFile string) (string, error) {
	// 1. Explicit override takes precedence
	if explicitFolder != "" {
		return explicitFolder, nil
	}

	// Derive HF repo name from defaults.yaml mapping, fall back to model name
	var defaultsErr error
	hfRepo, err := GetHFRepo(model, defaultsFile)
	if err != nil {
		defaultsErr = err
		logrus.Warnf("--roofline: could not read hf_repo from defaults: %v (HuggingFace fetch may fail due to case-sensitivity)", err)
	}
	if hfRepo == "" {
		hfRepo = model
	}

	// Sanitize model name for filesystem paths (replace / with -)
	cacheModelID := strings.ReplaceAll(model, "/", "-")

	// 2. Check local cache (validate JSON integrity to detect corrupted cache)
	cacheDir, err := hfCacheDir(cacheModelID)
	if err != nil {
		logrus.Warnf("--roofline: cannot determine cache directory: %v", err)
	} else {
		cachePath := filepath.Join(cacheDir, hfConfigFile)
		if data, err := os.ReadFile(cachePath); err == nil {
			if json.Valid(data) {
				logrus.Infof("--roofline: using cached config from %s", cacheDir)
				return cacheDir, nil
			}
			logrus.Warnf("--roofline: cached config at %s is not valid JSON, removing", cachePath)
			if removeErr := os.Remove(cachePath); removeErr != nil {
				logrus.Warnf("--roofline: failed to remove corrupted cache file %s: %v", cachePath, removeErr)
			}
		}
	}

	// 3. Try HF fetch (uses fetchHFConfigFunc for testability)
	fetchedDir, err := fetchHFConfigFunc(hfRepo, cacheModelID)
	if err == nil {
		logrus.Infof("--roofline: fetched and cached config for %s", model)
		return fetchedDir, nil
	}
	logrus.Warnf("--roofline: HF fetch failed for %s: %v", model, err)

	// 4. Bundled fallback - try model_configs/<short-name>/
	bundledDir, err := bundledModelConfigDir(model, "")
	if err != nil {
		return "", fmt.Errorf("--roofline: invalid model name %q: %w", model, err)
	}
	bundledPath := filepath.Join(bundledDir, hfConfigFile)
	if _, err := os.Stat(bundledPath); err == nil {
		logrus.Infof("--roofline: using bundled config from %s", bundledDir)
		return bundledDir, nil
	}

	errMsg := fmt.Sprintf(
		"--roofline: could not find config.json for model %q.\n"+
			"  Tried: cache, HuggingFace (%s/%s), bundled (%s).\n"+
			"  Provide --model-config-folder explicitly",
		model, hfBaseURL, hfRepo, bundledDir,
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

// fetchHFConfigFunc is the function used to fetch HF configs. Package-level
// variable allows tests to inject a mock without hitting real HuggingFace.
var fetchHFConfigFunc = fetchHFConfig

// fetchHFConfig downloads config.json from HuggingFace and caches it locally.
// Supports HF_TOKEN env var for gated models.
func fetchHFConfig(hfRepo, cacheModelID string) (string, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", hfBaseURL, hfRepo, hfConfigFile)
	return fetchHFConfigFromURL(url, cacheModelID)
}

// fetchHFConfigFromURL fetches config.json from the given URL and caches it.
// Extracted for testability (allows injecting test server URLs).
func fetchHFConfigFromURL(url, cacheModelID string) (string, error) {

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

	// Validate that the response is valid JSON before caching — prevents caching
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

	// Write to cache — if cache write fails, fall back to a temp directory
	// so that successfully fetched data is never lost (I-4: decouple fetch/cache).
	cacheDir, err := hfCacheDir(cacheModelID)
	if err != nil {
		logrus.Warnf("--roofline: cannot determine cache directory: %v; using temp dir", err)
		return writeToTempDir(cacheModelID, body)
	}
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		logrus.Warnf("--roofline: cannot create cache dir %s: %v; using temp dir", cacheDir, err)
		return writeToTempDir(cacheModelID, body)
	}

	cachePath := filepath.Join(cacheDir, hfConfigFile)
	if err := os.WriteFile(cachePath, body, 0o600); err != nil {
		logrus.Warnf("--roofline: cannot write cache file %s: %v; using temp dir", cachePath, err)
		return writeToTempDir(cacheModelID, body)
	}

	return cacheDir, nil
}

// hfCacheDir returns the cache directory for a given model.
// Returns an error if the user's home directory cannot be determined.
func hfCacheDir(cacheModelID string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", err)
	}
	return filepath.Join(home, blisCacheDir, modelConfigsDir, cacheModelID), nil
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

// writeToTempDir writes config.json to a temporary directory as a fallback
// when the normal cache directory is unavailable.
func writeToTempDir(cacheModelID string, body []byte) (string, error) {
	tmpDir := filepath.Join(os.TempDir(), "blis-hfconfig-"+cacheModelID)
	if err := os.MkdirAll(tmpDir, 0o700); err != nil {
		return "", fmt.Errorf("create temp dir: %w", err)
	}
	tmpPath := filepath.Join(tmpDir, hfConfigFile)
	if err := os.WriteFile(tmpPath, body, 0o600); err != nil {
		return "", fmt.Errorf("write temp config file: %w", err)
	}
	return tmpDir, nil
}

// bundledModelConfigDir returns the expected path for bundled model configs.
// Model names like "meta-llama/llama-3.1-8b-instruct" map to "<baseDir>/model_configs/llama-3.1-8b-instruct/".
// When baseDir is empty, returns a relative path (resolved relative to the process's
// working directory — callers must ensure CWD is the repo root).
// Returns an error if the model name contains path traversal sequences.
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
