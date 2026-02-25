package cmd

import (
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
	hfRepo := GetHFRepo(model, defaultsFile)
	if hfRepo == "" {
		hfRepo = model
	}

	// Sanitize model name for filesystem paths (replace / with -)
	cacheModelID := strings.ReplaceAll(model, "/", "-")

	// 2. Check local cache
	cacheDir := hfCacheDir(cacheModelID)
	cachePath := filepath.Join(cacheDir, hfConfigFile)
	if _, err := os.Stat(cachePath); err == nil {
		logrus.Infof("--roofline: using cached config from %s", cacheDir)
		return cacheDir, nil
	}

	// 3. Try HF fetch
	fetchedDir, err := fetchHFConfig(hfRepo, cacheModelID)
	if err == nil {
		logrus.Infof("--roofline: fetched and cached config for %s", model)
		return fetchedDir, nil
	}
	logrus.Warnf("--roofline: HF fetch failed for %s: %v", model, err)

	// 4. Bundled fallback - try model_configs/<short-name>/
	bundledDir := bundledModelConfigDir(model)
	bundledPath := filepath.Join(bundledDir, hfConfigFile)
	if _, err := os.Stat(bundledPath); err == nil {
		logrus.Infof("--roofline: using bundled config from %s", bundledDir)
		return bundledDir, nil
	}

	return "", fmt.Errorf(
		"--roofline: could not find config.json for model %q.\n"+
			"  Tried: cache (%s), HuggingFace (%s/%s), bundled (%s).\n"+
			"  Provide --model-config-folder explicitly",
		model, cachePath, hfBaseURL, hfRepo, bundledDir,
	)
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

	client := &http.Client{Timeout: httpTimeout}
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

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}

	// Write to cache
	cacheDir := hfCacheDir(cacheModelID)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", fmt.Errorf("create cache dir %s: %w", cacheDir, err)
	}

	cachePath := filepath.Join(cacheDir, hfConfigFile)
	if err := os.WriteFile(cachePath, body, 0o644); err != nil {
		return "", fmt.Errorf("write cache file %s: %w", cachePath, err)
	}

	return cacheDir, nil
}

// hfCacheDir returns the cache directory for a given model.
func hfCacheDir(cacheModelID string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, blisCacheDir, modelConfigsDir, cacheModelID)
}

// bundledModelConfigDir returns the expected path for bundled model configs.
// Model names like "meta-llama/llama-3.1-8b-instruct" map to "model_configs/llama-3.1-8b-instruct/".
func bundledModelConfigDir(model string) string {
	// Use the part after the org prefix (after the /)
	parts := strings.SplitN(model, "/", 2)
	shortName := model
	if len(parts) == 2 {
		shortName = parts[1]
	}
	return filepath.Join(modelConfigsDir, shortName)
}
