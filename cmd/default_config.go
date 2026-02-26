package cmd

import (
	"bytes"
	"os"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// Workload describes a preset workload configuration in defaults.yaml.
type Workload struct {
	PrefixTokens      int `yaml:"prefix_tokens"`
	PromptTokensMean  int `yaml:"prompt_tokens"`
	PromptTokensStdev int `yaml:"prompt_tokens_stdev"`
	PromptTokensMin   int `yaml:"prompt_tokens_min"`
	PromptTokensMax   int `yaml:"prompt_tokens_max"`
	OutputTokensMean  int `yaml:"output_tokens"`
	OutputTokensStdev int `yaml:"output_tokens_stdev"`
	OutputTokensMin   int `yaml:"output_tokens_min"`
	OutputTokensMax   int `yaml:"output_tokens_max"`
}

// Config represents the full defaults.yaml structure.
// All top-level sections must be listed to satisfy KnownFields(true) strict parsing (R10).
type Config struct {
	Models    []Model                  `yaml:"models"`
	Defaults  map[string]DefaultConfig `yaml:"defaults"`
	Version   string                   `yaml:"version"`
	Workloads map[string]Workload      `yaml:"workloads"`
}

// Define the inner structure for default config given model
type DefaultConfig struct {
	GPU               string `yaml:"GPU"`
	TensorParallelism int    `yaml:"tensor_parallelism"`
	VLLMVersion       string `yaml:"vllm_version"`
}

type Model struct {
	GPU               string    `yaml:"GPU"`
	AlphaCoeffs       []float64 `yaml:"alpha_coeffs"`
	BetaCoeffs        []float64 `yaml:"beta_coeffs"`
	ID                string    `yaml:"id"`
	TensorParallelism int       `yaml:"tensor_parallelism"`
	VLLMVersion       string    `yaml:"vllm_version"`
	TotalKVBlocks     int64     `yaml:"total_kv_blocks"`
	BestLoss          float64   `yaml:"best_loss"` // Calibration metric from coefficient fitting; not used at runtime
}

func GetDefaultSpecs(LLM string) (GPU string, TensorParallelism int, VLLMVersion string) {
	data, err := os.ReadFile(defaultsFilePath)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file: %v", err)
	}

	// Parse YAML with strict field checking (R10: typos must cause errors)
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
	}

	if _, modelExists := cfg.Defaults[LLM]; modelExists {
		return cfg.Defaults[LLM].GPU, cfg.Defaults[LLM].TensorParallelism, cfg.Defaults[LLM].VLLMVersion
	} else {
		return "", 0, ""
	}
}

// loadDefaultsConfig parses defaults.yaml into a Config struct.
// Uses strict field checking (R10).
func loadDefaultsConfig(path string) Config {
	data, err := os.ReadFile(path)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file: %v", err)
	}
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
	}
	return cfg
}

func GetCoefficients(LLM string, tp int, GPU string, vllmVersion string, defaultsFilePath string) ([]float64, []float64, int64) {
	data, err := os.ReadFile(defaultsFilePath)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file %s: %v", defaultsFilePath, err)
	}

	// Parse YAML with strict field checking (R10: typos must cause errors)
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
	}

	for _, model := range cfg.Models {
		if model.ID == LLM && model.TensorParallelism == tp && model.GPU == GPU && model.VLLMVersion == vllmVersion {
			return model.AlphaCoeffs, model.BetaCoeffs, model.TotalKVBlocks
		}
	}
	return nil, nil, 0
}
