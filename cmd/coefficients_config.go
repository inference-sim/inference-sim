package cmd

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Define struct for YAML
type Config struct {
	Models   []Model                  `yaml:"models"`
	Defaults map[string]DefaultConfig `yaml:"defaults"`
	Version  string                   `yaml:"version"`
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
}

func GetDefaultConfig(LLM string) (GPU string, TensorParallelism int, VLLMVersion string) {
	// Read YAML file
	data, err := os.ReadFile(coeffsFilePath)
	if err != nil {
		panic(err)
	}

	// Parse YAML
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		panic(err)
	}

	if _, modelExists := cfg.Defaults[LLM]; modelExists {
		return cfg.Defaults[LLM].GPU, cfg.Defaults[LLM].TensorParallelism, cfg.Defaults[LLM].VLLMVersion
	} else {
		return "", 0, ""
	}
}

func GetCoefficients(LLM string, tp int, GPU string, vllmVersion string, coeffsFilePath string) ([]float64, []float64, int64) {
	// Read YAML file
	data, err := os.ReadFile(coeffsFilePath)
	if err != nil {
		panic(err)
	}

	// Parse YAML
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		panic(err)
	}

	for _, model := range cfg.Models {
		if model.ID == LLM && model.TensorParallelism == tp && model.GPU == GPU && model.VLLMVersion == vllmVersion {
			return model.AlphaCoeffs, model.BetaCoeffs, model.TotalKVBlocks
		}
	}
	return nil, nil, 0
}
