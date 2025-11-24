package cmd

import (
	"os"

	"gopkg.in/yaml.v3"
)

const TRAINED_COEFFICIENTS_FILEPATH string = "coefficients.yaml"

// Define struct for YAML
type Config struct {
	Models  []Model `yaml:"models"`
	Version string  `yaml:"version"`
}

type Model struct {
	GPU               string    `yaml:"GPU"`
	AlphaCoeffs       []float64 `yaml:"alpha_coeffs"`
	BetaCoeffs        []float64 `yaml:"beta_coeffs"`
	ID                string    `yaml:"id"`
	TensorParallelism int       `yaml:"tensor_parallelism"`
	VLLMVersion       string    `yaml:"vllm_version"`
}

func GetCoefficients(LLM string, tp int, GPU string, vllmVersion string) ([]float64, []float64) {
	// Read YAML file
	data, err := os.ReadFile(TRAINED_COEFFICIENTS_FILEPATH)
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
			return model.AlphaCoeffs, model.BetaCoeffs
		}
	}
	return nil, nil
}
