package cmd

import (
	"os"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// Define struct for YAML
type WorkloadConfig struct {
	Workloads map[string]Workload `yaml:"workloads"`
}

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

func GetWorkloadConfig(workloadFilePath string, workloadType string, rate float64, numRequests int) *sim.GuideLLMConfig {
	// Read YAML file
	data, err := os.ReadFile(workloadFilePath)
	if err != nil {
		panic(err)
	}

	// Parse YAML
	var cfg WorkloadConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		panic(err)
	}

	if workload, workloadExists := cfg.Workloads[workloadType]; workloadExists {
		logrus.Infof("Using preset workload %v\n", workloadType)
		return sim.NewGuideLLMConfig(
			rate, numRequests,
			workload.PrefixTokens, workload.PromptTokensMean,
			workload.PromptTokensStdev, workload.PromptTokensMin, workload.PromptTokensMax,
			workload.OutputTokensMean, workload.OutputTokensStdev,
			workload.OutputTokensMin, workload.OutputTokensMax,
		)
	} else {
		return nil
	}
}

func GetDefaultSpecs(LLM string) (GPU string, TensorParallelism int, VLLMVersion string) {
	// Read YAML file
	data, err := os.ReadFile(defaultsFilePath)
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

func GetCoefficients(LLM string, tp int, GPU string, vllmVersion string, defaultsFilePath string) ([]float64, []float64, int64) {
	// Read YAML file
	data, err := os.ReadFile(defaultsFilePath)
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
