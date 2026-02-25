package cmd

import (
	"bytes"
	"os"

	sim "github.com/inference-sim/inference-sim/sim"
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
	HFRepo            string `yaml:"hf_repo,omitempty"`
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

func GetWorkloadConfig(workloadFilePath string, workloadType string, rate float64, numRequests int) *sim.GuideLLMConfig {
	data, err := os.ReadFile(workloadFilePath)
	if err != nil {
		logrus.Fatalf("Failed to read defaults file: %v", err)
	}

	// Parse into Config (not WorkloadConfig) because defaults.yaml has all top-level
	// sections (models, defaults, workloads, version). KnownFields(true) requires all
	// sections to be declared in the target struct (R10).
	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		logrus.Fatalf("Failed to parse defaults YAML: %v", err)
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

// GetHFRepo returns the HuggingFace repository path for the given model from defaults.yaml.
// Returns empty string if the model has no hf_repo mapping.
func GetHFRepo(modelName string, defaultsFile string) string {
	data, err := os.ReadFile(defaultsFile)
	if err != nil {
		return ""
	}

	var cfg Config
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return ""
	}

	if dc, ok := cfg.Defaults[modelName]; ok {
		return dc.HFRepo
	}
	return ""
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
