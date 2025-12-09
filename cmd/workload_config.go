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
	PromptTokensMean  int `yaml:"prompt_tokens"`
	PromptTokensStdev int `yaml:"prompt_tokens_stdev"`
	PromptTokensMin   int `yaml:"prompt_tokens_min"`
	PromptTokensMax   int `yaml:"prompt_tokens_max"`
	OutputTokensMean  int `yaml:"output_tokens"`
	OutputTokensStdev int `yaml:"output_tokens_stdev"`
	OutputTokensMin   int `yaml:"output_tokens_min"`
	OutputTokensMax   int `yaml:"output_tokens_max"`
}

func GetWorkloadConfig(workloadFilePath string, workloadType string, rate float64, maxPrompts int) *sim.GuideLLMConfig {
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
		return &sim.GuideLLMConfig{Rate: rate, MaxPrompts: maxPrompts,
			PromptTokens: workload.PromptTokensMean, PromptTokensStdDev: workload.PromptTokensStdev,
			PromptTokensMin: workload.PromptTokensMin, PromptTokensMax: workload.PromptTokensMax,
			OutputTokens: workload.OutputTokensMean, OutputTokensStdDev: workload.OutputTokensStdev,
			OutputTokensMin: workload.OutputTokensMin, OutputTokensMax: workload.OutputTokensMax}
	} else {
		return nil
	}
}
