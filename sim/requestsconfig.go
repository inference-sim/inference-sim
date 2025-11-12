package sim

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// RequestGenConfig enables the simulator to generate data with specified arrival rates
type RequestGenConfig struct {
	Format         string          `yaml:"format"` // "GuideLLM" is the only supported format at the moment
	Seed           int64           `yaml:"seed"`   // Random generation seed
	GuideLLMConfig *GuideLLMConfig `yaml:",inline"`
}

// GuideLLMConfig supports GuideLLM style request generation
type GuideLLMConfig struct {
	DataConfig DataConfig `yaml:"data"` // prompt, prefix and output token config
	RateConfig RateConfig `yaml:"rate"` // prompt arrival and count config
}

// DataConfig supports GuideLLM style data config for request generation
type DataConfig struct {
	PromptTokens       int `yaml:"prompt_tokens"`       // Average Prompt Token Count
	PromptTokensStdDev int `yaml:"prompt_tokens_stdev"` // Stddev Prompt Token Count
	PromptTokensMin    int `yaml:"prompt_tokens_min"`   // Min Prompt Token Count
	PromptTokensMax    int `yaml:"prompt_tokens_max"`   // Max Prompt Token Count
	OutputTokens       int `yaml:"output_tokens"`       // Average Output Token Count
	OutputTokensStdDev int `yaml:"output_tokens_stdev"` // Stddev Output Token Count
	OutputTokensMin    int `yaml:"output_tokens_min"`   // Min Output Token Count
	OutputTokensMax    int `yaml:"output_tokens_max"`   // Max Output Token Count
	// Each prompt has PrefixTokens shared prefix tokens, followed by random tokens
	PrefixTokens int `yaml:"prefix_tokens"` // Shared Prefix Token Count, additive to random tokens
}

type ArrivalType string

// RateConfig supports GuideLLM style rate config for request generation
type RateConfig struct {
	ArrivalType ArrivalType `yaml:"arrival_type"` // "Constant" is the only supported type at the moment
	Rate        float64     `yaml:"rate"`         // Requests per second
	MaxPrompts  int         `yaml:"max-requests"` // Max number of prompts to be generated
}

// ReadRequestGenConfig returns request gen config from YAML file
func ReadRequestGenConfig(fileName string) (*RequestGenConfig, error) {
	yamlFile, err := os.ReadFile(fileName)
	if err != nil {
		return nil, fmt.Errorf("failed to read request generator config file: %w", err)
	}

	var config RequestGenConfig
	config.GuideLLMConfig = &GuideLLMConfig{}

	err = yaml.Unmarshal(yamlFile, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal request generator config YAML: %w", err)
	}
	// convert RPS from requests per second to requests per microsec (or tick)
	config.GuideLLMConfig.RateConfig.Rate = config.GuideLLMConfig.RateConfig.Rate / 1e6

	return &config, nil
}
