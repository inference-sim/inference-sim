package sim

import "fmt"

// RequestGenConfig enables the simulator to generate prompt and output tokens
// It also includes rate specifications
type RequestGenConfig struct {
	Format         string // "GuideLLM" is the only supported format at the moment
	GuideLLMConfig *GuideLLMConfig
}

// GuideLLMConfig supports GuideLLM style request generation
type GuideLLMConfig struct {
	DataConfig DataConfig
	RateConfig RateConfig
}

// DataConfig supports GuideLLM style data config for request generation
type DataConfig struct {
	PromptTokens       int
	PromptTokensStdDev int
	PromptTokensMin    int
	PromptTokensMax    int
	OutputTokens       int
	OutputTokensStdDev int
	OutputTokensMin    int
	OutputTokensMax    int
	PrefixTokens       int
}

type ArrivalType string

// ConstantArrivalType is the only supported type at the moment
const ConstantArrivalType ArrivalType = "Constant"

// RateConfig supports GuideLLM style rate config for request generation
type RateConfig struct {
	ArrivalType ArrivalType
	Rate        float64
	MaxPrompts  int
}

// ReadRequestGenConfig return request gen config from file
func ReadRequestGenConfig(fileName string) (*RequestGenConfig, error) {
	return nil, fmt.Errorf("not implemented")
}
