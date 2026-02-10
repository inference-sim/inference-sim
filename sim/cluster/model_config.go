package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// MoEConfig represents Mixture of Experts configuration
type MoEConfig struct {
	NumLocalExperts              int // Total experts per layer (e.g., 8 for Mixtral)
	NumExpertsPerTok             int // Active experts per token (e.g., 2)
	ExpertIntermediateSize       int // Expert MLP hidden size
	NumSharedExperts             int // Shared experts (DeepSeek-style, 0 if none)
	SharedExpertIntermediateSize int // Shared expert MLP size
}

// RopeScalingConfig represents RoPE scaling configuration
type RopeScalingConfig struct {
	Type   string  // e.g., "linear", "dynamic"
	Factor float64 // scaling factor
}

// HFModelConfig wraps HuggingFace config.json with computed properties for simulation
type HFModelConfig struct {
	// Identity
	ModelID       ModelID
	ModelType     string
	Architectures []string

	// Dimensions
	NumLayers              int
	HiddenSize             int
	IntermediateSize       int
	NumAttentionHeads      int
	NumKVHeads             int
	VocabSize              int
	MaxPositionEmbeddings  int

	// Precision
	TorchDtype    string
	BytesPerParam int

	// MoE (nil for dense models)
	MoE *MoEConfig

	// RoPE scaling (optional)
	RopeScaling *RopeScalingConfig

	// Raw HF config (preserves all fields)
	Raw *sim.HFConfig
}

// IsMoE returns true if this is a Mixture of Experts model
func (m *HFModelConfig) IsMoE() bool {
	return m.MoE != nil && m.MoE.NumLocalExperts > 1
}

// HeadDim returns the dimension of each attention head
func (m *HFModelConfig) HeadDim() int {
	if m.NumAttentionHeads == 0 {
		return 0
	}
	return m.HiddenSize / m.NumAttentionHeads
}

// KVCacheBytesPerToken calculates memory required per token for KV cache
// Formula: 2 × NumLayers × NumKVHeads × HeadDim × BytesPerParam
func (m *HFModelConfig) KVCacheBytesPerToken() int64 {
	return int64(2 * m.NumLayers * m.NumKVHeads * m.HeadDim() * m.BytesPerParam)
}

// TotalParameters calculates the total number of parameters in the model
func (m *HFModelConfig) TotalParameters() int64 {
	// Embedding layer
	embedding := int64(m.VocabSize * m.HiddenSize)

	// Attention layers
	qkvProjection := int64(m.NumLayers * m.HiddenSize * (m.NumAttentionHeads*m.HeadDim() + 2*m.NumKVHeads*m.HeadDim()))
	attnOutput := int64(m.NumLayers * m.NumAttentionHeads * m.HeadDim() * m.HiddenSize)
	attention := qkvProjection + attnOutput

	// MLP layers
	var mlp int64
	if m.IsMoE() {
		// MoE: gate + experts + shared experts
		gate := int64(m.NumLayers * m.HiddenSize * m.MoE.NumLocalExperts)
		experts := int64(m.NumLayers * m.MoE.NumLocalExperts * 3 * m.HiddenSize * m.MoE.ExpertIntermediateSize)
		shared := int64(0)
		if m.MoE.NumSharedExperts > 0 {
			shared = int64(m.NumLayers * m.MoE.NumSharedExperts * 3 * m.HiddenSize * m.MoE.SharedExpertIntermediateSize)
		}
		mlp = gate + experts + shared
	} else {
		// Dense: standard FFN
		mlp = int64(m.NumLayers * 3 * m.HiddenSize * m.IntermediateSize)
	}

	// Output layer
	output := int64(m.HiddenSize * m.VocabSize)

	// Layer norms (approximation: 2 per layer + final)
	layerNorms := int64((2*m.NumLayers + 1) * m.HiddenSize)

	return embedding + attention + mlp + output + layerNorms
}

// ActiveParametersPerToken calculates parameters activated per token
// For dense models, this equals TotalParameters
// For MoE models, only active experts contribute
func (m *HFModelConfig) ActiveParametersPerToken() int64 {
	if !m.IsMoE() {
		return m.TotalParameters()
	}

	// Embedding layer
	embedding := int64(m.VocabSize * m.HiddenSize)

	// Attention layers (always active)
	qkvProjection := int64(m.NumLayers * m.HiddenSize * (m.NumAttentionHeads*m.HeadDim() + 2*m.NumKVHeads*m.HeadDim()))
	attnOutput := int64(m.NumLayers * m.NumAttentionHeads * m.HeadDim() * m.HiddenSize)
	attention := qkvProjection + attnOutput

	// MoE: gate + active experts + shared experts
	gate := int64(m.NumLayers * m.HiddenSize * m.MoE.NumLocalExperts)
	activeExperts := int64(m.NumLayers * m.MoE.NumExpertsPerTok * 3 * m.HiddenSize * m.MoE.ExpertIntermediateSize)
	shared := int64(0)
	if m.MoE.NumSharedExperts > 0 {
		shared = int64(m.NumLayers * m.MoE.NumSharedExperts * 3 * m.HiddenSize * m.MoE.SharedExpertIntermediateSize)
	}
	mlp := gate + activeExperts + shared

	// Output layer
	output := int64(m.HiddenSize * m.VocabSize)

	// Layer norms
	layerNorms := int64((2*m.NumLayers + 1) * m.HiddenSize)

	return embedding + attention + mlp + output + layerNorms
}

// Validate checks if the model configuration is valid
func (m *HFModelConfig) Validate() error {
	if m.NumLayers <= 0 {
		return fmt.Errorf("NumLayers must be positive, got %d", m.NumLayers)
	}
	if m.HiddenSize <= 0 {
		return fmt.Errorf("HiddenSize must be positive, got %d", m.HiddenSize)
	}
	if m.NumAttentionHeads <= 0 {
		return fmt.Errorf("NumAttentionHeads must be positive, got %d", m.NumAttentionHeads)
	}
	if m.NumKVHeads <= 0 {
		return fmt.Errorf("NumKVHeads must be positive, got %d", m.NumKVHeads)
	}
	if m.NumKVHeads > m.NumAttentionHeads {
		return fmt.Errorf("NumKVHeads (%d) cannot exceed NumAttentionHeads (%d)", m.NumKVHeads, m.NumAttentionHeads)
	}
	if m.HiddenSize%m.NumAttentionHeads != 0 {
		return fmt.Errorf("HiddenSize (%d) must be divisible by NumAttentionHeads (%d)", m.HiddenSize, m.NumAttentionHeads)
	}
	if m.BytesPerParam != 1 && m.BytesPerParam != 2 && m.BytesPerParam != 4 {
		return fmt.Errorf("BytesPerParam must be 1, 2, or 4, got %d", m.BytesPerParam)
	}

	// MoE validation
	if m.IsMoE() {
		if m.MoE.NumLocalExperts <= 1 {
			return fmt.Errorf("MoE NumLocalExperts must be > 1, got %d", m.MoE.NumLocalExperts)
		}
		if m.MoE.NumExpertsPerTok < 1 {
			return fmt.Errorf("MoE NumExpertsPerTok must be >= 1, got %d", m.MoE.NumExpertsPerTok)
		}
		if m.MoE.NumExpertsPerTok > m.MoE.NumLocalExperts {
			return fmt.Errorf("MoE NumExpertsPerTok (%d) cannot exceed NumLocalExperts (%d)", m.MoE.NumExpertsPerTok, m.MoE.NumLocalExperts)
		}
	}

	return nil
}
