package cluster

import "fmt"

// VLLMMoEConfig represents vLLM-specific MoE deployment settings
type VLLMMoEConfig struct {
	EnableAllToAll     bool   // All-to-all communication for expert routing
	UseFusedMoEKernel  bool   // Use optimized MoE kernels
	MoEKernelBackend   string // "triton", "cuda", or "auto"
}

// VLLMEngineConfig captures vLLM deployment parameters affecting simulation behavior
type VLLMEngineConfig struct {
	// Identity
	VLLMVersion string

	// Parallelism
	TensorParallelSize   int
	PipelineParallelSize int
	DataParallelSize     int

	// MoE (nil for dense models)
	MoEConfig *VLLMMoEConfig

	// Batch limits
	MaxNumSeqs           int
	MaxNumBatchedTokens  int

	// KV cache
	BlockSize              int
	GPUMemoryUtilization   float64
	TotalKVBlocks          int // Optional: if 0, compute from memory
	SwapSpace              int // CPU swap space in bytes

	// Scheduling
	EnableChunkedPrefill   bool
	MaxNumPartialPrefills  int
	ChunkSize              int

	// Caching
	EnablePrefixCaching    bool
}

// TotalGPUs returns total number of GPUs used by this deployment
func (v *VLLMEngineConfig) TotalGPUs() int {
	return v.TensorParallelSize * v.PipelineParallelSize * v.DataParallelSize
}

// EffectiveExpertParallelism returns the effective expert parallelism for MoE models
// In vLLM, expert parallelism (EP) is implicit as DP × TP
func (v *VLLMEngineConfig) EffectiveExpertParallelism() int {
	if !v.IsMoEDeployment() {
		return 0
	}
	return v.DataParallelSize * v.TensorParallelSize
}

// IsMoEDeployment returns true if this is a MoE deployment
func (v *VLLMEngineConfig) IsMoEDeployment() bool {
	return v.MoEConfig != nil
}

// Validate checks if the engine configuration is valid
func (v *VLLMEngineConfig) Validate() error {
	if v.TensorParallelSize < 1 {
		return fmt.Errorf("TensorParallelSize must be >= 1, got %d", v.TensorParallelSize)
	}
	if v.PipelineParallelSize < 1 {
		return fmt.Errorf("PipelineParallelSize must be >= 1, got %d", v.PipelineParallelSize)
	}
	if v.DataParallelSize < 1 {
		return fmt.Errorf("DataParallelSize must be >= 1, got %d", v.DataParallelSize)
	}
	if v.MaxNumSeqs <= 0 {
		return fmt.Errorf("MaxNumSeqs must be > 0, got %d", v.MaxNumSeqs)
	}
	if v.MaxNumBatchedTokens <= 0 {
		return fmt.Errorf("MaxNumBatchedTokens must be > 0, got %d", v.MaxNumBatchedTokens)
	}
	if v.BlockSize <= 0 {
		return fmt.Errorf("BlockSize must be > 0, got %d", v.BlockSize)
	}
	if v.GPUMemoryUtilization <= 0.0 || v.GPUMemoryUtilization > 1.0 {
		return fmt.Errorf("GPUMemoryUtilization must be in (0, 1], got %.2f", v.GPUMemoryUtilization)
	}
	if v.TotalKVBlocks < 0 {
		return fmt.Errorf("TotalKVBlocks must be >= 0, got %d", v.TotalKVBlocks)
	}

	return nil
}

// DefaultVLLMEngineConfig returns a default vLLM engine configuration
func DefaultVLLMEngineConfig() *VLLMEngineConfig {
	return &VLLMEngineConfig{
		VLLMVersion:            "0.4.2",
		TensorParallelSize:     1,
		PipelineParallelSize:   1,
		DataParallelSize:       1,
		MaxNumSeqs:             256,
		MaxNumBatchedTokens:    8192,
		BlockSize:              16,
		GPUMemoryUtilization:   0.90,
		TotalKVBlocks:          0, // Computed
		SwapSpace:              0,
		EnableChunkedPrefill:   true,
		MaxNumPartialPrefills:  1,
		ChunkSize:              8192,
		EnablePrefixCaching:    true,
	}
}
