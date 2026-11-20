// Original to blis-vllm.

package vllm

import "github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"

// MemoryUtilization is a release's default CacheConfig.gpu_memory_utilization and where it is
// written.
type MemoryUtilization struct {
	Value  float64
	Source string
}

// memoryUtilizationDefaults are vLLM's default gpu_memory_utilization (0.92, with bounds
// gt=0, le=1) in the release the engine models. The cost model applies no default (SUB-6), so
// the engine resolves it before the kernel is built (core/kernelmodel resolveEngine).
var memoryUtilizationDefaults = map[string]MemoryUtilization{
	upstream.Version: {0.92, "vLLM main head's DEFAULT_GPU_MEMORY_UTILIZATION when the extensible KV cache is off (vllm/config/cache.py:186,490-496@0fda86a3); the head default with it on, 1.0, is not modelled"},
}

// DefaultMemoryUtilization is the release's default gpu_memory_utilization, and false for a
// release the engine does not model.
func DefaultMemoryUtilization(release string) (MemoryUtilization, bool) {
	d, ok := memoryUtilizationDefaults[release]
	return d, ok
}
