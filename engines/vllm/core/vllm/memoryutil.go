// Original to blis-vllm.

package vllm

// MemoryUtilization is a release's default CacheConfig.gpu_memory_utilization and where it is
// written.
type MemoryUtilization struct {
	Value  float64
	Source string
}

// memoryUtilizationDefaults are vLLM's default gpu_memory_utilization (0.92, with bounds
// gt=0, le=1) in each release the engine models. The cost model applies no default (SUB-6), so
// the engine resolves it before the kernel is built (compat/kernelmodel resolveEngine).
var memoryUtilizationDefaults = map[string]MemoryUtilization{
	"0.29.0": {0.92, "vLLM v0.29.0's default (vllm/config/cache.py:111@98dff2a81d)"},
	"0.31.0": {0.92, "vLLM v0.31.0's default (vllm/config/cache.py:103@db9527a468)"},
	"0.32.0": {0.92, "vLLM v0.32.0's default (vllm/config/cache.py:102@5df63a7203)"},
}

// DefaultMemoryUtilization is the release's default gpu_memory_utilization, and false for a
// release the engine does not model.
func DefaultMemoryUtilization(release string) (MemoryUtilization, bool) {
	d, ok := memoryUtilizationDefaults[release]
	return d, ok
}
