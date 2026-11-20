// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

// The engine reads UCX_CUDA_IPC_ENABLE_MNNVL (the KV handoff path) and
// VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS (the KV budget, core/kernelmodel) and refuses
// every other environment variable it does not read, by name.
func TestTheEngineReadsTheMNNVLFlagAndRefusesOtherUnreadEnv(t *testing.T) {
	if err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"UCX_CUDA_IPC_ENABLE_MNNVL": "y"}}); err != nil {
		t.Errorf("UCX_CUDA_IPC_ENABLE_MNNVL is refused: %v", err)
	}
	if err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"}}); err != nil {
		t.Errorf("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is refused although the KV budget reads it: %v", err)
	}
	err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"UCX_TLS": "cuda_copy,rc"}})
	if err == nil || !strings.Contains(err.Error(), "UCX_TLS") {
		t.Errorf("a stated UCX_TLS must be refused by name, got %v", err)
	}
}
