// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
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

// The cache-location variables are proven inert (oracle/inertness/<name>.json): no file of vLLM's serving
// process uses them outside the model-file path, so an engine block that states one prices and schedules as
// one that does not, and the engine accepts it where it refuses every other variable it does not read.
func TestTheEngineAcceptsTheModelCacheLocationVariablesItProvesInert(t *testing.T) {
	for _, name := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"} {
		if err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{name: "/models/cache"}}); err != nil {
			t.Errorf("%s is refused although fields.json proves it inert: %v", name, err)
		}
	}
	err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"HF_HOME": "/models/cache"}})
	if err == nil || !strings.Contains(err.Error(), "HF_HOME") {
		t.Errorf("HF_HOME, which no proof covers, must still be refused by name, got %v", err)
	}
}
