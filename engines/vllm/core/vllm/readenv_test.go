// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// The engine reads UCX_CUDA_IPC_ENABLE_MNNVL (the KV handoff path, compat/kernelmodel) and refuses
// every other environment variable it does not read, by name.
func TestTheEngineReadsTheMNNVLFlagAndRefusesOtherUnreadEnv(t *testing.T) {
	if err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"UCX_CUDA_IPC_ENABLE_MNNVL": "y"}}, false); err != nil {
		t.Errorf("UCX_CUDA_IPC_ENABLE_MNNVL is refused: %v", err)
	}
	err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"UCX_TLS": "cuda_copy,rc"}}, false)
	if err == nil || !strings.Contains(err.Error(), "UCX_TLS") {
		t.Errorf("a stated UCX_TLS must be refused by name, got %v", err)
	}
}
