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

// vLLM's platform environment check (vllm/platforms/interface.py validate_environ, called from
// EngineArgs.create_engine_config) accepts VLLM_USE_V2_MODEL_RUNNER=1 with a warning that it is
// redundant, and raises for every other value because Model Runner V1 has been removed. The engine
// resolves 1 to the V2 runner and refuses every other value, as vLLM does.
func TestTheV2RunnerEnvKeyIsOneOrRefused(t *testing.T) {
	if _, err := build(t, "use_v2_model_runner: null\nenv: {VLLM_USE_V2_MODEL_RUNNER: \"1\"}\n")(); err != nil {
		t.Errorf("VLLM_USE_V2_MODEL_RUNNER=1 is refused: %v", err)
	}
	for _, v := range []string{"0", "true", "2", ""} {
		_, err := build(t, "use_v2_model_runner: null\nenv: {VLLM_USE_V2_MODEL_RUNNER: \""+v+"\"}\n")()
		if err == nil || !strings.Contains(err.Error(), "no longer supported") {
			t.Errorf("VLLM_USE_V2_MODEL_RUNNER=%q: err %v, want vLLM's refusal (no longer supported)", v, err)
		}
	}
}

// VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD is named by no file of the pinned vLLM release, so a stated
// value reaches no code (oracle/inertness/VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD.json): the engine accepts it
// and still refuses a neighbouring VLLM_ variable no proof covers.
func TestTheEngineAcceptsTheRoutedDownProjThresholdItProvesInert(t *testing.T) {
	if err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD": "0"}}); err != nil {
		t.Errorf("VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD is refused although fields.json proves it inert: %v", err)
	}
	err := refuseUnmodelled(engineconfig.Engine{Env: map[string]string{"VLLM_SOME_OTHER_THRESHOLD": "0"}})
	if err == nil || !strings.Contains(err.Error(), "VLLM_SOME_OTHER_THRESHOLD") {
		t.Errorf("an unproven VLLM_ variable must still be refused by name, got %v", err)
	}
}
