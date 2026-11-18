// Original to blis-vllm.

package kernelmodel

import (
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

func kvMemoryWithEnv(t *testing.T, env map[string]string) (*Model, KVMemory, error) {
	t.Helper()
	in, err := latencykernel.OpenInputs("gpt-oss-120b-h200-fp4-vllm-tp4.yaml", repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	editPoolEngine(t, &in.Deployment.Pools[0], func(e *engineconfig.Engine) { e.Env = env })
	m, err := fromInputs(in)
	if err != nil {
		t.Fatal(err)
	}
	mem, err := m.KVMemory()
	return m, mem, err
}

// VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 stops the worker taking the graph-capture estimate out of
// the KV budget (vllm/v1/worker/gpu_worker.py:720-724 and :753-757@4ba787ec), so the pool grows by
// exactly the capture cost; 1 and an unstated variable are the same default.
func TestAStatedZeroGraphEstimateGivesTheKVPoolTheCaptureCost(t *testing.T) {
	m, base, err := kvMemoryWithEnv(t, nil)
	if err != nil {
		t.Fatal(err)
	}
	capture := m.k.FixedBytes().CUDAGraph
	if capture <= 0 {
		t.Fatalf("the fixture captures no graph (%d bytes); the case does not separate the two settings", capture)
	}
	_, one, err := kvMemoryWithEnv(t, map[string]string{envProfilerEstimatesGraphs: "1"})
	if err != nil {
		t.Fatal(err)
	}
	if one.AllocatableBytes != base.AllocatableBytes {
		t.Errorf("a stated 1 gives %d allocatable bytes, the default %d", one.AllocatableBytes, base.AllocatableBytes)
	}
	_, zero, err := kvMemoryWithEnv(t, map[string]string{envProfilerEstimatesGraphs: "0"})
	if err != nil {
		t.Fatal(err)
	}
	if got := zero.AllocatableBytes - base.AllocatableBytes; got != capture {
		t.Errorf("a stated 0 adds %d bytes to the KV pool, want the capture cost %d", got, capture)
	}
}

func TestAGraphEstimateThatIsNotAnIntegerIsRefusedAsVLLMDoes(t *testing.T) {
	_, _, err := kvMemoryWithEnv(t, map[string]string{envProfilerEstimatesGraphs: "false"})
	if err == nil || !strings.Contains(err.Error(), envProfilerEstimatesGraphs) {
		t.Fatalf("want a refusal naming the variable, got %v", err)
	}
}
