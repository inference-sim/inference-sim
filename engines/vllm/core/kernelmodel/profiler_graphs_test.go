// Original to blis-vllm.

package kernelmodel

import (
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

func kvMemoryWithEnv(t *testing.T, env map[string]string, edit ...func(*engineconfig.Engine)) (*Model, KVMemory, error) {
	t.Helper()
	in, err := latencykernel.OpenInputs("gpt-oss-120b-h200-fp4-vllm-tp4.yaml", repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	editPoolEngine(t, &in.Deployment.Pools[0], func(e *engineconfig.Engine) {
		e.Env = env
		for _, f := range edit {
			f(e)
		}
	})
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

// The variable is read where vLLM reads it, after the stated-kv_cache_memory_bytes return
// (gpu_worker.py:655-686@4ba787ec): a stated cache size skips profiling, so a value int() would reject is
// never looked at and the cache is the stated one.
func TestAStatedCacheSizeNeverReadsTheGraphEstimate(t *testing.T) {
	stated := int64(8) << 30
	_, mem, err := kvMemoryWithEnv(t, map[string]string{envProfilerEstimatesGraphs: "false"},
		func(e *engineconfig.Engine) { e.KVCacheMemoryBytes = &stated })
	if err != nil {
		t.Fatalf("a stated kv_cache_memory_bytes was refused for a variable vLLM never reads: %v", err)
	}
	if mem.AllocatableBytes != stated {
		t.Errorf("allocatable %d, want the stated %d", mem.AllocatableBytes, stated)
	}
}

// The forms Python's int() reads, as vLLM reads the variable; anything else is refused by name.
func TestTheGraphEstimateIsReadAsPythonsIntReadsIt(t *testing.T) {
	for _, c := range []struct {
		in      string
		nonzero bool
		ok      bool
	}{
		{"1", true, true}, {"0", false, true}, {" 0 ", false, true}, {"+1", true, true}, {"-0", false, true},
		{"1_0", true, true}, {"0_0", false, true}, {"00", false, true},
		{"", false, false}, {"1__0", false, false}, {"_1", false, false}, {"1_", false, false}, {"+-1", false, false},
		{"0x1", false, false}, {"1.0", false, false}, {"true", false, false}, {"\u0661", false, false},
	} {
		nonzero, ok := pythonIntNonzero(c.in)
		if nonzero != c.nonzero || ok != c.ok {
			t.Errorf("%q: got (%v, %v), want (%v, %v)", c.in, nonzero, ok, c.nonzero, c.ok)
		}
	}
}
