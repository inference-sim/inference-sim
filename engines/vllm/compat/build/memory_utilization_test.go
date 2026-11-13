// Original to blis-vllm.

package build

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// A harness restates a measured point's gpu_memory_utilization on a scenario with ScenarioVariant.
// The stated value must reach the pool the engine sizes: vLLM takes ceil(total memory x util) as
// the memory it may use, less what profiling finds is not KV cache, and cuts the rest into blocks
// (request_memory, vllm/v1/worker/utils.py:716-726; Worker.determine_available_memory,
// vllm/v1/worker/gpu_worker.py:733-745; get_kv_cache_config_from_groups,
// vllm/v1/core/kv_cache_utils.py:1588, all at v0.32.0). So 0.95 must give more blocks than 0.9, by
// exactly the extra 5% of the device over the page, and the stated 0.95 and 0.9 must bracket the
// pool of an unstated value (vLLM's default, 0.92), which would be the case only if the stated
// value, not the default, sized it.
func TestAStatedMemoryUtilizationSizesTheKVPoolByVLLMsRule(t *testing.T) {
	src := filepath.Join(kernelmodel.DefaultScenarios(), "gpt-oss-120b-h200-fp4-vllm-tp4.yaml")
	base, err := kernelmodel.Open(filepath.Base(src), kernelmodel.DefaultRepos())
	if err != nil {
		t.Fatal(err)
	}
	rs, err := StatedRunSettings(base)
	if err != nil {
		t.Fatal(err)
	}

	type pool struct {
		blocks, page, fixed, device int64
		util                        float64
		source                      string
	}
	open := func(util float64) pool {
		t.Helper()
		rs := rs
		rs.GPUMemoryUtilization = util
		body, err := ScenarioVariant(src, rs, nil)
		if err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), body, 0o644); err != nil {
			t.Fatal(err)
		}
		r := kernelmodel.DefaultRepos()
		r.Scenarios = dir
		m, err := kernelmodel.Open("variant.yaml", r)
		if err != nil {
			t.Fatalf("utilization %v: %v", util, err)
		}
		b, err := m.KVBudget()
		if err != nil {
			t.Fatalf("utilization %v: %v", util, err)
		}
		mem, err := m.KVMemory()
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Engine()
		if err != nil {
			t.Fatal(err)
		}
		return pool{b.PerRankBlocks, b.PerBlockBytes, mem.FixedBytes, mem.DeviceBytes, e.GPUMemoryUtilization, m.EngineSources()["gpu_memory_utilization"]}
	}

	lo, hi, unstated := open(0.9), open(0.95), open(0)
	for _, p := range []pool{lo, hi} {
		if p.source != "stated" {
			t.Errorf("utilization %v is reported %q, want stated", p.util, p.source)
		}
		// vLLM's rule, written out: the requested memory less the same non-KV memory, in whole pages.
		want := (int64(math.Ceil(float64(p.device)*p.util)) - p.fixed) / p.page
		if p.blocks != want {
			t.Errorf("utilization %v: %d blocks, vLLM's rule gives %d", p.util, p.blocks, want)
		}
	}
	if unstated.util != 0.92 || unstated.source == "stated" {
		t.Fatalf("an unstated utilization resolved to %v (%q), want vLLM's default 0.92", unstated.util, unstated.source)
	}
	if lo.blocks >= unstated.blocks || unstated.blocks >= hi.blocks {
		t.Errorf("blocks at 0.9, default 0.92 and 0.95: %d, %d, %d; a stated value must size the pool, not the default",
			lo.blocks, unstated.blocks, hi.blocks)
	}
}
