// Original to blis-vllm.

package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// d010 is deepseek-v4-pro on one 8-GPU H200 node, tensor parallel 8, with gpu_memory_utilization
// 0.92 on its engine block. The tests below replace that line.
const d010Utilization = "      gpu_memory_utilization: 0.92\n"

// d010With is the d010 scenario with its utilization line replaced by engineLines, and each
// further (old, new) pair of replacements applied to the file.
func d010With(t *testing.T, engineLines string, replace ...string) engine.Spec {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "d010-032.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), d010Utilization) {
		t.Fatal("the fixture states no gpu_memory_utilization: 0.92 to replace")
	}
	body := strings.Replace(string(raw), d010Utilization, engineLines, 1)
	for i := 0; i+1 < len(replace); i += 2 {
		if !strings.Contains(body, replace[i]) {
			t.Fatalf("the fixture has no %q to replace", replace[i])
		}
		body = strings.Replace(body, replace[i], replace[i+1], 1)
	}
	path := filepath.Join(t.TempDir(), "v.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: map[string]string{RootScenario: path}}
}

// A stated kv_cache_memory_bytes is the KV cache's bytes on each GPU, taken whole: the worker
// reserves exactly that many and skips memory profiling, so gpu_memory_utilization does not decide
// the pool (vllm/v1/worker/gpu_worker.py:656-681@4ba787ec; vllm/config/cache.py:374-382). The pool
// is then the stated bytes over the bytes one block costs, the same block a utilization-sized pool
// divides by. Before this the key was refused (blis-vllm#81).
func TestAStatedKVCacheMemoryBytesSizesThePoolWhateverTheUtilization(t *testing.T) {
	Register()
	const stated = 20 << 30
	var blocks [2]int64
	var bytesPerBlock int64
	for i, util := range []string{"0.92", "0.80"} {
		spec := d010With(t, "      gpu_memory_utilization: "+util+"\n      kv_cache_memory_bytes: 21474836480\n")
		p, err := open(spec, nil)
		if err != nil {
			t.Fatalf("utilization %s: %v", util, err)
		}
		blocks[i] = p.blocks
		if p.kvBytes != stated {
			t.Errorf("utilization %s: the pool was sized from %d bytes, want the stated %d", util, p.kvBytes, stated)
		}
		runs, err := p.model.KVLayers()
		if err != nil {
			t.Fatal(err)
		}
		sp, err := stagePoolsOf(p.model, runs)
		if err != nil {
			t.Fatal(err)
		}
		bytesPerBlock = sp.perBlock[0]
	}
	if blocks[0] != blocks[1] {
		t.Errorf("the pool is %d blocks at utilization 0.92 and %d at 0.80: a stated kv_cache_memory_bytes ignores it", blocks[0], blocks[1])
	}
	if want := int64(stated) / bytesPerBlock; blocks[0] != want {
		t.Errorf("the pool is %d blocks, want the stated %d bytes over %d bytes a block = %d", blocks[0], int64(stated), bytesPerBlock, want)
	}
	// Without the key the same deployment is sized from the utilization, and moves with it.
	var unstated [2]int64
	for i, util := range []string{"0.92", "0.80"} {
		p, err := open(d010With(t, "      gpu_memory_utilization: "+util+"\n"), nil)
		if err != nil {
			t.Fatal(err)
		}
		unstated[i] = p.blocks
	}
	if unstated[0] <= unstated[1] {
		t.Errorf("an unstated kv_cache_memory_bytes: %d blocks at utilization 0.92 and %d at 0.80, want more at the higher", unstated[0], unstated[1])
	}
	// The engine built from the block runs the pool it states.
	e, err := engine.New(d010With(t, "      gpu_memory_utilization: 0.92\n      kv_cache_memory_bytes: 21474836480\n"))
	if err != nil {
		t.Fatal(err)
	}
	if got := e.Snapshot().KVBlocksTotal; got != blocks[0]-1 {
		t.Errorf("the engine holds %d usable blocks, want the stated pool's %d beside the null block", got, blocks[0]-1)
	}
}

// A cache larger than the GPU cannot be reserved: with the weights and the rank's other fixed
// memory it exceeds the device, and vLLM's allocation fails. The engine refuses the deployment
// with that reason rather than size a pool no GPU holds.
func TestAStatedKVCacheMemoryBytesBeyondTheDeviceIsRefused(t *testing.T) {
	Register()
	_, err := engine.New(d010With(t, "      gpu_memory_utilization: 0.92\n      kv_cache_memory_bytes: 1099511627776\n"))
	if err == nil || !strings.Contains(err.Error(), "kv_cache_memory_bytes") || !strings.Contains(err.Error(), "exceed") {
		t.Fatalf("err %v, want a refusal that the stated cache and the fixed occupancy exceed the device", err)
	}
}
