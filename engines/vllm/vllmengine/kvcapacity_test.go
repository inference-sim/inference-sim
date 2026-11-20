// Original to blis-vllm.

package vllmengine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// startupFigure reads the engine's start-up KV figure, "GPU KV cache size: N tokens" and its
// "Maximum concurrency", from the method every vLLM engine core of this repository has
// (Engine.KVCacheCapacity and DPGroup.KVCacheCapacity). It is read by reflection so that a build
// without the method fails this test with a message, not at compilation.
func startupFigure(t *testing.T, e engine.Engine) (tokens int64, concurrency float64) {
	t.Helper()
	m := reflect.ValueOf(e).MethodByName("KVCacheCapacity")
	if !m.IsValid() {
		t.Fatalf("%T reports no KV cache size: it has no KVCacheCapacity method, so the pool's size is only blocks x block size", e)
	}
	c := m.Call(nil)[0]
	return c.FieldByName("Tokens").Int(), c.FieldByName("Concurrency").Float()
}

// poolCapacityOracle is vLLM v0.33.0's own get_kv_cache_capacity for deepseek-v4-pro's registered
// spec classes (oracle/lower/extract.py kv_pool_cases; core/lower/testdata/vllm_lower.json): the
// answer for a model length, max_in_flight_tokens, retained tokens and pool of blocks.
func poolCapacityOracle(t *testing.T, mml, inFlight, retained int, blocks int64) (tokens int64, concurrency float64) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "core", "lower", "testdata", "vllm_lower.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		KVPool []struct {
			Model    string `json:"model"`
			Capacity []struct {
				MaxModelLen   int     `json:"max_model_len"`
				InFlight      int     `json:"max_in_flight_tokens"`
				Retained      int     `json:"extra_retained_tokens"`
				NumBlocks     int64   `json:"num_blocks"`
				Tokens        int64   `json:"tokens"`
				MaxConcurrent float64 `json:"max_concurrency"`
			} `json:"capacity"`
		} `json:"kv_pool"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	for _, c := range f.KVPool {
		for _, a := range c.Capacity {
			if c.Model == "deepseek-v4-pro" && a.MaxModelLen == mml && a.InFlight == inFlight && a.Retained == retained && a.NumBlocks == blocks {
				return a.Tokens, a.MaxConcurrent
			}
		}
	}
	t.Fatalf("the oracle file has no deepseek-v4-pro capacity at max_model_len %d, in flight %d, retained %d, %d blocks", mml, inFlight, retained, blocks)
	return 0, 0
}

// A DeepSeek-V4-Pro engine reports the size vLLM's engine core logs at start-up. The pool is pinned to
// 17,475 blocks of 1,435,968 bytes (the bytes vLLM's own _get_kv_cache_bytes_per_block gives over
// the model's five KV cache groups, core/kv's TestPoolBytesPerBlockIsVLLMsAtFP8) with a stated
// kv_cache_memory_bytes, a synchronous batch queue of 512 tokens, and two model lengths. The
// expected figures are vLLM's own get_kv_cache_capacity over the same groups (the oracle), not
// numbers written here. Before this the engine's only size was its blocks x 256 tokens, 4.47
// million whatever the length: 6 to 23 times the logged figure on the dashboard's V4-Pro points
// (blis-vllm#129).
func TestDeepSeekV4ProEngineReportsVLLMsStartupKVFigure(t *testing.T) {
	Register()
	const blocks, perBlock = 17475, 1435968
	for _, mml := range []int{2304, 9472} {
		spec := d010With(t, "      gpu_memory_utilization: 0.92\n      kv_cache_memory_bytes: "+itoa(blocks*perBlock)+"\n",
			"max_model_len: 4096", "max_model_len: "+itoa(mml),
			"max_num_batched_tokens: 8192", "max_num_batched_tokens: 512",
			"max_num_seqs: 256", "max_num_seqs: 256\n      async_scheduling: false")
		e, err := engine.New(spec)
		if err != nil {
			t.Fatalf("max_model_len %d: %v", mml, err)
		}
		if got := e.Snapshot().KVBlocksTotal; got != blocks-1 {
			t.Fatalf("max_model_len %d: the pool holds %d usable blocks, want %d beside the null block", mml, got, blocks-1)
		}
		tokens, conc := startupFigure(t, e)
		wantTokens, wantConc := poolCapacityOracle(t, mml, 512, 0, blocks)
		if tokens != wantTokens || conc != wantConc {
			t.Errorf("max_model_len %d: the engine reports %d tokens at %.4fx, vLLM logs %d at %.4fx", mml, tokens, conc, wantTokens, wantConc)
		}
		if naive := (e.Snapshot().KVBlocksTotal + 1) * int64(e.Snapshot().KVBlockSize); naive < 4*tokens {
			t.Errorf("max_model_len %d: blocks x block size is %d against the startup figure %d: the two are not the same quantity here", mml, naive, tokens)
		}
	}
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}
