// Original to blis-vllm.

package kv

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// ocCapacity is one capacity answer of the oracle's kv_pool cases (oracle/lower/extract.py
// kv_pool_cases): vLLM's own get_kv_cache_capacity over the groups it forms for the case's layers.
type ocCapacity struct {
	MaxModelLen       int     `json:"max_model_len"`
	MaxInFlightTokens int     `json:"max_in_flight_tokens"`
	ExtraRetained     int     `json:"extra_retained_tokens"`
	NumBlocks         int64   `json:"num_blocks"`
	Tokens            int64   `json:"tokens"`
	MaxConcurrency    float64 `json:"max_concurrency"`
}

type ocCapacityCase struct {
	Model     string       `json:"model"`
	BlockSize int          `json:"block_size"`
	Runs      []ocLayer    `json:"runs"`
	Backend   string       `json:"backend"`
	Capacity  []ocCapacity `json:"capacity"`
}

func loadCapacityOracle(t *testing.T) []ocCapacityCase {
	t.Helper()
	raw, err := os.ReadFile("../lower/testdata/vllm_lower.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		KVPool []ocCapacityCase `json:"kv_pool"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	var out []ocCapacityCase
	for _, c := range f.KVPool {
		if len(c.Capacity) > 0 {
			out = append(out, c)
		}
	}
	return out
}

// The size an engine core logs at start-up, "GPU KV cache size: N tokens", is vLLM's
// get_kv_cache_capacity over the pool: the pool's blocks over the blocks one max_model_len request
// holds across the KV cache groups, times max_model_len. It is not the pool's blocks times the block
// size, which is 6 to 23 times larger for a DeepSeek-V4 model (blis-vllm#129): that is the figure
// the Result carried as the engine's KV capacity. The expected numbers are vLLM v0.33.0's own, from
// oracle/lower/extract.py over DeepSeek-V4-Pro's and V4.1-Flash's registered spec classes, for two
// model lengths, four in-flight budgets (a synchronous and an asynchronous batch queue at several
// token budgets), with and without a multi-token drafter's retained tokens, and two pool sizes.
func TestKVCapacityIsVLLMsStartupFigure(t *testing.T) {
	cases := loadCapacityOracle(t)
	models := map[string]int{}
	for _, c := range cases {
		models[c.Model]++
		layers := layersOfCase(t, ocCase{Runs: c.Runs})
		for _, want := range c.Capacity {
			got, err := KVCapacity(layers, GroupOptions{}, c.BlockSize, RequestLimits{
				MaxModelLen: want.MaxModelLen, MaxInFlightTokens: want.MaxInFlightTokens, ExtraRetained: want.ExtraRetained,
			}, want.NumBlocks)
			if err != nil {
				t.Fatalf("%s %+v: %v", c.Model, want, err)
			}
			if got.Tokens != want.Tokens || math.Abs(got.Concurrency-want.MaxConcurrency) > 1e-9 {
				t.Errorf("%s at max_model_len %d, in flight %d, retained %d, %d blocks: %d tokens at %.6fx, vLLM logs %d tokens at %.6fx",
					c.Model, want.MaxModelLen, want.MaxInFlightTokens, want.ExtraRetained, want.NumBlocks,
					got.Tokens, got.Concurrency, want.Tokens, want.MaxConcurrency)
			}
		}
	}
	for _, m := range []string{"deepseek-v4-pro", "deepseek-v4.1-flash"} {
		if models[m] == 0 {
			t.Errorf("the oracle file holds no capacity answers for %s", m)
		}
	}
}

// A pool of one full-attention group holds a request in cdiv(max_model_len, block_size) blocks, so
// its start-up figure is its block count over that, times max_model_len, which is a block count
// times the block size only when max_model_len is a whole number of blocks.
func TestKVCapacityOfOneFullAttentionGroup(t *testing.T) {
	got, err := KVCapacity(nil, GroupOptions{}, 16, RequestLimits{MaxModelLen: 100}, 700)
	if err != nil {
		t.Fatal(err)
	}
	// cdiv(100, 16) = 7 blocks a request, 700 blocks hold 100 of them: int(100 x 100) tokens.
	if got.BlocksPerRequest != 7 || got.Concurrency != 100 || got.Tokens != 10000 {
		t.Errorf("capacity %+v, want 7 blocks a request, 100x, 10000 tokens", got)
	}
}
