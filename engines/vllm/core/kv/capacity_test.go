// Original to blis-vllm.

package kv

import (
	"encoding/json"
	"os"
	"testing"
)

// capCase is one case of testdata/vllm_kv_capacity.json, printed by oracle/kvcapacity/extract.py:
// the layers, the engine settings and each worker's memory for KV, and vLLM's own answers.
type capCase struct {
	Name           string     `json:"name"`
	BlockSize      int        `json:"block_size"`
	NumKVHeads     int        `json:"num_kv_heads"`
	HeadSize       int        `json:"head_size"`
	DType          string     `json:"dtype"`
	SlidingWindow  int        `json:"sliding_window"`
	Layers         string     `json:"layers"`
	Stages         [][2]int   `json:"stages"`
	MaxModelLen    int        `json:"max_model_len"`
	MNBT           int        `json:"max_num_batched_tokens"`
	Batches        int        `json:"max_concurrent_batches"`
	Lookahead      int        `json:"num_prefill_lookahead_tokens"`
	AvailableBytes []int64    `json:"available_bytes"`
	VLLM           capAnswers `json:"vllm"`
}

type capAnswers struct {
	Groups []struct {
		Kind   string `json:"kind"`
		Layers int    `json:"layers"`
	} `json:"groups"`
	PaddingLayers  int     `json:"padding_layers"`
	NumBlocks      int64   `json:"num_blocks"`
	BlocksPerReq   int64   `json:"blocks_per_request"`
	Tokens         int64   `json:"kv_cache_size_tokens"`
	MaxConcurrency float64 `json:"max_concurrency"`
}

func loadCapacityOracle(t *testing.T) []capCase {
	t.Helper()
	raw, err := os.ReadFile("testdata/vllm_kv_capacity.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []capCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("no cases in the oracle file")
	}
	return f.Cases
}

func (c capCase) layers(t *testing.T) []Layer {
	t.Helper()
	elem := map[string]int{"fp8": 1, "bf16": 2}[c.DType]
	if elem == 0 {
		t.Fatalf("%s: dtype %q", c.Name, c.DType)
	}
	out := make([]Layer, len(c.Layers))
	for i, k := range c.Layers {
		l := Layer{Spec: LayerFull, KVHeads: c.NumKVHeads, HeadSize: c.HeadSize, ElemBytes: elem}
		if k == 'S' {
			l.Spec, l.Window = LayerSlidingWindow, c.SlidingWindow
		}
		out[i] = l
	}
	return out
}

func (c capCase) limits() RequestLimits {
	return RequestLimits{MaxModelLen: c.MaxModelLen, MaxInFlightTokens: c.Batches * c.MNBT,
		ExtraRetained: max(0, c.Lookahead-1)}
}

// poolBlocks is the pool every worker keeps: each worker's memory over its stage's bytes per
// block, the smallest of them.
func (c capCase) poolBlocks(t *testing.T, layers []Layer) int64 {
	t.Helper()
	pool := int64(-1)
	for s, span := range c.Stages {
		on := make([]bool, len(layers))
		for i := span[0]; i < span[1]; i++ {
			on[i] = true
		}
		per, err := StagePoolBytesPerBlock(layers, on, GroupOptions{}, c.BlockSize)
		if err != nil {
			t.Fatalf("%s stage %d: %v", c.Name, s, err)
		}
		if b := c.AvailableBytes[s] / per; pool < 0 || b < pool {
			pool = b
		}
	}
	return pool
}

// The KV pool of a hybrid sliding-window model, with and without a sliding-window drafter's
// layers after the target's and under pipeline parallelism, is the pool vLLM sizes and reports:
// for each case of vLLM's own get_kv_cache_configs and get_kv_cache_capacity
// (testdata/vllm_kv_capacity.json, oracle/kvcapacity/extract.py), the groups and their layers,
// the padding layers vLLM warns of, num_blocks, a max_model_len request's blocks, and the
// "GPU KV cache size" tokens and maximum concurrency it logs at start-up.
//
// The cases are Muse Spark Open's layout: 46 sliding-window and 16 full-attention layers group as
// three window groups and one full group of 16 (2 padding layers); with DSpark's five window layers
// they group as four window groups and one full group (13 padding layers, vLLM's "may waste at
// most 25.49%"). The start-up figure is not blocks x block_size: each request also holds a
// window's blocks in every window group, so the drafter's extra group lowers it.
func TestStartupCapacityIsVLLMs(t *testing.T) {
	for _, c := range loadCapacityOracle(t) {
		t.Run(c.Name, func(t *testing.T) {
			layers := c.layers(t)
			o := GroupOptions{}
			groups, members, err := groupLayers(layers, o.at(c.BlockSize))
			if err != nil {
				t.Fatal(err)
			}
			if len(groups) != len(c.VLLM.Groups) {
				t.Fatalf("%d groups, vLLM forms %d", len(groups), len(c.VLLM.Groups))
			}
			for i, g := range groups {
				kind := map[GroupKind]string{FullAttention: "full", SlidingWindow: "sliding_window"}[g.Kind]
				if want := c.VLLM.Groups[i]; kind != want.Kind || len(members[i]) != want.Layers {
					t.Errorf("group %d: %s of %d layers, vLLM's is %s of %d", i, kind, len(members[i]), want.Kind, want.Layers)
				}
			}
			pad, err := PaddingLayers(layers, o, c.BlockSize)
			if err != nil {
				t.Fatal(err)
			}
			if pad != c.VLLM.PaddingLayers {
				t.Errorf("%d padding layers, vLLM adds %d", pad, c.VLLM.PaddingLayers)
			}
			blocks := c.poolBlocks(t, layers)
			if blocks != c.VLLM.NumBlocks {
				t.Errorf("num_blocks %d, vLLM's %d", blocks, c.VLLM.NumBlocks)
			}
			got, err := StartupCapacity(layers, o, c.BlockSize, c.limits(), c.VLLM.NumBlocks)
			if err != nil {
				t.Fatal(err)
			}
			if got.RequestBlocks != c.VLLM.BlocksPerReq || got.Tokens != c.VLLM.Tokens || got.Concurrency != c.VLLM.MaxConcurrency {
				t.Errorf("%d blocks a request, %d tokens, %vx concurrency; vLLM: %d, %d, %vx",
					got.RequestBlocks, got.Tokens, got.Concurrency, c.VLLM.BlocksPerReq, c.VLLM.Tokens, c.VLLM.MaxConcurrency)
			}
		})
	}
}

// The KV manager built over a pool reports that pool's start-up figure as vLLM does: the same
// cases, through the manager the scheduler runs (its groups, max_model_len, max_in_flight_tokens
// and the windows' extra retained tokens).
func TestManagerStartupCapacityIsVLLMs(t *testing.T) {
	for _, c := range loadCapacityOracle(t) {
		t.Run(c.Name, func(t *testing.T) {
			groups, err := GroupLayers(c.layers(t), GroupOptions{BlockSize: c.BlockSize})
			if err != nil {
				t.Fatal(err)
			}
			m := NewManagerWith(int(c.VLLM.NumBlocks), c.BlockSize, c.MaxModelLen, true, 0,
				Options{Groups: groups, MaxInFlightTokens: c.Batches * c.MNBT, PrefillLookahead: c.Lookahead})
			got, err := m.StartupCapacity()
			if err != nil {
				t.Fatal(err)
			}
			if got.Tokens != c.VLLM.Tokens || got.Concurrency != c.VLLM.MaxConcurrency || got.NumBlocks != c.VLLM.NumBlocks {
				t.Errorf("%d tokens, %vx over %d blocks; vLLM: %d tokens, %vx over %d", got.Tokens, got.Concurrency,
					got.NumBlocks, c.VLLM.Tokens, c.VLLM.MaxConcurrency, c.VLLM.NumBlocks)
			}
		})
	}
}
