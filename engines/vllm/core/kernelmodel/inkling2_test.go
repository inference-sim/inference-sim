package kernelmodel

import (
	"strings"
	"testing"
)

// Inkling 2 registers two caches a layer, the conv state then the attention
// (vllm/models/inkling2/nvidia/model.py@0fda86a3). At TP4 each rank holds 2 of the 8 KV heads, so
// the conv state's head is the next power of two over 2 x 128 + 2 x (4608 / 2) = 4864, which is
// 8192; the 40 local layers' attention is a sliding window of 1024, the 8 full layers' is full.
// Its page, 4 tokens x 2 heads x 8192 x 2 B, is above every attention page at block 16, so the
// attention blocks are scaled to it (unify_kv_cache_spec_page_size).
func TestInkling2KVLayersAreTheEnginesConvStateAndAttention(t *testing.T) {
	m, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml", "")
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	conv, local, full, order := 0, 0, 0, 0
	for i, r := range runs {
		switch {
		case r.Spec == "sliding_window" && r.SlidingWindow == 4:
			conv += r.Count
			if r.BlockSize != 4 || r.NumKVHeads != 2 || r.HeadSize != 8192 || r.StateBytes != 8192*2 {
				t.Errorf("conv-state run %+v, want block 4, 2 heads of 8192 elements, 16384 B a state", r)
			}
			if i+1 >= len(runs) || runs[i+1].Spec == "sliding_window" && runs[i+1].SlidingWindow == 4 {
				t.Errorf("conv-state run %d is not followed by its layer's attention", i)
			}
			order++
		case r.Spec == "sliding_window" && r.SlidingWindow == 1024:
			local += r.Count
		case r.Spec == "full":
			full += r.Count
		default:
			t.Errorf("unexpected run %+v", r)
		}
		if r.Spec != "" && (r.NumKVHeads != 2 || r.ElemBytes != 2) {
			t.Errorf("run %+v: want 2 KV heads a rank and bf16 elements", r)
		}
	}
	if conv != 48 || local != 40 || full != 8 {
		t.Errorf("conv %d, local %d, full %d layers; want 48, 40, 8", conv, local, full)
	}
	// The conv state is the layer's own: the model prices it as no Mamba page.
	e, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	if e.BlockSize != 16 {
		t.Errorf("block size %d, want the stated 16", e.BlockSize)
	}
}

// A DSpark drafter's five attention layers register after the target's, as the target's own
// cache config (dspark.py:118-181): three windows of 1024 around one full layer.
func TestInkling2DSparkDraftLayersAreDraftAttention(t *testing.T) {
	m, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml",
		"speculative_config: {method: dspark, num_speculative_tokens: 7}")
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	var window, full int
	for _, r := range runs {
		if !r.Draft {
			continue
		}
		switch r.Spec {
		case "sliding_window":
			if r.SlidingWindow != 1024 {
				t.Errorf("draft window %d, want 1024", r.SlidingWindow)
			}
			window += r.Count
		case "full":
			full += r.Count
		default:
			t.Errorf("draft run %+v", r)
		}
	}
	if window != 4 || full != 1 {
		t.Errorf("draft layers: %d windowed, %d full; want 4 and 1", window, full)
	}
	// The attention layer asserts a cache dtype of auto or bfloat16.
	m2, err := openScenarioDir(t, "inkling-2-small", "inkling-2-small-gb200-tp4.yaml", "kv_cache_dtype: fp8")
	if err == nil {
		_, err = m2.KVLayers()
	}
	if err == nil || !strings.Contains(err.Error(), "attention.py:215") {
		t.Errorf("fp8 cache: got %v, want the attention layer's assertion named", err)
	}
}
