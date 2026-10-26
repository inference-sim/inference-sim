// Original to blis-vllm.

package kv

import (
	"reflect"
	"testing"
)

// deepseekV4ProLayers are deepseek-v4-pro's KV layers as vLLM v0.32.0's modules register them
// (vllm/models/deepseek_v4/attention.py:233-420, compressor.py:139-193,
// vllm/v1/attention/backends/mla/sparse_swa.py:74-140@5df63a72), at fp8_ds_mla: per layer, a
// ratio-4 layer's indexer key cache and its compressor state, the sliding window, then a
// compressed layer's MLA cache and compressor state. mtp adds the MTP layer, which is pure
// sliding window (compress_ratio 1, attention.py:229-235).
func deepseekV4ProLayers(blockSize int, ratios []int, mtp bool) []Layer {
	swa := Layer{Spec: LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 64, StateBytes: 584, Alignment: 576}
	var out []Layer
	n := len(ratios)
	if mtp {
		n++
	}
	for i := 0; i < n; i++ {
		cr := 1
		if i < len(ratios) {
			cr = max(1, ratios[i])
		}
		if cr == 4 {
			out = append(out,
				Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 132, BlockSize: blockSize, TokensPerState: 4, Alignment: 576},
				Layer{Spec: LayerSlidingWindowMLA, Window: 8, KVHeads: 1, HeadSize: 512, BlockSize: 4, ElemBytes: 4, Alignment: 576})
		}
		out = append(out, swa)
		if cr > 1 {
			coff := 1
			if cr == 4 {
				coff = 2
			}
			state := Layer{Spec: LayerSlidingWindowMLA, Window: coff * cr, KVHeads: 1, HeadSize: 2 * coff * 512, BlockSize: 8, ElemBytes: 4, Alignment: 576}
			if cr == 4 {
				state.BlockSize = 4
			}
			out = append(out,
				Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: blockSize, TokensPerState: cr, StateBytes: 584, Alignment: 576},
				state)
		}
	}
	return out
}

// deepseekV4ProRatios is deepseek-v4-pro's compress_ratios (its config.json), 61 layers.
func deepseekV4ProRatios() []int {
	r := []int{128, 128}
	for i := 0; i < 29; i++ {
		r = append(r, 4, 128)
	}
	return append(r, 4)
}

// deepseekV41FlashLayers are deepseek-v4.1-flash's KV layers as vLLM v0.32.0's modules register
// them (vllm/models/deepseek_v41/attention.py:420-600, 1138-1240, compressor.py:136-167
// @5df63a72): per layer, a kv-source layer's indexer key cache (only kv sources own one; other
// index sources share it, :433-455), the sliding window (block 32), then a kv-source layer's MLA
// cache and, at ratio 2, its compressor ring of max(8, next power of two of K + 2) rows; then a
// DSpark drafter's three sliding-window layers.
func deepseekV41FlashLayers(blockSize, numSpec int, replay bool) []Layer {
	ratios := []int{0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	kvSource := map[int]bool{2: true, 8: true, 14: true, 20: true}
	indexSource := map[int]bool{2: true, 8: true, 14: true, 20: true, 24: true, 28: true, 32: true, 36: true}
	ring := 8
	for ring < numSpec+2 {
		ring *= 2
	}
	swa := Layer{Spec: LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 32, StateBytes: 584, Alignment: 576, BoundedReplay: replay}
	var out []Layer
	for i, cr := range ratios {
		if indexSource[i] && kvSource[i] {
			out = append(out, Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 132, BlockSize: blockSize, TokensPerState: max(cr, 1), Alignment: 576})
		}
		out = append(out, swa)
		if kvSource[i] {
			out = append(out, Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: blockSize, TokensPerState: max(cr, 1), StateBytes: 584, Alignment: 576})
			if cr > 1 {
				out = append(out, Layer{Spec: LayerCircularBuffer, KVHeads: 1, HeadSize: 1024, BlockSize: ring, ElemBytes: 4})
			}
		}
	}
	if numSpec > 0 {
		out = append(out, swa, swa, swa)
	}
	return out
}

// The groups vLLM v0.32.0's get_kv_cache_groups builds for DeepSeek V4's layers, with its own
// spec classes and the block-outermost layout the DeepSeek V4 indexer backend declares
// (oracle/run_scheduler.py's "deepseek" kv_layers, the dashboard-deepseek-* scenarios): each
// test case is vLLM's output for the layers above. Page sizes differ, so the layers pack into
// groups of several block sizes (_get_packed_kv_cache_groups, kv_cache_utils.py:1860-2014).
func TestPackedGroupsAreVLLMs(t *testing.T) {
	swa := func(bs int) Group { return Group{Kind: SlidingWindowMLA, Window: 128, BlockSize: bs} }
	repeat := func(n int, g Group) []Group {
		out := make([]Group, n)
		for i := range out {
			out[i] = g
		}
		return out
	}
	cat := func(gs ...[]Group) []Group {
		var out []Group
		for _, g := range gs {
			out = append(out, g...)
		}
		return out
	}
	for _, c := range []struct {
		name   string
		layers []Layer
		want   []Group
	}{
		// d000: 243 layers; two window groups at 64 (31 and 30 layers), the compressed MLA and
		// indexer caches at 256 (91), the ratio-128 compressor states at 8 and the ratio-4 ones at
		// 4 (window 8).
		{"deepseek-v4-pro", deepseekV4ProLayers(256, deepseekV4ProRatios(), false), cat(repeat(2, swa(64)),
			[]Group{{Kind: FullAttention, BlockSize: 256}, {Kind: SlidingWindowMLA, Window: 128, BlockSize: 8},
				{Kind: SlidingWindowMLA, Window: 8, BlockSize: 4}})},
		// d001: the MTP layer adds a window layer, the second window group's 31st.
		{"deepseek-v4-pro mtp", deepseekV4ProLayers(256, deepseekV4ProRatios(), true), cat(repeat(2, swa(64)),
			[]Group{{Kind: FullAttention, BlockSize: 256}, {Kind: SlidingWindowMLA, Window: 128, BlockSize: 8},
				{Kind: SlidingWindowMLA, Window: 8, BlockSize: 4}})},
		// d030: four window groups at 32, the 8 MLA caches at 128, one ring group of 3 rings.
		{"deepseek-v4.1-flash 128", deepseekV41FlashLayers(128, 0, false), cat(repeat(4, swa(32)),
			[]Group{{Kind: FullAttention, BlockSize: 128}, {Kind: CircularBuffer, BlockSize: 8}})},
		{"deepseek-v4.1-flash 128 dspark", deepseekV41FlashLayers(128, 2, false), cat(repeat(4, swa(32)),
			[]Group{{Kind: FullAttention, BlockSize: 128}, {Kind: CircularBuffer, BlockSize: 8}})},
		// d033-d035: block 64 halves the MLA group's page, so the 43 window layers split into
		// eight groups.
		{"deepseek-v4.1-flash 64 dspark", deepseekV41FlashLayers(64, 2, false), cat(repeat(8, swa(32)),
			[]Group{{Kind: FullAttention, BlockSize: 64}, {Kind: CircularBuffer, BlockSize: 8}})},
		// swa_bounded_replay keeps the window out of prefix caching; the grouping is the same.
		{"deepseek-v4.1-flash bounded replay", deepseekV41FlashLayers(128, 2, true), cat(
			repeat(4, Group{Kind: SlidingWindowMLA, Window: 128, BlockSize: 32, BoundedReplay: true}),
			[]Group{{Kind: FullAttention, BlockSize: 128}, {Kind: CircularBuffer, BlockSize: 8}})},
		// A ring bucket wider than the block the others need is split until a group fits it:
		// 9 rings of 512 bytes beside a 3072-byte block are 3 groups (kv_cache_utils.py:1960-1968).
		{"rings split", append(repeatLayers(2, Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 64, BlockSize: 16},
			Layer{Spec: LayerSlidingWindowMLA, Window: 8, KVHeads: 1, HeadSize: 32, BlockSize: 16}),
			repeatLayers(9, Layer{Spec: LayerCircularBuffer, KVHeads: 1, HeadSize: 64, BlockSize: 8})...),
			cat([]Group{{Kind: FullAttention, BlockSize: 16}, {Kind: SlidingWindowMLA, Window: 8, BlockSize: 16}},
				repeat(3, Group{Kind: CircularBuffer, BlockSize: 8}))},
	} {
		got, err := GroupsOfLayers(c.layers, false)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: groups\n%v\nwant\n%v", c.name, got, c.want)
		}
	}
}

func repeatLayers(n int, ls ...Layer) []Layer {
	var out []Layer
	for range n {
		out = append(out, ls...)
	}
	return out
}

// Page bytes are vLLM's page_size_bytes for each DeepSeek V4 spec, as the oracle's grouping
// reports them (a group's page is the sum of its layers'): a window of 64 tokens at 584 bytes
// is 37376 padded to 576, 37440; the ratio-128 MLA cache at block 256 holds 2 latents, 1168
// padded to 1728; a ratio-4 indexer page at 256 is 64 x 132 = 8448, padded to 8640.
func TestPageBytesAreVLLMs(t *testing.T) {
	for _, c := range []struct {
		name  string
		layer Layer
		want  int
	}{
		{"window", Layer{Spec: LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 64, StateBytes: 584, Alignment: 576}, 37440},
		{"mla 128", Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: 256, TokensPerState: 128, StateBytes: 584, Alignment: 576}, 1728},
		{"mla 4", Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: 256, TokensPerState: 4, StateBytes: 584, Alignment: 576}, 37440},
		{"indexer 4", Layer{Spec: LayerMLA, KVHeads: 1, HeadSize: 132, BlockSize: 256, TokensPerState: 4, Alignment: 576}, 8640},
		{"ring", Layer{Spec: LayerCircularBuffer, KVHeads: 1, HeadSize: 1024, BlockSize: 8, ElemBytes: 4}, 32768},
	} {
		if got := pageBytes(c.layer); got != c.want {
			t.Errorf("%s: page %d bytes, want %d", c.name, got, c.want)
		}
	}
}

// _approximate_gcd picks the divisor that pads least, the larger on a tie, from the floor up.
func TestApproximateGCD(t *testing.T) {
	for _, c := range []struct {
		values      []int
		lower, want int
	}{
		{[]int{30, 31, 30}, 30, 31},
		{[]int{12, 13}, 1, 1},
		{[]int{31, 30, 91}, 31, 31},
		{[]int{6, 4}, 1, 2},
		{[]int{5}, 9, 9},
	} {
		if got := approximateGCD(c.values, c.lower); got != c.want {
			t.Errorf("approximateGCD(%v, %d) = %d, want %d", c.values, c.lower, got, c.want)
		}
	}
}

// With a trailing drafter (MTP, or DeepSeek V4's DSpark), the group holding the model's last
// layer is the drafter's (_annotate_eagle_groups, kv_cache_utils.py:2051-2109@5df63a72): the
// probes above mark it, deepseek-v4-pro's second window group with MTP and
// deepseek-v4.1-flash's third with DSpark.
func TestPackedGroupsFlagTheDrafterGroup(t *testing.T) {
	for _, c := range []struct {
		name   string
		layers []Layer
		eagle  int
	}{
		{"deepseek-v4-pro mtp", deepseekV4ProLayers(256, deepseekV4ProRatios(), true), 1},
		{"deepseek-v4.1-flash 128 dspark", deepseekV41FlashLayers(128, 2, false), 2},
		{"deepseek-v4.1-flash 64 dspark", deepseekV41FlashLayers(64, 2, false), 2},
	} {
		got, err := GroupLayers(c.layers, GroupOptions{TrailingDrafter: true})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		for i, g := range got {
			if g.Eagle != (i == c.eagle) {
				t.Errorf("%s: group %d Eagle %t, want only group %d", c.name, i, g.Eagle, c.eagle)
			}
		}
	}
}
