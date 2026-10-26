// Original to blis-vllm.

package kv

import (
	"reflect"
	"testing"
)

// The grouping of a dashboard model's attention layers is vLLM v0.32.0's get_kv_cache_groups
// (vllm/v1/core/kv_cache_utils.py:2159-2253@5df63a72), as the oracle's dashboard-* scenarios
// check against vLLM itself.
func TestGroupsOfLayers(t *testing.T) {
	full := Layer{Spec: LayerFull, KVHeads: 8, HeadSize: 64}
	win := Layer{Spec: LayerSlidingWindow, Window: 128, KVHeads: 8, HeadSize: 64}
	indexer := Layer{Spec: LayerIndexer, KVHeads: 1, HeadSize: 128}
	repeat := func(n int, ls ...Layer) []Layer {
		var out []Layer
		for i := 0; i < n; i++ {
			out = append(out, ls...)
		}
		return out
	}
	for _, c := range []struct {
		name   string
		layers []Layer
		want   []Group
	}{
		// One spec for every layer: one group (_get_kv_cache_groups_uniform_spec, :2185-2189).
		{"dense", repeat(4, full), []Group{{Kind: FullAttention}}},
		// gpt-oss-120b: windowed and full layers alternate at one page size, so the uniform-page
		// split makes one group of each type (_get_kv_cache_groups_uniform_page_size,
		// :1362-1502), in the order the layers first appear.
		{"gpt-oss", repeat(18, win, full), []Group{{Kind: SlidingWindow, Window: 128}, {Kind: FullAttention}}},
		// MiniMax-M3: full attention and key-only MLA indexer layers are all of the
		// FullAttentionSpec type at one block size, one uniform-type group (:2189-2193).
		{"minimax-m3", append(repeat(3, Layer{Spec: LayerFull, KVHeads: 4, HeadSize: 128}),
			repeat(57, Layer{Spec: LayerFull, KVHeads: 4, HeadSize: 128}, indexer)...), []Group{{Kind: FullAttention}}},
		// Two window sizes are two types: one group per type at an equal page.
		{"two windows", repeat(2, full, win, Layer{Spec: LayerSlidingWindow, Window: 64, KVHeads: 8, HeadSize: 64}),
			[]Group{{Kind: FullAttention}, {Kind: SlidingWindow, Window: 128}, {Kind: SlidingWindow, Window: 64}}},
		// Unequal counts: the group size is the fewest layers of a type, so 3 full layers
		// beside 1 window are 3 full groups, and 2 full beside 3 windows (3 < 1.5 x 2 fails)
		// are 1 full and 2 window groups, the second padded.
		{"3 full, 1 window", []Layer{full, full, full, win}, []Group{{Kind: FullAttention}, {Kind: FullAttention},
			{Kind: FullAttention}, {Kind: SlidingWindow, Window: 128}}},
		{"2 full, 3 windows", []Layer{full, win, full, win, win}, []Group{{Kind: FullAttention},
			{Kind: SlidingWindow, Window: 128}, {Kind: SlidingWindow, Window: 128}}},
		// 12 windows beside 13 full layers (gpt-oss-20b with a drafter layer): 13 < 1.5 x 12,
		// so the group size is 13 and each type is one group.
		{"12 windows, 13 full", append(repeat(12, win, full), full), []Group{{Kind: SlidingWindow, Window: 128},
			{Kind: FullAttention}}},
	} {
		got, err := GroupsOfLayers(c.layers, false)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: groups %v, want %v", c.name, got, c.want)
		}
	}
}

// What the uniform-page split cannot express is refused rather than grouped some other way:
// windowed and full layers whose pages differ go to vLLM's packed grouping, which this port
// does not model.
func TestGroupsOfLayersRefusesWhatItDoesNotModel(t *testing.T) {
	full := Layer{Spec: LayerFull, KVHeads: 8, HeadSize: 64}
	for _, c := range []struct {
		name   string
		layers []Layer
	}{
		{"mixed pages", []Layer{full, {Spec: LayerSlidingWindow, Window: 128, KVHeads: 4, HeadSize: 64}}},
		{"none", nil},
	} {
		if _, err := GroupsOfLayers(c.layers, false); err == nil {
			t.Errorf("%s: grouped, want a refusal", c.name)
		}
	}
}

// With the hybrid manager disabled, windows beside full attention are promoted to it: one
// group (unify_hybrid_kv_cache_specs, kv_cache_utils.py:1799-1822@5df63a72).
func TestGroupsOfLayersDisableHybrid(t *testing.T) {
	full := Layer{Spec: LayerFull, KVHeads: 8, HeadSize: 64}
	win := Layer{Spec: LayerSlidingWindow, Window: 128, KVHeads: 8, HeadSize: 64}
	got, err := GroupsOfLayers([]Layer{win, full, win, full}, true)
	if err != nil || !reflect.DeepEqual(got, []Group{{Kind: FullAttention}}) {
		t.Errorf("groups %v (%v), want one full-attention group", got, err)
	}
}
