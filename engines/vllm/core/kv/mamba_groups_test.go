// Original to blis-vllm.

package kv

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

// The numbers below are worked by hand from vLLM v0.32.0's source: a layer of 2 KV heads of 8
// elements at 2 bytes, 16-token blocks, has the page 2 x 16 x 2 x 8 x 2 = 1024 bytes
// (AttentionSpec.page_size_bytes, vllm/v1/kv_cache_interface.py:546-563@5df63a72), and a Mamba
// state of 600 bytes is padded to that page at start-up (vllm/platforms/interface.py:1118-1126).
const hybridBlock = 16

func hybridAttn() Layer { return Layer{Spec: LayerFull, KVHeads: 2, HeadSize: 8, ElemBytes: 2} }

func hybridMamba(mode string, tokens int) Layer {
	return Layer{Spec: LayerMamba, StateBytes: 600, PageSizePadded: 1024, BlockSize: tokens, MambaCacheMode: mode}
}

// nemotronLike is 12 Mamba layers and 4 attention layers in the pattern m m m a.
func nemotronLike(mode string, tokens int) []Layer {
	var out []Layer
	for range 4 {
		out = append(out, hybridMamba(mode, tokens), hybridMamba(mode, tokens), hybridMamba(mode, tokens), hybridAttn())
	}
	return out
}

// Attention beside Mamba layers of the same page is vLLM's uniform-page split
// (_get_kv_cache_groups_uniform_page_size, kv_cache_utils.py:1362-1502@5df63a72): 12 Mamba and
// 4 attention layers are two types of 12 and 4 layers; 12 is not under 1.5 x 4, so the group
// size is the fewest, 4, the attention layers are one group and the Mamba layers are cdiv(12, 4)
// = 3 groups, the i-th taking layers [i::3] of the type (:1495-1501). Mamba's block size is its
// own: the engine's in align mode, max_model_len in none mode.
func TestHybridMambaAttentionGroupsAtAnEqualPage(t *testing.T) {
	for _, c := range []struct {
		name   string
		mode   string
		tokens int
	}{
		{"align", "align", hybridBlock},
		{"none", "none", 64},
	} {
		t.Run(c.name, func(t *testing.T) {
			layers := nemotronLike(c.mode, c.tokens)
			groups, members, err := groupLayers(layers, GroupOptions{BlockSize: hybridBlock})
			if err != nil {
				t.Fatal(err)
			}
			m := Group{Kind: Mamba, BlockSize: c.tokens, MambaCacheMode: c.mode}
			want := []Group{m, m, m, {Kind: FullAttention}}
			if !reflect.DeepEqual(groups, want) {
				t.Errorf("groups %+v, want %+v", groups, want)
			}
			wantMembers := [][]int{{0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}}
			if !reflect.DeepEqual(members, wantMembers) {
				t.Errorf("members %v, want %v", members, wantMembers)
			}
		})
	}
}

// A model of nothing but Mamba layers of one spec is one group (UniformTypeKVCacheSpecs,
// kv_cache_utils.py:2189-2193@5df63a72); Mamba layers of two specs (here, two speculative-block
// counts) are two types (MambaSpec.is_uniform_with_collection, kv_cache_interface.py:1093-1103),
// each its own group beside the attention layers.
func TestMambaLayersOfOneSpecAreOneType(t *testing.T) {
	pure := []Layer{hybridMamba("none", 64), hybridMamba("none", 64), hybridMamba("none", 64)}
	groups, members, err := groupLayers(pure, GroupOptions{})
	if err != nil || len(groups) != 1 || groups[0].Kind != Mamba || !reflect.DeepEqual(members, [][]int{{0, 1, 2}}) {
		t.Fatalf("pure Mamba: groups %+v members %v (%v), want one Mamba group of all three", groups, members, err)
	}
	a, b := hybridMamba("align", 16), hybridMamba("align", 16)
	b.NumSpeculativeBlocks = 1
	groups, _, err = groupLayers([]Layer{a, a, b, b, hybridAttn(), hybridAttn()}, GroupOptions{BlockSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 3 || groups[0].MambaSpeculativeBlocks != 0 || groups[1].MambaSpeculativeBlocks != 1 || groups[2].Kind != FullAttention {
		t.Errorf("groups %+v, want a Mamba group per spec, then full attention", groups)
	}
}

// A Mamba page that differs from the attention page, where both are known, is
// unify_kv_cache_spec_page_size's to pad or scale (kv_cache_utils.py:1292-1354@5df63a72): refused
// by name, not grouped as if equal.
func TestMambaPageUnequalToAttentionIsRefused(t *testing.T) {
	m := hybridMamba("align", 16)
	m.PageSizePadded = 2048
	_, _, err := groupLayers([]Layer{m, hybridAttn()}, GroupOptions{BlockSize: 16})
	if err == nil || !strings.Contains(err.Error(), "unify_kv_cache_spec_page_size") {
		t.Fatalf("err %v, want a refusal naming unify_kv_cache_spec_page_size", err)
	}
}

// With the hybrid manager disabled, unify_hybrid_kv_cache_specs promotes the sliding windows of
// the attention layers to full attention and leaves the Mamba layers, which form their own part
// (kv_cache_utils.py:1799-1822, 1684-1761@5df63a72): the model is a full-attention group and a
// Mamba group, not one full-attention group. Mamba layers of more than one spec cannot be
// promoted to one type, and vLLM raises (:1759-1760).
func TestDisableHybridKeepsMambaGroups(t *testing.T) {
	win := Layer{Spec: LayerSlidingWindow, Window: 128, KVHeads: 2, HeadSize: 8, ElemBytes: 2}
	mamba := hybridMamba("none", 64)
	groups, members, err := groupLayers([]Layer{win, hybridAttn(), mamba, mamba}, GroupOptions{DisableHybrid: true, BlockSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	want := []Group{{Kind: FullAttention}, {Kind: Mamba, BlockSize: 64, MambaCacheMode: "none"}}
	if !reflect.DeepEqual(groups, want) || !reflect.DeepEqual(members, [][]int{{0, 1}, {2, 3}}) {
		t.Errorf("groups %+v members %v, want %+v over {0,1} and {2,3}", groups, members, want)
	}
	other := mamba
	other.NumSpeculativeBlocks = 1
	_, _, err = groupLayers([]Layer{hybridAttn(), mamba, other}, GroupOptions{DisableHybrid: true, BlockSize: 16})
	if err == nil || !strings.Contains(err.Error(), "Failed to promote local KV cache specs") {
		t.Errorf("two Mamba specs: err %v, want vLLM's promotion failure", err)
	}
}

// A mamba_cache_mode this port does not run ("all") is refused by name.
func TestMambaCacheModeAllIsRefused(t *testing.T) {
	_, _, err := groupLayers([]Layer{hybridMamba("all", 16)}, GroupOptions{})
	if err == nil || !strings.Contains(err.Error(), `"all"`) {
		t.Errorf("err %v, want a refusal naming mode all", err)
	}
}

// _get_packed_kv_cache_groups on a block-outermost layout with Mamba (kv_cache_utils.py:
// 1860-2014@5df63a72): 2 attention layers of page 1024 and 8 Mamba layers of page 512 are two
// buckets, neither mixing page sizes, so repeats_per_group is None. The anchor is the widest
// group a bucket must hold: attention 2 x 1024 = 2048, Mamba at one layer a group 512, so 2048
// (:1948-1960). Mamba is a state bucket (is_state_bucket, :1941-1946) and splits to the states a
// block fits, 2048 // 512 = 4, so cdiv(8, 4) = 2 groups taking layers [i::2] (:1964-1983). Were
// Mamba not a state bucket it would stay one group of 8 and widen every block to 4096.
func TestPackedGroupsSplitMambaToTheAnchorBlock(t *testing.T) {
	attn := hybridAttn()
	mamba := Layer{Spec: LayerMamba, StateBytes: 512, BlockSize: 16, MambaCacheMode: "align"}
	layers := append([]Layer{attn, attn}, slices.Repeat([]Layer{mamba}, 8)...)
	o := GroupOptions{BlockOutermost: true, BlockSize: 16}
	groups, members, err := groupLayers(layers, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 3 || groups[0].Kind != FullAttention || groups[1].Kind != Mamba || groups[2].Kind != Mamba {
		t.Fatalf("groups %+v, want full attention and two Mamba groups", groups)
	}
	if want := [][]int{{0, 1}, {2, 4, 6, 8}, {3, 5, 7, 9}}; !reflect.DeepEqual(members, want) {
		t.Errorf("members %v, want %v", members, want)
	}
	per, err := PoolBytesPerBlock(layers, o, 16)
	if err != nil || per != 2048 {
		t.Errorf("pool bytes per block %d (%v), want 2048: the widest group, 4 x 512", per, err)
	}
}
