// Original to blis-vllm.

package kv

import (
	"strings"
	"testing"
)

// gemma4Layers are Gemma 4 31B's KV layers as a vLLM deployment registers them: sixty layers in the
// pattern five sliding windows of 1,024 tokens (16 KV heads of 256) then one full attention layer (4 KV
// heads of 512), in 16-token blocks of bf16. A sliding layer's page is 262,144 bytes and a full layer's
// 131,072: unequal, so vLLM's unify_kv_cache_spec_page_size raises the full layers' block size to 32.
func gemma4Layers() []Layer {
	var out []Layer
	for i := 0; i < 10; i++ {
		for j := 0; j < 5; j++ {
			out = append(out, Layer{Spec: LayerSlidingWindow, Window: 1024, KVHeads: 16, HeadSize: 256, ElemBytes: 2})
		}
		out = append(out, Layer{Spec: LayerFull, KVHeads: 4, HeadSize: 512, ElemBytes: 2})
	}
	return out
}

// TestUnifyPageSizesAreVLLMs: a layer of a smaller page takes block_size * (max_page / page) when the largest is a
// multiple of its page (unify_kv_cache_spec_page_size, kv_cache_planning.py:346-408@4ba787ec); equal pages are left as
// they are; what vLLM pads or raises on is refused by name.
func TestUnifyPageSizesAreVLLMs(t *testing.T) {
	small := Layer{Spec: LayerFull, KVHeads: 2, HeadSize: 128, ElemBytes: 2}
	big := Layer{Spec: LayerSlidingWindow, Window: 64, KVHeads: 8, HeadSize: 128, ElemBytes: 2}
	got, err := unifyPageSizes([]Layer{big, small, big, small}, 16)
	if err != nil {
		t.Fatal(err)
	}
	// 2*8*128*16*2 = 65,536 bytes against 2*2*128*16*2 = 16,384: the small layers' block is x4.
	for i, l := range got {
		want := 0
		if i%2 == 1 {
			want = 64
		}
		if l.BlockSize != want {
			t.Errorf("layer %d: block size %d, want %d", i, l.BlockSize, want)
		}
		if p := pageAt(l, 16); p != 65536 {
			t.Errorf("layer %d: page %d, want 65536", i, p)
		}
	}
	// A layer that states its own block size scales that one.
	s := small
	s.BlockSize = 8
	if got, err = unifyPageSizes([]Layer{big, s}, 16); err != nil || got[1].BlockSize != 8*8 {
		t.Errorf("a stated block size of 8 beside 65,536 bytes: %+v, %v (want 64: page 2*2*128*8*2 = 8,192, x8)", got, err)
	}
	// Equal pages: untouched, and the caller's slice is not written.
	eq := []Layer{big, big}
	if got, err = unifyPageSizes(eq, 16); err != nil || got[0].BlockSize != 0 || got[1].BlockSize != 0 {
		t.Errorf("equal pages changed: %+v, %v", got, err)
	}
	in := []Layer{big, small}
	if _, err = unifyPageSizes(in, 16); err != nil || in[1].BlockSize != 0 {
		t.Errorf("the input was written: %+v, %v", in, err)
	}
	// No engine block size: the pages cannot be compared in bytes.
	if got, err = unifyPageSizes([]Layer{big, small}, 0); err != nil || got[1].BlockSize != 0 {
		t.Errorf("without a block size: %+v, %v", got, err)
	}
	// A page the largest is not a multiple of is padded by vLLM: refused by name.
	odd := Layer{Spec: LayerFull, KVHeads: 3, HeadSize: 128, ElemBytes: 2}
	if _, err = unifyPageSizes([]Layer{big, odd}, 16); err == nil || !strings.Contains(err.Error(), "does not divide") {
		t.Errorf("a page that does not divide: %v", err)
	}
	// An MLA-kind layer whose page divides the largest is unified by the ratio, as vLLM does for any non-Mamba layer
	// (unify_kv_cache_spec_page_size :392-395@4ba787ec tests divisibility before the kind). Before this was refused.
	for _, spec := range []LayerSpec{LayerMLA, LayerIndexer, LayerSlidingWindowMLA, LayerCircularBuffer} {
		l := Layer{Spec: spec, KVHeads: 1, HeadSize: 256, ElemBytes: 2, StateBytes: 512}
		if spec == LayerSlidingWindowMLA {
			l.Window = 128
		}
		if p := pageAt(l, 16); p != 8192 {
			t.Fatalf("spec %v: page %d, want 8192", spec, p)
		}
		got, err := unifyPageSizes([]Layer{big, l}, 16)
		if err != nil || got[1].BlockSize != 16*8 || pageAt(got[1], 16) != 65536 {
			t.Errorf("spec %v beside 65,536 bytes: %+v, %v (want block size 128, page 65,536)", spec, got, err)
		}
	}
	// A page that does not divide: an MLA-kind layer is vLLM's own NotImplementedError (no padding for MLA) ...
	for _, spec := range []LayerSpec{LayerMLA, LayerIndexer} {
		l := Layer{Spec: spec, KVHeads: 1, HeadSize: 192, ElemBytes: 2, StateBytes: 384} // 16 x 384 = 6,144; 65,536 / 6,144 is not whole
		_, err := unifyPageSizes([]Layer{big, l}, 16)
		if err == nil || !strings.Contains(err.Error(), "NotImplementedError") || !strings.Contains(err.Error(), "MLA-kind") {
			t.Errorf("spec %v that does not divide: %v (want vLLM's NotImplementedError)", spec, err)
		}
	}
	// ... and any other attention layer is padded by vLLM, which the port does not model: refused as such, with the issue.
	for _, spec := range []LayerSpec{LayerSlidingWindow, LayerSlidingWindowMLA, LayerCircularBuffer} {
		l := Layer{Spec: spec, KVHeads: 1, HeadSize: 192, ElemBytes: 2, StateBytes: 384, Window: 128}
		_, err := unifyPageSizes([]Layer{big, l}, 16)
		if err == nil || strings.Contains(err.Error(), "NotImplementedError") || !strings.Contains(err.Error(), "not modelled (inference-sim/inference-sim#2661)") {
			t.Errorf("spec %v that does not divide: %v (want the padding refusal citing inference-sim/inference-sim#2661)", spec, err)
		}
	}
}

// TestGemma4LayersGroupAsVLLMDoes: sixty layers of unequal pages are grouped after unify as
// _get_kv_cache_groups_uniform_page_size does: the sliding windows (50 layers) and the full layers (10) are two
// buckets, the group size is the fewest layers of a bucket (10; 50 is not under 1.5 times 10), so five groups of
// sliding windows at the engine's block size and one group of full attention at 32; the pool pages the widest group.
func TestGemma4LayersGroupAsVLLMDoes(t *testing.T) {
	o := GroupOptions{BlockSize: 16}
	groups, members, sized, err := groupAndLayers(gemma4Layers(), o)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 6 {
		t.Fatalf("%d groups, want 6: %+v", len(groups), groups)
	}
	full := 0
	for i, g := range groups {
		if len(members[i]) != 10 {
			t.Errorf("group %d holds %d layers, want 10", i, len(members[i]))
		}
		switch g.Kind {
		case FullAttention:
			full++
			if g.BlockSize != 32 {
				t.Errorf("the full-attention group's block size is %d, want 32", g.BlockSize)
			}
		case SlidingWindow:
			if g.BlockSize != 0 || g.Window != 1024 {
				t.Errorf("a sliding group: %+v, want the engine's block size and a window of 1024", g)
			}
		default:
			t.Errorf("group %d of kind %v", i, g.Kind)
		}
	}
	if full != 1 {
		t.Errorf("%d full-attention groups, want 1", full)
	}
	for i, l := range sized {
		if p := pageAt(l, 16); p != 262144 {
			t.Errorf("layer %d pages %d bytes, want 262144", i, p)
		}
	}
	if per, err := PoolBytesPerBlock(gemma4Layers(), o, 16); err != nil || per != 10*262144 {
		t.Errorf("pool bytes per block %d, %v; want 10 layers x 262144 = %d", per, err, 10*262144)
	}
	pages, err := GroupPages(gemma4Layers(), o, 16)
	if err != nil || len(pages) != 6 {
		t.Fatalf("%v, %d", err, len(pages))
	}
	for i, p := range pages {
		want := 16
		if groups[i].Kind == FullAttention {
			want = 32
		}
		if p.BlockTokens != want || p.BytesPerBlock != 10*262144 {
			t.Errorf("group %d: %+v, want %d tokens and %d bytes", i, p, want, 10*262144)
		}
	}
}

// TestTheManagerRunsGroupsOfTwoBlockSizes: the stage-1 question. The manager sizes the scheduler's block at the LCM of
// the groups' (32) and the hash block at their GCD (16) with prefix caching, allocates each group at its own block
// size, and frees without leaking, exactly as for the packed groups of several block sizes.
func TestTheManagerRunsGroupsOfTwoBlockSizes(t *testing.T) {
	o := GroupOptions{BlockSize: 16}
	groups, err := GroupLayers(gemma4Layers(), o)
	if err != nil {
		t.Fatal(err)
	}
	for _, caching := range []bool{false, true} {
		m := NewManagerWith(4000, 16, 8192, caching, 0, Options{Groups: groups})
		if m.BlockSize() != 32 {
			t.Errorf("caching=%v: scheduler block size %d, want 32", caching, m.BlockSize())
		}
		wantHash := 32
		if caching {
			wantHash = 16
		}
		if m.HashBlockSize() != wantHash {
			t.Errorf("caching=%v: hash block size %d, want %d", caching, m.HashBlockSize(), wantHash)
		}
		free0 := m.Pool.NumFree()
		if !m.Allocate("r", Alloc{New: 5000, NumTokens: 5000}) {
			t.Fatalf("caching=%v: a 5,000-token request does not fit 4,000 blocks", caching)
		}
		for i, g := range m.groups {
			n := len(g.reqBlocks["r"])
			if groups[i].Kind == FullAttention {
				if want := (5000 + 31) / 32; n != want {
					t.Errorf("full group: %d blocks, want ceil(5000/32) = %d", n, want)
				}
			} else if n == 0 || n > (5000+15)/16 {
				t.Errorf("sliding group %d: %d blocks", i, n)
			}
		}
		m.Free("r")
		if m.Pool.NumFree() != free0 {
			t.Errorf("caching=%v: %d free blocks after the request is freed, want %d", caching, m.Pool.NumFree(), free0)
		}
	}
}
