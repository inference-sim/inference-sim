package kv

import "testing"

// pages is n layers of one page, each of alignment 1.
func pages(n int, bytes int64) PageGroup {
	g := make(PageGroup, n)
	for i := range g {
		g[i] = PageLayer{PageBytes: bytes, Alignment: 1}
	}
	return g
}

// The pool's block is the widest group's, not every layer's: gpt-oss-120b's 18 window and 18
// full-attention layers (8 KV heads of 64 at TP1, bf16, 16-token blocks: 32768 bytes a page) are
// two groups of 18, so a block is 18 pages and the pool holds twice the blocks one group of all
// 36 layers would. MiniMax-M3's 60 full-attention layers and 57 key-only indexers are one
// uniform-type group of mixed pages, whose block is all of them.
func TestBytesPerBlockIsTheWidestGroups(t *testing.T) {
	m3 := append(pages(60, 16384), pages(57, 4096)...)
	for _, c := range []struct {
		name           string
		groups         []PageGroup
		blockOutermost bool
		want           int64
	}{
		{"gpt-oss two groups", []PageGroup{pages(18, 32768), pages(18, 32768)}, false, 18 * 32768},
		{"gpt-oss two groups, block-outer", []PageGroup{pages(18, 32768), pages(18, 32768)}, true, 18 * 32768},
		{"gpt-oss as one group", []PageGroup{pages(36, 32768)}, false, 36 * 32768},
		{"minimax-m3 one group of mixed pages", []PageGroup{m3}, false, 60*16384 + 57*4096},
		{"unequal groups: the larger sets the block", []PageGroup{pages(7, 100), pages(6, 100)}, false, 700},
		// Layer-outer: each layer's stride is its page padded to its own alignment.
		{"layer-outer pads each page", []PageGroup{{{PageBytes: 100, Alignment: 64}, {PageBytes: 30, Alignment: 1}}}, false, 128 + 30},
		// Block-outer: the summed block is padded to the lcm of every layer's alignment (192
		// for 64 and 48), and no page on its own.
		{"block-outer pads the block to the lcm", []PageGroup{{{PageBytes: 150, Alignment: 64}, {PageBytes: 50, Alignment: 48}}}, true, 384},
		{"block-outer pads no page", []PageGroup{{{PageBytes: 70, Alignment: 64}, {PageBytes: 70, Alignment: 64}}}, true, 192},
	} {
		got, err := BytesPerBlock(c.groups, c.blockOutermost)
		if err != nil || got != c.want {
			t.Errorf("%s: BytesPerBlock = %d (%v), want %d", c.name, got, err, c.want)
		}
	}
}

// The block count is the available memory over the widest group's block, floored.
func TestNumBlocksFloorsTheAvailableMemory(t *testing.T) {
	const available = 110912008213 // d036's KV memory, the kernel's allocatable bytes
	got, err := NumBlocks(available, []PageGroup{pages(18, 32768), pages(18, 32768)}, false)
	if err != nil || got != available/(18*32768) {
		t.Errorf("NumBlocks = %d (%v), want %d", got, err, available/(18*32768))
	}
}

func TestBytesPerBlockRefusesWhatHasNoPage(t *testing.T) {
	for _, groups := range [][]PageGroup{nil, {{}}, {{{PageBytes: 0, Alignment: 1}}}, {{{PageBytes: 8, Alignment: 0}}}} {
		if _, err := BytesPerBlock(groups, false); err == nil {
			t.Errorf("BytesPerBlock(%v) gave no error", groups)
		}
	}
}

// layersOf is n layers of spec s, KV heads and head size, at 2 bytes an element.
func layersOf(n int, s LayerSpec, window, heads, head int) []Layer {
	out := make([]Layer, n)
	for i := range out {
		out[i] = Layer{Spec: s, Window: window, KVHeads: heads, HeadSize: head, ElemBytes: 2}
	}
	return out
}

// The pool's block from a model's layers as vLLM groups them: the groups' members are what
// get_kv_cache_groups puts in each (a bucket of n layers split into ceil(n / size) groups of its
// layers [i::groups], kv_cache_utils.py:1462-1501@5df63a72), each layer's page its spec's.
func TestPoolBytesPerBlockSizesTheGroupsVLLMBuilds(t *testing.T) {
	var gptOSS []Layer
	for range 18 {
		gptOSS = append(gptOSS, layersOf(1, LayerSlidingWindow, 128, 8, 64)[0], layersOf(1, LayerFull, 0, 8, 64)[0])
	}
	var m3 []Layer
	m3 = append(m3, layersOf(3, LayerFull, 0, 2, 128)...)
	for range 57 {
		m3 = append(m3, layersOf(1, LayerIndexer, 0, 1, 128)[0], layersOf(1, LayerFull, 0, 2, 128)[0])
	}
	const page = 16 * 2 * 8 * 64 * 2 // a gpt-oss layer's page at 16 tokens: K and V, 8 heads of 64, bf16
	for _, c := range []struct {
		name   string
		layers []Layer
		want   int64
	}{
		// Two groups of 18: the block is one group's 18 pages.
		{"gpt-oss-120b", gptOSS, 18 * page},
		// One uniform-type group of 60 full-attention pages and 57 key-only indexer pages.
		{"minimax-m3", m3, 60*16*2*2*128*2 + 57*16*1*128*2},
		{"full attention only", layersOf(36, LayerFull, 0, 8, 64), 36 * page},
		// 30 windows and 10 full: groups of 10, three of windows and one of full.
		{"3:1", append(layersOf(30, LayerSlidingWindow, 128, 8, 64), layersOf(10, LayerFull, 0, 8, 64)...), 10 * page},
		// 13 windows and 12 full: under 1.5 times, so groups of 13, the full group padded.
		{"13:12", append(layersOf(13, LayerSlidingWindow, 128, 8, 64), layersOf(12, LayerFull, 0, 8, 64)...), 13 * page},
	} {
		got, err := PoolBytesPerBlock(c.layers, GroupOptions{}, 16)
		if err != nil || got != c.want {
			t.Errorf("%s: PoolBytesPerBlock = %d (%v), want %d", c.name, got, err, c.want)
		}
	}
	// With the hybrid manager off, gpt-oss's windows fold into full attention: one group.
	if got, _ := PoolBytesPerBlock(gptOSS, GroupOptions{DisableHybrid: true}, 16); got != 36*page {
		t.Errorf("gpt-oss-120b, hybrid manager off: PoolBytesPerBlock = %d, want %d", got, 36*page)
	}
}
