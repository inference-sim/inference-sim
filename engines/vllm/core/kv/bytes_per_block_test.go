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
