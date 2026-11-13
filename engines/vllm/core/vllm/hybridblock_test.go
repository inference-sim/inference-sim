// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"
)

// The aligned block size is the smallest multiple of the kernel block alignment whose attention
// page holds one Mamba page. Each case is worked by hand from the formula in
// Platform._align_hybrid_block_size (vllm/platforms/interface.py:1086-1126@5df63a72):
//
//	Kimi-K3 at tp 8, bf16 cache: one MLA token is 576 x 2 = 1152 bytes, the KDA page 814080;
//	  alignment max(64, 16, 128) = 128; 128 x ceil(814080 / (128 x 1152)) = 128 x 6 = 768;
//	  the attention page 768 x 1152 = 884736 holds it, so the Mamba page pads to 884736.
//	the same at an fp8 cache: a token is 576 bytes, 128 x ceil(814080 / 73728) = 128 x 12 = 1536.
//	Nemotron-3-Ultra at tp 8: one GQA token is 2 x 1 x 128 x 2 = 512 bytes, the Mamba-2 page 1062400;
//	  16 x ceil(1062400 / (16 x 512)) = 16 x 130 = 2080, a page of 1064960.
func TestHybridBlockSizeHoldsOneMambaPage(t *testing.T) {
	for _, c := range []struct {
		name   string
		in     HybridBlockInput
		block  int
		mamba  int
		padded int64
	}{
		{"kimi-k3 tp8 bf16", HybridBlockInput{Block: 16, AttnPage1Token: 1152, MambaPage: 814080, KernelBlockMin: 64,
			BackendBlockLCM: 64, MLA: true, MambaCacheMode: "align"}, 768, 768, 884736},
		{"kimi-k3 tp8 fp8", HybridBlockInput{Block: 16, AttnPage1Token: 576, MambaPage: 814080, KernelBlockMin: 64,
			BackendBlockLCM: 64, MLA: true, MambaCacheMode: "align"}, 1536, 1536, 884736},
		{"nemotron-3-ultra tp8 bf16", HybridBlockInput{Block: 16, AttnPage1Token: 512, MambaPage: 1062400, KernelBlockMin: 16,
			BackendBlockLCM: 16, MambaCacheMode: "align"}, 2080, 2080, 1064960},
		// A stated block already past the aligned one stays, and pads the Mamba page to its page.
		{"stated block 4096", HybridBlockInput{Block: 4096, AttnPage1Token: 512, MambaPage: 1062400, KernelBlockMin: 16,
			BackendBlockLCM: 16, MambaCacheMode: "align"}, 4096, 4096, 2097152},
		// A page that is exactly one block's pages is not padded.
		{"exact page", HybridBlockInput{Block: 16, AttnPage1Token: 512, MambaPage: 16 * 512 * 4, KernelBlockMin: 16,
			BackendBlockLCM: 16, MambaCacheMode: "align"}, 64, 64, 0},
		// The backends' smallest kernel blocks round the size up: 16 x ceil(100000 / (16 x 512)) = 16 x 13 = 208,
		// then a multiple of lcm 48 is 240.
		{"backend lcm", HybridBlockInput{Block: 16, AttnPage1Token: 512, MambaPage: 100000, KernelBlockMin: 16,
			BackendBlockLCM: 48, MambaCacheMode: "align"}, 240, 240, 240 * 512},
		// In none mode the Mamba block stays the window it was given, and the page still pads.
		{"none mode", HybridBlockInput{Block: 16, AttnPage1Token: 1152, MambaPage: 814080, KernelBlockMin: 64,
			BackendBlockLCM: 64, MLA: true, MambaCacheMode: "none", MambaBlockSize: 262144}, 768, 262144, 884736},
		// No Mamba page: nothing to align.
		{"no mamba page", HybridBlockInput{Block: 16, AttnPage1Token: 512, KernelBlockMin: 16, BackendBlockLCM: 16,
			MambaCacheMode: "align"}, 16, 16, 0},
	} {
		got, err := AlignHybridBlockSize(c.in)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if got.BlockSize != c.block || got.MambaBlockSize != c.mamba || got.PageSizePadded != c.padded {
			t.Errorf("%s: block %d, mamba block %d, padded %d; want %d, %d, %d", c.name,
				got.BlockSize, got.MambaBlockSize, got.PageSizePadded, c.block, c.mamba, c.padded)
		}
		if page := int64(got.BlockSize) * c.in.AttnPage1Token; c.in.MambaPage > page {
			t.Errorf("%s: attention page %d does not hold the Mamba page %d", c.name, page, c.in.MambaPage)
		}
	}
}

// An MLA model's alignment is at least 128 whichever dense MLA backend runs: their smallest kernel
// blocks (16, 32, 64, 128) all divide it, and a first choice of block size up to 128 is below it,
// so neither the backend nor the first choice changes the result. That is what lets a Kimi-K3
// deployment that states no block_size be aligned without the backend selection.
func TestMLAHybridBlockSizeDoesNotDependOnTheBackend(t *testing.T) {
	var want HybridBlock
	for i, min := range []int{16, 32, 64, 128} {
		for _, first := range []int{16, 32, 64, 128} {
			got, err := AlignHybridBlockSize(HybridBlockInput{Block: first, AttnPage1Token: 1152, MambaPage: 814080,
				KernelBlockMin: min, BackendBlockLCM: min, MLA: true, MambaCacheMode: "align"})
			if err != nil {
				t.Fatal(err)
			}
			if i == 0 && first == 16 {
				want = got
			}
			if got != want {
				t.Errorf("smallest kernel block %d, first block %d: %+v, want %+v", min, first, got, want)
			}
		}
	}
	if want.BlockSize != 768 {
		t.Errorf("block %d, want 768", want.BlockSize)
	}
}

func TestHybridBlockSizeRefusesInputsItCannotAlign(t *testing.T) {
	ok := HybridBlockInput{Block: 16, AttnPage1Token: 512, MambaPage: 1000, KernelBlockMin: 16, BackendBlockLCM: 16, MambaCacheMode: "align"}
	for name, mut := range map[string]func(*HybridBlockInput){
		"block":       func(in *HybridBlockInput) { in.Block = 0 },
		"page":        func(in *HybridBlockInput) { in.AttnPage1Token = 0 },
		"kernel":      func(in *HybridBlockInput) { in.KernelBlockMin = 0 },
		"cache mode":  func(in *HybridBlockInput) { in.MambaCacheMode = "all" },
		"lcm":         func(in *HybridBlockInput) { in.BackendBlockLCM = 0 },
		"mamba bytes": func(in *HybridBlockInput) { in.MambaPage = -1 },
	} {
		in := ok
		mut(&in)
		if _, err := AlignHybridBlockSize(in); err == nil || !strings.Contains(err.Error(), "hybrid block alignment") {
			t.Errorf("%s: %v, want it refused", name, err)
		}
	}
}

// The KV pool's block count for a hybrid model: the attention layers take their element size from
// the memory model's bytes for one block of the attention layers, as before, and a Mamba layer
// does not count among them. Twelve bf16 attention layers of 8 heads x 64 beside six Mamba layers
// whose page was padded to the attention page make groups of six layers (the fewest of a type,
// _get_kv_cache_groups_uniform_page_size, kv_cache_utils.py:1362-1502@5df63a72), so one block is
// the 6 pages of the widest group; counting the Mamba layers' stated state bytes as a packed state
// left the attention layers at one byte an element, which halves the page again.
func TestKVPoolBytesPerBlockOfAHybridModelKeepsTheAttentionElementSize(t *testing.T) {
	const page = 16 * 2 * 8 * 64 * 2 // one attention layer's page at 16 tokens, bf16
	runs := []KVLayerRun{
		{Count: 12, Spec: "full", NumKVHeads: 8, HeadSize: 64},
		{Count: 6, Spec: "mamba", StateBytes: 30000, BlockSize: 16, PageSizePadded: page, MambaCacheMode: "align",
			NumPrefillCheckpointBlocks: 1, PrefillCheckpointAlignment: 1},
	}
	got, err := KVPoolBytesPerBlock(runs, false, 16, 12*page)
	if err != nil || got != 6*page {
		t.Errorf("KVPoolBytesPerBlock = %d (%v), want %d: the widest group's 6 pages of %d bytes", got, err, 6*page, page)
	}
	blocks, err := KVPoolBlocks(runs, false, 16, 12*page, 1_000_000_007)
	if err != nil || blocks != 1_000_000_007/(6*page) {
		t.Errorf("KVPoolBlocks = %d (%v), want %d", blocks, err, 1_000_000_007/(6*page))
	}
}
