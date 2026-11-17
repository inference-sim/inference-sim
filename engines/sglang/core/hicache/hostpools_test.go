// Original to blis-sglang.

package hicache

import (
	"strings"
	"testing"
)

// The MHA host pool's bytes per token are SGLang's, in every trace: MHATokenToKVPoolHost's
// get_size_per_token, as its constructor answered (pool_host/mha.py:290-296@297e54fd).
func TestTheMHAHostPoolSizesATokenAsSGLangDoes(t *testing.T) {
	checked := 0
	for name, tr := range loadTraces(t) {
		for _, c := range tr.Calls {
			if c.Iface != "host_pool" || c.Method != "__init__" || c.Class != "MHATokenToKVPoolHost" {
				continue
			}
			var s struct {
				HeadNum      int    `json:"head_num"`
				HeadDim      int    `json:"head_dim"`
				VHeadDim     int    `json:"v_head_dim"`
				LayerNum     int    `json:"layer_num"`
				DType        string `json:"dtype"`
				SizePerToken int64  `json:"size_per_token"`
			}
			if err := c.Answer(&s); err != nil {
				t.Fatal(err)
			}
			if s.DType != "torch.bfloat16" {
				t.Fatalf("%s: a %s pool; the test reads bfloat16 as 2 bytes", name, s.DType)
			}
			got := MHAHost(s.HeadNum, s.HeadDim, s.VHeadDim, s.LayerNum, 2, 1).SizePerToken()
			if got != s.SizePerToken {
				t.Errorf("%s: %d bytes per token; SGLang's host pool takes %d", name, got, s.SizePerToken)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no trace builds an MHA host pool")
	}
}

// MLA's host row is the latent cache, kv_lora_rank + qk_rope_head_dim wide, or the device pool's
// packed width (pool_host/mla.py:127-133, 229-231@297e54fd); DSA's index row is the index keys
// plus one fp32 scale per quant block (pool_host/dsa.py:71-76), and an index page that does not
// split over the KV page is refused as SGLang refuses it (:113-124).
func TestTheMLAAndDSAHostPoolsSizeATokenAsSGLangDoes(t *testing.T) {
	// DeepSeek-V3's latent cache: rank 512, RoPE 64, 61 layers, bfloat16.
	if got := MLAHost(512, 64, 0, 61, 2).SizePerToken(); got != (512+64)*2*61 {
		t.Errorf("MLA: %d bytes per token", got)
	}
	// An FP8 DSA pool packs latent, RoPE and scales into a 656-byte row of one-byte elements.
	if got := MLAHost(512, 64, 656, 61, 1).SizePerToken(); got != 656*61 {
		t.Errorf("packed MLA: %d bytes per token", got)
	}
	idx, err := DSAIndexerHost(128, 128, 64, 1, 61)
	if err != nil {
		t.Fatal(err)
	}
	if got := idx.BytesPerTokenPerLayer(); got != 132 {
		t.Errorf("DSA index row: %d bytes, want 128 key bytes and one 4-byte scale", got)
	}
	if got := idx.SizePerToken(); got != 132*61 {
		t.Errorf("DSA: %d bytes per token", got)
	}
	// index_kpool=4 stores a quarter-page index row over the KV page.
	if quarter, err := DSAIndexerHost(128, 128, 64, 4, 61); err != nil || quarter.BytesPerTokenPerLayer() != 33 {
		t.Errorf("index_kpool 4: %v, %v", quarter.BytesPerTokenPerLayer(), err)
	}
	if _, err := DSAIndexerHost(128, 128, 64, 3, 61); err == nil || !strings.Contains(err.Error(), "multiple of index_kpool") {
		t.Errorf("a page that index_kpool does not divide: %v", err)
	}
	if _, err := DSAIndexerHost(100, 128, 64, 64, 61); err == nil || !strings.Contains(err.Error(), "does not split over") {
		t.Errorf("an index page that does not split over the KV page: %v", err)
	}
}

// Every layout moves the same bytes; it changes only the contiguous runs a copy is cut into,
// from each layout's buffer dims (pool_host/mha.py:305-352@297e54fd). A backup of n tokens
// moves n tokens' rows of every layer; a load-back moves them a layer at a time.
func TestEachHostLayoutPricesACopyByItsRuns(t *testing.T) {
	// Qwen2.5-0.5B: 2 KV heads of 64, 24 layers, bfloat16; 4 pages of 16 tokens.
	s := MHAHost(2, 64, 64, 24, 2, 1)
	const page, tokens = 16, 64
	row := int64(2 * 64 * 2) // one head row, K or V
	want := map[string]struct{ backupRuns, loadRuns int64 }{
		// (2, layer, token, head, dim): one run per token, layer and component.
		"layer_first": {2 * 24 * tokens, 2 * tokens},
		// (2, token, layer, head, dim): a token's layers are one run per component.
		"page_first": {2 * tokens, 2 * tokens},
		// (2, page, layer, page_size, head, dim): a page's layers are one run per component.
		"page_first_direct": {2 * 4, 2 * 4},
		// (2, page, head, page_size, layer, dim): one run per page, head and component; a layer
		// is strided, one run per head row.
		"page_head": {2 * 2 * 4, 2 * 2 * tokens},
		// (page, group, layer, 2, page_size, heads, dim): one run per page and head group.
		"page_unified": {4, 4},
	}
	for _, layout := range Layouts {
		b, err := s.Backup(layout, page, tokens)
		if err != nil {
			t.Fatalf("%s: %v", layout, err)
		}
		l, err := s.LoadLayer(layout, page, tokens)
		if err != nil {
			t.Fatalf("%s: %v", layout, err)
		}
		if b.Bytes != s.SizePerToken()*tokens || l.Bytes*24 != b.Bytes || l.Bytes != 2*row*tokens {
			t.Errorf("%s: a backup moves %d bytes and a layer %d; want %d and %d", layout, b.Bytes, l.Bytes,
				s.SizePerToken()*tokens, 2*row*tokens)
		}
		w := want[layout]
		if b.Runs != w.backupRuns || l.Runs != w.loadRuns {
			t.Errorf("%s: %d backup runs and %d load runs; want %d and %d", layout, b.Runs, l.Runs, w.backupRuns, w.loadRuns)
		}
	}
	if _, err := MLAHost(512, 64, 0, 61, 2).Backup("page_head", page, tokens); err == nil {
		t.Error("an MLA pool priced the MHA-only page_head layout")
	}
	if _, err := s.Backup("layer_first", page, 10); err == nil {
		t.Error("a copy of part of a page was priced")
	}
}

// SGLang's server arguments resolve a layout and an IO backend that cannot copy together
// (arg_groups/hicache_hook.py:171-195@297e54fd), and the host tier refuses what it does not
// model by name, citing its open issue; every other combination starts.
func TestTheHostTierResolvesAndRefusesItsSettingsAsSGLangDoes(t *testing.T) {
	for _, tc := range []struct{ layout, io, wantLayout, wantIO string }{
		{"page_first_direct", "kernel", "page_first_direct", "direct"},
		{"page_first", "direct", "page_first_direct", "direct"},
		{"page_first", "kernel", "page_first", "kernel"},
		{"layer_first", "direct", "layer_first", "direct"},
	} {
		if l, io := ResolveLayoutIO(tc.layout, tc.io); l != tc.wantLayout || io != tc.wantIO {
			t.Errorf("%s+%s resolved to %s+%s; SGLang's is %s+%s", tc.layout, tc.io, l, io, tc.wantLayout, tc.wantIO)
		}
	}
	ok := L2Config{WritePolicy: "write_through", IOBackend: "direct", Layout: "layer_first"}
	for _, tc := range []struct {
		edit       func(*L2Config)
		want, cite string
	}{
		{func(c *L2Config) { c.StorageBackend = "file" }, "--hicache-storage-backend file", StorageRefusalIssue},
		{func(c *L2Config) { c.HostMemoryMode = "buffer_only" }, "buffer_only", L2RefusalIssue},
		{func(c *L2Config) { c.IOBackend = "kernel_ascend" }, "kernel_ascend", L2RefusalIssue},
		{func(c *L2Config) { c.IOBackend, c.Layout = "kernel", "page_first_kv_split" }, "page_first_kv_split", L2RefusalIssue},
		{func(c *L2Config) { c.WritePolicy = "write_sometimes" }, "invalid write policy", ""},
		{func(c *L2Config) { c.Layout = "page_head" }, "for io backend 'direct'", ""},
	} {
		c := ok
		tc.edit(&c)
		err := c.Check()
		if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), tc.cite) {
			t.Errorf("%+v: %v, want a refusal naming %q citing %q", c, err, tc.want, tc.cite)
		}
	}
	for _, p := range WritePolicies {
		for _, io := range []string{"direct", "kernel"} {
			for _, layout := range Layouts {
				l, i := ResolveLayoutIO(layout, io)
				c := L2Config{WritePolicy: p, IOBackend: i, Layout: l}
				if i == "direct" && l != "layer_first" && l != "page_first_direct" {
					continue
				}
				if err := c.Check(); err != nil {
					t.Errorf("%+v refused: %v", c, err)
				}
			}
		}
	}
}
