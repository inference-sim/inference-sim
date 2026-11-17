// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/pool_host/mha.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/pool_host/mla.py,
// python/sglang/srt/mem_cache/pool_host/dsa.py, python/sglang/srt/mem_cache/pool_host/page_unified.py,
// python/sglang/srt/mem_cache/pool_host/host_pool_decl.py and python/sglang/srt/arg_groups/hicache_hook.py.

package hicache

import (
	"fmt"
	"slices"
)

// Layouts are the five host layouts the CUDA host pools build (pool_host/mha.py:305-360
// @297e54fd): the axis order of the host buffer, which decides how a copy is cut into runs.
var Layouts = []string{"layer_first", "page_first", "page_first_direct", "page_head", "page_unified"}

// ResolveLayoutIO is resolve_layout_io_compatibility (arg_groups/hicache_hook.py:171-195
// @297e54fd): the kernel backend cannot copy page_first_direct, so it falls back to direct; the
// direct backend cannot copy page_first, so the layout becomes page_first_direct.
func ResolveLayoutIO(layout, io string) (string, string) {
	if layout == "page_first_direct" && io == "kernel" {
		io = "direct"
	}
	if layout == "page_first" && io == "direct" {
		layout = "page_first_direct"
	}
	return layout, io
}

// HostShape is one host pool's geometry: what its buffer holds per token, per layer.
type HostShape struct {
	// Pool is the SGLang host pool class the shape is: MHATokenToKVPoolHost,
	// AsymmetricMHATokenToKVPoolHost (K and V of different widths), MLATokenToKVPoolHost or
	// DSAIndexerPoolHost. It decides which layouts the pool can lay out (CheckLayout).
	Pool string
	// Components is 2 for K and V (MHA), 1 for MLA's latent cache and DSA's index keys.
	Components int
	// HeadNum and HeadDim are the per-layer row of one component; MLA and DSA have one head of
	// the row's width.
	HeadNum, HeadDim int
	// VHeadDim is the V row's head width (MHA), equal to HeadDim unless the model says otherwise.
	VHeadDim int
	// RowBytes overrides the per-token, per-layer row of a pool whose row is not heads × width
	// (DSA's packed index keys with their scales).
	RowBytes int
	LayerNum int
	ItemSize int
	// HeadGroupNum is page_unified's head partition (1 when there is none).
	HeadGroupNum int
}

// MHAHost is MHATokenToKVPoolHost's geometry (pool_host/mha.py:135-240): K and V rows of
// head_num heads, over the target layers and any packed draft layers.
func MHAHost(headNum, headDim, vHeadDim, layerNum, itemSize, headGroupNum int) HostShape {
	pool := "MHATokenToKVPoolHost"
	if headDim != vHeadDim {
		// get_mha_host_pool_cls (mha.py:1684-1699).
		pool = "AsymmetricMHATokenToKVPoolHost"
	}
	return HostShape{Pool: pool, Components: 2, HeadNum: headNum, HeadDim: headDim, VHeadDim: vHeadDim, LayerNum: layerNum,
		ItemSize: itemSize, HeadGroupNum: max(1, headGroupNum)}
}

// MLAHost is MLATokenToKVPoolHost's geometry (pool_host/mla.py:99-237): one latent row of
// kv_lora_rank + qk_rope_head_dim elements per layer, unless the device pool packs a wider one
// (overrideDim, the FP8 DSA row).
func MLAHost(kvLoraRank, qkRopeHeadDim, overrideDim, layerNum, itemSize int) HostShape {
	dim := kvLoraRank + qkRopeHeadDim
	if overrideDim > 0 {
		dim = overrideDim
	}
	return HostShape{Pool: "MLATokenToKVPoolHost", Components: 1, HeadNum: 1, HeadDim: dim, VHeadDim: 0, LayerNum: layerNum, ItemSize: itemSize, HeadGroupNum: 1}
}

// DSAIndexerBytesPerTokenPerLayer is dsa_indexer_bytes_per_token_per_layer (pool_host/dsa.py:71-76):
// the packed index keys and one fp32 scale per quant block, as uint8.
func DSAIndexerBytesPerTokenPerLayer(indexHeadDim, quantBlockSize int) int {
	return indexHeadDim + indexHeadDim/quantBlockSize*4
}

// DSAIndexerHost is DSAIndexerPoolHost's geometry (pool_host/dsa.py:113-260): the index keys of
// the layers that own an index buffer, riding on the KV host pool's slots. A device pool whose
// index page is smaller than its KV page (index_kpool > 1) stores its index row over the KV page
// (make_dsa_indexer_pool_decl, :113-140).
func DSAIndexerHost(indexHeadDim, quantBlockSize, pageSize, indexKPool, ownedLayers int) (HostShape, error) {
	if indexKPool <= 0 || pageSize%indexKPool != 0 {
		return HostShape{}, fmt.Errorf("page_size %d must be a multiple of index_kpool %d", pageSize, indexKPool)
	}
	indexPageBytes := pageSize / indexKPool * DSAIndexerBytesPerTokenPerLayer(indexHeadDim, quantBlockSize)
	if indexPageBytes%pageSize != 0 {
		return HostShape{}, fmt.Errorf("index page of %d bytes does not split over page_size=%d tokens", indexPageBytes, pageSize)
	}
	return HostShape{Pool: "DSAIndexerPoolHost", Components: 1, HeadNum: 1, RowBytes: indexPageBytes / pageSize, LayerNum: ownedLayers, ItemSize: 1,
		HeadGroupNum: 1}, nil
}

func (s HostShape) componentRow(c int) int64 {
	if s.RowBytes > 0 {
		return int64(s.RowBytes)
	}
	d := s.HeadDim
	if c == 1 && s.Components == 2 {
		d = s.VHeadDim
	}
	return int64(s.HeadNum * d * s.ItemSize)
}

// BytesPerTokenPerLayer is one token's row of one layer, every component.
func (s HostShape) BytesPerTokenPerLayer() int64 {
	var b int64
	for c := range s.Components {
		b += s.componentRow(c)
	}
	return b
}

// SizePerToken is get_size_per_token (mha.py:290-296, mla.py:301-302, dsa.py:261-262): the bytes
// one token takes on host, over every layer. It sizes the host pool (HostSizing.SizePerToken).
func (s HostShape) SizePerToken() int64 { return s.BytesPerTokenPerLayer() * int64(s.LayerNum) }

// TransferPrice is what a copy between the device and host pools moves, as the simulator prices it
// (design §6.1): the bytes, and the contiguous runs of the host buffer they fall into. The layout
// changes the runs, never the bytes.
type TransferPrice struct {
	Bytes int64
	// Runs is the contiguous host spans; RunBytes their mean size (K and V rows may differ).
	Runs     int64
	RunBytes int64
}

// CheckLayout is the host pool's own refusal of a layout it cannot lay out, raised when HiCache
// builds it at start-up: init_kv_buffer (mha.py:305-352, 1358-1381; mla.py:307-430;
// dsa.py:280-320@297e54fd) and PageUnifiedLayout's checks (page_unified.py:63-89).
func (s HostShape) CheckLayout(layout string) error {
	allowed := map[string][]string{
		"MHATokenToKVPoolHost":           Layouts,
		"AsymmetricMHATokenToKVPoolHost": {"page_first", "page_first_direct"},
		"MLATokenToKVPoolHost":           {"layer_first", "page_first", "page_first_direct", "page_unified"},
		"DSAIndexerPoolHost":             {"layer_first", "page_first", "page_first_direct"},
	}[s.Pool]
	if !slices.Contains(allowed, layout) {
		if s.Pool == "AsymmetricMHATokenToKVPoolHost" {
			return fmt.Errorf("Unsupported layout for models with head_dim != v_head_dim: %s; expected 'page_first' or 'page_first_direct'.", layout)
		}
		return fmt.Errorf("Unsupported layout: %s", layout)
	}
	if layout == "page_unified" {
		g := max(1, s.HeadGroupNum)
		if s.HeadNum%g != 0 {
			return fmt.Errorf("head_group_num=%d must divide the pool's %d kv heads", g, s.HeadNum)
		}
		if group := s.componentRow(0) / int64(g); group%16 != 0 {
			return fmt.Errorf("one head group's token row is %d bytes, which the page_unified transfer kernels cannot move: they require a multiple of 16", group)
		}
	}
	return nil
}

// hostRun is the contiguous host span, in bytes, that one copy unit of the layout covers when
// layers [0, layers) are copied, for page-aligned tokens, and the units per page. From each
// layout's buffer dims (mha.py:305-352, mla.py:307-345, page_unified.py:91-128).
func (s HostShape) hostRun(layout string, pageSize, layers int) (int64, int64, error) {
	row := s.componentRow(0)
	switch layout {
	case "layer_first":
		// (components, layer, token, row): a token's row of one layer and one component.
		return row, int64(pageSize * s.Components * layers), nil
	case "page_first":
		// (components, token, layer, row): a token's layers of one component.
		return row * int64(layers), int64(pageSize * s.Components), nil
	case "page_first_direct":
		// (components, page, layer, page_size, row): a page's layers of one component.
		return row * int64(pageSize*layers), int64(s.Components), nil
	case "page_head":
		if s.Components != 2 {
			return 0, 0, fmt.Errorf("hicache: the page_head layout is MHA's only")
		}
		// (components, page, head, page_size, layer, head_dim): one head's tokens and layers.
		hb := row / int64(s.HeadNum)
		if layers != s.LayerNum {
			// A layer's slice is strided: one head_dim run per (head, token).
			return hb, int64(s.Components * s.HeadNum * pageSize), nil
		}
		return hb * int64(pageSize*layers), int64(s.Components * s.HeadNum), nil
	case "page_unified":
		// (page, head group, layer, components, page_size, heads per group, head_dim): one head
		// group's layers, both components.
		g := max(1, s.HeadGroupNum)
		if s.HeadNum%g != 0 {
			return 0, 0, fmt.Errorf("head_group_num=%d must divide the pool's %d kv heads", g, s.HeadNum)
		}
		group := row / int64(g)
		return group * int64(s.Components*pageSize*layers), int64(g), nil
	}
	return 0, 0, fmt.Errorf("hicache: unsupported layout: %s", layout)
}

// Backup prices backup_from_device_all_layer: tokens' rows of every layer, device to host.
func (s HostShape) Backup(layout string, pageSize, tokens int) (TransferPrice, error) {
	return s.price(layout, pageSize, tokens, s.LayerNum)
}

// LoadLayer prices load_to_device_per_layer: tokens' rows of one layer, host to device. A load
// back copies LayerNum of these, one per layer, so a forward pass can start on layer 0.
func (s HostShape) LoadLayer(layout string, pageSize, tokens int) (TransferPrice, error) {
	return s.price(layout, pageSize, tokens, 1)
}

func (s HostShape) price(layout string, pageSize, tokens, layers int) (TransferPrice, error) {
	if pageSize <= 0 || tokens%pageSize != 0 {
		return TransferPrice{}, fmt.Errorf("hicache: %d tokens are not whole pages of %d", tokens, pageSize)
	}
	if err := s.CheckLayout(layout); err != nil {
		return TransferPrice{}, err
	}
	_, perPage, err := s.hostRun(layout, pageSize, layers)
	if err != nil {
		return TransferPrice{}, err
	}
	bytes := s.BytesPerTokenPerLayer() * int64(tokens*layers)
	runs := perPage * int64(tokens/pageSize)
	runBytes := int64(0)
	if runs > 0 {
		runBytes = bytes / runs
	}
	return TransferPrice{Bytes: bytes, Runs: runs, RunBytes: runBytes}, nil
}
