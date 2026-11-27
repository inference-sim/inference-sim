// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/memory_pool.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/index_key_cache.py,
// python/sglang/srt/model_executor/pool_configurator.py (DefaultPoolConfigurator._compute_cell_size
// and _compute_dsa_indexer_cell_size) and python/sglang/srt/mem_cache/kv_cache_configurator.py
// (calculate_mla_kv_cache_dim).

package cache

import (
	"fmt"
	"strings"
)

// DType is a KV cache dtype by its torch name, without the "torch." prefix.
type DType string

// The KV cache dtypes a modelled pool stores. FP4 and MXFP8 select their own pool classes, which
// are refused (refuse.go).
const (
	BFloat16   DType = "bfloat16"
	Float16    DType = "float16"
	Float32    DType = "float32"
	Float8E4M3 DType = "float8_e4m3fn"
	Float8E5M2 DType = "float8_e5m2"
)

var itemSize = map[DType]int{BFloat16: 2, Float16: 2, Float32: 4, Float8E4M3: 1, Float8E5M2: 1, "uint8": 1}

// ParseDType reads a dtype as the trace records it ("torch.bfloat16") or bare.
func ParseDType(s string) (DType, error) {
	d := DType(strings.TrimPrefix(s, "torch."))
	if _, ok := itemSize[d]; !ok || d == "uint8" {
		return "", fmt.Errorf("KV cache dtype %q is not one a modelled pool stores", s)
	}
	return d, nil
}

// ItemSize is torch._utils._element_size of the dtype.
func (d DType) ItemSize() int { return itemSize[d] }

// Store is the dtype the buffers hold: FP8 is stored as uint8, because index_put is not
// implemented for it (memory_pool.py:2029-2033@297e54fd).
func (d DType) Store() DType {
	if d == Float8E4M3 || d == Float8E5M2 {
		return "uint8"
	}
	return d
}

// MHAParams is MHATokenToKVPool's constructor arguments that size it (memory_pool.py:2187-2209
// @297e54fd). HeadNum is the KV heads on this rank. VHeadDim nil means HeadDim.
type MHAParams struct {
	Size, PageSize    int
	DType             DType
	HeadNum, HeadDim  int
	LayerNum          int
	VHeadDim          *int
	PostCaptureActive bool
}

// MHAPool is MHATokenToKVPool (memory_pool.py:2184@297e54fd) as sizing, in the NHD layout: a K
// and a V buffer per layer, each of size+page_size rows (the padded page holds slot 0).
type MHAPool struct {
	size, pageSize, layerNum int
	DType                    DType
	HeadNum, HeadDim         int
	VHeadDim                 int
}

// NewMHAPool is MHATokenToKVPool.__init__ (memory_pool.py:2187-2259@297e54fd) as sizing.
func NewMHAPool(p MHAParams) (*MHAPool, error) {
	if p.PageSize < 1 || p.Size < 0 || p.LayerNum < 1 || p.HeadNum < 1 || p.HeadDim < 1 {
		return nil, fmt.Errorf("MHATokenToKVPool: size %d, page size %d, %d layers, %d heads of %d",
			p.Size, p.PageSize, p.LayerNum, p.HeadNum, p.HeadDim)
	}
	if _, ok := itemSize[p.DType]; !ok {
		return nil, fmt.Errorf("MHATokenToKVPool: dtype %q", p.DType)
	}
	if p.PostCaptureActive {
		// Post-capture sizing backs the pool with CUDA virtual memory after graph capture
		// (memory_pool.py:2216-2220, 2700-2750); it is not modelled.
		return nil, fmt.Errorf("MHATokenToKVPool with post_capture_active is not modelled; refused until it is (%s)",
			RefusedPoolsIssue)
	}
	size := p.Size
	v := p.HeadDim
	if p.VHeadDim != nil {
		v = *p.VHeadDim
	}
	return &MHAPool{size: size, pageSize: p.PageSize, layerNum: p.LayerNum, DType: p.DType,
		HeadNum: p.HeadNum, HeadDim: p.HeadDim, VHeadDim: v}, nil
}

func (m *MHAPool) Size() int     { return m.size }
func (m *MHAPool) PageSize() int { return m.pageSize }
func (m *MHAPool) LayerNum() int { return m.layerNum }

// BytesPerToken is the cell size DefaultPoolConfigurator charges an MHA token
// (pool_configurator.py:491-498@297e54fd): K and V heads over every layer.
func (m *MHAPool) IDSpaces() []string { return []string{"FULL"} }

func (m *MHAPool) BytesPerToken() int64 {
	return int64(m.HeadNum) * int64(m.HeadDim+m.VHeadDim) * int64(m.layerNum) * int64(m.DType.ItemSize())
}

// KVSizeBytes is get_kv_size_bytes (memory_pool.py:2753-2767@297e54fd), K and V together.
func (m *MHAPool) KVSizeBytes() (k, v int64) {
	rows := int64(m.size + m.pageSize)
	item := int64(m.DType.Store().ItemSize())
	layers := int64(m.layerNum)
	return layers * rows * int64(m.HeadNum*m.HeadDim) * item, layers * rows * int64(m.HeadNum*m.VHeadDim) * item
}

// MLAParams is MLATokenToKVPool's constructor arguments that size it (memory_pool.py:4755-4770
// @297e54fd).
type MLAParams struct {
	Size, PageSize            int
	DType                     DType
	KVLoraRank, QKRopeHeadDim int
	LayerNum                  int
	UseDSA                    bool
	OverrideKVCacheDim        *int
}

// MLAPool is MLATokenToKVPool (memory_pool.py:4754@297e54fd) as sizing: one latent buffer per
// layer of size+page_size rows of KVCacheDim.
type MLAPool struct {
	size, pageSize, layerNum  int
	DType                     DType
	KVLoraRank, QKRopeHeadDim int
	KVCacheDim                int
}

// NewMLAPool is MLATokenToKVPool.__init__ (memory_pool.py:4755-4810@297e54fd) as sizing.
func NewMLAPool(p MLAParams) (*MLAPool, error) {
	if p.PageSize < 1 || p.Size < 0 || p.LayerNum < 1 || p.KVLoraRank < 1 || p.QKRopeHeadDim < 0 {
		return nil, fmt.Errorf("MLATokenToKVPool: size %d, page size %d, %d layers, rank %d, rope %d",
			p.Size, p.PageSize, p.LayerNum, p.KVLoraRank, p.QKRopeHeadDim)
	}
	if _, ok := itemSize[p.DType]; !ok {
		return nil, fmt.Errorf("MLATokenToKVPool: dtype %q", p.DType)
	}
	dim := p.KVLoraRank + p.QKRopeHeadDim
	// An override is used only for DSA's packed FP8 rows (memory_pool.py:4786-4797).
	if p.UseDSA && p.DType == Float8E4M3 && p.OverrideKVCacheDim != nil {
		dim = *p.OverrideKVCacheDim
	}
	return &MLAPool{size: p.Size, pageSize: p.PageSize, layerNum: p.LayerNum, DType: p.DType,
		KVLoraRank: p.KVLoraRank, QKRopeHeadDim: p.QKRopeHeadDim, KVCacheDim: dim}, nil
}

func (m *MLAPool) Size() int     { return m.size }
func (m *MLAPool) PageSize() int { return m.pageSize }
func (m *MLAPool) LayerNum() int { return m.layerNum }

// BytesPerToken is the cell size DefaultPoolConfigurator charges an MLA token
// (pool_configurator.py:379-386@297e54fd): the latent row over every layer, at the KV dtype's
// element size. The configurator's row is calculate_mla_kv_cache_dim, which is the KVCacheDim
// the builder hands the pool.
func (m *MLAPool) IDSpaces() []string { return []string{"FULL"} }

func (m *MLAPool) BytesPerToken() int64 {
	return int64(m.KVCacheDim) * int64(m.layerNum) * int64(m.DType.ItemSize())
}

// KVSizeBytes is get_kv_size_bytes (memory_pool.py:4876-4881@297e54fd).
func (m *MLAPool) KVSizeBytes() int64 {
	return int64(m.layerNum) * int64(m.size+m.pageSize) * int64(m.KVCacheDim) * int64(m.DType.Store().ItemSize())
}

// MLAKVCacheDim is calculate_mla_kv_cache_dim (kv_cache_configurator.py:3412-3475@297e54fd): the
// latent row width. A DSA model stores FP8 rows packed with per-block scales and a bf16 rope
// part, unless its attention backends read the raw layout (trtllm on the local role, TileLang
// on both, or the HIP kernels), which rawLayout says.
func MLAKVCacheDim(kvLoraRank, qkRopeHeadDim int, dtype DType, isDSA, rawLayout bool) (int, error) {
	dim := kvLoraRank + qkRopeHeadDim
	if !isDSA || rawLayout || dtype != Float8E4M3 {
		return dim, nil
	}
	if kvLoraRank%dsaQuantBlockSize != 0 {
		return 0, fmt.Errorf("kv_lora_rank %d must be multiple of quant_block_size %d", kvLoraRank, dsaQuantBlockSize)
	}
	return kvLoraRank + kvLoraRank/dsaQuantBlockSize*4 + qkRopeHeadDim*BFloat16.ItemSize(), nil
}

// DSATokenToKVPool's class constants (memory_pool.py:5280-5282@297e54fd): the indexer keys are
// FP8 in blocks of 128 with a 4-byte scale each, stored as uint8.
const (
	dsaQuantBlockSize = 128
	dsaIndexPageSize  = 64
)

// DSAParams is DSATokenToKVPool's constructor arguments that size it (memory_pool.py:5284-5305
// @297e54fd). KVCacheDim is what the builder passes, calculate_mla_kv_cache_dim. IndexBufSize nil
// is the default, the pool's size. SkipTopKLayers marks the layers that reuse the previous
// layer's top-k and so hold no index keys; nil is none.
type DSAParams struct {
	Size, PageSize            int
	DType                     DType
	KVLoraRank, QKRopeHeadDim int
	LayerNum                  int
	IndexHeadDim              int
	KVCacheDim                int
	IndexBufSize              *int
	IndexKPool                int
	SkipTopKLayers            []bool
}

// DSAPool is DSATokenToKVPool (memory_pool.py:5279@297e54fd) as sizing: the MLA latent buffers
// and the indexer key cache (index_key_cache.py:21@297e54fd), one buffer per layer that computes
// its own top-k.
type DSAPool struct {
	MLAPool
	IndexHeadDim, IndexKPool, IndexBufSize int
	SkipTopKLayers                         []bool
}

// NewDSAPool is DSATokenToKVPool.__init__ (memory_pool.py:5284-5385@297e54fd) as sizing, on the
// CUDA path: 64-token index pages.
func NewDSAPool(p DSAParams) (*DSAPool, error) {
	var override *int
	if p.KVCacheDim != p.KVLoraRank+p.QKRopeHeadDim {
		d := p.KVCacheDim
		override = &d
	}
	mla, err := NewMLAPool(MLAParams{Size: p.Size, PageSize: p.PageSize, DType: p.DType, KVLoraRank: p.KVLoraRank,
		QKRopeHeadDim: p.QKRopeHeadDim, LayerNum: p.LayerNum, UseDSA: true, OverrideKVCacheDim: override})
	if err != nil {
		return nil, fmt.Errorf("DSATokenToKVPool: %w", err)
	}
	kpool := p.IndexKPool
	if kpool == 0 {
		kpool = 1
	}
	if p.PageSize%kpool != 0 {
		return nil, fmt.Errorf("page_size %d must be a multiple of index_kpool %d", p.PageSize, kpool)
	}
	if p.IndexHeadDim != 128 {
		return nil, fmt.Errorf("DSATokenToKVPool: index_head_dim %d; DSA's index keys are one head of 128", p.IndexHeadDim)
	}
	if p.PageSize/kpool != dsaIndexPageSize {
		return nil, fmt.Errorf("DSA requires 64-token physical pages, got page_size=%d with index_kpool=%d", p.PageSize, kpool)
	}
	buf := p.Size
	if p.IndexBufSize != nil {
		buf = *p.IndexBufSize
	}
	skip := p.SkipTopKLayers
	if skip == nil {
		skip = make([]bool, p.LayerNum)
	}
	if len(skip) != p.LayerNum {
		return nil, fmt.Errorf("DSATokenToKVPool: %d skip_topk_layers for %d layers", len(skip), p.LayerNum)
	}
	return &DSAPool{MLAPool: *mla, IndexHeadDim: p.IndexHeadDim, IndexKPool: kpool, IndexBufSize: buf,
		SkipTopKLayers: append([]bool(nil), skip...)}, nil
}

// indexRowBytes is one token's index key and its scales (pool_configurator.py:566-569@297e54fd,
// index_key_cache.py:40-47).
func (d *DSAPool) indexRowBytes() int {
	return d.IndexHeadDim + d.IndexHeadDim/dsaQuantBlockSize*4
}

func (d *DSAPool) indexerLayers() int {
	n := 0
	for _, s := range d.SkipTopKLayers {
		if !s {
			n++
		}
	}
	return n
}

// BytesPerToken is the MLA cell plus the indexer cell, as DefaultPoolConfigurator charges a DSA
// token (pool_configurator.py:399-409, 560-635@297e54fd): one index row per layer that holds
// index keys, shared by index_kpool tokens.
func (d *DSAPool) IDSpaces() []string { return []string{"FULL"} }

func (d *DSAPool) BytesPerToken() int64 {
	indexer := int64(ceilDiv(d.indexRowBytes(), d.IndexKPool)) * int64(d.indexerLayers())
	return d.MLAPool.BytesPerToken() + indexer
}

// KVSizeBytes is get_kv_size_bytes (memory_pool.py:5659-5663@297e54fd): the latent buffers and
// the index key buffers, whose pages hold page_size/index_kpool rows each
// (index_key_cache.py:22-47).
func (d *DSAPool) KVSizeBytes() int64 {
	pages := int64((d.IndexBufSize + d.pageSize + 1) / d.pageSize)
	row := int64(d.pageSize / d.IndexKPool * d.indexRowBytes())
	return d.MLAPool.KVSizeBytes() + pages*row*int64(d.indexerLayers())
}
