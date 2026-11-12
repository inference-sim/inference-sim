// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/platforms/interface.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// Platform._align_hybrid_block_size (:939-1126), which update_block_size_for_backend runs for a
// hybrid attention/Mamba model after the attention backends' block size is chosen (:750-794).

package vllm

import (
	"fmt"
)

// HybridBlockInput is what _align_hybrid_block_size reads. The caller states the backend
// facts; the function is the arithmetic.
type HybridBlockInput struct {
	// Block is cache_config.block_size when the alignment starts: the stated block_size, or the
	// size every attention backend accepts (update_block_size_for_backend phase 1,
	// interface.py:771-783@5df63a72).
	Block int
	// AttnPage1Token is the attention page of one token on a rank: a FullAttentionSpec (or, for
	// an MLA model, MLAAttentionSpec) of block_size 1 (interface.py:970-1038@5df63a72), in bytes.
	AttnPage1Token int64
	// MambaPage is the Mamba page, MambaSpec(block_size=-1).page_size_bytes: the state tensors'
	// bytes (interface.py:1040-1056@5df63a72). Zero returns the input unchanged (:1057-1058).
	MambaPage int64
	// KernelBlockMin is the smallest kernel block of the first attention backend
	// (get_supported_kernel_block_sizes, a MultipleOf's base), and BackendBlockLCM the least
	// common multiple of every attention backend's smallest (interface.py:1060-1077@5df63a72).
	KernelBlockMin, BackendBlockLCM int
	// MLA is model_config.use_mla: the kernel block alignment is at least 128 so a TRTLLM or
	// FlashInfer MLA decode kernel's physical blocks stay aligned (interface.py:1078-1082).
	MLA bool
	// MambaCacheMode is the mode that runs, "align" or "none". In align mode mamba_block_size is
	// the aligned block size (interface.py:1102-1103). In none mode it stays what
	// MambaModelConfig set, MambaBlockSize (max_model_len).
	MambaCacheMode string
	MambaBlockSize int
}

// HybridBlock is the engine settings _align_hybrid_block_size leaves.
type HybridBlock struct {
	// BlockSize is cache_config.block_size: the attention block, no smaller than the input's.
	BlockSize int
	// MambaBlockSize is cache_config.mamba_block_size.
	MambaBlockSize int
	// PageSizePadded is cache_config.mamba_page_size_padded: the attention page, to which the
	// Mamba page is padded so the two are equal; zero when they already are (:1106-1126).
	PageSizePadded int64
}

// AlignHybridBlockSize is _align_hybrid_block_size (vllm/platforms/interface.py:939-1126@5df63a72):
// the smallest attention block size, a multiple of the kernel block alignment and of every
// backend's smallest kernel block, whose attention page holds one Mamba page; the engine's block
// size is that or the input's, whichever is larger.
func AlignHybridBlockSize(in HybridBlockInput) (HybridBlock, error) {
	out := HybridBlock{BlockSize: in.Block, MambaBlockSize: in.MambaBlockSize}
	if in.MambaCacheMode == "align" {
		out.MambaBlockSize = in.Block
	}
	switch {
	case in.Block <= 0:
		return HybridBlock{}, fmt.Errorf("vllm: hybrid block alignment: block size %d", in.Block)
	case in.MambaPage < 0 || in.AttnPage1Token <= 0:
		return HybridBlock{}, fmt.Errorf("vllm: hybrid block alignment: attention page %d bytes a token and mamba page %d bytes",
			in.AttnPage1Token, in.MambaPage)
	case in.KernelBlockMin <= 0 || in.BackendBlockLCM <= 0:
		return HybridBlock{}, fmt.Errorf("vllm: hybrid block alignment: kernel block minimum %d and backend lcm %d",
			in.KernelBlockMin, in.BackendBlockLCM)
	case in.MambaCacheMode != "align" && in.MambaCacheMode != "none":
		return HybridBlock{}, fmt.Errorf("vllm: hybrid block alignment: mamba_cache_mode %q", in.MambaCacheMode)
	}
	if in.MambaPage == 0 {
		return out, nil
	}
	align := int64(max(in.KernelBlockMin, in.Block))
	if in.MLA {
		align = max(align, 128)
	}
	attn := align * hybridCeilDiv(in.MambaPage, align*in.AttnPage1Token)
	lcm := int64(in.BackendBlockLCM)
	attn = lcm * hybridCeilDiv(attn, lcm)
	if int64(in.Block) < attn {
		out.BlockSize = int(attn)
	}
	if in.MambaCacheMode == "align" {
		out.MambaBlockSize = out.BlockSize
	}
	if page := int64(out.BlockSize) * in.AttnPage1Token; page != in.MambaPage {
		out.PageSizePadded = page
	}
	return out, nil
}

func hybridCeilDiv(a, b int64) int64 { return (a + b - 1) / b }
