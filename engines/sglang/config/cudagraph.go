// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/model_executor/cuda_graph_config.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package config

import "encoding/json"

// The capture backends a phase can use (cuda_graph_config.py:40-49@297e54fd).
const (
	backendFull      = "full"
	backendBreakable = "breakable"
	backendDisabled  = "disabled"
)

// PhaseConfig is one phase's CUDA-graph settings (cuda_graph_config.py:89-116@297e54fd).
type PhaseConfig struct {
	Backend                      string
	MaxBs                        *int
	Bs                           []int
	TcCompiler                   string
	MaxContextSize               *int
	FullPrefillMaxReq            *int
	FullPrefillPrefixChunkTokens *int
	MaxSeqLen                    *int
}

// CudaGraphConfig is the decode and prefill phases' settings (cuda_graph_config.py:145-193).
type CudaGraphConfig struct {
	Decode  PhaseConfig
	Prefill PhaseConfig
}

// defaultCudaGraphConfig is CudaGraphConfig() on the CUDA platform: decode captures full graphs,
// prefill breakable ones (default_prefill_backend, cuda_graph_config.py:119-123, is breakable
// when is_cuda()). Resolve models only the CUDA platform.
func defaultCudaGraphConfig() *CudaGraphConfig {
	return &CudaGraphConfig{
		Decode:  PhaseConfig{Backend: backendFull, TcCompiler: "eager"},
		Prefill: PhaseConfig{Backend: backendBreakable, TcCompiler: "eager"},
	}
}

func (p PhaseConfig) clone() PhaseConfig {
	q := p
	q.MaxBs = clonePtr(p.MaxBs)
	if p.Bs != nil {
		q.Bs = append([]int{}, p.Bs...)
	}
	q.MaxContextSize = clonePtr(p.MaxContextSize)
	q.FullPrefillMaxReq = clonePtr(p.FullPrefillMaxReq)
	q.FullPrefillPrefixChunkTokens = clonePtr(p.FullPrefillPrefixChunkTokens)
	q.MaxSeqLen = clonePtr(p.MaxSeqLen)
	return q
}

func (c *CudaGraphConfig) clone() *CudaGraphConfig {
	if c == nil {
		return nil
	}
	return &CudaGraphConfig{Decode: c.Decode.clone(), Prefill: c.Prefill.clone()}
}

// withPhase is with_phase (cuda_graph_config.py:126-142): a copy of c with one phase changed.
func (c *CudaGraphConfig) withPhase(phase string, change func(*PhaseConfig)) *CudaGraphConfig {
	d := c.clone()
	if phase == "decode" {
		change(&d.Decode)
	} else {
		change(&d.Prefill)
	}
	return d
}

// diff is _diff_phase (cuda_graph_config.py:201-207): the fields that differ from the default.
func (p PhaseConfig) diff(base PhaseConfig) map[string]any {
	out := map[string]any{}
	if p.Backend != base.Backend {
		out["backend"] = p.Backend
	}
	if !equalJSON(p.MaxBs, base.MaxBs) {
		out["max_bs"] = p.MaxBs
	}
	if !equalJSON(p.Bs, base.Bs) {
		out["bs"] = p.Bs
	}
	if p.TcCompiler != base.TcCompiler {
		out["tc_compiler"] = p.TcCompiler
	}
	if !equalJSON(p.MaxContextSize, base.MaxContextSize) {
		out["max_context_size"] = p.MaxContextSize
	}
	if !equalJSON(p.FullPrefillMaxReq, base.FullPrefillMaxReq) {
		out["full_prefill_max_req"] = p.FullPrefillMaxReq
	}
	if !equalJSON(p.FullPrefillPrefixChunkTokens, base.FullPrefillPrefixChunkTokens) {
		out["full_prefill_prefix_chunk_tokens"] = p.FullPrefillPrefixChunkTokens
	}
	if !equalJSON(p.MaxSeqLen, base.MaxSeqLen) {
		out["max_seq_len"] = p.MaxSeqLen
	}
	return out
}

// MarshalJSON writes SGLang's own form of the configuration, to_dict (cuda_graph_config.py:
// 160-168): each phase's fields that differ from the default.
func (c *CudaGraphConfig) MarshalJSON() ([]byte, error) {
	base := defaultCudaGraphConfig()
	return json.Marshal(map[string]any{
		"decode":  c.Decode.diff(base.Decode),
		"prefill": c.Prefill.diff(base.Prefill),
	})
}

func equalJSON(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

// generatePrefillBatchSizes is generate_prefill_cuda_graph_batch_sizes
// (arg_groups/cuda_graph_hook.py:480-497@297e54fd): the token counts prefill graphs capture.
func generatePrefillBatchSizes(maxBs int) []int {
	var sizes []int
	add := func(lo, hi, step int) {
		for s := lo; s < hi; s += step {
			sizes = append(sizes, s)
		}
	}
	add(4, 33, 4)
	add(48, 257, 16)
	add(288, 513, 32)
	add(576, 1024+1, 64)
	add(1280, 4096+1, 256)
	add(4608, maxBs+1, 512)
	out := []int{}
	for _, s := range sizes {
		if s <= maxBs {
			out = append(out, s)
		}
	}
	return out
}

// generateDecodeBatchSizes is generate_decode_cuda_graph_batch_sizes
// (arg_groups/cuda_graph_hook.py:500-532@297e54fd).
func generateDecodeBatchSizes(disablePadding, speculative bool, maxBs int) []int {
	var sizes []int
	add := func(lo, hi, step int) {
		for s := lo; s < hi; s += step {
			sizes = append(sizes, s)
		}
	}
	switch {
	case disablePadding:
		add(1, maxBs+1, 1)
	case !speculative:
		sizes = append(sizes, 1, 2, 4, 8, 12)
		add(16, 257, 8)
		add(272, 512, 16)
		add(512, maxBs+1, 32)
	default:
		add(1, 9, 1)
		add(10, 33, 2)
		add(40, 65, 4)
		add(72, 257, 8)
		add(272, maxBs+1, 16)
	}
	out := []int{}
	for _, s := range sizes {
		if s <= maxBs {
			out = append(out, s)
		}
	}
	found := false
	for _, s := range out {
		found = found || s == maxBs
	}
	if !found {
		out = append(out, maxBs)
	}
	return out
}
