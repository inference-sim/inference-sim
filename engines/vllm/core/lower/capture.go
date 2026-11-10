// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/vllm.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// VllmConfig._set_cudagraph_sizes (:2292-2548) and DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES
// (:74-100). Also vllm/config/compilation.py (CUDAGraphMode, :53-110).

// Package lower is the engine's half of the boundary with the cost model (design section 6,
// SUB-6): the decisions vLLM makes about a launch, which the cost model must receive and not
// make. It resolves the CUDA-graph capture sizes, pads a batch to the graph the runner would
// replay, synchronises the padding across data-parallel ranks, and sizes the KV groups a plan
// states, all as vLLM v0.32.0 does. Its outputs are PoolPlan and StepShape fields.
//
// Each rule here is the pinned release's own, and each has a test against that release's code
// run unmodified (oracle/lower/extract.py), except where a function says it has none.
package lower

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// GraphMode is CUDAGraphMode (vllm/config/compilation.py:53-110). The first three are runtime
// modes; the last two name a decode mode and a mixed mode together.
type GraphMode int

const (
	GraphNone GraphMode = iota
	GraphPiecewise
	GraphFull
	GraphFullDecodeOnly
	GraphFullAndPiecewise
)

var graphModeNames = []string{"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}

func (m GraphMode) String() string { return graphModeNames[m] }

// ParseGraphMode is the mode a cudagraph_mode setting names.
func ParseGraphMode(s string) (GraphMode, error) {
	for i, n := range graphModeNames {
		if strings.EqualFold(s, n) {
			return GraphMode(i), nil
		}
	}
	return 0, fmt.Errorf("lower: cudagraph_mode %q is not one of %s", s, strings.Join(graphModeNames, ", "))
}

// separate is CUDAGraphMode.separate_routine: a tuple mode, one routine for decode and one for
// every other batch.
func (m GraphMode) separate() bool { return m == GraphFullDecodeOnly || m == GraphFullAndPiecewise }

// decode is CUDAGraphMode.decode_mode.
func (m GraphMode) decode() GraphMode {
	switch m {
	case GraphFullDecodeOnly, GraphFullAndPiecewise:
		return GraphFull
	}
	return m
}

// mixed is CUDAGraphMode.mixed_mode.
func (m GraphMode) mixed() GraphMode {
	switch m {
	case GraphFullDecodeOnly:
		return GraphNone
	case GraphFullAndPiecewise:
		return GraphPiecewise
	}
	return m
}

// CaptureInput is what VllmConfig._set_cudagraph_sizes reads.
type CaptureInput struct {
	// Mode is compilation_config.cudagraph_mode, after the attention backend's override.
	Mode GraphMode
	// EnforceEager is model_config.enforce_eager.
	EnforceEager bool
	// MaxCaptureSize and Sizes are compilation_config.max_cudagraph_capture_size and
	// cudagraph_capture_sizes; nil is unstated.
	MaxCaptureSize *int
	Sizes          []int
	// MaxNumSeqs and MaxNumBatchedTokens are the scheduler's resolved limits.
	MaxNumSeqs, MaxNumBatchedTokens int
	// SpecTokens is num_speculative_tokens, 0 without speculation. Dynamic speculative decoding,
	// which gives each batch size its own draft length, is not modelled.
	SpecTokens int
	// Blackwell is current_platform.is_device_capability_family(100): the platform default ceiling
	// is 1024 and not 512.
	Blackwell bool
	// Interactivity is performance_mode == "interactivity", which captures every size to 32.
	Interactivity bool
}

// Capture is the resolved compilation_config.max_cudagraph_capture_size and
// cudagraph_capture_sizes.
type Capture struct {
	Max   int
	Sizes []int
}

// CaptureSizes is VllmConfig._set_cudagraph_sizes (vllm/config/vllm.py:2292-2548@5df63a72).
// Where it asserts or raises, it returns an error with the same condition.
func CaptureSizes(in CaptureInput) (Capture, error) {
	if in.EnforceEager || in.Mode == GraphNone {
		return Capture{Sizes: []int{}}, nil
	}
	ceiling := 0
	var uniformDecode []int
	if in.MaxCaptureSize != nil {
		ceiling = *in.MaxCaptureSize
	} else {
		platform := 512
		if in.Blackwell {
			platform = 1024
		}
		dql := 1 + in.SpecTokens
		ceiling = min(in.MaxNumSeqs*dql*2, platform)
		if dql > 1 {
			// request_counts (:2392-2402): a request-count ladder, scaled by the query length.
			counts := func(maxReqs int) []int {
				maxReqs = min(maxReqs, platform)
				var c []int
				for _, n := range []int{1, 2, 4} {
					if n <= maxReqs {
						c = append(c, n)
					}
				}
				for n := 8; n < min(maxReqs+1, 256); n += 8 {
					c = append(c, n)
				}
				for n := 256; n <= maxReqs; n += 16 {
					c = append(c, n)
				}
				return dedupSorted(append(c, maxReqs))
			}
			for _, n := range counts(in.MaxNumSeqs) {
				if n*dql <= ceiling {
					uniformDecode = append(uniformDecode, n*dql)
				}
			}
			uniformDecode = dedupSorted(uniformDecode)
		} else if in.MaxNumSeqs <= ceiling {
			uniformDecode = []int{in.MaxNumSeqs}
		}
	}
	ceiling = min(in.MaxNumBatchedTokens, ceiling)
	if ceiling < 1 {
		return Capture{}, fmt.Errorf("lower: the maximum cudagraph size is %d; it is at least 1 when graphs are on "+
			"(vllm/config/vllm.py:2448)", ceiling)
	}
	var sizes []int
	if in.Sizes != nil {
		if len(in.Sizes) == 0 {
			return Capture{}, fmt.Errorf("lower: cudagraph_capture_sizes is empty; it holds at least one size when graphs are on " +
				"(vllm/config/vllm.py:2454)")
		}
		for _, s := range dedupSorted(append([]int(nil), in.Sizes...)) {
			if s <= in.MaxNumBatchedTokens {
				sizes = append(sizes, s)
			}
		}
	} else {
		if in.Interactivity {
			for i := 1; i <= min(ceiling, 32); i++ {
				sizes = append(sizes, i)
			}
		} else {
			for _, s := range []int{1, 2, 4} {
				if s <= ceiling {
					sizes = append(sizes, s)
				}
			}
		}
		if ceiling >= 8 {
			for s := 8; s < min(ceiling+1, 256); s += 8 {
				sizes = append(sizes, s)
			}
		}
		if ceiling >= 256 {
			for s := 256; s <= ceiling; s += 16 {
				sizes = append(sizes, s)
			}
		}
		if in.MaxNumBatchedTokens <= ceiling && !slices.Contains(sizes, in.MaxNumBatchedTokens) {
			sizes = append(sizes, in.MaxNumBatchedTokens)
		}
		for _, s := range uniformDecode {
			if s <= in.MaxNumBatchedTokens {
				sizes = append(sizes, s)
			}
		}
		sizes = dedupSorted(sizes)
	}
	valid := 0
	if len(sizes) > 0 {
		valid = sizes[len(sizes)-1]
	}
	if in.MaxCaptureSize != nil && *in.MaxCaptureSize != valid && in.Sizes != nil {
		return Capture{}, fmt.Errorf("lower: max_cudagraph_capture_size %d is not the largest of cudagraph_capture_sizes (%d); "+
			"vLLM refuses the pair at start-up (vllm/config/vllm.py:2511-2517)", *in.MaxCaptureSize, valid)
	}
	if sizes == nil {
		sizes = []int{}
	}
	return Capture{Max: valid, Sizes: sizes}, nil
}

func dedupSorted(x []int) []int {
	sort.Ints(x)
	return slices.Compact(x)
}

// Breakable is whether an architecture runs breakable CUDA graphs by default on CUDA
// (default_breakable_cudagraph_architectures, vllm/config/vllm.py:104-121@5df63a72;
// DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES, :74-100). Such a piecewise replay runs one piece for
// each eager region, which the engine states in the plan and the cost model prices.
func Breakable(architecture string) bool {
	_, ok := breakable[architecture]
	return ok
}

// BreakableArchitectures is the list, sorted.
func BreakableArchitectures() []string {
	out := make([]string, 0, len(breakable))
	for a := range breakable {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

var breakable = map[string]struct{}{
	"DeepseekV32MTPModel": {}, "DeepseekV32ForCausalLM": {}, "DeepseekV4ForCausalLM": {},
	"DeepseekV4ForConditionalGeneration": {}, "DeepSeekV4MTPModel": {}, "DeepseekV41ForCausalLM": {},
	"Dots3NoteForCausalLM": {}, "Dots3NoteMTPModel": {}, "Glm5NextForCausalLM": {},
	"Glm5NextForConditionalGeneration": {}, "Glm5NextMTPModel": {}, "GlmMoeDsaForCausalLM": {},
	"HYV4ForCausalLM": {}, "HYV4MTPModel": {}, "InklingForCausalLM": {},
	"InklingForConditionalGeneration": {}, "KimiK3ForConditionalGeneration": {}, "KimiK3MTPModel": {},
	"KimiLinearForCausalLM": {}, "MiniMaxM3SparseForCausalLM": {}, "MiniMaxM3SparseForConditionalGeneration": {},
	"Qwen4ExpForCausalLM": {}, "Qwen4ExpForConditionalGeneration": {}, "Qwen4ExpMTP": {},
}
