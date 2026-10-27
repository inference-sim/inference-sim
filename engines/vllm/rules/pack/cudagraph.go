// Moved from inference-sim/blis-schemas/rules/cudagraph.go at 70246460a379f8b1898fbf6c36a3091364118950 (CUDAGraphCaptureSizes).

package pack

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-schemas/rules"
)

// CUDAGraphCaptureSizes resolves the CUDA graph capture a deployment's stated knobs give,
// as both registered releases' VllmConfig._set_cudagraph_sizes does (v0.31.0
// vllm/config/vllm.py:2388-2660, v0.29.0 :1940-2210). It reports false when the
// deployment states neither max_cudagraph_capture_size nor cudagraph_capture_sizes and
// leaves cudagraph_mode on: the release default then depends on the platform and the
// decode query length, and is the consumer's to compute.
//
//   - cudagraph_mode NONE captures nothing.
//   - Stated sizes are de-duplicated, those above max_num_batched_tokens dropped, and
//     sorted. A stated maximum must equal the largest kept, or the engine refuses it.
//   - A stated maximum alone is capped at max_num_batched_tokens and expanded to 1, 2, 4,
//     every multiple of 8 below 256 and every multiple of 16 from 256, up to it, with
//     max_num_batched_tokens added when it fits. The maximum that runs is the largest of
//     those, which is below the stated one when it falls off that stride.
//
// enforce_eager disables capture too, as cudagraph_mode NONE does (_set_cudagraph_sizes
// tests both), but the schema has no field for it, so a consumer pricing an eager
// deployment must not read these sizes.
//
// It needs max_num_batched_tokens, which the engine otherwise derives. It does not model
// performance_mode interactivity, which captures every size up to 32, or the
// sequence-parallel rounding the engine applies under tensor parallelism with SP enabled;
// the schema states neither.
func (p *Pack) CUDAGraphCaptureSizes(in rules.CUDAGraphCaptureInput) (rules.CUDAGraphCapture, bool, error) {
	if in.Mode == "NONE" {
		return rules.CUDAGraphCapture{Sizes: []int{}}, true, nil
	}
	if in.MaxCaptureSize == nil && in.CaptureSizes == nil {
		return rules.CUDAGraphCapture{}, false, nil
	}
	mnbt := in.MaxNumBatchedTokens
	if mnbt < 1 {
		return rules.CUDAGraphCapture{}, true, fmt.Errorf("resolving CUDA graph capture sizes needs max_num_batched_tokens, which the deployment does not state")
	}
	var out rules.CUDAGraphCapture
	if in.CaptureSizes != nil {
		seen := map[int]bool{}
		for _, s := range in.CaptureSizes {
			if seen[s] {
				continue
			}
			seen[s] = true
			if s <= mnbt {
				out.Sizes = append(out.Sizes, s)
			} else {
				out.Dropped = append(out.Dropped, s)
			}
		}
		sort.Ints(out.Sizes)
		sort.Ints(out.Dropped)
	} else {
		m := *in.MaxCaptureSize
		if m > mnbt {
			m, out.Capped = mnbt, true
		}
		set := map[int]bool{}
		for _, s := range []int{1, 2, 4} {
			if s <= m {
				set[s] = true
			}
		}
		if m >= 8 {
			for s := 8; s < m+1 && s < 256; s += 8 {
				set[s] = true
			}
		}
		if m >= 256 {
			for s := 256; s <= m; s += 16 {
				set[s] = true
			}
		}
		if mnbt <= m {
			set[mnbt] = true
		}
		for s := range set {
			out.Sizes = append(out.Sizes, s)
		}
		sort.Ints(out.Sizes)
	}
	if out.Sizes == nil {
		out.Sizes = []int{}
	}
	if n := len(out.Sizes); n > 0 {
		out.Max = out.Sizes[n-1]
	}
	if m := in.MaxCaptureSize; m != nil && *m != out.Max {
		if in.CaptureSizes != nil {
			return out, true, rules.ErrCUDAGraphCaptureInconsistent
		}
		out.Truncated = out.Max < min(*m, mnbt)
	}
	return out, true, nil
}
