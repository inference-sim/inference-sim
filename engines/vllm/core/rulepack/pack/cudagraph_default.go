// Original to blis-vllm.

package pack

import (
	"math"
	"sort"
)

// The capture a deployment gets when it states neither max_cudagraph_capture_size nor
// cudagraph_capture_sizes, as vLLM main head 0fda86a35419c9c70dbfada224c838c10fbb62ad
// resolves it: VllmConfig._set_cudagraph_sizes (vllm/config/vllm.py:2465-2690) builds the
// shared token ladder, then _split_cudagraph_capture_sizes (:2698-2870) resolves a prefill
// token grid and a decode request grid from the platform's CUDAGraphCaptureDefaults
// (vllm/config/compilation.py) and records the union of every graph's token count as
// cudagraph_capture_sizes. The platform split is vllm/platforms/cuda.py:640-652: compute
// capability family 100 gets 1024 / 8192 / max_num_seqs / 8192, every other platform the
// interface default (vllm/platforms/interface.py:1497-1523), which sets only the legacy 512
// and leaves both split limits unset.
//
// Not modelled: performance_mode interactivity (the dense head of the prefill grid),
// dynamic speculative decoding (one query length per tier; a deployment states a single
// num_speculative_tokens here), adaptive verification's varlen decode grid, and the
// multi-module MTP carve-outs index_share_for_mtp_iteration and minimax_m3_mtp.

// DefaultCaptureInput is what the default capture depends on.
type DefaultCaptureInput struct {
	// MaxNumSeqs and MaxNumBatchedTokens are the resolved scheduler limits.
	MaxNumSeqs, MaxNumBatchedTokens int
	// DecodeQueryLen is VllmConfig.uniform_decode_query_len: 1 + num_speculative_tokens.
	DecodeQueryLen int
	// Family100 is whether the device is compute capability family 100 (SM100).
	Family100 bool
	// MixedGraphs is whether the cudagraph mode captures prefill and mixed steps
	// (cudagraph_mode.mixed_mode() is not NONE). FULL_DECODE_ONLY captures decode graphs
	// only, so the prefill grid does not extend the recorded sizes.
	MixedGraphs bool
	// DraftDecode is whether the speculative method has one-token draft decode steps that
	// get their own FULL graphs (get_draft_decode_cudagraph_request_sizes is non-empty): the
	// method is eagle, eagle3, draft_model, or a single-module mtp, with
	// num_speculative_tokens of at least 2.
	DraftDecode bool
}

// DefaultCapture is the resolved default capture.
type DefaultCapture struct {
	// Sizes are the token counts captured, ascending: the engine's cudagraph_capture_sizes.
	Sizes []int
	// Max is the largest, the engine's max_cudagraph_capture_size; zero when nothing is
	// captured.
	Max int
	// LegacyMax is the shared-ladder ceiling, min(max_num_seqs * decode_query_len * 2,
	// platform legacy maximum, max_num_batched_tokens), the first stage computes.
	LegacyMax int
	// PrefillSizes and DecodeRequestSizes are the two split grids.
	PrefillSizes, DecodeRequestSizes []int
}

// platformCaptureDefaults is CUDAGraphCaptureDefaults for the device. A negative split limit
// is unset.
func platformCaptureDefaults(family100 bool, maxNumSeqs int) (legacyMax, maxPrefill, maxDecodeReqs, maxDecodeTokens int) {
	if family100 {
		return 1024, 8192, maxNumSeqs, 8192
	}
	return 512, -1, -1, -1
}

// walkGrid is compilation.py _walk_grid: sizes 1 to limit, each the next multiple of the
// largest power of two not above step(size).
func walkGrid(limit int, step func(float64) float64) []int {
	if limit <= 0 {
		return []int{}
	}
	grid := []int{}
	size := 1
	for size < limit {
		grid = append(grid, size)
		e := int(math.Floor(math.Log2(math.Max(1, step(float64(size))))))
		stride := 1 << uint(max(0, e))
		size = (size/stride + 1) * stride
	}
	return append(grid, limit)
}

// PrefillCUDAGraphCaptureGrid is compilation.py prefill_cudagraph_capture_grid without the
// interactivity dense head: 1, 2, 4, every 8 up to 64, four sizes per doubling up to 1024,
// then every 256 tokens, and every 512 once 7% of the size exceeds 512.
func PrefillCUDAGraphCaptureGrid(maxTokens int) []int {
	return walkGrid(maxTokens, func(t float64) float64 {
		return math.Min(t, math.Min(math.Max(8, t/4), 0.07*math.Max(t, 4096)))
	})
}

// DecodeCUDAGraphCaptureGrid is compilation.py decode_cudagraph_capture_grid: the request
// counts up to maxRequests, every count up to 6, then a power-of-two step of at most
// 0.65 * n^(2/3).
func DecodeCUDAGraphCaptureGrid(maxRequests int) []int {
	return walkGrid(maxRequests, func(n float64) float64 { return 0.65 * math.Pow(n, 2.0/3.0) })
}

// decodeCaptureTokens is compilation.py decode_cudagraph_capture_shapes reduced to the token
// counts n*q of its (num_reqs, query_len) shapes, for the one query length q whose batches
// have at most maxReqs requests: the sizes up to and including the first at or above the
// limit, the last captured at the limit.
func decodeCaptureTokens(sizes []int, q, maxReqs, maxTokens int) []int {
	limit := min(maxReqs, maxTokens/q)
	if limit < 1 {
		return nil
	}
	sorted := append([]int(nil), sizes...)
	sort.Ints(sorted)
	var out []int
	for _, n := range sorted {
		out = append(out, min(n, limit)*q)
		if n >= limit {
			break
		}
	}
	return out
}

// DefaultCUDAGraphCapture resolves the capture an unstated deployment gets. The input limits
// must be positive.
func DefaultCUDAGraphCapture(in DefaultCaptureInput) DefaultCapture {
	q := max(in.DecodeQueryLen, 1)
	seqs, mnbt := in.MaxNumSeqs, in.MaxNumBatchedTokens
	legacyMax, maxPrefill, maxDecodeReqs, maxDecodeTokens := platformCaptureDefaults(in.Family100, seqs)

	// _set_cudagraph_sizes: the shared ladder.
	m := min(seqs*q*2, legacyMax)
	var uniformDecode []int
	if q > 1 {
		reqs := min(seqs, legacyMax)
		counts := map[int]bool{reqs: true}
		for _, n := range []int{1, 2, 4} {
			if n <= reqs {
				counts[n] = true
			}
		}
		for n := 8; n < min(reqs+1, 256); n += 8 {
			counts[n] = true
		}
		for n := 256; n <= reqs; n += 16 {
			counts[n] = true
		}
		for n := range counts {
			if n*q <= m {
				uniformDecode = append(uniformDecode, n*q)
			}
		}
	} else if seqs <= m {
		uniformDecode = []int{seqs}
	}
	m = min(mnbt, m)
	shared := map[int]bool{}
	for _, s := range []int{1, 2, 4} {
		if s <= m {
			shared[s] = true
		}
	}
	for s := 8; m >= 8 && s < min(m+1, 256); s += 8 {
		shared[s] = true
	}
	for s := 256; m >= 256 && s <= m; s += 16 {
		shared[s] = true
	}
	if mnbt <= m {
		shared[mnbt] = true
	}
	for _, s := range uniformDecode {
		if s <= mnbt {
			shared[s] = true
		}
	}
	sharedSizes := sortedSizes(shared)
	sharedMax := 0
	if n := len(sharedSizes); n > 0 {
		sharedMax = sharedSizes[n-1]
	}

	// _split_cudagraph_capture_sizes: the prefill and decode grids, each from its own limit.
	prefill := sharedSizes
	if maxPrefill >= 0 {
		prefill = PrefillCUDAGraphCaptureGrid(min(maxPrefill, mnbt))
	}
	var decode []int
	if maxDecodeReqs >= 0 {
		decode = DecodeCUDAGraphCaptureGrid(min(maxDecodeReqs, seqs))
	} else {
		set := map[int]bool{}
		for _, s := range sharedSizes {
			if n := (s + q - 1) / q; n*q <= sharedMax {
				set[min(n, seqs)] = true
			}
		}
		decode = sortedSizes(set)
	}
	decodeMaxTokens := mnbt
	if maxDecodeTokens >= 0 {
		decodeMaxTokens = min(mnbt, maxDecodeTokens)
	}
	tokens := map[int]bool{}
	for _, s := range decodeCaptureTokens(decode, q, seqs, decodeMaxTokens) {
		tokens[s] = true
	}
	if in.DraftDecode && len(decode) > 0 {
		for _, s := range decodeCaptureTokens(DecodeCUDAGraphCaptureGrid(seqs), 1, seqs, decodeMaxTokens) {
			tokens[s] = true
		}
	}
	if in.MixedGraphs {
		for _, s := range prefill {
			tokens[s] = true
		}
	}
	out := DefaultCapture{Sizes: sortedSizes(tokens), LegacyMax: m, PrefillSizes: prefill, DecodeRequestSizes: decode}
	if n := len(out.Sizes); n > 0 {
		out.Max = out.Sizes[n-1]
	}
	return out
}

func sortedSizes(set map[int]bool) []int {
	out := make([]int, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Ints(out)
	return out
}
