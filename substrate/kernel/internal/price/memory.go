package price

import "math"

// Memory laws: how many bytes a deployment holds, and how that divides across ranks.
//
// These are the terms a feasibility check needs. They are separate from the timing laws
// because a caller asks whether a batch fits before asking how long it takes, and the fit
// question has no timing content.

// KVBytesPerToken returns per-rank KV bytes for one token.
//
//	tensors * max(1, nKV/tp) * headDim * bytes(dtype) * layers
//
// where tensors is 2 -- a key and a value -- for full and windowed attention, and 1 for
// latent attention. Four details decide the answer, and getting any of them wrong changes a
// capacity verdict by a large factor:
//
// The head count floors at one. An engine divides KV heads by the tensor-parallel width
// and replicates rather than splitting a head, so a model with eight KV heads stops
// dividing at tp 8: at tp 16 each rank still holds one head, and per-rank bytes are the
// same as at tp 8 rather than half.
//
// Latent attention holds ONE vector per token, with no separate value: "MLA stores a single
// latent vector per state; there is no separate V", head_size_v = 0
// (vllm/v1/kv_cache_interface.py:674-676 at v0.31.0), and a page is
// (head_size + head_size_v) * dtype per head unless a spec states its own state size
// (:521-525). So a latent layer is charged one tensor, not two. Its head count is one
// whatever the config says and no tensor-parallel width reduces it; the model graph records
// nKV = 1 for those kinds. An earlier form charged a latent layer two tensors -- every MLA
// and sparse-MLA model's cache at double its size -- against a decode rate blis-registry
// fitted on the single vector (scripts/fit_attention_mla.py: "kv_bytes = batch * step *
// 576 * dtype_width").
//
// The layouts that state their own size are applied by the kernel where a deployment names
// them, or where its backend repacks the cache (the packed ds_mla caches; see Kernel.lift and
// sparseMLABackend in the root package). COVERAGE LIMIT: per-token-head quantization adds two
// fp32 scales a token (vllm/model_executor/layers/attention/mla_attention.py:1580-1593), so
// the kernel refuses those cache dtypes rather than size them here; padded pages are sized by
// their own specs (vllm/v1/kv_cache_interface.py:527-536), and a compressed cache (a
// compress_ratio layer) holds fewer states than tokens. Neither is modelled HERE: a graph
// that states its caches (blis-schemas model.Cache) is sized from them in the root package
// (caches.go), and this law is what a layer that states none is sized by.
//
// The cache dtype is independent of the weight dtype. A bf16 model with an fp8 cache
// halves this, which is the difference between a deployment holding one long sequence and
// two.
func KVBytesPerToken(nKV, tp, headDim, layers int, dtypeBytes float64, latent bool) float64 {
	if nKV <= 0 || headDim <= 0 || layers <= 0 || dtypeBytes <= 0 {
		return 0
	}
	perRankHeads := nKV
	if tp > 1 {
		perRankHeads = nKV / tp
		if perRankHeads < 1 {
			// The replication floor: a head is never split across ranks.
			perRankHeads = 1
		}
	}
	tensors := 2.0
	if latent {
		tensors = 1
	}
	return tensors * float64(perRankHeads) * float64(headDim) * dtypeBytes * float64(layers)
}

// PagedBytes rounds a token count up to whole pages before charging for it.
//
// KV is allocated in blocks, so a 17-token sequence at a block size of 16 occupies two
// blocks rather than 17/16 of one. The rounding matters most where it is least expected:
// at a large block size a short sequence wastes most of a page, and a capacity estimate
// that ignored it would overcount how many sequences fit.
func PagedBytes(tokens, blockSize int, bytesPerToken float64) int64 {
	if tokens <= 0 || bytesPerToken <= 0 {
		return 0
	}
	if blockSize <= 1 {
		return int64(float64(tokens) * bytesPerToken)
	}
	pages := (tokens + blockSize - 1) / blockSize
	return int64(float64(pages) * float64(blockSize) * bytesPerToken)
}

// RecurrentStateBytes returns per-sequence bytes for one recurrent layer.
//
// The mode decides whether this is a fixed cost or a context-proportional one, which is a
// difference of three orders of magnitude rather than a detail:
//
//   - RecurrentCacheNone: one page plus the speculative blocks. Prefix caching off.
//   - RecurrentCacheAlign: two pages plus speculation and prefill checkpoints. The
//     default when prefix caching is on.
//   - RecurrentCacheAll: one page per block position up to the context bound, which makes
//     the state proportional to maxModelLen and so a variable cost rather than a fixed one.
//
// A memory model that treated the state as fixed in every mode would understate the third
// by the block count — 8192x at a 131072 context and 16-token blocks.
func RecurrentStateBytes(pageBytes float64, mode RecurrentCacheMode,
	maxModelLen, blockSize, specBlocks, prefillCheckpointBlocks int) float64 {
	if pageBytes <= 0 {
		return 0
	}
	switch mode {
	case RecurrentCacheAll:
		if blockSize <= 0 {
			return pageBytes
		}
		pages := (maxModelLen + blockSize - 1) / blockSize
		return pageBytes * float64(pages+specBlocks)
	case RecurrentCacheAlign:
		return pageBytes * float64(2+specBlocks+prefillCheckpointBlocks)
	default:
		return pageBytes * float64(1+specBlocks)
	}
}

// RecurrentCacheMode is the engine's cache strategy for recurrent layers.
type RecurrentCacheMode string

const (
	RecurrentCacheNone  RecurrentCacheMode = "none"
	RecurrentCacheAlign RecurrentCacheMode = "align"
	RecurrentCacheAll   RecurrentCacheMode = "all"
)

// ProportionalToContext reports whether a mode makes the state grow with the context
// bound. Callers use it to decide which memory term the state belongs in: a fixed
// per-sequence cost, or one that varies with prompt length.
func (m RecurrentCacheMode) ProportionalToContext() bool {
	return m == RecurrentCacheAll
}

// ExpertsPerRank returns how many experts each rank of an expert-parallel group holds,
// and the static imbalance the placement carries.
//
// An engine distributes experts as evenly as it can: with a remainder, the first ranks
// take one extra. So an indivisible split is legal and uneven, and the imbalance is a
// step-time term rather than an error — the grouped GEMM on a rank with one extra expert
// is larger, and the collective waits for the slowest rank.
//
// Under expert-parallel load balancing the physical count must divide the width, and an
// indivisible layout does not start at all. That check belongs to the resolver, which has
// the scenario; this function reports what the placement would be.
func ExpertsPerRank(experts, ep int) (base, withExtra, ranksWithExtra int, imbalance float64) {
	if ep <= 0 || experts <= 0 {
		return 0, 0, 0, 1
	}
	base = experts / ep
	ranksWithExtra = experts % ep
	withExtra = base
	if ranksWithExtra > 0 {
		withExtra = base + 1
	}
	if base == 0 {
		// More ranks than experts: some ranks hold none, and the layer cannot run as a
		// balanced collective. Reported rather than smoothed over.
		return 0, withExtra, ranksWithExtra, 0
	}
	return base, withExtra, ranksWithExtra, float64(withExtra) / float64(base)
}

// ExpertsTouched returns the expected number of DISTINCT local experts a step reads.
//
// The cost model's decode assumption is that every local expert is touched, so a rank
// reads its whole expert shard however small the batch. That holds at the batch sizes the
// assumption was written for and fails badly below them: with 224 experts and top-8
// routing, a single token can touch at most 8 of them and a rank holding 14 reads one.
// Charging the whole shard at one token overstates the memory term by fourteen times, and
// the error is largest exactly where a latency-sensitive deployment operates.
//
// A local expert goes untouched when every one of the step's tokens routes elsewhere.
// Treating each token's choice of k from E as independent across tokens,
//
//	P(one local expert untouched) = ((E - k) / E) ^ tokens
//
// so the expected count is localExperts times one minus that. The independence assumption
// is the router's own behaviour at the granularity that matters: a token's top-k is
// chosen without replacement WITHIN the token, which this respects by using k/E as the
// per-token hit probability, and across tokens the router is not coordinated.
//
// The result saturates quickly — at 224 experts and top-8 it is within one percent of the
// full shard by 128 tokens — so this changes decode pricing at small batch and nothing
// above it. It is deliberately a continuous expectation rather than a rounded count: a
// step reads whole experts, but a cost model averaging over steps wants the mean.
func ExpertsTouched(tokens, totalExperts, topK int, localExperts float64) float64 {
	if localExperts <= 0 || tokens <= 0 {
		return 0
	}
	if totalExperts <= 0 || topK <= 0 || topK >= totalExperts {
		// Every expert is reachable by any token, so the whole shard is read.
		return localExperts
	}
	missOnce := float64(totalExperts-topK) / float64(totalExperts)
	// math.Pow with a large exponent underflows to zero, which is the right limit: at
	// many tokens nothing goes untouched.
	untouched := math.Pow(missOnce, float64(tokens))
	return localExperts * (1 - untouched)
}
