package latencykernel

import (
	"time"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// pricesQwen4Exp prices the two kinds no allowed data measures: sparse GQA attention (the QSA
// Triton kernel) and the short convolution. It is the stand-in, and says so.
//
// Sparse GQA. A layer reads index_topk selected tokens of per-head K and V, and its indexer
// scorer reads a key cache compressed by its ratio. Both are bounded reads the kernel already
// prices for every attention kind (selectedKVTokensFor), so a stand-in needs only a rate: the
// part-wide attention pair, which is a dense grouped-query attention rate measured on another
// kernel. A measured attention_decode_rate_sparse_gqa replaces it with no code change, and
// is used when the registry carries one. Until then Provenance records the substitution
// under method assumed, so a point priced here is capped at the lowest scored grade.
//
// Short convolution. A depthwise convolution over a window of inputs moves channels x taps
// values per token, a few tens of kilobytes. No decode form is registered for it, so it is
// not charged beyond the projections the graph prices as GEMM nodes; the substitution is
// recorded too.
func (k *Kernel) pricesQwen4Exp(c *resolve.Coefficients) {
	usedAttn, usedRec := false, false
	for _, l := range k.plan.Layers {
		usedAttn = usedAttn || (l.AttnQHeads > 0 && l.AttnKind == model.AttentionSparseGQA)
		usedRec = usedRec || l.RecurrentKind == model.RecurrentShortConv
	}
	if usedAttn {
		if rate := c.ValueOr("attention_decode_rate_sparse_gqa", 0); rate > 0 {
			fr := floorRate{floor: k.attentionFloor, rate: rate * 1e6}
			if f := c.ValueOr("attention_decode_floor_sparse_gqa", 0); f > 0 {
				fr.floor = time.Duration(f * float64(time.Microsecond))
			}
			k.attentionByKind[model.AttentionSparseGQA] = fr
		} else {
			k.assume("attention_decode_rate_sparse_gqa", "substituted",
				"the QSA Triton kernel has no measurement in an allowed data source "+
					"(blis-substrate#251), so sparse-GQA attention and its indexer scorer are priced "+
					"with the part-wide grouped-query attention pair over the selected tokens; "+
					"a point priced so is capped at the lowest scored grade")
		}
	}
	if usedRec && !c.Has("recurrent_decode_rate_short_conv") {
		k.assume("recurrent_decode_rate_short_conv", "substituted",
			"no measured decode form for the short_conv mixer, so its convolution step is not "+
				"charged beyond the projections the graph prices; a point priced so is capped at "+
				"the lowest scored grade")
	}
}
