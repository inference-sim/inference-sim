package latencykernel

import (
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// Measured prefill and mHC forms for DeepSeek-V4.1-Flash's own kernels, read where the
// registry carries them (blis-registry PR #47, fitted by scripts/fit_deepseek_v41_prefill.py)
// and priced as before where it does not:
//
//	attention_prefill_{floor,per_request,per_token,rate}_<kind>  a primary attention kind's
//	                                                               prefill (sparse_mla)
//	attention_prefill_{floor,per_token,rate}_<role>              a secondary kernel's prefill
//	                                                               (block_index_scores)
//	mhc_{floor,rate}_{pre,post}                                  the hyper-connection kernels
//
// Each family is all or none: a member missing beside the others is recorded as a fallback
// and the family is not used, since a form fitted jointly is not the sum of its parts.

// liftPrefillForms resolves the forms for the layers a step can price.
func (k *Kernel) liftPrefillForms(c *resolve.Coefficients, layers []price.PlannedLayer,
	hidden int) {
	k.hiddenSize = hidden
	k.attentionPrefillByKind = map[model.AttentionKind]prefillForm{}
	k.attentionPrefillByRole = map[string]prefillForm{}
	kinds, roles, mhc := map[model.AttentionKind]bool{}, map[string]bool{}, false
	for _, l := range layers {
		if l.AttnQHeads > 0 && l.AttnKind == model.AttentionSparseMLA {
			kinds[l.AttnKind] = true
		}
		for _, a := range l.Attentions {
			if !a.HoldsKV() && a.Role != "" {
				roles[a.Role] = true
			}
		}
		mhc = mhc || len(l.MHC) > 0
	}
	for kind := range kinds {
		suffix := string(kind)
		if f, ok := k.prefillFamily(c, suffix, true,
			"no measured prefill form for "+suffix+" attention, so its prefill is priced "+
				"with the part-wide prefill pair on the attention FLOPs"); ok {
			k.attentionPrefillByKind[kind] = f
		}
	}
	for role := range roles {
		if f, ok := k.prefillFamily(c, role, false,
			"no measured prefill form for the "+role+" kernel, so its prefill is priced "+
				"with the part-wide prefill pair on its scoring FLOPs"); ok {
			k.attentionPrefillByRole[role] = f
		}
	}
	if mhc && hidden > 0 {
		k.mhcPre = k.mhcComponent(c, "pre")
		k.mhcPost = k.mhcComponent(c, "post")
		// Both or neither. The pre form times the hc_*fn GEMM with it, and V4.1's fused
		// sites run post and pre in one node, so one component alone would leave a site
		// charged as stated bytes beside a dropped GEMM.
		if (k.mhcPre.rate > 0) != (k.mhcPost.rate > 0) {
			k.assume("mhc", "fallback", "only one of the mHC pre and post forms resolves, "+
				"so neither is used and the hyper-connection nodes are charged their "+
				"stated bytes and the hc_*fn GEMMs on the dense ramp")
			k.mhcPre, k.mhcPost = floorRate{}, floorRate{}
		}
	}
}

// prefillFamily reads one prefill family by its suffix, recording the rate's absence as
// optional does and a partial family as a fallback. perRequest is whether the family has a
// per-request member.
func (k *Kernel) prefillFamily(c *resolve.Coefficients, suffix string, perRequest bool,
	without string) (prefillForm, bool) {
	rate := k.optional(c, "attention_prefill_rate_"+suffix, true, without)
	if rate <= 0 {
		return prefillForm{}, false
	}
	members := []string{"attention_prefill_floor_" + suffix,
		"attention_prefill_per_token_" + suffix}
	if perRequest {
		members = append(members, "attention_prefill_per_request_"+suffix)
	}
	var missing []string
	for _, m := range members {
		if !c.Has(m) {
			missing = append(missing, m)
		}
	}
	if len(missing) > 0 {
		k.assume("attention_prefill_"+suffix, "fallback",
			"the family is incomplete ("+strings.Join(missing, ", ")+" absent), so the "+
				"measured form is not used: "+without)
		return prefillForm{}, false
	}
	us := func(name string) float64 { return c.ValueOr(name, 0) * 1e-6 }
	return prefillForm{
		floor:      us("attention_prefill_floor_" + suffix),
		perRequest: us("attention_prefill_per_request_" + suffix),
		perToken:   us("attention_prefill_per_token_" + suffix),
		rate:       rate * 1e6,
	}, true
}

// mhcComponent reads one mHC component's floor and rate, both or neither.
func (k *Kernel) mhcComponent(c *resolve.Coefficients, comp string) floorRate {
	without := "no measured mHC " + comp + " form, so the hyper-connection nodes are charged " +
		"their stated bytes at HBM bandwidth and the hc_*fn GEMMs on the dense ramp"
	rate := k.optional(c, "mhc_rate_"+comp, true, without)
	if rate <= 0 {
		return floorRate{}
	}
	if !c.Has("mhc_floor_" + comp) {
		k.assume("mhc_"+comp, "fallback", "mhc_floor_"+comp+" absent beside its rate, so "+
			"the measured form is not used: "+without)
		return floorRate{}
	}
	return floorRate{
		floor: time.Duration(c.ValueOr("mhc_floor_"+comp, 0) * float64(time.Microsecond)),
		rate:  rate * 1e6,
	}
}

// mhcBytesPerHidden is the mHC form's work per token per hidden element: the residual
// stream's hc_mult = 4 copies at BF16 (vllm/models/deepseek_v41/nvidia/model.py, hc_mult;
// the 2-byte stream the tilelang mhc_shifted_post_pre kernels read), which is the work
// variable the registry's mhc rates were fitted on.
const mhcBytesPerHidden = 8

// mhcSeconds prices a layer's mHC nodes with the measured form where it covers them, and
// returns the elementwise bytes those nodes no longer carry. A node is covered when every
// component it runs resolves; one that is not keeps its stated bytes.
func (k *Kernel) mhcSeconds(l *price.PlannedLayer, s *stepShape, sp bool) (seconds,
	replacedBytes float64) {
	if k.mhcPre.rate <= 0 && k.mhcPost.rate <= 0 {
		return 0, 0
	}
	for _, m := range l.MHC {
		if (!m.Pre && !m.Post) || (m.Pre && k.mhcPre.rate <= 0) ||
			(m.Post && k.mhcPost.rate <= 0) {
			continue
		}
		rows := s.tokensF
		if m.TokenSharded && sp {
			rows = s.spRows
		}
		// The bytes the node added to its elementwise sum, divided as that sum divides.
		switch {
		case m.Replicated:
			replacedBytes += rows * m.Bytes
		default:
			replacedBytes += s.tokensF * m.Bytes / k.tp
		}
		w := rows * float64(k.hiddenSize) * mhcBytesPerHidden
		if m.Pre {
			seconds += k.mhcPre.floor.Seconds() + w/k.mhcPre.rate
		}
		if m.Post {
			seconds += k.mhcPost.floor.Seconds() + w/k.mhcPost.rate
		}
	}
	return seconds, replacedBytes
}

// sumBlocks is the sum over positions p in [lo, hi] of ceil(min(floor(p / ratio), cap) /
// block), with cap <= 0 meaning no cap: one term of FlashMLA's per-query block count. It
// steps over the runs where the term is constant, so a chunk costs at most cap / block steps
// rather than one per token.
func sumBlocks(lo, hi, ratio, cap, block int) float64 {
	var total float64
	for p := lo; p <= hi; {
		m := p / ratio
		if cap > 0 && m >= cap {
			total += float64((cap+block-1)/block) * float64(hi-p+1)
			break
		}
		v := (m + block - 1) / block
		next := (v*block + 1) * ratio // the first position whose term exceeds v
		end := min(next-1, hi)
		total += float64(v) * float64(end-p+1)
		p = end + 1
	}
	return total
}

// sumCeil is the sum over positions p in [lo, hi] of ceil(p / ratio), in closed form.
func sumCeil(lo, hi, ratio int) float64 {
	// S(n) = sum_{p=0}^{n} floor(p / r); ceil(p / r) = floor((p + r - 1) / r).
	s := func(n int) float64 {
		if n < 0 {
			return 0
		}
		q, rem := n/ratio, n%ratio
		return float64(ratio)*float64(q)*float64(q-1)/2 + float64(q)*float64(rem+1)
	}
	return s(hi+ratio-1) - s(lo+ratio-2)
}

// sparsePrefillReadBytes is the bytes a sparse-MLA layer's prefill queries read, as the
// registry's form counts them: per query at absolute position p (1-based), FlashMLA's 64-token
// blocks over its window, min(p, window), and over its selection, min(floor(p / ratio),
// index_topk), times the cache record and the padded heads over 64. A layer with no ratio
// selects nothing; one with a ratio and no top-k reads its whole compressed stream.
func sparsePrefillReadBytes(chunks []chunk, window, topk, ratio, record int,
	heads float64) float64 {
	var blocks float64
	for _, c := range chunks {
		lo, hi := c.prefix+1, c.prefix+c.sched
		if hi < lo {
			continue
		}
		if window > 0 {
			blocks += sumBlocks(lo, hi, 1, window, 64)
		}
		if ratio > 0 {
			blocks += sumBlocks(lo, hi, ratio, topk, 64)
		}
	}
	return blocks * 64 * float64(record) * heads / 64
}

// scorerPrefillReadBytes is the bytes a block-index scorer's prefill queries read, as the
// registry's form counts them: per query at absolute position p, ceil(p / ratio) index keys
// of record bytes, times the kernel's query heads over 64.
func scorerPrefillReadBytes(chunks []chunk, ratio, record, heads int) float64 {
	var keys float64
	for _, c := range chunks {
		lo, hi := c.prefix+1, c.prefix+c.sched
		if hi >= lo {
			keys += sumCeil(lo, hi, max(ratio, 1))
		}
	}
	return keys * float64(record) * float64(heads) / 64
}

// primaryRecord is the bytes of one record of a sparse layer's window cache, the record its
// prefill reads, or zero where the graph states none.
func primaryRecord(l *price.PlannedLayer) int {
	for _, c := range l.AttnCaches {
		if c.Kind == model.CacheWindow {
			return c.EntryBytes
		}
	}
	if len(l.AttnCaches) > 0 {
		return l.AttnCaches[0].EntryBytes
	}
	return 0
}
