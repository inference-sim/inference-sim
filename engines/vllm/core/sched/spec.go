// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/speculative.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/config/vllm.py (num_lookahead_tokens, num_prefill_lookahead_tokens).

package sched

// Spec is what speculative decoding changes in the scheduler and the KV cache manager. Its
// zero value is speculative decoding off.
type Spec struct {
	// NumTokens is num_speculative_tokens, K: the drafts each decode step verifies.
	NumTokens int
	// DraftSlots is max_num_new_slots_for_drafting: input-budget slots each scheduled request
	// takes beyond its scheduled tokens (vllm/config/speculative.py:1887-1926@5df63a72).
	DraftSlots int
	// Lookahead is num_lookahead_tokens: KV slots past the scheduled tokens the drafter
	// writes (vllm/config/vllm.py:614-641@5df63a72).
	Lookahead int
	// PrefillLookahead is num_prefill_lookahead_tokens: prefill tokens past the computed
	// range the drafter reads (vllm/config/vllm.py:643-665@5df63a72).
	PrefillLookahead int
	// EagleBlockDrop is use_eagle_block_drop (vllm/config/speculative.py:1950-1952@5df63a72).
	EagleBlockDrop bool
}

// SpecOptions are the speculative-decoding settings a deployment states, and the drafter
// facts its checkpoint carries.
type SpecOptions struct {
	// Method is speculative_config.method: eagle, eagle3, mtp, dflash, dspark, draft_model,
	// ngram, ngram_gpu, suffix, ...
	Method string
	// NumTokens is num_speculative_tokens.
	NumTokens int
	// ParallelDrafting is parallel_drafting (P-EAGLE, PARD); dflash and dspark always draft
	// in parallel (vllm/config/speculative.py:1468-1469@5df63a72).
	ParallelDrafting bool
	// NextnPredictLayers is an MTP drafter's num_nextn_predict_layers; 0 means 1.
	NextnPredictLayers int
	// SampleFromAnchor is a DSpark drafter's sample_from_anchor; nil means true, vLLM's
	// default (vllm/config/vllm.py:628-630@5df63a72).
	SampleFromAnchor *bool
	// DisableEagleBlockDrop is disable_eagle_block_drop.
	DisableEagleBlockDrop bool
}

// ResolveSpec derives the scheduler's speculative-decoding numbers exactly as vLLM's
// SpeculativeConfig and VllmConfig properties do.
func ResolveSpec(o SpecOptions) Spec {
	if o.NumTokens <= 0 {
		return Spec{}
	}
	k := o.NumTokens
	useEagle := o.Method == "eagle" || o.Method == "eagle3" || o.Method == "mtp" ||
		o.Method == "dflash" || o.Method == "dspark" // use_eagle, speculative.py:1944-1948
	draftModel := o.Method == "draft_model" // uses_draft_model, :1963-1964
	parallel := o.ParallelDrafting || o.Method == "dflash" || o.Method == "dspark"
	s := Spec{NumTokens: k, EagleBlockDrop: useEagle && !o.DisableEagleBlockDrop}

	// max_num_new_slots_for_drafting (speculative.py:1907-1926).
	switch {
	case o.Method == "dflash":
		s.DraftSlots = k
	case parallel && draftModel:
		s.DraftSlots = k
	case parallel:
		s.DraftSlots = k - 1
	case draftModel:
		s.DraftSlots = 1
	}

	// num_lookahead_tokens (vllm.py:625-641).
	anchor := o.SampleFromAnchor == nil || *o.SampleFromAnchor
	switch {
	case o.Method == "dflash" || (o.Method == "dspark" && !anchor):
		s.Lookahead = k + 1
	case useEagle || draftModel:
		s.Lookahead = k
	}

	// num_prefill_lookahead_tokens (vllm.py:655-665); use_multi_module_mtp is
	// speculative.py:1982-1987.
	if useEagle {
		s.PrefillLookahead = 1
		if o.Method == "mtp" && min(max(o.NextnPredictLayers, 1), k) > 1 {
			s.PrefillLookahead = k
		}
	}
	return s
}
