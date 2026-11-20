// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/speculative.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
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
	// EagleBlockDrop is use_eagle_block_drop (vllm/config/speculative.py:2147-2149@4ba787ec): the
	// drafter's KV at a position depends on the next token, so the trailing prefix-cache block
	// is volatile. It holds for eagle, eagle3 and mtp only (needs_eagle_block_drop, :2122-2129);
	// dflash and dspark build that KV from the target's hidden state alone. The scheduler turns
	// it off when the block hashes cover the lookahead instead (HashLookahead).
	EagleBlockDrop bool
	// DraftQueryPass is use_dflash() or use_dspark() (vllm/config/speculative.py@4ba787ec): the
	// drafter runs a pass of num_lookahead_tokens query rows a request inside
	// max_num_batched_tokens, which caps max_num_active_seqs (VllmConfig._set_max_num_scheduled_tokens,
	// vllm/config/vllm.py:2377-2396@4ba787ec).
	DraftQueryPass bool
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
	// in parallel (vllm/config/speculative.py:1581-1582@4ba787ec).
	ParallelDrafting bool
	// NextnPredictLayers is an MTP drafter's num_nextn_predict_layers; 0 means 1.
	NextnPredictLayers int
	// SampleFromAnchor is a DSpark drafter's sample_from_anchor; nil means true, vLLM's
	// default (vllm/config/vllm.py:639-641@4ba787ec).
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
	needsDrop := useEagle && o.Method != "dflash" && o.Method != "dspark" // needs_eagle_block_drop
	s := Spec{NumTokens: k, EagleBlockDrop: needsDrop && !o.DisableEagleBlockDrop,
		DraftQueryPass: o.Method == "dflash" || o.Method == "dspark"}

	// max_num_new_slots_for_drafting (vllm/config/speculative.py:2058-2097@4ba787ec). DFlash and
	// DSpark draft in a separate query pass and take no slots; v0.32.0 gave them K and K-1.
	switch {
	case o.Method == "dflash" || o.Method == "dspark":
		s.DraftSlots = 0
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
