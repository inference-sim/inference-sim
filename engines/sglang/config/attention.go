// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/attention_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/model_override_base.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package config

// handleAttentionBackendCompatibility is handle_attention_backend_compatibility
// (attention_hook.py:110-278) on CUDA: the default backend, then the page sizes the chosen
// backends require.
func (r *resolver) handleAttentionBackendCompatibility() {
	r.attentionBackendDefault()
	r.mlaBackendPageConstraints()
	// The TRTLLM MHA checks need SM90, SM100 or SM120, which every accepted chip is.
	r.fa4PageConstraint()
	// torch_native, flex_attention, Whisper, cutedsl_mla, an fp8_e5m2 cache on fa3, aiter, the
	// Intel backends and XPU decode context parallelism are not reached by an accepted input.
}

// attentionBackends is attention_backends_of (model_override_base.py:138-152): the prefill and
// decode backends, each falling back to attention_backend.
func (r *resolver) attentionBackends() (prefill, decode *string) {
	prefill, decode = r.cfg.PrefillAttentionBackend, r.cfg.DecodeAttentionBackend
	if prefill == nil || *prefill == "" {
		prefill = r.cfg.AttentionBackend
	}
	if decode == nil || *decode == "" {
		decode = r.cfg.AttentionBackend
	}
	return prefill, decode
}

// attentionBackendDefault is _attention_backend_default (overrides.py:1126-1140).
func (r *resolver) attentionBackendDefault() {
	const src = "_attention_backend_default"
	p, d := r.cfg.PrefillAttentionBackend, r.cfg.DecodeAttentionBackend
	if p != nil && d != nil && *p == *d {
		r.declare(src, "attention_backend", *p)
		return
	}
	if r.cfg.AttentionBackend == nil {
		r.declare(src, "attention_backend", r.defaultAttentionBackend())
	}
}

// noSpecOrTopkOne is is_no_spec_infer_or_topk_one (utils/common.py:3960-3966).
func (r *resolver) noSpecOrTopkOne() bool {
	k := r.cfg.SpeculativeEagleTopk
	return k == nil || (*k == 1 && (r.cfg.PageSize == nil || *r.cfg.PageSize == 1))
}

// defaultAttentionBackend is get_default_attn_backend (model_override_base.py:277-351) on CUDA.
func (r *resolver) defaultAttentionBackend() string {
	if !r.model.useMLA() {
		switch {
		case r.chip.isHopperWithCUDA123() && r.noSpecOrTopkOne():
			return "fa3"
		case r.chip.isSM100() && r.noSpecOrTopkOne() &&
			(r.cfg.SpeculativeAlgorithm == nil || r.cfg.SpeculativeEagleTopk != nil || upper(*r.cfg.SpeculativeAlgorithm) == "DFLASH"):
			if r.model.HasAsymmetricKV {
				return "fa4"
			}
			return "trtllm_mha"
		default:
			// SGLang's CUDA image installs flashinfer; FlashInfer has no attention sinks.
			if !r.model.HasAttentionSinks {
				return "flashinfer"
			}
			return "triton"
		}
	}
	switch {
	case r.chip.isHopperWithCUDA123():
		return "fa3"
	case r.chip.isSM100():
		return "flashinfer"
	}
	return "triton"
}

// mlaBackendPageConstraints is _mla_backend_page_constraints (overrides.py:1143-1200): each
// backend's page sizes, the adjustments chained on one value.
func (r *resolver) mlaBackendPageConstraints() {
	page := r.cfg.PageSize
	in := func(sizes ...int) bool {
		if page == nil {
			return false
		}
		for _, s := range sizes {
			if *page == s {
				return true
			}
		}
		return false
	}
	a, d, p, sd := r.cfg.AttentionBackend, r.cfg.DecodeAttentionBackend, r.cfg.PrefillAttentionBackend,
		r.cfg.SpeculativeDraftAttentionBackend
	if is(a, "flashmla") || is(d, "flashmla") {
		page = ptr(64)
	}
	if (is(a, "trtllm_mla") || is(d, "trtllm_mla")) && !in(32, 64) {
		page = ptr(64)
	}
	if (is(a, "cutedsl_mla") || is(d, "cutedsl_mla") || is(p, "cutedsl_mla")) && !in(32, 64) {
		page = ptr(64)
	}
	if (is(a, "trtllm_mha") || is(d, "trtllm_mha") || is(p, "trtllm_mha") || is(sd, "trtllm_mha")) && !in(16, 32, 64, 128) {
		page = ptr(64)
	}
	if (is(a, "hpc_ops") || is(d, "hpc_ops") || is(p, "hpc_ops")) && !in(64) {
		page = ptr(64)
	}
	if !equalJSON(page, r.cfg.PageSize) {
		r.declare("_mla_backend_page_constraints", "page_size", page)
	}
}

// fa4PageConstraint is _fa4_page_constraint (overrides.py:1273-1297) for the accepted
// architectures, none of which is DiffusionGemma.
func (r *resolver) fa4PageConstraint() {
	a, d, p := r.cfg.AttentionBackend, r.cfg.DecodeAttentionBackend, r.cfg.PrefillAttentionBackend
	topk := 0
	if r.cfg.SpeculativeEagleTopk != nil {
		topk = *r.cfg.SpeculativeEagleTopk
	}
	if (is(a, "fa4") || is(d, "fa4") || is(p, "fa4")) && !r.model.useMLA() && r.chip.isSM100() && topk <= 1 {
		r.declare("_fa4_page_constraint", "page_size", 128)
	}
}

func is(p *string, v string) bool { return p != nil && *p == v }

func upper(s string) string {
	b := []byte(s)
	for i, c := range b {
		if 'a' <= c && c <= 'z' {
			b[i] = c - 'a' + 'A'
		}
	}
	return string(b)
}
