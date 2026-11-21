// Original to blis-sglang.

package config

import (
	"testing"
)

// The refusals below are SGLang's own (SGL-6, SIM-9): each is raised with SGLang's text at the site
// SGLang raises it, and each test pairs the refused input with the nearest accepted ones, so that
// a refusal removed, or widened to refuse what SGLang accepts, fails. The passes are called on a
// resolver directly, as the Qwen3.5 tests do, because the inputs that reach them are not all ones
// Resolve accepts yet. Citations are path:lines@297e54fd, the pin of SGLang v0.5.23.

// refusalAt runs f and returns the refusal it raised, or nil.
func refusalAt(f func()) (ref *UpstreamRefusal) {
	defer func() {
		if p := recover(); p != nil {
			rf, ok := p.(refusal)
			if !ok {
				panic(p)
			}
			ref = rf.err
		}
	}()
	f()
	return nil
}

type pairCase struct {
	name string
	set  func(*ServerArgs)
	// refused is the message SGLang raises, or "" for the input it accepts; a refusal with no
	// message is an assert, named by assertion.
	refused   string
	assertion bool
}

func f64(v float64) *float64 { return &v }
func i64(v int) *int         { return &v }

func runPairs(t *testing.T, model ModelFacts, site string, pass func(*resolver), cases []pairCase) {
	t.Helper()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := passResolver(model, func(a *ServerArgs) {
				a.CudaGraphConfig = defaultCudaGraphConfig()
				if tc.set != nil {
					tc.set(a)
				}
			})
			got := refusalAt(func() { pass(r) })
			if tc.refused == "" && !tc.assertion {
				if got != nil {
					t.Fatalf("refused an input SGLang accepts: %v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("accepted an input SGLang refuses (want %q at %s)", tc.refused, site)
			}
			if got.Site != "sglang/srt/arg_groups/"+site {
				t.Errorf("refused at %q, want %q", got.Site, site)
			}
			if got.Message != tc.refused {
				t.Errorf("refusal message %q, want %q", got.Message, tc.refused)
			}
		})
	}
}

// kv_cache_hook.py handle_cache_compatibility: the hierarchical cache and the disabled radix cache
// are each accepted alone and refused together.
func TestHierarchicalCacheWithDisabledRadixCacheIsRefused(t *testing.T) {
	runPairs(t, ModelFacts{}, "kv_cache_hook.py:handle_cache_compatibility", (*resolver).handleCacheCompatibility, []pairCase{
		{name: "neither"},
		{name: "hicache alone", set: func(a *ServerArgs) { a.EnableHierarchicalCache = true }},
		{name: "no radix cache alone", set: func(a *ServerArgs) { a.DisableRadixCache = true }},
		{name: "both", set: func(a *ServerArgs) { a.EnableHierarchicalCache, a.DisableRadixCache = true, true },
			refused: "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive and cannot be used at the same time. Please use only one of them."},
	})
}

// hicache_hook.py handle_hicache_ratio_default: the host-memory fraction is in (0, 1].
func TestHicacheHostMemoryFractionOutsideZeroToOneIsRefused(t *testing.T) {
	const msg = "--hicache-host-memory-fraction must be in (0, 1]."
	runPairs(t, ModelFacts{}, "hicache_hook.py:handle_hicache_ratio_default", (*resolver).handleHicacheRatioDefault, []pairCase{
		{name: "unset"},
		{name: "half", set: func(a *ServerArgs) { a.HicacheHostMemoryFraction = f64(0.5) }},
		{name: "one", set: func(a *ServerArgs) { a.HicacheHostMemoryFraction = f64(1) }},
		{name: "zero", set: func(a *ServerArgs) { a.HicacheHostMemoryFraction = f64(0) }, refused: msg},
		{name: "negative", set: func(a *ServerArgs) { a.HicacheHostMemoryFraction = f64(-0.25) }, refused: msg},
		{name: "above one", set: func(a *ServerArgs) { a.HicacheHostMemoryFraction = f64(1.5) }, refused: msg},
	})
}

// parallel_hook.py handle_data_parallelism: the attention data-parallel size is positive, is not
// combined with data-parallel replicas, and divides the tensor-parallel size.
func TestDataParallelismCombinationsAreRefused(t *testing.T) {
	dp := func(tp, dpSize, attn int) func(*ServerArgs) {
		return func(a *ServerArgs) {
			a.TpSize, a.DpSize, a.AttnDpSize = tp, dpSize, attn
			a.ChunkedPrefillSize = i64(8192)
		}
	}
	runPairs(t, ModelFacts{}, "parallel_hook.py:handle_data_parallelism", (*resolver).handleDataParallelism, []pairCase{
		{name: "replicas alone", set: dp(2, 2, 1)},
		{name: "attention DP alone", set: dp(8, 1, 4)},
		{name: "attention DP 0", set: dp(8, 1, 0), refused: "--attn-dp-size must be positive (got 0)."},
		{name: "replicas with attention DP", set: dp(8, 2, 4),
			refused: "--dp-size 2 with --attn-dp-size 4: data-parallel replicas combined with attention data parallelism are not supported."},
		{name: "attention DP does not divide TP", set: dp(8, 1, 3), assertion: true},
	})
}

// overrides.py _dsa_kv_cache_dtype_default: learnable sinks need a bfloat16 cache, and a DSA model
// takes bf16 or fp8_e4m3 only.
func TestDSAKVCacheDtypeRefusals(t *testing.T) {
	sinks := ModelFacts{LearnableSink: true}
	const site = "overrides.py:_dsa_kv_cache_dtype_default"
	kv := func(d string) func(*ServerArgs) { return func(a *ServerArgs) { a.KvCacheDtype = d } }
	runPairs(t, sinks, site, (*resolver).dsaKVCacheDtypeDefault, []pairCase{
		{name: "sinks auto", set: kv("auto")},
		{name: "sinks bf16", set: kv("bf16")},
		{name: "sinks bfloat16", set: kv("bfloat16")},
		{name: "sinks fp8", set: kv("fp8_e4m3"), refused: "Learnable DSA attention sinks require a bfloat16 KV cache; got kv_cache_dtype=fp8_e4m3."},
	})
	runPairs(t, ModelFacts{}, site, (*resolver).dsaKVCacheDtypeDefault, []pairCase{
		{name: "no sinks fp8", set: kv("fp8_e4m3")},
		{name: "no sinks bf16", set: kv("bf16")},
		{name: "no sinks fp8_e5m2", set: kv("fp8_e5m2"), refused: "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"},
	})
}

// deepseek_v4_hook.py apply_deepseek_v4_defaults and overrides.py _deepseek_v4_kv_cache_dtype.
func TestDeepseekV4Refusals(t *testing.T) {
	v4 := ModelFacts{Architecture: "DeepseekV4ForCausalLM", ModelType: "deepseek_v4"}
	algo := func(a string, topk *int) func(*ServerArgs) {
		return func(s *ServerArgs) { s.SpeculativeAlgorithm, s.SpeculativeEagleTopk = &a, topk }
	}
	kv := func(d string) func(*ServerArgs) { return func(a *ServerArgs) { a.KvCacheDtype = d } }
	runPairs(t, v4, "overrides.py:_deepseek_v4_kv_cache_dtype", (*resolver).applyDeepseekV4Defaults, []pairCase{
		{name: "kv bf16", set: kv("bfloat16")},
		{name: "kv fp8_e4m3", set: kv("fp8_e4m3")},
		{name: "kv fp8_e5m2", set: kv("fp8_e5m2"), refused: "fp8_e5m2 is not supported for DeepseekV4ForCausalLM"},
	})
	const site = "deepseek_v4_hook.py:apply_deepseek_v4_defaults"
	runPairs(t, v4, site, (*resolver).applyDeepseekV4Defaults, []pairCase{
		{name: "DSPARK", set: algo("DSPARK", nil)},
		{name: "EAGLE topk 1", set: algo("EAGLE", i64(1))},
		{name: "NGRAM", set: algo("NGRAM", nil), refused: "Only EAGLE and DSPARK speculative algorithms are supported for DeepseekV4ForCausalLM"},
		{name: "EAGLE topk 2", set: algo("EAGLE", i64(2)), refused: "Only EAGLE speculative algorithm with topk == 1 is supported for DeepseekV4ForCausalLM"},
		{name: "EAGLE topk unset", set: algo("EAGLE", nil), refused: "Only EAGLE speculative algorithm with topk == 1 is supported for DeepseekV4ForCausalLM"},
	})
}

// deepseek_v4_hook.py validate_deepseek_v41_features: each feature V4.1 does not support is
// refused by name, and the same feature on another model is not looked at.
func TestDeepseekV41UnsupportedFeaturesAreRefused(t *testing.T) {
	const site = "deepseek_v4_hook.py:validate_deepseek_v41_features"
	v41 := ModelFacts{Architecture: "DeepseekV4ForCausalLM", ModelType: "deepseek_v41"}
	refuse := func(feature string) string {
		return "DeepSeek-V4.1 does not support " + feature + " yet; disable it to serve this model."
	}
	algo := func(a string) func(*ServerArgs) { return func(s *ServerArgs) { s.SpeculativeAlgorithm = &a } }
	cases := []pairCase{
		{name: "defaults"},
		{name: "DSPARK", set: algo("DSPARK")},
		{name: "EAGLE", set: algo("EAGLE"), refused: refuse("speculative decoding other than DSpark")},
		{name: "HiSparse", set: func(a *ServerArgs) { a.EnableHisparse = true }, refused: refuse("HiSparse")},
		{name: "two-batch overlap", set: func(a *ServerArgs) { a.EnableTwoBatchOverlap = true }, refused: refuse("two-batch overlap")},
		{name: "pipeline parallelism", set: func(a *ServerArgs) { a.PpSize = 2 }, refused: refuse("pipeline parallelism")},
	}
	runPairs(t, v41, site, (*resolver).validateDeepseekV41Features, cases)
	// The same settings on DeepSeek-V4 (not V4.1) are not this check's concern.
	for _, tc := range cases[1:] {
		tc.refused = ""
		t.Run("not V4.1 "+tc.name, func(t *testing.T) {
			r := passResolver(ModelFacts{Architecture: "DeepseekV4ForCausalLM", ModelType: "deepseek_v4"}, func(a *ServerArgs) {
				a.CudaGraphConfig = defaultCudaGraphConfig()
				tc.set(a)
			})
			if got := refusalAt(r.validateDeepseekV41Features); got != nil {
				t.Fatalf("refused a model that is not V4.1: %v", got)
			}
		})
	}
}

// mamba_hook.py validate_mamba_extra_buffer and overrides.py mamba_cache_chunk_size.
func TestMambaExtraBufferRefusals(t *testing.T) {
	const site = "mamba_hook.py:validate_mamba_extra_buffer"
	extra := func(set func(*ServerArgs)) func(*ServerArgs) {
		return func(a *ServerArgs) {
			a.LinearAttnBackend, a.MambaRadixCacheStrategy = "triton", "extra_buffer"
			if set != nil {
				set(a)
			}
		}
	}
	runPairs(t, qwen35Facts(), site, (*resolver).validateMambaExtraBuffer, []pairCase{
		{name: "extra_buffer", set: extra(nil)},
		{name: "extra_buffer_lazy, no PD", set: extra(func(a *ServerArgs) { a.MambaRadixCacheStrategy = "extra_buffer_lazy" })},
		{name: "extra_buffer_lazy under PD", set: extra(func(a *ServerArgs) {
			a.MambaRadixCacheStrategy, a.DisaggregationMode = "extra_buffer_lazy", "decode"
		}), refused: "extra_buffer_lazy unsupported under PD disaggregation; use --mamba-radix-cache-strategy extra_buffer."},
		{name: "track interval below the draft tokens", set: extra(func(a *ServerArgs) {
			a.MambaTrackInterval, a.SpeculativeNumDraftTokens = 4, i64(8)
		}), assertion: true},
		{name: "track interval not a multiple of the page", set: extra(func(a *ServerArgs) {
			a.MambaTrackInterval, a.PageSize = 256, i64(96)
		}), assertion: true},
	})
	// An architecture the strategy is not supported for, and the same model on another linear backend.
	runPairs(t, ModelFacts{Architecture: "LlamaForCausalLM"}, site, (*resolver).validateMambaExtraBuffer, []pairCase{
		{name: "unsupported architecture", set: extra(nil), refused: "extra_buffer is not supported for LlamaForCausalLM; use no_buffer."},
	})
	runPairs(t, qwen35Facts(), site, (*resolver).validateMambaExtraBuffer, []pairCase{
		{name: "unsupported linear backend", set: func(a *ServerArgs) { a.LinearAttnBackend = "flashinfer" },
			refused: "extra_buffer is not supported for Qwen3_5MoeForConditionalGeneration; use no_buffer."},
	})
	// The page size and FLA's chunk (64) divide one another: 32 and 128 do, 96 does not.
	chunk := func(p, interval int) func(*ServerArgs) {
		return extra(func(a *ServerArgs) { a.PageSize, a.MambaTrackInterval = i64(p), interval })
	}
	runPairs(t, qwen35Facts(), "overrides.py:mamba_cache_chunk_size", (*resolver).validateMambaExtraBuffer, []pairCase{
		{name: "page 32", set: chunk(32, 576)},
		{name: "page 128", set: chunk(128, 512)},
		{name: "page 96", set: chunk(96, 576),
			refused: "For SSM models, either chunk_size or page_size must be divisible by the other, got chunk_size=64, page_size=96"},
	})
}

// speculative_hook.py _handle_eagle_family: the TRT-LLM MHA backend takes topk 1 only, and topk > 1
// with a page larger than 1 needs flashinfer, fa3 or triton.
func TestEagleTopkRefusals(t *testing.T) {
	const site = "speculative_hook.py:_handle_eagle_family"
	eagle := func(topk, page int, backend string) func(*ServerArgs) {
		return func(a *ServerArgs) {
			algo := "EAGLE"
			a.SpeculativeAlgorithm = &algo
			a.SpeculativeNumSteps, a.SpeculativeEagleTopk, a.SpeculativeNumDraftTokens = i64(3), i64(topk), i64(4)
			if topk > 1 {
				a.SpeculativeNumDraftTokens = i64(8)
			}
			a.PageSize = i64(page)
			if backend != "" {
				a.AttentionBackend = &backend
			}
		}
	}
	model := ModelFacts{Architecture: "Qwen3ForCausalLM"}
	runPairs(t, model, site, (*resolver).handleEagleFamily, []pairCase{
		{name: "trtllm_mha topk 1", set: eagle(1, 1, "trtllm_mha")},
		{name: "fa3 topk 2", set: eagle(2, 1, "fa3")},
		{name: "trtllm_mha topk 2", set: eagle(2, 1, "trtllm_mha"), refused: "trtllm_mha backend only supports topk = 1 for speculative decoding."},
		{name: "topk 2 page 16 on fa3", set: eagle(2, 16, "fa3")},
		{name: "topk 2 page 16 on flashinfer", set: eagle(2, 16, "flashinfer")},
		{name: "topk 2 page 16 on triton", set: eagle(2, 16, "triton")},
		{name: "topk 1 page 16 on cutlass_mla", set: eagle(1, 16, "cutlass_mla")},
		{name: "topk 2 page 1 on cutlass_mla", set: eagle(2, 1, "cutlass_mla")},
		{name: "topk 2 page 16 on cutlass_mla", set: eagle(2, 16, "cutlass_mla"), refused: "speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton'); got attention_backend='cutlass_mla'. Use page_size == 1 or one of those backends."},
		{name: "topk 2 page 16 on no backend", set: eagle(2, 16, ""), refused: "speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton'); got attention_backend=''. Use page_size == 1 or one of those backends."},
	})
}
