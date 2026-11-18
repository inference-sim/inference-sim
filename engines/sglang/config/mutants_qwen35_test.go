// Original to blis-sglang.

package config

import (
	"math"
	"strings"
	"testing"
)

// passResolver is a resolver for the Qwen3.5 passes with the SM100 chip and a Qwen3.5 model, its
// arguments the defaults but for what a test sets. The passes are called on it directly, because
// the fields they read (the MoE runner, the all-to-all backend, the overlap schedule) are not all
// ones Resolve accepts yet.
func passResolver(model ModelFacts, set func(*ServerArgs)) *resolver {
	a := Defaults()
	if set != nil {
		set(&a)
	}
	return &resolver{raw: a, cfg: a, by: map[string]string{}, chip: Chip{Capability: [2]int{10, 0}, MemoryMiB: 183359},
		model: model, cgLocked: map[[2]string]bool{}}
}

func qwen35Facts() ModelFacts {
	return ModelFacts{Architecture: "Qwen3_5MoeForConditionalGeneration", IsMultimodal: true, HasVisionConfig: true}
}

// refusal runs f and reports whether it raised SGLang's refusal, and the message it raised.
func refusalOf(f func()) (raised bool, msg string) {
	defer func() {
		if p := recover(); p != nil {
			rf, ok := p.(refusal)
			if !ok {
				panic(p)
			}
			raised, msg = true, rf.err.Message
		}
	}()
	f()
	return false, ""
}

func str(s string) *string { return &s }

// The W4A16 NVFP4 checkpoint on SM100 (qwen3_moe.py _qwen3_moe_family_overrides): the runner left on
// auto becomes marlin; marlin stays; any other runner is refused with SGLang's text.
func TestW4A16CheckpointChoosesMarlinOrRefusesTheRunner(t *testing.T) {
	facts := qwen35Facts()
	facts.QuantMethod = str("modelopt_mixed")
	facts.MixedPrecisionMoEQuantAlgos = []string{"W4A16_NVFP4"}
	for _, tc := range []struct {
		runner  string
		refuses bool
		marlin  bool
	}{{"auto", false, true}, {"marlin", false, false}, {"triton", true, false}, {"flashinfer_cutedsl", true, false}} {
		r := passResolver(facts, func(a *ServerArgs) { a.MoeRunnerBackend = tc.runner })
		raised, msg := refusalOf(r.qwen3MoeFamilyOverrides)
		if raised != tc.refuses || (tc.refuses && !strings.Contains(msg, "W4A16_NVFP4 MoE layers require --moe-runner-backend=marlin")) {
			t.Errorf("runner %s: refusal %q (raised %v), want refusal %v", tc.runner, msg, raised, tc.refuses)
		}
		if got := r.by["moe_runner_backend"] == "_qwen3_moe_family_overrides"; got != tc.marlin {
			t.Errorf("runner %s: marlin declared %v, want %v", tc.runner, got, tc.marlin)
		}
		if tc.marlin && r.cfg.MoeRunnerBackend != "marlin" {
			t.Errorf("runner %s resolved to %q, want marlin", tc.runner, r.cfg.MoeRunnerBackend)
		}
	}
}

// W4A16 is read only with the modelopt_mixed quantization: the same algos on an fp8 checkpoint take
// the FlashInfer TRT-LLM runner like any fp8 one.
func TestW4A16NeedsTheMixedQuantization(t *testing.T) {
	facts := qwen35Facts()
	facts.QuantMethod = str("fp8")
	facts.MixedPrecisionMoEQuantAlgos = []string{"W4A16_NVFP4"}
	r := passResolver(facts, nil)
	if raised, msg := refusalOf(r.qwen3MoeFamilyOverrides); raised {
		t.Fatalf("refused: %s", msg)
	}
	if r.cfg.MoeRunnerBackend != "flashinfer_trtllm" {
		t.Errorf("runner %q, want flashinfer_trtllm", r.cfg.MoeRunnerBackend)
	}
}

// The FlashInfer TRT-LLM runner is chosen only for the quantizations it serves, with no all-to-all
// backend and the runner left on auto.
func TestTRTLLMRunnerNeedsItsQuantizationAndNoAllToAll(t *testing.T) {
	for _, tc := range []struct {
		name  string
		quant *string
		a2a   string
		run   string
		want  string
	}{
		{"fp8", str("fp8"), "none", "auto", "flashinfer_trtllm"},
		{"none declared", nil, "none", "auto", "flashinfer_trtllm"},
		{"modelopt_fp4", str("modelopt_fp4"), "none", "auto", "flashinfer_trtllm"},
		{"another quantization", str("awq"), "none", "auto", "auto"},
		{"an all-to-all backend", str("fp8"), "deepep", "auto", "auto"},
		{"a stated runner", str("fp8"), "none", "triton", "triton"},
		{"an all-to-all backend and a stated runner", str("fp8"), "deepep", "triton", "triton"},
		{"another quantization and a stated runner", str("awq"), "none", "triton", "triton"},
	} {
		facts := qwen35Facts()
		facts.QuantMethod = tc.quant
		r := passResolver(facts, func(a *ServerArgs) { a.MoeA2aBackend, a.MoeRunnerBackend = tc.a2a, tc.run })
		if raised, msg := refusalOf(r.qwen3MoeFamilyOverrides); raised {
			t.Errorf("%s: refused: %s", tc.name, msg)
			continue
		}
		if r.cfg.MoeRunnerBackend != tc.want {
			t.Errorf("%s: runner %q, want %q", tc.name, r.cfg.MoeRunnerBackend, tc.want)
		}
	}
}

// no_buffer (validate_mamba_no_buffer): a page size other than one, the overlap schedule left on and
// the TRT-LLM MHA attention backend are each refused, with SGLang's text for the first that applies.
func TestNoBufferRefusesWhatItDoesNotSupport(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*ServerArgs)
		want string
	}{
		{"page 64", func(a *ServerArgs) { a.PageSize = ptr(64); a.DisableOverlapSchedule = true }, "no_buffer only supports page_size=1."},
		{"overlap", func(a *ServerArgs) { a.PageSize = ptr(1) }, "no_buffer do not support overlap schedule."},
		{"trtllm_mha", func(a *ServerArgs) { a.DisableOverlapSchedule = true; a.AttentionBackend = str("trtllm_mha") }, "no_buffer do not support trtllm_mha attention backend."},
		{"triton", func(a *ServerArgs) { a.DisableOverlapSchedule = true; a.AttentionBackend = str("triton") }, ""},
		{"no attention backend, page 1", func(a *ServerArgs) { a.DisableOverlapSchedule = true; a.PageSize = ptr(1) }, ""},
	} {
		r := passResolver(qwen35Facts(), tc.set)
		raised, msg := refusalOf(r.validateMambaNoBuffer)
		if raised != (tc.want != "") || !strings.Contains(msg, tc.want) {
			t.Errorf("%s: refusal %q (raised %v), want %q", tc.name, msg, raised, tc.want)
		}
	}
}

// extra_buffer (validate_mamba_extra_buffer): the track interval must cover the draft tokens (equal
// is enough), and the lazy strategy is refused under a prefill/decode split but not outside one.
func TestExtraBufferValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		set     func(*ServerArgs)
		refuses bool
		text    string
	}{
		{"interval below the draft tokens", func(a *ServerArgs) { a.SpeculativeNumDraftTokens = ptr(8); a.MambaTrackInterval = 7 }, true, ""},
		{"interval equal to the draft tokens", func(a *ServerArgs) { a.SpeculativeNumDraftTokens = ptr(8); a.MambaTrackInterval = 8 }, false, ""},
		{"lazy under a prefill split", func(a *ServerArgs) { a.MambaRadixCacheStrategy = "extra_buffer_lazy"; a.DisaggregationMode = "prefill" }, true, "extra_buffer_lazy unsupported under PD disaggregation"},
		{"lazy outside a split", func(a *ServerArgs) { a.MambaRadixCacheStrategy = "extra_buffer_lazy"; a.DisaggregationMode = "null" }, false, ""},
		{"extra_buffer under a split", func(a *ServerArgs) { a.MambaRadixCacheStrategy = "extra_buffer"; a.DisaggregationMode = "prefill" }, false, ""},
	} {
		r := passResolver(qwen35Facts(), func(a *ServerArgs) {
			a.MambaRadixCacheStrategy, a.LinearAttnBackend, a.DisaggregationMode = "extra_buffer", "triton", "null"
			a.MambaTrackInterval = 256
			tc.set(a)
		})
		raised, msg := refusalOf(r.validateMambaExtraBuffer)
		if raised != tc.refuses || !strings.Contains(msg, tc.text) {
			t.Errorf("%s: refusal %q (raised %v), want raised %v with %q", tc.name, msg, raised, tc.refuses, tc.text)
		}
	}
}

// The strategy "auto" (_mamba_radix_cache_resolution): extra_buffer when the overlap schedule or a
// paged cache wants it and the architecture supports it with the Triton linear-attention backend;
// otherwise no_buffer with the overlap schedule off.
func TestAutoStrategyFollowsOverlapPagingAndSupport(t *testing.T) {
	for _, tc := range []struct {
		name     string
		overlapO bool // disable_overlap_schedule
		page     *int
		linear   string
		want     string
	}{
		{"overlap on, triton", false, nil, "triton", "extra_buffer"},
		{"overlap off, page 64, triton", true, ptr(64), "triton", "extra_buffer"},
		{"overlap off, page 1, triton", true, ptr(1), "triton", "no_buffer"},
		{"overlap off, no page, triton", true, nil, "triton", "no_buffer"},
		{"overlap on, another linear backend", false, nil, "flashinfer", "no_buffer"},
		{"overlap off, page 64, another linear backend", true, ptr(64), "flashinfer", "no_buffer"},
	} {
		r := passResolver(qwen35Facts(), func(a *ServerArgs) {
			a.MambaRadixCacheStrategy, a.DisableOverlapSchedule, a.PageSize, a.LinearAttnBackend = "auto", tc.overlapO, tc.page, tc.linear
		})
		r.mambaRadixCacheResolution()
		if r.cfg.MambaRadixCacheStrategy != tc.want {
			t.Errorf("%s: strategy %q, want %q", tc.name, r.cfg.MambaRadixCacheStrategy, tc.want)
		}
		if tc.want == "no_buffer" && !r.cfg.DisableOverlapSchedule {
			t.Errorf("%s: no_buffer left the overlap schedule on", tc.name)
		}
	}
}

// A model is a VLM for the memory adjustment only when it is multimodal and is neither run
// language-only nor as a decode server.
func TestIsVLMNeedsEveryCondition(t *testing.T) {
	for _, tc := range []struct {
		name  string
		multi bool
		set   func(*ServerArgs)
		want  bool
	}{
		{"multimodal", true, nil, true},
		{"text only", false, nil, false},
		{"language_only", true, func(a *ServerArgs) { a.LanguageOnly = true }, false},
		{"language_model_only", true, func(a *ServerArgs) { a.LanguageModelOnly = true }, false},
		{"decode server", true, func(a *ServerArgs) { a.DisaggregationMode = "decode" }, false},
		{"prefill server", true, func(a *ServerArgs) { a.DisaggregationMode = "prefill" }, true},
		{"text only, language_only", false, func(a *ServerArgs) { a.LanguageOnly = true }, false},
	} {
		f := qwen35Facts()
		f.IsMultimodal = tc.multi
		r := passResolver(f, tc.set)
		if got := r.isVLM(); got != tc.want {
			t.Errorf("%s: isVLM %v, want %v", tc.name, got, tc.want)
		}
	}
}

// The memory fraction is scaled by 0.95 times the tower's size relative to ViT-L/14 (24 layers, width
// 1024), clamped to [0.8, 1.05]; a tower with no stated depth or width counts as ViT-L/14's.
func TestVLMMemoryAdjustmentReadsTheTowerSize(t *testing.T) {
	scale := func(layers, hidden *int) float64 {
		f := qwen35Facts()
		f.VisionNumHiddenLayers, f.VisionHiddenSize = layers, hidden
		return passResolver(f, nil).adjustMemFractionForVLM(1)
	}
	want := func(layers, hidden int) float64 {
		ratio := float64(layers*hidden*hidden) / float64(24*1024*1024)
		return 0.95 * math.Max(0.8, math.Min(1.05, 1-0.1*(ratio-1)))
	}
	for _, tc := range []struct{ layers, hidden int }{{24, 1152}, {12, 1152}, {48, 1152}, {24, 1024}, {2, 64}} {
		if got := scale(ptr(tc.layers), ptr(tc.hidden)); math.Abs(got-want(tc.layers, tc.hidden)) > 1e-12 {
			t.Errorf("%d layers, width %d: factor %v, want %v", tc.layers, tc.hidden, got, want(tc.layers, tc.hidden))
		}
	}
	if got := scale(nil, nil); math.Abs(got-0.95) > 1e-12 {
		t.Errorf("no stated tower: factor %v, want 0.95", got)
	}
	if got := scale(nil, ptr(1152)); math.Abs(got-want(24, 1152)) > 1e-12 {
		t.Errorf("only the width stated: factor %v, want %v", got, want(24, 1152))
	}
	if got := scale(ptr(12), nil); math.Abs(got-want(12, 1024)) > 1e-12 {
		t.Errorf("only the depth stated: factor %v, want %v", got, want(12, 1024))
	}
}
