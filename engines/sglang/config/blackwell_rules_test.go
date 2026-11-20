// Original to blis-sglang.

package config

import "testing"

// caseInput is the inputs of a case of the oracle, with the changes the test makes.
func caseInput(t *testing.T, name string, args map[string]any, change func(*Chip, *ModelFacts)) Input {
	t.Helper()
	for _, c := range readAnswers(t).Cases {
		if c.Name != name {
			continue
		}
		in := map[string]any{"model_path": "<model>"}
		for k, v := range c.Args {
			in[k] = v
		}
		for k, v := range args {
			in[k] = v
		}
		chip, facts := c.Chip, c.ModelFacts
		if change != nil {
			change(&chip, &facts)
		}
		return Input{Args: in, Chip: chip, Model: facts}
	}
	t.Fatalf("the oracle has no case %s", name)
	return Input{}
}

func resolveOK(t *testing.T, in Input) *Resolved {
	t.Helper()
	got, err := Resolve(in)
	if err != nil {
		t.Fatalf("Resolve refused: %v", err)
	}
	return got
}

func quantOf(r *Resolved) string {
	if r.Args.Quantization == nil {
		return "<unset>"
	}
	return *r.Args.Quantization
}

func orUnset(p *string) string {
	if p == nil {
		return "<unset>"
	}
	return *p
}

// On SM100, a DeepseekV3ForCausalLM checkpoint that declares no quantization is served as fp8 exactly
// when the first expert tensor of its header is float8_e4m3fn (_deepseek_moe_quant_resolution,
// overrides.py); a checkpoint that declares one keeps it, and another architecture is not given fp8 by
// this step whatever its header holds.
func TestADeepseekV3CheckpointWithAnFP8HeaderAndNoDeclaredQuantizationIsServedAsFP8(t *testing.T) {
	v3 := func(fp8 bool, method *string) func(*Chip, *ModelFacts) {
		return func(_ *Chip, m *ModelFacts) {
			m.HasFP8WeightsInCheckpoint, m.QuantMethod = fp8, method
		}
	}
	fp4 := "modelopt_fp4"

	got := resolveOK(t, caseInput(t, "dsr1-nvfp4v2-b200-tp8", nil, v3(true, nil)))
	if q := quantOf(got); q != "fp8" || got.DeclaredBy["quantization"] != "_deepseek_moe_quant_resolution" {
		t.Errorf("no declared quantization, fp8 header: quantization %s by %q, SGLang resolves fp8", q, got.DeclaredBy["quantization"])
	}
	got = resolveOK(t, caseInput(t, "dsr1-nvfp4v2-b200-tp8", nil, v3(false, nil)))
	if q := quantOf(got); q != "<unset>" {
		t.Errorf("no declared quantization, no fp8 header: quantization %s, SGLang leaves it unset", q)
	}
	got = resolveOK(t, caseInput(t, "dsr1-nvfp4v2-b200-tp8", nil, v3(true, &fp4)))
	if q := quantOf(got); q != "modelopt_fp4" {
		t.Errorf("a declared quantization with an fp8 header: quantization %s, the checkpoint's own is kept", q)
	}
	got = resolveOK(t, caseInput(t, "glm5-nvfp4-b200-tp8", nil, func(_ *Chip, m *ModelFacts) {
		m.HasFP8WeightsInCheckpoint, m.QuantMethod = true, nil
	}))
	if q := quantOf(got); q != "<unset>" {
		t.Errorf("a GlmMoeDsaForCausalLM checkpoint is not given %s by the DeepseekV3 rule", q)
	}
}

// _dsa_split_backend_resolution: Blackwell (compute capability 10) defaults the DSA backends to trtllm
// where Hopper (9) takes flashmla_kv with an fp8 KV cache and fa3 for decode otherwise.
func TestTheDSABackendsDefaultByComputeCapability(t *testing.T) {
	chip := func(major int) func(*Chip, *ModelFacts) {
		return func(c *Chip, _ *ModelFacts) { c.Capability = [2]int{major, 0} }
	}
	for _, tc := range []struct {
		kv                      string
		major                   int
		wantPrefill, wantDecode string
	}{
		{"fp8_e4m3", 10, "trtllm", "trtllm"},
		{"fp8_e4m3", 9, "flashmla_kv", "flashmla_kv"},
		{"bfloat16", 10, "flashmla_sparse", "trtllm"},
		{"bfloat16", 9, "flashmla_sparse", "fa3"},
	} {
		got := resolveOK(t, caseInput(t, "glm5-h200-tp8-recipe", map[string]any{"kv_cache_dtype": tc.kv}, chip(tc.major)))
		if p, d := orUnset(got.Args.DsaPrefillBackend), orUnset(got.Args.DsaDecodeBackend); p != tc.wantPrefill || d != tc.wantDecode {
			t.Errorf("kv_cache_dtype %s on compute capability %d: prefill %s, decode %s; SGLang resolves %s, %s",
				tc.kv, tc.major, p, d, tc.wantPrefill, tc.wantDecode)
		}
	}
}

// disable_prefill_cuda_graph_for_deepseek_trtllm_mla disables the captured prefill graph of a
// DeepseekV3ForCausalLM on trtllm_mla, and returns at once when the prefill graph is already disabled
// (cuda_graph_hook.py:279-312): a graph an earlier step disabled stays decided by that step.
func TestThePrefillGraphIsDisabledForTRTLLMMLAAndAnEarlierDisableStandsAsTheDecider(t *testing.T) {
	const step = "_disable_prefill_cuda_graph_for_deepseek_trtllm_mla"
	got := resolveOK(t, caseInput(t, "dsr1-fp8-b200-tp8", nil, nil))
	if got.DeclaredBy["cuda_graph_config"] != step || got.Args.CudaGraphConfig.Prefill.Backend != backendDisabled {
		t.Fatalf("DeepSeek-R1 on trtllm_mla: prefill graph %s decided by %q, SGLang disables it in %s",
			got.Args.CudaGraphConfig.Prefill.Backend, got.DeclaredBy["cuda_graph_config"], step)
	}
	// A multimodal model whose breakable graph is unsupported has its prefill graph disabled before this
	// step (apply_cuda_graph_compatibility); this step leaves it, and stays out of DeclaredBy.
	got = resolveOK(t, caseInput(t, "dsr1-fp8-b200-tp8", nil, func(_ *Chip, m *ModelFacts) {
		m.IsMultimodal, m.MultimodalBreakableGraphSupported = true, false
	}))
	if got.Args.CudaGraphConfig.Prefill.Backend != backendDisabled || got.DeclaredBy["cuda_graph_config"] == step {
		t.Errorf("an already disabled prefill graph: backend %s decided by %q, SGLang leaves the earlier step as its decider",
			got.Args.CudaGraphConfig.Prefill.Backend, got.DeclaredBy["cuda_graph_config"])
	}
}

// The prefill attention backend, where one is stated and not empty, is the one whose graph the step
// weighs (cuda_graph_hook.py:279-312); the attention backend stands for both phases otherwise. Resolve
// accepts no prefill_attention_backend yet, so the step is run on a resolver built for the purpose.
func TestThePrefillGraphFollowsThePrefillBackendWhereOneIsStated(t *testing.T) {
	const step = "_disable_prefill_cuda_graph_for_deepseek_trtllm_mla"
	for _, tc := range []struct {
		name            string
		attention       string
		prefill         *string
		wantDisabledNow bool
	}{
		{"trtllm_mla for both phases", "trtllm_mla", nil, true},
		{"an empty prefill backend defers to the attention backend", "trtllm_mla", ptr(""), true},
		{"a prefill backend that is not trtllm_mla keeps its graph", "trtllm_mla", ptr("fa3"), false},
		{"a prefill backend of trtllm_mla disables it", "fa3", ptr("trtllm_mla"), true},
		{"neither is trtllm_mla", "fa3", nil, false},
	} {
		r := &resolver{cfg: Defaults(), by: map[string]string{}, cgLocked: map[[2]string]bool{},
			model: ModelFacts{Architecture: "DeepseekV3ForCausalLM"}}
		r.cfg.CudaGraphConfig = defaultCudaGraphConfig()
		a := tc.attention
		r.cfg.AttentionBackend, r.cfg.PrefillAttentionBackend = &a, tc.prefill
		r.disablePrefillCudaGraphForDeepseekTrtllmMla()
		if got := r.by["cuda_graph_config"] == step; got != tc.wantDisabledNow {
			t.Errorf("%s: step ran = %v, want %v", tc.name, got, tc.wantDisabledNow)
		}
	}
}
