package v0_33

import "testing"

// The pack's constants are transcribed from one engine release's configuration
// surface. These tests pin each set's membership so a transcription slip is a
// failure rather than a silently narrower vocabulary — which is how an accepted
// backend name comes to be rejected, or a rejected one accepted.
//
// When a new release changes one of these, the correct response is a NEW pack under
// the new version, not an edit here. Editing here would retroactively change what a
// scenario pinned to this version means.

func TestAll2AllBackendMembership(t *testing.T) {
	want := []string{
		"allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency",
		"deepep_v2", "flashinfer_all2allv", "flashinfer_nvlink_one_sided",
		"flashinfer_nvlink_two_sided", "mori_high_throughput", "mori_low_latency",
		"mori", "moonep", "naive", "nixl_ep", "passthrough", "pplx",
		"alltoall_batched", "auto",
	}
	assertSet(t, "All2AllBackends", Pack().All2AllBackends, want)
}

// The sequence-parallel subset is smaller than the full backend set, and the
// engine's default backend is a member — so this path is reached by ordinary
// configurations rather than exotic ones.
func TestSPMoEBackendMembership(t *testing.T) {
	want := []string{
		"allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency",
		"deepep_v2", "flashinfer_nvlink_one_sided", "mori_high_throughput",
		"mori_low_latency", "mori", "nixl_ep", "passthrough",
	}
	p := Pack()
	if p.SPMoEBackends["moonep"] {
		t.Error("moonep is not in the engine's sequence-parallel subset")
	}
	assertSet(t, "SPMoEBackends", p.SPMoEBackends, want)
	if !p.SPMoEBackends["allgather_reducescatter"] {
		t.Error("the default backend must be in the sequence-parallel subset")
	}
	for name := range p.SPMoEBackends {
		if !p.All2AllBackends[name] {
			t.Errorf("%q is sequence-parallel-capable but not an accepted backend", name)
		}
	}
}

func TestSmallVocabularies(t *testing.T) {
	p := Pack()
	// The engine has no allreduce-backend enum; the deployment field records whether
	// the SM-consuming kernel is requested or declined.
	assertSet(t, "AllReduceBackends", p.AllReduceBackends,
		[]string{"custom", "nccl"})
	// Five graph modes, not four: FULL_DECODE_ONLY captures decode and runs prefill
	// eagerly, which is a distinct host-term profile.
	assertSet(t, "CUDAGraphModes", p.CUDAGraphModes,
		[]string{"FULL", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "NONE",
			"PIECEWISE"})
	assertSet(t, "DCPCommBackends", p.DCPCommBackends,
		[]string{"a2a", "ag_rs"})
	// Twenty-three cache dtypes: the previous release's twenty-two and mxfp4_mla.
	if len(p.CacheDTypes) != 23 {
		t.Errorf("CacheDTypes: size = %d, want 23", len(p.CacheDTypes))
	}
	for _, want := range []string{"auto", "fp8", "nvfp4", "fp8_ds_mla", "nvfp4_ds_mla",
		"int8_per_token_head", "turboquant_k8v4", "ultraquant_4bit", "octave_k3v3_compact", "mxfp4_mla"} {
		if !p.CacheDTypes[want] {
			t.Errorf("CacheDTypes: missing %q", want)
		}
	}
	assertSet(t, "SchedulerPolicies", p.SchedulerPolicies,
		[]string{"fcfs", "priority"})
	assertSet(t, "OffloadSpecs", p.OffloadSpecs,
		[]string{"CPUOffloadingSpec", "TieringOffloadingSpec"})
	// Thirty-five quantization methods and eighteen connectors. As with the cache dtypes the
	// count is the guard: a narrower set warns on an in-tree method or connector the
	// engine ships, which trains a reader to ignore the warning. The members spot-checked
	// are the ones the report corpus and offload paths exercise.
	if len(p.Quantizations) != 35 {
		t.Errorf("Quantizations: size = %d, want 35", len(p.Quantizations))
	}
	for _, want := range []string{"fp8", "compressed-tensors", "gptq_marlin",
		"deepseek_v4_fp8", "mxfp4", "w4afp8", "modelopt_int8"} {
		if !p.Quantizations[want] {
			t.Errorf("Quantizations: missing %q", want)
		}
	}
	if p.Quantizations["fp_quant"] {
		t.Error("fp_quant left the QuantizationMethods literal")
	}
	if len(p.Connectors) != 18 {
		t.Errorf("Connectors: size = %d, want 18", len(p.Connectors))
	}
	for _, want := range []string{"NixlConnector", "HiSparseConnector", "LMCacheConnectorV1",
		"OffloadingConnector", "MultiConnector", "UMBPStoreConnector"} {
		if !p.Connectors[want] {
			t.Errorf("Connectors: missing %q", want)
		}
	}
	// Two in-tree eviction policies (lru, arc). The engine extends the registry out of
	// tree, so the set is small and the rule warns; the count still guards a silent drop.
	assertSet(t, "EvictionPolicies", p.EvictionPolicies, []string{"arc", "lru"})
}

func TestNumericConstants(t *testing.T) {
	p := Pack()
	if p.DefaultDBODecodeThreshold != 32 {
		t.Errorf("DBO decode threshold = %d, want 32", p.DefaultDBODecodeThreshold)
	}
	if p.DefaultDBOPrefillThreshold != 512 {
		t.Errorf("DBO prefill threshold = %d, want 512", p.DefaultDBOPrefillThreshold)
	}
	if p.TritonFetchSMs != 12 {
		t.Errorf("Triton fetch SMs = %d, want 12", p.TritonFetchSMs)
	}
	if p.TritonFetchPageBytes != 28*1024 {
		t.Errorf("Triton page threshold = %d, want %d", p.TritonFetchPageBytes, 28*1024)
	}
	if p.CascadeAttnOptIn {
		t.Error("cascade attention was removed in this release; the pack must not model it")
	}
	want := map[int]bool{2: true, 4: true, 6: true, 8: true, 16: true}
	if len(p.CustomAllReduceWorldSizes) != len(want) {
		t.Errorf("custom all-reduce world sizes = %v, want %v",
			p.CustomAllReduceWorldSizes, want)
	}
	for k := range want {
		if !p.CustomAllReduceWorldSizes[k] {
			t.Errorf("world size %d is supported but missing", k)
		}
	}
	// A width the kernel does not support must be absent, or the reachability rule
	// would never fire.
	for _, absent := range []int{1, 3, 5, 7, 12, 32} {
		if p.CustomAllReduceWorldSizes[absent] {
			t.Errorf("world size %d is not supported by the kernel", absent)
		}
	}
}

// With the V1 model runner gone, async scheduling stays enabled for every speculative
// method (vllm/config/vllm.py no longer has the V1 runner's incompatibilities), so the
// async-compatible set is the accepted set. A method accepted but absent from it would be
// priced with the host term exposed that the engine hides.
func TestAsyncCompatibleSpecIsEveryAcceptedMethod(t *testing.T) {
	p := Pack()
	assertSet(t, "AsyncCompatibleSpec", p.AsyncCompatibleSpec, keysOf(p.SpeculativeMethods))
	for _, m := range []string{"ngram", "suffix", "dflash", "dfly", "ngram_gpu", "draft_model", "dspark"} {
		if compatible, known := p.AsyncCompatibleSpeculative(m); !known || !compatible {
			t.Errorf("%q keeps async scheduling: compatible=%v known=%v", m, compatible, known)
		}
	}
	if p.AsyncCompatibleSpec["custom_class"] || p.SpeculativeMethods["custom_class"] {
		t.Error("custom_class left the SpeculativeMethod literal")
	}
}

// The speculative sets are the engine's whole literal, not a curated subset: 39 accepted
// methods (custom_class removed, dfly and nemotron3_5_transcribe_mtp added), every one of
// which keeps async scheduling on. The members spot-checked are the ones the v0.31
// campaign models need: DeepSeek-V4.1's dspark and GLM-5.3-Flash's glm5_next_mtp, which
// v0.29.0 does not accept.
func TestSpeculativeMethods(t *testing.T) {
	p := Pack()
	if len(p.SpeculativeMethods) != 39 {
		t.Errorf("SpeculativeMethods: size = %d, want 39", len(p.SpeculativeMethods))
	}
	for _, m := range []string{"dspark", "glm5_next_mtp", "mtp", "deepseek_mtp", "suffix",
		"dfly", "nemotron3_5_transcribe_mtp"} {
		if !p.SpeculativeMethods[m] {
			t.Errorf("%q should be accepted", m)
		}
	}
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func assertSet(t *testing.T, name string, got map[string]bool, want []string) {
	t.Helper()
	if len(got) != len(want) {
		have := make([]string, 0, len(got))
		for k := range got {
			have = append(have, k)
		}
		t.Fatalf("%s: size = %d, want %d (have %v)", name, len(got), len(want), have)
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("%s: missing %q", name, w)
		}
	}
}
