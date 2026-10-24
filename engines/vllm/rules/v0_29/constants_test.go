package v0_29

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
		"naive", "nixl_ep", "pplx",
	}
	assertSet(t, "All2AllBackends", Pack().All2AllBackends, want)
}

// The sequence-parallel subset is smaller than the full backend set, and the
// engine's default backend is a member — so this path is reached by ordinary
// configurations rather than exotic ones.
func TestSPMoEBackendMembership(t *testing.T) {
	want := []string{
		"allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency",
		"flashinfer_nvlink_one_sided", "mori_high_throughput", "mori_low_latency",
		"nixl_ep",
	}
	p := Pack()
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
	assertSet(t, "MambaCacheModes", p.MambaCacheModes,
		[]string{"align", "all", "none"})
	assertSet(t, "DCPCommBackends", p.DCPCommBackends,
		[]string{"a2a", "ag_rs"})
	// Seventeen cache dtypes. The count is the point: a narrower set silently
	// rejects formats the engine accepts, including nvfp4.
	if len(p.CacheDTypes) != 17 {
		t.Errorf("CacheDTypes: size = %d, want 17", len(p.CacheDTypes))
	}
	for _, want := range []string{"auto", "fp8", "nvfp4", "fp8_ds_mla",
		"int8_per_token_head", "turboquant_k8v4"} {
		if !p.CacheDTypes[want] {
			t.Errorf("CacheDTypes: missing %q", want)
		}
	}
	assertSet(t, "SchedulerPolicies", p.SchedulerPolicies,
		[]string{"fcfs", "priority"})
	assertSet(t, "OffloadSpecs", p.OffloadSpecs,
		[]string{"CPUOffloadingSpec", "TieringOffloadingSpec"})
	// Thirty quantization methods and sixteen connectors. As with the cache dtypes the
	// count is the guard: a narrower set warns on an in-tree method or connector the
	// engine ships, which trains a reader to ignore the warning. The members spot-checked
	// are the ones the report corpus and offload paths exercise.
	if len(p.Quantizations) != 30 {
		t.Errorf("Quantizations: size = %d, want 30", len(p.Quantizations))
	}
	for _, want := range []string{"fp8", "compressed-tensors", "gptq_marlin",
		"deepseek_v4_fp8", "mxfp4"} {
		if !p.Quantizations[want] {
			t.Errorf("Quantizations: missing %q", want)
		}
	}
	if len(p.Connectors) != 16 {
		t.Errorf("Connectors: size = %d, want 16", len(p.Connectors))
	}
	for _, want := range []string{"NixlConnector", "LMCacheConnectorV1",
		"OffloadingConnector", "MultiConnector"} {
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
	if !p.CascadeAttnOptIn {
		t.Error("cascade attention is disabled by default in this release")
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

// The async-compatible subset must be a subset of the accepted methods; otherwise a
// method could be async-compatible and simultaneously unrecognized.
func TestAsyncCompatibleSpecIsASubset(t *testing.T) {
	p := Pack()
	for m := range p.AsyncCompatibleSpec {
		if !p.SpeculativeMethods[m] {
			t.Errorf("%q is async-compatible but not an accepted method", m)
		}
	}
	// Medusa is accepted and is NOT async-compatible, which is the case that makes
	// the distinction worth encoding.
	if !p.SpeculativeMethods["medusa"] {
		t.Error("medusa should be an accepted method")
	}
	if p.AsyncCompatibleSpec["medusa"] {
		t.Error("medusa disables async scheduling, so it is not in the subset")
	}
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
