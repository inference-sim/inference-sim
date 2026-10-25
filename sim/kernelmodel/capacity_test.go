package kernelmodel

import (
	"os"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/sim/engine"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// A KV budget must be derived from the kernel's own memory answers, and it must be
// internally consistent: the blocks it reports, times the bytes one block costs, must fit
// in what is left of the device after fixed occupancy. A budget that did not satisfy this
// would admit requests the modelled device cannot hold.
func TestKVBudgetFitsTheDeviceItWasDerivedFrom(t *testing.T) {
	m := open(t, "gpt-oss-120b-h200-fp4-vllm-tp8.yaml")
	b, err := m.KVBudget()
	if err != nil {
		t.Fatalf("KVBudget: %v", err)
	}
	if b.TotalBlocks <= 0 {
		t.Fatalf("budget has %d blocks", b.TotalBlocks)
	}
	if used := b.TotalBlocks * b.PerBlockBytes; used > b.AllocatableBytes {
		t.Errorf("%d blocks x %d bytes = %d exceeds the %d allocatable",
			b.TotalBlocks, b.PerBlockBytes, used, b.AllocatableBytes)
	}
	if b.FixedBytes+b.AllocatableBytes != b.BudgetBytes {
		t.Errorf("fixed %d + allocatable %d != budget %d",
			b.FixedBytes, b.AllocatableBytes, b.BudgetBytes)
	}
	if b.BudgetBytes >= b.DeviceBytes {
		t.Errorf("budget %d is not below device %d; utilization was not applied",
			b.BudgetBytes, b.DeviceBytes)
	}
}

// A narrower tensor-parallel width must yield FEWER KV blocks per rank, because each rank
// holds a larger KV share. This is the behaviour a resident-batch bound depends on: if the
// budget did not move with the layout, every deployment would admit the same number of
// requests regardless of how it was sharded.
func TestKVBudgetGrowsWithTensorParallelWidth(t *testing.T) {
	widths := []struct {
		tp       int
		scenario string
	}{
		{1, "gpt-oss-120b-h200-fp4-vllm-tp1.yaml"},
		{2, "gpt-oss-120b-h200-fp4-vllm-tp2.yaml"},
		{4, "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"},
		{8, "gpt-oss-120b-h200-fp4-vllm-tp8.yaml"},
	}
	var prev int64
	for i, w := range widths {
		b, err := open(t, w.scenario).KVBudget()
		if err != nil {
			t.Fatalf("tp=%d: %v", w.tp, err)
		}
		if i > 0 && b.TotalBlocks <= prev {
			t.Errorf("tp=%d gives %d blocks, not above the %d at the narrower width",
				w.tp, b.TotalBlocks, prev)
		}
		prev = b.TotalBlocks
	}
}

// A per-rank occupancy must be physically possible: it cannot exceed the part it sits on.
// The guard in KVBudget exists because an impossible figure divided into a budget yields a
// plausible-looking capacity built on a wrong weight term, and the resident batch is the
// quantity this experiment measures.
//
// This asserts the guard does not fire on any evaluation deployment -- i.e. that every one
// of them reports a possible occupancy and a derivable budget. It replaced a test that
// asserted the opposite: before the upstream /expertTensorShards fix, 17 of 39 deployments
// reported more per-rank memory than their device had.
func TestEveryEvaluationDeploymentHasADerivableBudget(t *testing.T) {
	for _, scenario := range evaluationScenarios(t) {
		m, err := Open(scenario, repos())
		if err != nil {
			t.Errorf("%s: Open: %v", scenario, err)
			continue
		}
		b, err := m.KVBudget()
		if err != nil {
			t.Errorf("%s: %v", scenario, err)
			continue
		}
		if b.TotalBlocks <= 0 {
			t.Errorf("%s: %d blocks", scenario, b.TotalBlocks)
		}
		if b.FixedBytes >= b.DeviceBytes {
			t.Errorf("%s: per-rank occupancy %.1f GiB does not fit a %.1f GiB device",
				scenario, float64(b.FixedBytes)/(1<<30),
				float64(b.DeviceBytes)/(1<<30))
		}
	}
}

// Expert weights must be TENSOR-SLICED when expert parallelism is off: a rank holds every
// expert, each divided across the tensor-parallel group. The signature is arithmetic rather
// than a recorded byte count.
//
// GLM-5 served fp8 has 75 MoE layers of 256 experts, three matrices each at 2048x6144, so
// 675 GiB of expert parameters for the whole model and 84.4 GiB for one rank of eight. A
// rank reporting the whole model's figure is the defect this test was written against; it
// now pins the fix.
func TestExpertWeightsAreTensorSlicedWhenExpertParallelismIsOff(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	if w := m.Kernel().Resolved().ExpertParallelWidth; w > 1 {
		t.Skipf("this scenario resolved expert-parallel width %d; the test needs it off", w)
	}
	const (
		wholeModelExpertBytes = 75.0 * 256 * 3 * 2048 * 6144
		tp                    = 8.0
	)
	perRank := wholeModelExpertBytes / tp
	got := float64(m.Kernel().FixedBytes().Weights)

	// Above the expert slice (dense layers, embeddings and the head add to it) and well
	// below twice it. The undivided figure is 8x, so this band excludes it decisively.
	if got < perRank || got > perRank*1.5 {
		t.Errorf("a tp=8 rank reports %.1f GiB of weights; the expert slice is %.1f GiB "+
			"and the dense remainder is small, so anything outside [%.1f, %.1f] GiB means "+
			"the expert term is sharded wrongly",
			got/(1<<30), perRank/(1<<30), perRank/(1<<30), perRank*1.5/(1<<30))
	}
}

// evaluationScenarios lists the scenario files the comparison runs over. Reading the
// directory rather than hardcoding a list means a scenario added upstream is covered
// automatically instead of silently skipped.
func evaluationScenarios(t *testing.T) []string {
	t.Helper()
	ents, err := os.ReadDir(repos().Scenarios)
	if err != nil {
		t.Fatalf("reading scenarios: %v", err)
	}
	var out []string
	for _, e := range ents {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".yaml") {
			out = append(out, e.Name())
		}
	}
	if len(out) == 0 {
		t.Fatal("no scenario files found, so this test would pass vacuously")
	}
	return out
}

// #1941: the budget charges each resident sequence what the engine reserves for it at
// admission whatever its length -- the kernel's SequenceAdmissionBytes, window rings at vLLM's
// admission cap plus compressor/indexer state -- in whole blocks of its per-block bytes, the
// smallest block count that holds it, and charges nothing for a model with no bounded
// per-sequence caches. The steady decode hold is reported beside it and never exceeds it.
func TestTheBudgetChargesTheKernelsAdmissionReservationInWholeBlocks(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	for _, tt := range []struct {
		scenario string
		charged  bool
	}{
		{dspark, true},
		{"llama-3.1-70b-instruct-h200-tp4-4node.yaml", false},
	} {
		m, err := Open(tt.scenario, r)
		if err != nil {
			t.Fatalf("%s: %v", tt.scenario, err)
		}
		b, err := m.KVBudget()
		if err != nil {
			t.Fatalf("%s: %v", tt.scenario, err)
		}
		reserved := m.Kernel().(admissionReserver).SequenceAdmissionBytes()
		if b.SequenceAdmissionBytes != reserved || b.SequenceSteadyBytes != m.Kernel().SequenceFixedBytes() {
			t.Errorf("%s: budget reports %d/%d bytes a sequence, the kernel %d/%d", tt.scenario,
				b.SequenceAdmissionBytes, b.SequenceSteadyBytes, reserved, m.Kernel().SequenceFixedBytes())
		}
		if b.SequenceSteadyBytes > b.SequenceAdmissionBytes {
			t.Errorf("%s: the steady hold %d exceeds the admission reservation %d", tt.scenario,
				b.SequenceSteadyBytes, b.SequenceAdmissionBytes)
		}
		for _, c := range []struct{ blocks, bytes int64 }{{b.SequenceAdmissionBlocks, reserved}, {b.SequenceSteadyBlocks, b.SequenceSteadyBytes}} {
			if got := c.blocks * b.PerBlockBytes; got < c.bytes || (c.blocks > 0 && got-b.PerBlockBytes >= c.bytes) {
				t.Errorf("%s: %d blocks of %d bytes is not the smallest whole-block cover of %d bytes",
					tt.scenario, c.blocks, b.PerBlockBytes, c.bytes)
			}
		}
		if (b.SequenceAdmissionBlocks > 0) != tt.charged {
			t.Errorf("%s: %d admission blocks a sequence; charged = %t, want %t",
				tt.scenario, b.SequenceAdmissionBlocks, b.SequenceAdmissionBlocks > 0, tt.charged)
		}
		s, err := m.Settings()
		if err != nil {
			t.Fatal(err)
		}
		if (s.SequenceState.Hold != nil) != tt.charged || s.SequenceAdmissionBlocks != b.SequenceAdmissionBlocks {
			t.Errorf("%s: Settings carries a state = %t with %d admission blocks; want %t and %d", tt.scenario,
				s.SequenceState.Hold != nil, s.SequenceAdmissionBlocks, tt.charged, b.SequenceAdmissionBlocks)
		}
	}
}

// Metamorphic (#1941): more memory never admits fewer sequences. Raising
// gpu_memory_utilization grows the pool while the per-sequence fixed term stays put, so the
// number of whole sequences of any length the pool holds -- blocks / (prompt blocks + fixed
// blocks) -- never falls.
func TestRaisingMemoryUtilizationNeverLowersTheSequenceCeiling(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	in, err := latencykernel.OpenInputs(dspark, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	budgetAt := func(util float64) KVBudget {
		in.Deployment.Pools[0].Engine.GPUMemoryUtilization = util
		m, err := fromInputs(in)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.KVBudget()
		if err != nil {
			t.Fatalf("utilization %v: %v", util, err)
		}
		return b
	}
	rapid.Check(t, func(rt *rapid.T) {
		lo := rapid.Float64Range(0.75, 0.95).Draw(rt, "lo")
		hi := rapid.Float64Range(lo, 0.95).Draw(rt, "hi")
		promptBlocks := rapid.Int64Range(1, 4096).Draw(rt, "promptBlocks")
		a, b := budgetAt(lo), budgetAt(hi)
		if a.SequenceSteadyBlocks != b.SequenceSteadyBlocks || a.SequenceAdmissionBlocks != b.SequenceAdmissionBlocks {
			rt.Fatalf("the per-sequence term moved with utilization: %d/%d at %v, %d/%d at %v",
				a.SequenceSteadyBlocks, a.SequenceAdmissionBlocks, lo, b.SequenceSteadyBlocks, b.SequenceAdmissionBlocks, hi)
		}
		ceil := func(x KVBudget) int64 { return x.PerRankBlocks / (promptBlocks + x.SequenceSteadyBlocks) }
		if ceil(b) < ceil(a) {
			rt.Fatalf("utilization %v holds %d sequences of %d blocks, %v holds only %d",
				lo, ceil(a), promptBlocks, hi, ceil(b))
		}
	})
}

// GOLDEN, from the engine's own log (#1941, blis-latency-kernel#46): DeepSeek-V4.1-Flash on 8x
// H200 under vLLM v0.31.0 -- tp 8, expert parallelism off, DSpark K=5, a 1,048,576-token
// context, 8,192 batched tokens, gpu_memory_utilization 0.80 -- reported 61.18 GiB of KV cache
// and "GPU KV cache size: 28,706,997 tokens": the pool's blocks over one request's blocks at
// max_model_len, times max_model_len (get_kv_cache_capacity,
// vllm/v1/core/kv_cache_utils.py:2444-2471), a request sized at its admission cap plus its
// tokens' pages. This is the STARTUP figure the engine reports, not a runtime hold: vLLM gates
// admission on that cap but holds a sequence's window rings on demand (INV-K3,
// TestSequenceStateHoldsWhatVLLMAllocatesOverALifecycle). In vLLM's 61.18 GiB, BLIS's per-
// request sizing gives vLLM's count, to the 0.005 GiB rounding of the logged figure (about
// 2,350 tokens).
//
// End to end BLIS's own pool is smaller, so it admits about 11% fewer tokens (25.5M): the
// kernel's fixed occupancy leaves about 54.4 GiB, not 61.18, its weights being 48.2 GiB
// against vLLM's logged 45.8. That gap is the kernel's FixedBytes (blis-latency-kernel#46), not
// the admission sizing this pins; the test logs it.
func TestKVCapacityMatchesTheEnginesLoggedFigure(t *testing.T) {
	const (
		scenario    = "deepseek-v4.1-flash-h200-tp8-dspark5-1m.yaml"
		maxModelLen = 1 << 20
		engine      = 28706997
		rounding    = 2400
	)
	loggedGiB := 61.18
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	m, err := Open(scenario, r)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if s.MaxModelLen != maxModelLen {
		t.Fatalf("the scenario serves max_model_len %d, want %d", s.MaxModelLen, maxModelLen)
	}
	tokenBlocks := m.Kernel().SequenceVariableBytes(maxModelLen) / s.KVBytesPerBlock
	perRequest := s.SequenceAdmissionBlocks + tokenBlocks
	capacity := func(poolBlocks int64) float64 {
		return float64(poolBlocks) / float64(perRequest) * maxModelLen
	}

	loggedPool := int64(loggedGiB*(1<<30)) / s.KVBytesPerBlock
	if d := capacity(loggedPool) - engine; d > rounding || d < -rounding {
		t.Errorf("a BLIS sequence of %d tokens holds %d blocks (%d admission + %d token); in vLLM's "+
			"%.2f GiB that admits %.0f tokens, against vLLM's logged %d", maxModelLen, perRequest,
			s.SequenceAdmissionBlocks, tokenBlocks, loggedGiB, capacity(loggedPool), engine)
	}
	if steady := m.Kernel().SequenceFixedBytes() / s.KVBytesPerBlock; float64(loggedPool)/float64(steady+tokenBlocks)*maxModelLen-engine <= rounding {
		t.Errorf("sized at the steady decode hold (%d blocks) the capacity is within rounding too; "+
			"the check does not distinguish the admission charge", steady)
	}

	t.Logf("BLIS pool %d blocks (%.2f GiB) admits %.0f tokens; vLLM's %.2f GiB admits %d",
		s.KVBlocks, float64(s.KVBlocks*s.KVBytesPerBlock)/(1<<30), capacity(s.KVBlocks), loggedGiB, engine)
}

// #1941, review C1: a sequence's bounded state is held on demand, as vLLM holds it, never at
// the admission cap for its life. vLLM's allocate_slots allocates a window group's blocks for
// the tokens scheduled and remove_skipped_blocks first frees those the window has passed
// (single_type_kv_cache_manager.py:178-238, 687-725, 1153-1175 at v0.31.0); the
// max_admission_blocks_per_request cap applies only to the admission gate (:216-229). Over a
// DeepSeek-V4.1 DSpark prefill -> decode lifecycle through the real KV store:
//
//   - after every allocation the store holds exactly the kernel's SequenceBoundedBytesAt for
//     that step, in blocks;
//   - a prefill chunk past the window holds the same however far the prompt has got (the
//     rings recycle), and more than a decode step (the chunk is in flight);
//   - once decoding the hold is the steady SequenceFixedBytes, every step;
//   - no step holds the admission cap, and the gate is min(cdiv(prompt), cap) per ring.
func TestSequenceStateHoldsWhatVLLMAllocatesOverALifecycle(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	m, err := Open(dspark, r)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	st := s.SequenceState
	if st.Hold == nil {
		t.Fatal("DeepSeek-V4.1 has bounded per-sequence state; Settings carries none")
	}
	k := m.Kernel().(admissionReserver)
	per := s.KVBytesPerBlock
	blocks := func(b int64) int64 { return (b + per - 1) / per }

	kvc := kv.NewKVCacheState(s.KVBlocks, int64(s.BlockSize))
	kvc.SequenceState = st
	const prompt, decodes = 20000, 40
	chunk := int64(s.MaxNumBatchedTokens)
	verify := int64(s.SpeculativeTokens + 1)
	req := &sim.Request{ID: "r", InputTokens: make([]sim.TokenID, prompt), OutputTokens: make([]sim.TokenID, 3*decodes)}
	for i := range req.InputTokens {
		req.InputTokens[i] = sim.TokenID(i + 1)
	}
	if got, want := st.Gate(prompt), min(blocks(k.SequenceBoundedBytesAt(0, prompt)), s.SequenceAdmissionBlocks); got != want {
		t.Errorf("gate for a %d-token prompt: %d blocks, want min(cdiv, cap) = %d", prompt, got, want)
	}
	var chunkHolds []int64
	for c := int64(0); c < prompt; c += chunk {
		end := min(c+chunk, prompt)
		if !kvc.AllocateKVBlocks(req, c, end, nil) {
			t.Fatalf("chunk at %d refused", c)
		}
		req.ProgressIndex = end
		got := int64(len(kvc.StateBlocks["r"]))
		if want := blocks(k.SequenceBoundedBytesAt(int(c), int(end-c))); got != want {
			t.Errorf("prefill chunk [%d, %d): holds %d state blocks, the kernel prices %d", c, end, got, want)
		}
		if end-c == chunk {
			chunkHolds = append(chunkHolds, got)
		}
		if got >= s.SequenceAdmissionBlocks {
			t.Errorf("prefill chunk [%d, %d) holds the admission cap (%d of %d blocks)", c, end, got, s.SequenceAdmissionBlocks)
		}
	}
	for i := 2; i < len(chunkHolds); i++ { // the first chunk has not filled the window yet
		if chunkHolds[i] != chunkHolds[1] {
			t.Errorf("full chunks past the window hold %v blocks; the rings recycle, so they hold alike", chunkHolds)
		}
	}
	for d := 0; d < decodes; d++ {
		pi := req.ProgressIndex
		if !kvc.AllocateKVBlocks(req, pi, pi+verify, nil) {
			t.Fatalf("decode %d refused", d)
		}
		if got := int64(len(kvc.StateBlocks["r"])); got != s.SequenceSteadyBlocks {
			t.Fatalf("decode %d: holds %d state blocks, want the steady %d (admission cap %d)", d, got,
				s.SequenceSteadyBlocks, s.SequenceAdmissionBlocks)
		}
		req.ProgressIndex += 2
	}
	if len(chunkHolds) > 1 && chunkHolds[1] <= s.SequenceSteadyBlocks {
		t.Errorf("a full chunk in flight holds %d blocks, no more than a decode's %d", chunkHolds[1], s.SequenceSteadyBlocks)
	}
	kvc.ReleaseKVBlocks(req)
	if kvc.UsedBlocks() != 0 {
		t.Errorf("%d blocks held after release", kvc.UsedBlocks())
	}
}
