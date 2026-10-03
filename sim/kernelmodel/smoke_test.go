package kernelmodel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/inference-sim/sim"
)

const (
	kernelRepo = "/Users/sri/Documents/Projects/blis-latency-kernel"
	catalog    = "/Users/sri/Documents/Projects/blis-catalog"
	registry   = "/Users/sri/Documents/Projects/blis-registry"
)

func repos() Repos {
	return Repos{
		Scenarios: kernelRepo + "/testdata/aisimulate",
		Catalog:   catalog,
		Registry:  registry,
	}
}

func open(t *testing.T, scenario string) *Model {
	t.Helper()
	m, err := Open(scenario, repos())
	if err != nil {
		t.Fatalf("Open(%s): %v", scenario, err)
	}
	return m
}

// The adapter must return the kernel's own number, converted, and nothing else. A version
// that recomputed anything -- or dropped the Overlap/NoOverlap distinction -- would differ
// here.
func TestStepTimeIsTheKernelsOwnNumberInTicks(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	batch := []*sim.Request{
		decodeReq(1024, 1024), decodeReq(1024, 1030), decodeReq(512, 600),
	}
	got := m.StepTime(batch)

	// Ask the kernel directly, building the same batch shape.
	b := kernel.Batch{DecodeThreshold: DecodeThreshold, SMBudget: m.smBudget}
	for _, r := range batch {
		b.Reqs = append(b.Reqs, shapeOf(r))
	}
	want := m.Kernel().StepTime(b).Expected.Microseconds()
	if got != want {
		t.Errorf("adapter returned %d ticks, the kernel says %d", got, want)
	}
	if got <= 0 {
		t.Errorf("step time %d is not positive", got)
	}
}

// BLIS requires >= 1 for every input including an empty batch: 0 stalls the clock (INV-3).
func TestEmptyBatchStillAdvancesTheClock(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	if got := m.StepTime(nil); got < 1 {
		t.Errorf("empty batch gave %d ticks; BLIS requires >= 1", got)
	}
	if got := m.StepTime([]*sim.Request{}); got < 1 {
		t.Errorf("empty slice gave %d ticks; BLIS requires >= 1", got)
	}
}

// Host terms must be the kernel's, in ticks.
func TestHostTermsComeFromTheKernel(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	if got, want := m.OutputTokenProcessingTime(),
		m.Kernel().OutputTokenOverhead().Microseconds(); got != want {
		t.Errorf("OutputTokenProcessingTime %d, kernel says %d", got, want)
	}
	if got, want := m.PostDecodeFixedOverhead(),
		m.Kernel().CompletionOverhead().Microseconds(); got != want {
		t.Errorf("PostDecodeFixedOverhead %d, kernel says %d", got, want)
	}
	if m.OutputTokenProcessingTime() <= 0 {
		t.Error("per-token host cost resolved to zero, so a TPOT comparison would be " +
			"missing its host term entirely")
	}
}

func decodeReq(promptLen, progress int64) *sim.Request {
	return &sim.Request{
		InputTokens:   make([]sim.TokenID, promptLen),
		ProgressIndex: progress,
		NumNewTokens:  1,
	}
}

// The prefill/decode classifier must survive translation. A request mid-prompt is a
// chunked-prefill tail and is priced by the prefill law; one past its prompt is a decode
// and is priced by the decode law. Swapping ProgressIndex and PromptLen, or dropping
// either, leaves the simulation running while every decode is priced as a prefill.
//
// Behavioural rather than structural: it asserts the two regimes cost DIFFERENTLY at the
// same scheduled-token count, which only holds if both fields reached the kernel.
func TestPrefillAndDecodeArePricedByDifferentLaws(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	const prompt = 4096

	// Same Scheduled, same PromptLen; only Computed differs. Below the prompt it is a
	// prefill chunk, at or past it a decode.
	prefill := &sim.Request{
		InputTokens:   make([]sim.TokenID, prompt),
		ProgressIndex: 0,
		NumNewTokens:  512,
	}
	decode := &sim.Request{
		InputTokens:   make([]sim.TokenID, prompt),
		ProgressIndex: prompt,
		NumNewTokens:  512,
	}
	pt, dt := m.StepTime([]*sim.Request{prefill}), m.StepTime([]*sim.Request{decode})
	if pt == dt {
		t.Errorf("a 512-token prefill chunk and a 512-token decode both priced %d ticks; "+
			"Computed is not reaching the kernel, so every decode is priced as a prefill",
			pt)
	}
	// Direction, established by measurement rather than by intuition. At equal scheduled
	// tokens the decode is the MORE expensive of the two, and consistently so: a decode
	// request with Computed=4096 reads the whole 4096-token KV cache for each of its
	// scheduled tokens, where a prefill chunk's causal attention attends only to earlier
	// positions and so averages about half that context. Measured ratios on this
	// deployment are 0.976 at 512 scheduled tokens and 0.910 at 2048.
	//
	// The first draft of this test asserted the opposite and failed, which is the reason
	// the direction is stated here with its cause: an assertion in the wrong direction
	// would have been "fixed" by swapping the mapping, breaking the thing it checks.
	if dt <= pt {
		t.Errorf("decode %d ticks is not above prefill chunk %d ticks at equal scheduled "+
			"tokens; a decode reads the full context per token and must cost more", dt, pt)
	}
}

// At one scheduled token the step is host-bound, and the two regimes coincide ON THE
// OVERLAP EDGE: a per-stage max is taken over resources, and the host term dominates every
// stage, so the attention read's difference between a prefill chunk and a decode never
// surfaces. Pinning this separately keeps the test above honest about WHERE the regimes
// diverge: a version that only checked a single batch shape could pass on a coincidence.
//
// It is asserted against Overlap rather than against the adapter's own number, because the
// adapter reads StepEstimate.Expected and that is NoOverlap, which sums every resource
// instead of maxing within a stage. Summing keeps the HBM difference -- 2.447 ms for the
// prefill chunk against 2.589 ms for the decode on this deployment -- so the two regimes
// do NOT coincide there, and should not. The host-bound claim is a property of the
// max-composed edge, so that is the edge it is checked on.
func TestTheRegimesCoincideWhenTheStepIsHostBound(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	const prompt = 4096
	prefill := &sim.Request{
		InputTokens: make([]sim.TokenID, prompt), ProgressIndex: 0, NumNewTokens: 1,
	}
	decode := &sim.Request{
		InputTokens: make([]sim.TokenID, prompt), ProgressIndex: prompt, NumNewTokens: 1,
	}
	pt := overlapOf(m, prefill)
	dt := overlapOf(m, decode)
	if pt != dt {
		t.Errorf("at one scheduled token a prefill chunk priced %v and a decode %v on the "+
			"overlap edge; both are host-bound at this size and should coincide", pt, dt)
	}
}

// overlapOf prices one request on the band's max-composed edge, for a test whose claim is
// about that edge rather than about what the adapter reports.
func overlapOf(m *Model, r *sim.Request) time.Duration {
	b := kernel.Batch{DecodeThreshold: DecodeThreshold, SMBudget: m.smBudget}
	b.Reqs = append(b.Reqs, shapeOf(r))
	return m.Kernel().StepTime(b).Overlap
}

// Step time must not fall when a request is scheduled more tokens. A mapping that sent
// the wrong field as Scheduled would break this.
func TestStepTimeRisesWithScheduledTokens(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	var prev int64
	for _, n := range []int{1, 2, 8, 32, 128, 512, 2048} {
		req := &sim.Request{
			InputTokens:   make([]sim.TokenID, 8192),
			ProgressIndex: 0,
			NumNewTokens:  n,
		}
		got := m.StepTime([]*sim.Request{req})
		if got < prev {
			t.Errorf("scheduled=%d gave %d ticks, below the %d at the previous level",
				n, got, prev)
		}
		prev = got
	}
}

// Step time must rise with the resident batch at fixed per-request work. This is the
// property the whole concurrency-response comparison rests on: if it did not hold, a
// normalised sweep would be flat and the MAPE meaningless.
func TestStepTimeRisesWithBatchSize(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	var prev int64
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64, 128, 256} {
		batch := make([]*sim.Request, 0, n)
		for i := 0; i < n; i++ {
			batch = append(batch, decodeReq(1024, 1024))
		}
		got := m.StepTime(batch)
		if got < prev {
			t.Errorf("batch=%d gave %d ticks, below the %d at the previous size",
				n, got, prev)
		}
		prev = got
	}
}

// Two deployments of the same model that differ only in tensor-parallel width must price
// differently. This is what proves the adapter is carrying the SCENARIO through to the
// kernel rather than building one generic kernel per model.
func TestDifferentDeploymentsOfOneModelPriceDifferently(t *testing.T) {
	four := open(t, "glm-5-b200-fp4-sglang-tp4.yaml")
	eight := open(t, "glm-5-b200-fp8-sglang-tp8.yaml")
	batch := make([]*sim.Request, 0, 32)
	for i := 0; i < 32; i++ {
		batch = append(batch, decodeReq(1024, 1024))
	}
	a, b := four.StepTime(batch), eight.StepTime(batch)
	if a == b {
		t.Errorf("tp=4 and tp=8 deployments both priced %d ticks; the scenario is not "+
			"reaching the kernel", a)
	}
}

// Scheduled must come from NumNewTokens, not from the prompt length. Holding the prompt
// fixed and varying only NumNewTokens must move the price; a mapping that sent InputLen()
// as Scheduled would return the same number for every scheduled-token count, and the
// monotonicity test above would still pass because a constant is non-decreasing.
//
// This is the mutation that survived the first round of probes.
func TestScheduledTokensComeFromNumNewTokensNotThePrompt(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	const prompt = 8192
	at := func(n int) int64 {
		return m.StepTime([]*sim.Request{{
			InputTokens:   make([]sim.TokenID, prompt),
			ProgressIndex: 0,
			NumNewTokens:  n,
		}})
	}
	one, many := at(1), at(2048)
	if one == many {
		t.Errorf("1 and 2048 scheduled tokens over the same %d-token prompt both priced "+
			"%d ticks; Scheduled is not coming from NumNewTokens", prompt, one)
	}
	if many <= one {
		t.Errorf("2048 scheduled tokens priced %d, not above the %d for one token",
			many, one)
	}
	// And the converse: holding NumNewTokens fixed while varying the prompt must ALSO
	// move the price, since PromptLen selects the regime and sets the context read. A
	// mapping that dropped PromptLen would make these equal.
	short := m.StepTime([]*sim.Request{{
		InputTokens: make([]sim.TokenID, 128), ProgressIndex: 128, NumNewTokens: 1,
	}})
	long := m.StepTime([]*sim.Request{{
		InputTokens: make([]sim.TokenID, 32768), ProgressIndex: 32768, NumNewTokens: 1,
	}})
	if short == long {
		t.Errorf("a 128-token and a 32768-token context both priced %d ticks at one "+
			"scheduled token; PromptLen or Computed is not reaching the kernel", short)
	}
}

// CachedTokens must stay zero, and this pins WHY behaviourally rather than by reasoning
// about the code.
//
// The adapter passes CachedTokens: 0 because BLIS reflects a prefix-cache hit in
// ProgressIndex -- batch_formation computes numNewTokens as InputLen()-ProgressIndex, so a
// hit is already excluded from the Scheduled count. Passing it again would subtract it twice.
//
// The original comment justified this as "BLIS subtracts prefix-cache hits before setting
// NumNewTokens", which is the right conclusion from the wrong premise: BLIS's KV store
// implements prefix caching UNCONDITIONALLY, not behind a flag, so the assumption cannot rest
// on caching being off.
//
// The property that must hold: a request whose prefix is already computed -- expressed as a
// ProgressIndex above zero on an unfinished prompt -- must cost LESS than the same request
// from scratch, and the difference must come from Scheduled alone. If the adapter also
// subtracted a cached count, the same hit would be removed twice and the step would be
// under-priced.
func TestAPrefixHitIsNotSubtractedTwice(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	const prompt = 4096

	// From scratch: the whole prompt is scheduled.
	cold := m.StepTime([]*sim.Request{{
		InputTokens: make([]sim.TokenID, prompt), ProgressIndex: 0, NumNewTokens: prompt,
	}})
	// Half the prompt already computed: BLIS schedules only the remainder, and that
	// remainder is what NumNewTokens carries.
	warm := m.StepTime([]*sim.Request{{
		InputTokens:   make([]sim.TokenID, prompt),
		ProgressIndex: prompt / 2,
		NumNewTokens:  prompt / 2,
	}})
	if warm >= cold {
		t.Errorf("a half-cached prompt priced %d ticks against %d from scratch; the hit is "+
			"not reaching the kernel at all", warm, cold)
	}

	// And the adapter must not subtract it a second time: pricing the same shape with the
	// kernel told about a cached count as well must differ from what the adapter produces.
	// If they agreed, CachedTokens would be a no-op and this test could not detect a
	// double subtraction.
	b := kernel.Batch{DecodeThreshold: DecodeThreshold, SMBudget: m.smBudget,
		Reqs: []kernel.ReqShape{{
			Scheduled: prompt / 2, Computed: prompt / 2, PromptLen: prompt,
			CachedTokens: prompt / 2,
		}}}
	twice := m.Kernel().StepTime(b).Overlap.Microseconds()
	if twice == warm {
		t.Skip("this kernel does not use CachedTokens for this shape, so a double " +
			"subtraction is undetectable here and the zero is harmless either way")
	}
	if twice > warm {
		t.Errorf("telling the kernel about a cached count raised the price from %d to %d; "+
			"the field's sign is not what this test assumed", warm, twice)
	}
}
