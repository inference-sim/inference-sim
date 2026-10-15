package kernelmodel

import (
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/inference-sim/sim"
)

// The artifact roots, which TestMain has already checked resolve.
func repos() Repos { return DefaultRepos() }

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
	// NoOverlap, matching the adapter. schemas v0.2.0 removed StepEstimate.Expected,
	// which the kernel had set to NoOverlap from measured evidence; the adapter now names
	// that edge directly. This test asserts the adapter forwards the kernel's answer
	// unchanged, so it has to ask for the same edge.
	want := m.Kernel().StepTime(b).NoOverlap.Microseconds()
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

// The law: WHERE a step is host-bound, the mixer choice must not move it; where it is
// compute-bound, it must. One scheduled token is the host-bound end and a full prompt is
// the compute-bound end, so the two ends together say the step model is sensitive to the
// prefill/decode distinction exactly where the hardware is.
//
// This is a metamorphic claim about the two ends rather than a value at either. An earlier
// version asserted exact float equality of the two prices at one token, which is the
// "exact formula reproduction" pattern the standards prohibit (principles.md): it broke on
// a coefficient refit that moved the host and attention terms by different amounts, while
// the physics it meant to pin -- host dominance at one token -- still held. The ratio bound
// below is loose because the claim is "negligible beside the host term", not a number.
//
// Asserted on the Overlap edge, where a per-stage max is taken over resources, because
// that is where a dominant host term can hide the attention difference at all. The adapter
// reads NoOverlap, which sums every resource, so the two regimes do NOT coincide there and
// should not. (Before schemas v0.2.0 the adapter read StepEstimate.Expected, which the
// kernel set to NoOverlap from measured evidence; the field is gone and the adapter names
// the edge directly, so this paragraph describes the same two regimes it always did.)
//
// The deployment is a DENSE-attention model on purpose. On a DSA sparse-attention model such
// as glm-5, each query attends a bounded top-k set rather than its whole context, so a
// decode at a 4,096-token context and a 4,096-token prefill chunk read nearly the same KV --
// the kernel prices that correctly, and the separation this test asserts would not exist.
func TestTheMixerChoiceMovesTheStepOnlyWhereComputeBinds(t *testing.T) {
	m := open(t, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	const prompt = 4096

	// hostBoundTolerance is how much of the host-bound price the mixer difference may
	// account for. The host term dominates every stage at one scheduled token, so the
	// attention read's contribution is a small fraction of it -- not zero, because
	// NoOverlap's resource sum still carries it into the stage max on some deployments.
	const hostBoundTolerance = 0.10
	// computeBoundSeparation is how far apart the two regimes must be, relatively, once
	// the prompt is priced. DIRECTION IS DELIBERATELY NOT ASSERTED: at 4,096 scheduled
	// tokens on this deployment the decode is the DEARER of the two (4,096 queries each
	// reading a 4,096-token cache, against a prefill chunk whose attention is triangular
	// over the chunk), and which side wins is a property of the shapes rather than of the
	// regime. The claim is separation, which is what a step model blind to the
	// distinction would fail.
	const computeBoundSeparation = 0.10

	at := func(scheduled int) (prefill, decode float64) {
		p := &sim.Request{
			InputTokens: make([]sim.TokenID, prompt), ProgressIndex: 0, NumNewTokens: scheduled,
		}
		d := &sim.Request{
			InputTokens: make([]sim.TokenID, prompt), ProgressIndex: prompt, NumNewTokens: scheduled,
		}
		return float64(overlapOf(m, p)), float64(overlapOf(m, d))
	}

	// Host-bound end: one scheduled token. The prices must be close.
	pOne, dOne := at(1)
	if pOne <= 0 || dOne <= 0 {
		t.Fatalf("both prices must be positive, got prefill %v decode %v", pOne, dOne)
	}
	if rel := math.Abs(dOne-pOne) / math.Min(pOne, dOne); rel > hostBoundTolerance {
		t.Errorf("at one scheduled token the mixer choice moved the step by %.1f%% "+
			"(prefill %v, decode %v); the host term dominates this size, so the attention "+
			"difference should be negligible beside it (tolerance %.0f%%)",
			rel*100, pOne, dOne, hostBoundTolerance*100)
	}

	// Compute-bound end: the whole prompt in one chunk. The prices must SEPARATE.
	pAll, dAll := at(prompt)
	if rel := math.Abs(dAll-pAll) / math.Min(pAll, dAll); rel < computeBoundSeparation {
		t.Errorf("pricing a %d-token chunk, prefill %v and decode %v differ by only %.1f%%; "+
			"a step model blind to the prefill/decode distinction would price them alike, "+
			"and this end is where the distinction must show (want >= %.0f%%)",
			prompt, pAll, dAll, rel*100, computeBoundSeparation*100)
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
