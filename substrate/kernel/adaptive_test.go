package latencykernel

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

// The budget is vLLM's get_num_tokens objective (vllm/v1/worker/gpu/spec_decode/
// adaptive_verification.py:282-350@5df63a72). The wants below are what this Python reference,
// a transcription of that method over the same inputs, prints with numpy 1.26:
//
//	import numpy as np
//	def get_num_tokens(sched_tokens, sched_drafts, sampling, conf, draft_cost, verify_cost,
//	                   max_total_logits, bonus=1):
//	    sched_tokens = np.array(sched_tokens); sched_drafts = np.array(sched_drafts)
//	    nreq = len(sched_tokens); nsteps = conf.shape[1]
//	    nd = sched_tokens - sched_drafts                                   # :303
//	    surv = np.cumprod(conf.astype(np.float64), axis=1)                 # :310
//	    valid = np.arange(nsteps)[None, :] < sched_drafts[:, None]         # :311-312
//	    scores = np.sort(surv[valid])[::-1]                                # :313
//	    ndt = int(nd.sum())
//	    maxb = min(int(sched_drafts.sum()), max(0, max_total_logits - nreq * bonus))  # :315-318
//	    scores = scores[:maxb]
//	    acc = np.concatenate(([sampling], sampling + np.cumsum(scores)))   # :325-327
//	    costs = draft_cost[nreq] + verify_cost[ndt:ndt + maxb + 1]         # :328-335
//	    return int(np.argmax(acc / costs))                                 # :344
//	def rates(length, n):  # SpeculativeConfig._acceptance_length_to_rates, config/speculative.py:556-563
//	    d = length - 1; f = int(d); return ([1.0] * f + [d - f] + [0.0] * (n - f - 1))[:n]
//	verify = np.array([1.0 + 0.05 * t + (2.0 if t > 16 else 0.0) for t in range(65)])
//	draft = np.array([0.5 + 0.01 * r for r in range(9)])
//	for al in [1.5, 2.84, 4.6, 8.0]:
//	    r = np.array(rates(al, 7)); conf = np.ones(7); prev = 1.0
//	    for i, x in enumerate(r):
//	        conf[i] = x / prev if prev > 0 else 0.0; prev = x   # cumprod(conf) == r
//	    for nreq in [1, 4, 8]:
//	        print(al, nreq, get_num_tokens([8] * nreq, [7] * nreq, nreq,
//	              np.tile(conf, (nreq, 1)), draft, verify, 10**6))
func TestAdaptiveBudgetIsVLLMsObjective(t *testing.T) {
	verify := make([]float64, 65)
	for i := range verify {
		verify[i] = 1.0 + 0.05*float64(i)
		if i > 16 {
			verify[i] += 2.0
		}
	}
	draft := make([]float64, 9)
	for i := range draft {
		draft[i] = 0.5 + 0.01*float64(i)
	}
	k := &Kernel{engine: enginesettings.Settings{Speculative: &enginesettings.Speculative{
		Method: "dspark", NumSpecTokens: 7, EnableAdaptiveVerification: true}}}
	for _, c := range []struct {
		al         float64
		reqs, want int
	}{
		{1.5, 1, 1}, {1.5, 4, 4}, {1.5, 8, 8},
		{2.84, 1, 2}, {2.84, 4, 8}, {2.84, 8, 8},
		{4.6, 1, 4}, {4.6, 4, 12}, {4.6, 8, 8},
		{8.0, 1, 7}, {8.0, 4, 12}, {8.0, 8, 56},
	} {
		surv, err := acceptance.MinimumVarianceRates(c.al, 7)
		if err != nil {
			t.Fatal(err)
		}
		a := &adaptiveVerify{survival: surv, draftCost: draft, verifyCost: verify, maxTotalLogits: 1e6}
		b := kernel.Batch{DecodeThreshold: 8}
		for range c.reqs {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 8, Computed: 100, PromptLen: 100})
		}
		got, _ := a.budget(k, b)
		if got != c.want {
			t.Errorf("AL %v, %d requests: budget %d, vLLM's objective picks %d", c.al, c.reqs, got, c.want)
		}
		v := admit(got, []int{7, 7, 7, 7, 7, 7, 7, 7}[:c.reqs])
		sum := 0
		for _, n := range v {
			sum += n
		}
		if sum != got {
			t.Errorf("AL %v: admitted %v, which is not the budget %d", c.al, v, got)
		}
	}
}

// The budget's slots go to every request's first step before any request's second, the
// global top-k over one non-increasing survival curve (_assign_draft_token_budget,
// adaptive_verification.py:46-72@5df63a72), and never past a request's own drafts.
func TestAdmitTakesTheBestSlotsFirst(t *testing.T) {
	got := admit(5, []int{3, 0, 1, 3})
	want := []int{2, 0, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("admit(5, [3 0 1 3]) = %v, want %v", got, want)
		}
	}
}

// The cost table is build_cost_tables_from_curves' build_table (adaptive_verification.py:
// 100-128@5df63a72): a step function at or below the capture limit, interpolation above it,
// extrapolation past the last point. The want is that function's numpy output over curve
// [(1,1),(4,2),(8,3),(16,5)], limit 8, size 20: only 16 lies above the limit, so np.interp
// clamps 9..16 to its 5.
func TestCostTableIsTheEnginesTable(t *testing.T) {
	got := costTable([][2]float64{{1, 1}, {4, 2}, {8, 3}, {16, 5}}, 20, 8, 0)
	want := []float64{1, 1, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5.25, 5.5, 5.75, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("table %v, want %v", got, want)
		}
	}
}

func adaptiveKernel(t *testing.T, on bool, al float64) *Kernel {
	t.Helper()
	k := &Kernel{engine: enginesettings.Settings{Speculative: &enginesettings.Speculative{
		Method: "dspark", NumSpecTokens: 7, EnableAdaptiveVerification: on}}, uniformDecodeWidth: 8}
	if !on {
		return k
	}
	surv, err := acceptance.MinimumVarianceRates(al, 7)
	if err != nil {
		t.Fatal(err)
	}
	// A verify cost of a fixed floor and a per-token slope, the shape of a bandwidth-bound decode.
	verify := make([]float64, 513)
	for i := range verify {
		verify[i] = 10e-3 + 20e-6*float64(i)
	}
	k.adaptive = &adaptiveVerify{survival: surv, draftCost: []float64{0, 1e-3, 1e-3, 1e-3, 1e-3,
		1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3},
		verifyCost: verify, maxTotalLogits: 1e6}
	return k
}

func adaptiveDecodeBatch(reqs int) kernel.Batch {
	b := kernel.Batch{DecodeThreshold: 8}
	for range reqs {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 8, Computed: 4000, PromptLen: 3000})
	}
	b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 512, Computed: 0, PromptLen: 4000})
	return b
}

// (a) With the flag off nothing is trimmed and nothing is reported: the step prices the batch
// as scheduled, through the same path as before (stepTime -> priceStep(b, b)), and the
// acceptance is not read, so a missing one is not refused.
func TestAdaptiveVerificationOffChangesNothing(t *testing.T) {
	k := adaptiveKernel(t, false, 0)
	if err := k.resolveAdaptive(nil); err != nil || k.adaptive != nil {
		t.Fatalf("flag off: resolveAdaptive %v, state %v", err, k.adaptive)
	}
	if v := k.VerifiedDrafts(adaptiveDecodeBatch(4)); v != nil {
		t.Errorf("flag off: VerifiedDrafts %v, want nil", v)
	}
	if k.decodeGraphed(kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 3, Computed: 10, PromptLen: 10}}}) {
		t.Error("flag off: a non-uniform decode replays the decode graph")
	}
	on := adaptiveKernel(t, false, 0)
	on.engine.Speculative.EnableAdaptiveVerification = true
	if err := on.resolveAdaptive(nil); err == nil {
		t.Error("flag on without an acceptance was not refused")
	}
}

// (b) At low acceptance the verifier runs fewer tokens than were scheduled; near full
// acceptance it runs them all. A prefill row is never trimmed, and a request never verifies
// more drafts than it scheduled.
func TestAdaptiveVerificationTrimsAtLowAcceptance(t *testing.T) {
	b := adaptiveDecodeBatch(16)
	low := adaptiveKernel(t, true, 1.3)
	trimmed := low.adaptive.trim(low, b)
	if trimmed.Tokens() >= b.Tokens() {
		t.Errorf("AL 1.3: the verifier runs %d of %d tokens, want fewer", trimmed.Tokens(), b.Tokens())
	}
	if last := trimmed.Reqs[len(trimmed.Reqs)-1]; last.Scheduled != 512 {
		t.Errorf("the prefill row was trimmed to %d", last.Scheduled)
	}
	for i, v := range low.VerifiedDrafts(b) {
		if want := low.scheduledDrafts(b.Reqs[i]); v > want || trimmed.Reqs[i].Scheduled-1 != v && want > 0 {
			t.Errorf("request %d verifies %d of %d drafts, runs %d tokens", i, v, want, trimmed.Reqs[i].Scheduled)
		}
	}
	full := adaptiveKernel(t, true, 8)
	if got := full.adaptive.trim(full, b).Tokens(); got != b.Tokens() {
		t.Errorf("AL 8 (every draft accepted): the verifier runs %d of %d tokens, want all", got, b.Tokens())
	}
	if !low.decodeGraphed(kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 3, Computed: 10, PromptLen: 10}, {Scheduled: 1, Computed: 10, PromptLen: 10}}}) {
		t.Error("a trimmed decode batch does not replay the varlen decode graph")
	}
}

// A tie keeps the smallest budget, as np.argmax keeps the first maximum
// (adaptive_verification.py:344@5df63a72): with nothing predicted to survive and a flat verify
// cost, every budget ties and the engine verifies no drafts.
func TestAdaptiveBudgetTiesKeepTheFirst(t *testing.T) {
	k := adaptiveKernel(t, true, 1)
	for i := range k.adaptive.verifyCost {
		k.adaptive.verifyCost[i] = 1
	}
	if got, _ := k.adaptive.budget(k, adaptiveDecodeBatch(4)); got != 0 {
		t.Errorf("all budgets tie: picked %d, want the first, 0", got)
	}
}

// Rates the resolver derived are the minimum-variance schedule for the mean acceptance length,
// and are disclosed as that, not as stated rates.
func TestDerivedRatesAreDisclosedAsTheSchedule(t *testing.T) {
	for _, derived := range []bool{true, false} {
		k := adaptiveKernel(t, false, 0)
		k.engine.Speculative.EnableAdaptiveVerification = true
		k.adaptive = nil
		k.graph = &model.Graph{}
		k.plan = &price.Plan{}
		acc := &acceptance.Resolution{AcceptanceLength: 2.84, PerPositionRates: []float64{1, 0.84, 0, 0, 0, 0, 0},
			RatesDerived: derived}
		// No capture sizes to profile, so it refuses; the assumption is recorded first.
		if err := k.resolveAdaptive(acc); err == nil {
			t.Error("profiled with no captured graph")
		}
		var scope string
		for _, a := range k.assumptions {
			if a.Name == "adaptive_verification_survival" {
				scope = a.Scope
			}
		}
		if got := strings.Contains(scope, "minimum-variance"); got != derived {
			t.Errorf("derived %v: assumption %q", derived, scope)
		}
	}
}
