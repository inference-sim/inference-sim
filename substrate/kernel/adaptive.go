package latencykernel

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// Adaptive verification (speculative_config.enable_adaptive_verification, vLLM v0.32.0,
// vllm/config/speculative.py:551-553@5df63a72): the verifier runs a draft budget the engine
// chooses each step, rather than every draft the scheduler budgeted. The scheduler still
// budgets 1 + K tokens a request (nothing under vllm/v1/core reads the flag), the drafter still
// runs full blocks (vllm/v1/attention/selector.py:143-152@5df63a72), and only the target's
// forward runs the trimmed width.

// adaptiveVerify is the resolved adaptive-verification state; nil when the engine does not
// enable it.
type adaptiveVerify struct {
	// survival[i] is the predicted probability that a request's first i+1 drafts are all
	// accepted, the score get_num_tokens ranks a (request, step) slot by.
	survival []float64
	// draftCost[r] and verifyCost[t] are the engine's profiled cost tables, in seconds: the
	// drafter at r requests and the target's forward at t tokens
	// (build_cost_tables_from_curves).
	draftCost, verifyCost []float64
	// maxTotalLogits bounds the verified logits rows (get_max_chunk_logits).
	maxTotalLogits int
}

// adaptiveProfileContext is VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN's default, the
// context each profiled request carries (vllm/envs.py:62,1111-1112@5df63a72).
const adaptiveProfileContext = 8192

// maxChunkBytes is the rejection sampler's chunk, MAX_CHUNK_BYTES
// (vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:33@5df63a72).
const maxChunkBytes = 1 << 30

// resolveAdaptive settles adaptive verification from the engine settings and the run's
// acceptance, and profiles the cost tables the budget is chosen from. With the flag off it
// leaves k.adaptive nil, and nothing else in the kernel reads the acceptance. It runs after
// every pricing term is settled, since the tables are this kernel's own step prices.
func (k *Kernel) resolveAdaptive(acc *acceptance.Resolution) error {
	s := k.engine.Speculative
	n := k.numSpecTokens()
	if s == nil || !s.EnableAdaptiveVerification || n == 0 {
		return nil
	}
	if acc == nil {
		return fmt.Errorf("speculative.enable_adaptive_verification is on, and the kernel prices the " +
			"verify budget the engine chooses from per-position acceptance: Inputs.Acceptance, the " +
			"run's resolved acceptance (blis-schemas acceptance.Resolve), is required")
	}
	if k.layout.PP > 1 {
		return fmt.Errorf("speculative.enable_adaptive_verification: vLLM refuses adaptive " +
			"verification with pipeline parallelism (vllm/config/vllm.py:3019-3028@5df63a72)")
	}
	survival := acc.PerPositionRates
	if len(survival) != n || acc.RatesDerived {
		r, err := acceptance.MinimumVarianceRates(acc.AcceptanceLength, n)
		if err != nil {
			return fmt.Errorf("speculative.enable_adaptive_verification: acceptance: %w", err)
		}
		survival = r
		k.assume("adaptive_verification_survival", fmt.Sprint(r),
			"the engine ranks each draft slot by its predicted survival, the running product of an "+
				"online per-position acceptance estimate (adaptive_verification.py:309-313, "+
				"acceptance_estimator.py@5df63a72); the kernel has the run's mean acceptance length "+
				strconv.FormatFloat(acc.AcceptanceLength, 'f', -1, 64)+" at num_spec_tokens "+
				strconv.Itoa(n)+" and no per-position rates, so survival is vLLM's minimum-variance "+
				"schedule for it (SpeculativeConfig._acceptance_length_to_rates, "+
				"vllm/config/speculative.py:556-563@5df63a72), the same for every request; natural "+
				"acceptance spreads more, so the trimmed budget is a lower bound")
	} else {
		survival = append([]float64(nil), survival...)
		k.assume("adaptive_verification_survival", fmt.Sprint(survival),
			"the run's stated per-position acceptance rates, the same for every request, stand in "+
				"for the engine's online per-request estimate (adaptive_verification.py:309-313@5df63a72)")
	}
	k.adaptive = &adaptiveVerify{survival: survival,
		maxTotalLogits: max(1, maxChunkBytes/max(k.graph.Global.VocabSize*4, 1))}
	if err := k.profileAdaptive(); err != nil {
		k.adaptive = nil
		return err
	}
	k.assume("adaptive_verification_cost_curves", "kernel step prices",
		"the engine profiles its draft and verify cost curves by timing dummy steps at the capture "+
			"sizes and tail sizes, each request at "+strconv.Itoa(adaptiveProfileContext)+" tokens of "+
			"context (batches_to_profile, set_initial_cost_curves, "+
			"vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:188-231@5df63a72); the kernel "+
			"prices the same dummy steps (NoOverlap) and builds the tables as the engine does "+
			"(build_cost_tables_from_curves, :80-128), so the budget follows this kernel's prices, "+
			"not measured timings; the drafter's cost is the step priced with it less without it")
	return nil
}

// profileAdaptive builds the cost tables from the dummy steps batches_to_profile runs
// (adaptive_verification.py:188-210@5df63a72), each priced by this kernel as _dummy_run shapes
// it (vllm/v1/worker/gpu/model_runner.py:841-876@5df63a72).
func (k *Kernel) profileAdaptive() error {
	maxTokens := k.batchedTokens
	sizes := append([]int(nil), k.captureSizes...)
	limit := 0
	if len(sizes) > 0 {
		limit = sizes[len(sizes)-1]
	}
	if size := limit; size > 0 {
		tail := map[int]bool{min(size+size/2, maxTokens): true}
		for size < maxTokens {
			size = min(size*2, maxTokens)
			tail[size] = true
		}
		for _, c := range k.captureSizes {
			delete(tail, c)
		}
		var extra []int
		for t := range tail {
			extra = append(extra, t)
		}
		sort.Ints(extra)
		sizes = append(sizes, extra...)
	}
	perReq := 1
	if d := k.plan.Draft; d != nil && d.Drafting == model.DraftingParallelBlock {
		perReq = k.numSpecTokens() // DSpark's anchor-as-first num_query_per_req (dspark/speculator.py:51-55)
	}
	var draftCurve, verifyCurve [][2]float64
	for _, tokens := range sizes {
		if tokens <= 0 {
			continue
		}
		reqs := min(tokens, k.maxNumSeqs, maxTokens/max(perReq, 1))
		tokens = min(tokens, maxTokens)
		if reqs <= 0 {
			continue
		}
		b := kernel.Batch{DecodeThreshold: k.uniformDecodeWidth}
		for i := range reqs {
			q := tokens / reqs
			if i >= reqs-tokens%reqs {
				q++
			}
			ctx := adaptiveProfileContext
			if k.maxModelLen > 0 {
				ctx = max(min(ctx, k.maxModelLen-(tokens+reqs-1)/reqs), 0)
			}
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: q, Computed: ctx, PromptLen: ctx})
		}
		target := k.priceStep(b, b, nil, false).NoOverlap.Seconds()
		verifyCurve = append(verifyCurve, [2]float64{float64(tokens), target})
		// Only graph-replay samples price the draft curve (set_initial_cost_curves, :221-227).
		if k.graphDecode == graphModeFull && k.stepGraphed(b) {
			whole := k.priceStep(b, b, nil, true).NoOverlap.Seconds()
			draftCurve = append(draftCurve, [2]float64{float64(reqs), whole - target})
		}
	}
	draftCurve, verifyCurve = medianCurve(draftCurve), medianCurve(verifyCurve)
	if len(draftCurve) == 0 || len(verifyCurve) == 0 {
		return fmt.Errorf("speculative.enable_adaptive_verification: \"Adaptive verification could " +
			"not profile step costs\" (adaptive_verification.py:241-246@5df63a72): the deployment " +
			"captures no full decode graph to profile the drafter under")
	}
	k.adaptive.draftCost = costTable(draftCurve, k.maxNumSeqs, 0, 0)
	k.adaptive.verifyCost = costTable(verifyCurve, maxTokens, limit, 1e-9)
	return nil
}

// medianCurve is set_initial_cost_curves' median_curve: one point per size, ascending.
func medianCurve(points [][2]float64) [][2]float64 {
	grouped := map[float64][]float64{}
	for _, p := range points {
		grouped[p[0]] = append(grouped[p[0]], p[1])
	}
	var out [][2]float64
	for x, ys := range grouped {
		sort.Float64s(ys)
		m := ys[len(ys)/2]
		if len(ys)%2 == 0 {
			m = (ys[len(ys)/2-1] + ys[len(ys)/2]) / 2
		}
		out = append(out, [2]float64{x, m})
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

// costTable is build_cost_tables_from_curves' build_table (adaptive_verification.py:100-128@
// 5df63a72): a running maximum over the curve, a step function up to the capture limit, linear
// interpolation between the points above it, linear extrapolation past the last, floored.
func costTable(curve [][2]float64, size, captureLimit int, floor float64) []float64 {
	xs := make([]float64, len(curve))
	ys := make([]float64, len(curve))
	for i, p := range curve {
		xs[i], ys[i] = p[0], p[1]
		if i > 0 {
			ys[i] = math.Max(ys[i], ys[i-1])
		}
	}
	out := make([]float64, size+1)
	for v := range out {
		x := float64(v)
		idx := sort.SearchFloat64s(xs, x)
		out[v] = ys[min(idx, len(xs)-1)]
		if captureLimit > 0 && v > captureLimit {
			// np.interp over the points above the limit, clamped at their ends.
			var ax, ay []float64
			for i := range xs {
				if xs[i] > float64(captureLimit) {
					ax, ay = append(ax, xs[i]), append(ay, ys[i])
				}
			}
			if len(ax) > 0 {
				out[v] = interp(x, ax, ay)
			}
		}
		if n := len(xs); n > 1 && x > xs[n-1] {
			slope := (ys[n-1] - ys[n-2]) / (xs[n-1] - xs[n-2])
			out[v] = ys[n-1] + slope*(x-xs[n-1])
		}
		out[v] = math.Max(out[v], floor)
	}
	return out
}

// interp is numpy.interp: linear between points, clamped at the ends.
func interp(x float64, xs, ys []float64) float64 {
	if x <= xs[0] {
		return ys[0]
	}
	if x >= xs[len(xs)-1] {
		return ys[len(ys)-1]
	}
	i := sort.SearchFloat64s(xs, x)
	if xs[i] == x {
		return ys[i]
	}
	f := (x - xs[i-1]) / (xs[i] - xs[i-1])
	return ys[i-1] + f*(ys[i]-ys[i-1])
}

// lookup reads a table, extrapolating past its end from its last two entries.
func lookup(t []float64, i int) float64 {
	if i < len(t) {
		return t[max(i, 0)]
	}
	n := len(t)
	if n < 2 {
		return t[n-1]
	}
	return t[n-1] + (t[n-1]-t[n-2])*float64(i-n+1)
}

// budget is get_num_tokens' draft budget B for a batch (adaptive_verification.py:282-350@
// 5df63a72): every scheduled (request, step) slot is scored by its survival, and B maximises
// (sampling requests + the top-B scores' sum) / (draft cost at the request count + verify cost
// at the non-draft tokens + B). It also returns each request's scheduled drafts.
func (a *adaptiveVerify) budget(k *Kernel, b kernel.Batch) (int, []int) {
	drafts := make([]int, len(b.Reqs))
	var scores []float64
	nonDraft, sampling, total := 0, 0, 0
	for i, r := range b.Reqs {
		drafts[i] = k.scheduledDrafts(r)
		total += drafts[i]
		nd := r.Scheduled - drafts[i]
		nonDraft += nd
		if r.Computed+nd >= r.PromptLen {
			sampling++
		}
		for j := 0; j < drafts[i] && j < len(a.survival); j++ {
			scores = append(scores, a.survival[j])
		}
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(scores)))
	maxBudget := min(total, max(0, a.maxTotalLogits-len(b.Reqs)))
	draft := lookup(a.draftCost, len(b.Reqs))
	best, bestRatio, accepted := 0, math.Inf(-1), float64(sampling)
	for B := 0; B <= maxBudget; B++ {
		if B > 0 {
			accepted += scores[B-1]
		}
		// np.argmax keeps the first maximum.
		if ratio := accepted / (draft + lookup(a.verifyCost, nonDraft+B)); ratio > bestRatio {
			best, bestRatio = B, ratio
		}
	}
	return best, drafts
}

// admit spreads a budget over the requests as reallocate_drafts does: the globally best slots
// win, and since every request here shares one survival curve, which never rises along the
// steps, that is every drafting request's first step, then every one's second, in batch order
// (_assign_draft_token_budget, adaptive_verification.py:46-72@5df63a72).
func admit(budget int, drafts []int) []int {
	out := make([]int, len(drafts))
	for step := 0; budget > 0; step++ {
		any := false
		for i := range drafts {
			if budget > 0 && drafts[i] > step {
				out[i]++
				budget--
				any = true
			}
		}
		if !any {
			break
		}
	}
	return out
}

// verified is each request's admitted drafts under the budget the engine picks for b.
func (a *adaptiveVerify) verified(k *Kernel, b kernel.Batch) []int {
	B, drafts := a.budget(k, b)
	return admit(B, drafts)
}

// trim is the batch the verifier runs: each request's non-draft tokens and its admitted drafts
// (reallocate_drafts, adaptive_verification.py:392-447@5df63a72).
func (a *adaptiveVerify) trim(k *Kernel, b kernel.Batch) kernel.Batch {
	v := a.verified(k, b)
	out := b
	out.Reqs = append([]kernel.ReqShape(nil), b.Reqs...)
	for i := range out.Reqs {
		out.Reqs[i].Scheduled -= k.scheduledDrafts(b.Reqs[i]) - v[i]
	}
	return out
}

// decodeGraphed is whether a step replays the decode graph. Under adaptive verification the
// engine captures varlen decode graphs, which take any mix of 1 to decode_query_len tokens a
// request (vllm/v1/worker/gpu/cudagraph_utils.py:297-313@5df63a72), so a trimmed decode batch
// still replays one; otherwise only a uniform decode does.
func (k *Kernel) decodeGraphed(b kernel.Batch) bool {
	if len(b.Reqs) == 0 || b.UniformDecode(k.uniformDecodeWidth) {
		return true
	}
	if k.adaptive == nil {
		return false
	}
	for _, r := range b.Reqs {
		if r.Computed < r.PromptLen || r.Scheduled < 1 || r.Scheduled > k.uniformDecodeWidth {
			return false
		}
	}
	return true
}

// VerifiedDrafts reports, for each request of b in order, how many of its scheduled drafts the
// verifier runs this step under adaptive verification: zero for a request that verifies none.
// It is nil when the engine does not enable adaptive verification, where every scheduled draft
// is verified. A simulator caps a request's accepted drafts at it, since a draft the verifier
// never ran cannot be accepted. Pass the batch StepTime prices: the budget depends on all of it.
func (k *Kernel) VerifiedDrafts(b kernel.Batch) []int {
	if k.adaptive == nil {
		return nil
	}
	return k.adaptive.verified(k, b)
}

// scheduledDrafts is how many drafts a request verifies as scheduled: a decode schedules its
// bonus position and its drafts (kernel.ReqShape.Scheduled), so Scheduled - 1 of them, at most
// num_spec_tokens; a prefill chunk verifies none.
func (k *Kernel) scheduledDrafts(r kernel.ReqShape) int {
	if r.Scheduled <= 1 || r.Computed < r.PromptLen {
		return 0
	}
	return min(r.Scheduled-1, k.numSpecTokens())
}
