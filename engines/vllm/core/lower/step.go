// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/worker/gpu/dp_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// sync_cudagraph_and_dp_padding (:44-210) and dispatch_cg_and_sync_dp (:213-325). Also
// vllm/v1/worker/gpu/model_runner.py (gather_batch_req_state, :1270-1355; _dummy_run, :815-905),
// vllm/v1/worker/utils.py (get_uniform_decode_token_count, :967-975),
// vllm/v1/attention/backend.py (max_decode_query_len, :569-586) and
// vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (_num_graph_context_tokens, :315-317, and the
// context store, :470-491).

package lower

import "fmt"

// RankBatch is what one data-parallel rank hands the runner for a step.
type RankBatch struct {
	// Reqs and Tokens are the rank's scheduled requests and their scheduled tokens. A rank with
	// nothing scheduled runs the dummy batch (DummyBatch).
	Reqs, Tokens int
	// MaxQueryLen is the largest tokens any request schedules.
	MaxQueryLen int
	// HasPrefill is the batch's has_prefill: some request is still prefilling its prompt
	// (model_runner.py:1305). Such a batch leaves out the query-length bound
	// (max_query_len = None, model_runner.py:1759-1762).
	HasPrefill bool
	// DecodeEligible is the batch's decode_graph_eligible: no request is prefilling, or each that
	// is runs as a one-token extension over existing context (model_runner.py:1323-1334).
	DecodeEligible bool
}

// Uniform is the per-request token count when the batch is a uniform decode, nil otherwise
// (get_uniform_decode_token_count: an eligible batch whose every request schedules MaxQueryLen).
func (b RankBatch) Uniform() *int {
	if b.DecodeEligible && b.Reqs > 0 && b.Tokens == b.MaxQueryLen*b.Reqs {
		return ptr(b.MaxQueryLen)
	}
	return nil
}

func (b RankBatch) queryBound() *int {
	if b.HasPrefill {
		return nil
	}
	return ptr(b.MaxQueryLen)
}

// DummyBatch is the batch a rank with no work runs to stay in the collective: a uniform decode of
// decode_query_len tokens for one request (execute_dummy_batch, gpu_worker.py:1431-1437;
// _dummy_run with uniform_decode, model_runner.py:838-850).
func DummyBatch(decodeQueryLen int) RankBatch {
	return RankBatch{Reqs: 1, Tokens: decodeQueryLen, MaxQueryLen: decodeQueryLen, DecodeEligible: true}
}

// Launch is how a rank runs a step.
type Launch struct {
	Mode GraphMode
	// Tokens is the launched token count, after graph and data-parallel padding.
	Tokens int
}

// Sync is what a rank learns from the agreement (DPSyncState): what a drafter pass reuses. Each
// rank holds its own copy, because the padded token count and the request bound come from the
// descriptor that rank's manager returned.
type Sync struct {
	// Reqs is the agreed upper bound on any rank's request count: the padded count a FULL
	// descriptor imposed, else the most any rank scheduled.
	Reqs int
	// Eager is whether the ranks agreed to run without a graph.
	Eager bool
	// Tokens is the agreed token count on every rank.
	Tokens int
}

// SyncDP is sync_cudagraph_and_dp_padding for ranks that do not microbatch: each rank asks its
// manager for a graph; the ranks all-reduce; if any rank wants no graph every rank runs eager at
// its own token count; otherwise every rank dispatches again at the largest count, with the agreed
// uniform-decode and query-length bounds, and runs the graph that returns. With one rank nothing
// is agreed and the rank's own dispatch stands (dispatch_cg_and_sync_dp, :256-262). A rank with no
// work is not empty: it runs the dummy batch (DummyBatch), so Tokens 0 on every rank means that no
// rank has work, and Tokens 0 on some ranks only is an error. The agreements returned are nil for
// one rank and for ranks that all have no work, as vLLM returns None.
func SyncDP(m *GraphManager, ranks []RankBatch) ([]Launch, []Sync, error) {
	if len(ranks) == 0 {
		return nil, nil, fmt.Errorf("lower: no ranks")
	}
	// A rank with no work runs the dummy batch, so a rank with Tokens 0 among ranks with work
	// would be priced as if it had agreed to run eager; vLLM never sees that input.
	idle := 0
	for _, r := range ranks {
		if r.Tokens == 0 {
			idle++
		}
	}
	if idle != 0 && idle != len(ranks) {
		return nil, nil, fmt.Errorf("lower: %d of %d ranks have no tokens; an idle rank runs the dummy batch (DummyBatch)", idle, len(ranks))
	}
	own := make([]Descriptor, len(ranks))
	for i, r := range ranks {
		own[i] = m.Dispatch(r.Reqs, r.Tokens, r.Uniform(), r.queryBound())
	}
	out := make([]Launch, len(ranks))
	syncs := make([]Sync, len(ranks))
	if len(ranks) == 1 {
		// dispatch_cg_and_sync_dp returns no agreement at dp_size 1 (:290).
		out[0] = Launch{Mode: own[0].Mode, Tokens: own[0].Tokens}
		return out, nil, nil
	}
	allZero, maxTokens, maxReqs, minMode := true, 0, 0, GraphFull
	for i, r := range ranks {
		allZero = allZero && r.Tokens == 0
		maxTokens, maxReqs = max(maxTokens, r.Tokens), max(maxReqs, r.Reqs)
		minMode = min(minMode, own[i].Mode)
	}
	if allZero {
		// vLLM returns None when no rank has work, and a caller tells "no agreement" by the nil: a
		// slice of zero agreements (Reqs 0, Tokens 0) would read as one that was reached.
		return out, nil, nil
	}
	if minMode == GraphNone {
		for i, r := range ranks {
			out[i] = Launch{Mode: GraphNone, Tokens: r.Tokens}
			syncs[i] = Sync{Reqs: maxReqs, Eager: true, Tokens: r.Tokens}
		}
		return out, syncs, nil
	}
	// The ranks agree on a uniform count only if they all state the same one, and on a query
	// bound only if every rank states one (:117-129).
	uniform := ranks[0].Uniform()
	boundMax, boundAll := 0, true
	for _, r := range ranks {
		if u := r.Uniform(); uniform == nil || u == nil || *u != *uniform {
			uniform = nil
		}
		if b := r.queryBound(); b == nil {
			boundAll = false
		} else {
			boundMax = max(boundMax, *b)
		}
	}
	var bound *int
	if boundAll {
		bound = ptr(boundMax)
	}
	for i, r := range ranks {
		d := m.Dispatch(r.Reqs, maxTokens, uniform, bound)
		out[i] = Launch{Mode: d.Mode, Tokens: d.Tokens}
		reqs := maxReqs
		if d.Mode == GraphFull && d.Reqs != nil {
			reqs = *d.Reqs
		}
		syncs[i] = Sync{Reqs: reqs, Tokens: d.Tokens}
	}
	return out, syncs, nil
}

// GraphContextTokens is DFlashSpeculator._num_graph_context_tokens (:315-317): the context rows a
// captured draft step stores, one full verify for each request the graph is padded to.
func GraphContextTokens(steps, maxNumTokens, reqs int) int {
	return min(reqs*(steps+1), maxNumTokens)
}

// MaxDecodeQueryLen is max_decode_query_len (vllm/v1/attention/backend.py:569-586@5df63a72): the
// widest request a spec-as-decode attention builder treats as a decode, and so the threshold
// above which a request launches the prefill kernels. It is 1 + num_speculative_tokens, and
// model runner V1's parallel drafter appends that many mask slots again.
func MaxDecodeQueryLen(specTokens *int, parallelDrafting, v2Runner bool) int {
	if specTokens == nil {
		return 1
	}
	q := 1 + *specTokens
	if parallelDrafting && !v2Runner {
		q += *specTokens
	}
	return q
}

// DrafterInput describes a parallel-block drafter's step (DFlash).
type DrafterInput struct {
	// Steps is num_speculative_steps; every draft step has 1+Steps query rows for each request.
	Steps int
	// MaxNumTokens is the runner's max_num_tokens.
	MaxNumTokens int
	// Reqs and TargetTokens are the step's requests and the target forward's tokens.
	Reqs, TargetTokens int
	// Sync is this rank's copy of the target's agreement when the pool has more than one
	// data-parallel rank; nil for one (SyncDP returns none for one rank). Only its request
	// bound is read: the drafter does not reuse a target's agreement to run eager.
	Sync *Sync
}

// DrafterPass is what a parallel-block drafter's step launches.
type DrafterPass struct {
	// Launch is the draft graph's mode and padded token count, agreed across ranks.
	Launch Launch
	// EagerContextRows are the target tokens whose key and value the drafter stores eagerly
	// before the replay: those beyond the rows the captured graph stores itself, or all of them
	// when no graph runs.
	EagerContextRows int
}

// Drafter is the drafter pass: the draft query rows are padded through the drafter's own graph
// manager m (a FULL_DECODE_ONLY manager of decode query length 1+Steps), and the eager context
// store is decided on the padded request count of the graph, not on the unpadded one
// (dflash/speculator.py:440-491). Beside other data-parallel ranks the batch is as wide as the
// target's agreement (_build_uniform_batch_dp_sync, spec_decode/speculator.py:547-564), but the
// drafter dispatches afresh on its own manager even when the target's ranks agreed to run eager:
// the rebuilt agreement carries eager=False, so dispatch_cg_and_sync_dp reads no eager reuse
// (dp_utils.py:261). The oracle runs both functions (oracle/lower/extract.py, "drafter").
func Drafter(m *GraphManager, in DrafterInput) (DrafterPass, error) {
	nq := 1 + in.Steps
	if in.Steps < 1 || in.Reqs < 1 {
		return DrafterPass{}, fmt.Errorf("lower: a drafter pass of %d steps and %d requests", in.Steps, in.Reqs)
	}
	tokens := in.Reqs * nq
	if in.Sync != nil {
		if in.Reqs > in.Sync.Reqs {
			return DrafterPass{}, fmt.Errorf("lower: a drafter batch of %d requests beside an agreement that covers %d "+
				"(spec_decode/speculator.py:553-555)", in.Reqs, in.Sync.Reqs)
		}
		tokens = in.Sync.Reqs * nq
	}
	d := m.Dispatch(in.Reqs, tokens, ptr(nq), nil)
	p := DrafterPass{Launch: Launch{Mode: d.Mode, Tokens: d.Tokens}}
	if d.Mode == GraphFull {
		if d.Reqs == nil {
			return DrafterPass{}, fmt.Errorf("lower: a FULL draft graph with no request count")
		}
		p.EagerContextRows = max(0, in.TargetTokens-GraphContextTokens(in.Steps, in.MaxNumTokens, *d.Reqs))
	} else {
		p.EagerContextRows = in.TargetTokens
	}
	return p, nil
}
