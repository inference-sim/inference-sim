// Moved from inference-sim/sim/kernelmodel/dp_lockstep.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// graphResolver is the kernel's typed report of its CUDA-graph resolution
// (blis-latency-kernel CaptureTokens and GraphModes, https://github.com/inference-sim/blis-latency-kernel/issues/54): the largest step a captured
// graph covers, and the modes -- "FULL", "PIECEWISE" or "NONE" -- a uniform decode step and
// every other step run under, after the kernel's overrides.
type graphResolver interface {
	CaptureTokens() int
	GraphModes() (decode, other string)
}

// dpPadding is how a lockstep data-parallel step pads this pool's ranks (#1909), resolved once.
type dpPadding struct {
	// ceiling is the cudagraph capture ceiling in tokens; 0 when the layout runs eagerly.
	ceiling int
	// decodeOnly is whether only uniform decode steps run under a graph (FULL_DECODE_ONLY:
	// every other step runs with none, so a mixed group step is not padded).
	decodeOnly bool
	// dummyWidth is the dummy batch's token count: the uniform decode query length.
	dummyWidth int
}

// dpPaddingOf resolves the padding from the kernel. It fails closed: a kernel that does not
// report its graph resolution, or runs graphs with no capture ceiling, would otherwise price
// every lockstep step unpadded with nothing reporting it.
func dpPaddingOf(k kernel.Kernel, specTokens int) (dpPadding, error) {
	p := dpPadding{dummyWidth: 1 + specTokens}
	g, ok := k.(graphResolver)
	if !ok {
		return p, fmt.Errorf("kernelmodel: the kernel (%T) does not report its CUDA-graph resolution "+
			"(CaptureTokens, GraphModes); lockstep data-parallel padding cannot be priced", k)
	}
	decode, other := g.GraphModes()
	decode, other = strings.ToUpper(decode), strings.ToUpper(other)
	if decode == "NONE" && other == "NONE" {
		return p, nil
	}
	if p.ceiling = g.CaptureTokens(); p.ceiling <= 0 {
		return p, fmt.Errorf("kernelmodel: the kernel runs CUDA graphs (decode %s, other %s) but reports "+
			"capture ceiling %d; lockstep data-parallel padding cannot be priced", decode, other, p.ceiling)
	}
	p.decodeOnly = other == "NONE"
	return p, nil
}

// StepTimesAcrossDPOfShapes prices one lockstep step of a data-parallel group, each rank's
// forward pass priced with the group's padding, an empty rank as the dummy batch. The shapes
// are the caller's: under speculation a decode's Scheduled is already its verify width.
//
// vLLM v0.31.0 all-reduces every rank's token count before each forward
// (coordinate_batch_across_dp, vllm/v1/worker/dp_utils.py:190-260) and, when the step runs
// under a CUDA graph on every rank, pads each rank to the largest count
// (_post_process_dp_padding, :99-111; should_dp_pad, :177-181). The graph mode is synced
// as the minimum across ranks (:114-119), so one rank past the capture ceiling, or a mixed
// batch under FULL_DECODE_ONLY, runs the step eagerly everywhere and nothing is padded.
// A rank with nothing ready runs the dummy batch: a uniform decode of
// uniform_decode_query_len tokens (vllm/v1/worker/gpu_worker.go:1376-1382) with no real
// tokens (num_tokens_unpadded 0), so all of it is pad, padded like any other rank.
//
// The experts of an attention-data-parallel group run over every rank's tokens gathered, so each
// rank's batch states the group's total (Batch.GroupTokens), not dp times its own.
//
// Pad tokens are the kernel's Batch.PadTokens: they run every GEMM, collective, MoE dispatch
// and expert at the padded count, and no attention, logits or sampling.
//
// This used to have a []*sim.Request-shaped sibling, StepTimesAcrossDP, which built its
// kernel.Batch with m.batchOf directly and so never ran the decoder replay that
// StepTimesAcrossDPOfStep applies (m.replay.tail). Nothing but its own tests ever called it --
// the shipped path has always gone through StepTimesAcrossDPOfStep (vllmengine.go's
// kernelPricer.StepTimesAcrossDP) -- so once #2681 made the two paths disagree it was removed
// rather than patched to match; its tests were redirected to this path (dp_lockstep_test.go).
func (m *Model) StepTimesAcrossDPOfShapes(ranks [][]ReqShape) []int64 {
	return m.StepTimesAcrossDPOfStep(ranks, false)
}

// StepTimesAcrossDPOfStep is StepTimesAcrossDPOfShapes for a step that states whether its
// drafters propose drafts: skipDrafts is the group's agreed skip_drafts, every rank's
// (dp_utils.py:75-88@4ba787ec), so each rank's batch carries it (StepTimeOfStep).
func (m *Model) StepTimesAcrossDPOfStep(ranks [][]ReqShape, skipDrafts bool) []int64 {
	kbs := make([]kernel.Batch, len(ranks))
	maxTokens := 0
	allDecode := true
	for i, reqs := range ranks {
		// Each rank trims its own decoder replay tail, exactly as the single-rank path does
		// (StepTimeOfStep): under DP every rank replays if any does (model_state.py:539-542,
		// :614-631@0fda86a3), and m.replay.tail is nil-safe, so a pool with no replay is unchanged.
		kbs[i] = m.replay.tail(kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget,
			SkipDrafts: skipDrafts})
		dummy := len(reqs) == 0
		if dummy {
			kbs[i].PadTokens = m.pad.dummyWidth
		}
		maxTokens = max(maxTokens, kbs[i].PaddedTokens())
		allDecode = allDecode && (dummy || kbs[i].UniformDecode(m.pad.dummyWidth))
	}
	m.padDrafters(kbs)
	padded := m.pad.ceiling > 0 && maxTokens <= m.pad.ceiling && (!m.pad.decodeOnly || allDecode)
	groupTokens := 0
	for i := range kbs {
		if padded {
			kbs[i].PadTokens += maxTokens - kbs[i].PaddedTokens()
		}
		groupTokens += kbs[i].PaddedTokens()
	}
	out := make([]int64, len(ranks))
	for i := range kbs {
		kbs[i].GroupTokens = groupTokens
		out[i] = max(1, ticks(m.k.StepTime(kbs[i]).NoOverlap))
	}
	return out
}

// padDrafters states each rank's drafter launch width (Batch.DraftPaddedTokens), the padding
// vLLM's drafters coordinate across the data-parallel group apart from the target's, which the
// kernel prices as it is told.
//
// The drafter's batch is built from the group's largest real request count, not from the rank's
// own: _build_uniform_batch_dp_sync (vllm/v1/worker/gpu/spec_decode/speculator.py:747-765 at
// 0fda86a35419) gives every rank num_reqs_across_dp.max() (gpu/dp_utils.py, num_reqs) times the
// drafter's query rows per request, and states it uniform and not eager. It is independent of
// the target's pad: it holds when the target steps run eagerly or past the capture ceiling, and
// the idle rank's dummy propose (model_runner.py:1014-1051) launches it too. A parallel-block
// drafter (DSpark, DFlash) runs one pass of num_speculative_tokens rows per request
// (dspark/speculator.py, num_query_per_req); an autoregressive one (MTP, EAGLE) runs its
// multi-step decode at one row per request (autoregressive/speculator.py:167-195).
//
// vLLM stops at one rank: dp_sync_state is None without data parallelism, so a single rank, a
// non-speculating pool and a step whose drafters are skipped state nothing. The graph bucket the
// width rounds up to is not modelled, as for the target's pad.
func (m *Model) padDrafters(kbs []kernel.Batch) {
	if len(kbs) < 2 || m.specTokens <= 0 || m.graph == nil || m.graph.Speculator == nil {
		return
	}
	rowsPerReq := 1
	if m.graph.Speculator.Drafting == model.DraftingParallelBlock {
		rowsPerReq = m.specTokens
	}
	groupReqs := 0
	for _, b := range kbs {
		if b.SkipDrafts {
			return
		}
		groupReqs = max(groupReqs, len(b.Reqs))
	}
	for i := range kbs {
		kbs[i].DraftPaddedTokens = groupReqs * rowsPerReq
	}
}
