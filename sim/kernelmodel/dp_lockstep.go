package kernelmodel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-schemas/kernel"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// graphResolver is the kernel's typed report of its CUDA-graph resolution
// (blis-latency-kernel CaptureTokens and GraphModes, kernel#54): the largest step a captured
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

// StepTimesAcrossDP prices one lockstep step of a data-parallel group, implementing
// sim.DPStepPricer.
//
// vLLM v0.31.0 all-reduces every rank's token count before each forward
// (coordinate_batch_across_dp, vllm/v1/worker/dp_utils.py:190-260) and, when the step runs
// under a CUDA graph on every rank, pads each rank to the largest count
// (_post_process_dp_padding, :99-111; should_dp_pad, :177-181). The graph mode is synced
// as the minimum across ranks (:114-119), so one rank past the capture ceiling, or a mixed
// batch under FULL_DECODE_ONLY, runs the step eagerly everywhere and nothing is padded.
// A rank with nothing ready runs the dummy batch: a uniform decode of
// uniform_decode_query_len tokens (vllm/v1/worker/gpu_worker.py:1376-1382) with no real
// tokens (num_tokens_unpadded 0), so all of it is pad, padded like any other rank.
//
// Pad tokens are the kernel's Batch.PadTokens: they run every GEMM, collective, MoE dispatch
// and expert at the padded count, and no attention, logits or sampling.
func (m *Model) StepTimesAcrossDP(batches [][]*sim.Request) []int64 {
	kbs := make([]kernel.Batch, len(batches))
	maxTokens := 0
	allDecode := true
	for i, b := range batches {
		kbs[i] = m.batchOf(b)
		dummy := len(kbs[i].Reqs) == 0
		if dummy {
			kbs[i].PadTokens = m.pad.dummyWidth
		}
		maxTokens = max(maxTokens, kbs[i].PaddedTokens())
		allDecode = allDecode && (dummy || kbs[i].UniformDecode(m.pad.dummyWidth))
	}
	padded := m.pad.ceiling > 0 && maxTokens <= m.pad.ceiling && (!m.pad.decodeOnly || allDecode)
	out := make([]int64, len(batches))
	for i := range kbs {
		if padded {
			kbs[i].PadTokens += maxTokens - kbs[i].PaddedTokens()
		}
		out[i] = max(1, ticks(m.k.StepTime(kbs[i]).NoOverlap))
	}
	return out
}

var _ sim.DPStepPricer = (*Model)(nil)
