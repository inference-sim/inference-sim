package kernelmodel

import (
	"strconv"
	"strings"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/inference-sim/sim"
)

// captureAssumption is the provenance entry in which blis-latency-kernel v0.2.0 records the
// cudagraph capture ceiling it resolved ("<tokens>: <reason>"). It is absent when the
// resolved layout runs no graphs at all.
const captureAssumption = "max_cudagraph_capture_size"

// dpPadding is how a lockstep data-parallel step pads its ranks (#1909), resolved once.
type dpPadding struct {
	// ceiling is the cudagraph capture ceiling in tokens; 0 when the layout runs eagerly.
	ceiling int
	// decodeOnly is whether only uniform decode batches are captured (FULL_DECODE_ONLY).
	decodeOnly bool
	// dummyWidth is the dummy batch's token count: the uniform decode query length.
	dummyWidth int
}

func dpPaddingOf(k kernel.Kernel, specTokens int) dpPadding {
	p := dpPadding{dummyWidth: 1 + specTokens}
	for _, o := range k.Provenance() {
		if o.Name != captureAssumption {
			continue
		}
		v, _, _ := strings.Cut(o.Scope, ":")
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			p.ceiling = n
		}
	}
	p.decodeOnly = strings.EqualFold(k.Deployment().Engine.CUDAGraphMode, "FULL_DECODE_ONLY")
	return p
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
// A rank with nothing ready runs the dummy batch: one uniform decode of
// uniform_decode_query_len tokens (vllm/v1/worker/gpu_worker.py:1376-1382), padded like any
// other rank.
//
// Padding is priced as decode rows of the dummy width on a one-token context: the padded
// slots run every GEMM and the MoE dispatch at the padded token count, and read no KV.
func (m *Model) StepTimesAcrossDP(batches [][]*sim.Request) []int64 {
	kbs := make([]kernel.Batch, len(batches))
	maxTokens := 0
	allDecode := true
	for i, b := range batches {
		kbs[i] = m.batchOf(b)
		if len(kbs[i].Reqs) == 0 {
			kbs[i].Reqs = m.padRows(kbs[i].Reqs, m.pad.dummyWidth)
		}
		maxTokens = max(maxTokens, kbs[i].Tokens())
		allDecode = allDecode && kbs[i].UniformDecode(m.pad.dummyWidth)
	}
	padded := m.pad.ceiling > 0 && maxTokens <= m.pad.ceiling && (!m.pad.decodeOnly || allDecode)
	out := make([]int64, len(batches))
	for i := range kbs {
		if padded {
			kbs[i].Reqs = m.padRows(kbs[i].Reqs, maxTokens-kbs[i].Tokens())
		}
		out[i] = max(1, ticks(m.k.StepTime(kbs[i]).NoOverlap))
	}
	return out
}

// padRows appends tokens padding tokens as dummy-width decode rows (the last one narrower
// when the width does not divide them).
func (m *Model) padRows(rows []kernel.ReqShape, tokens int) []kernel.ReqShape {
	w := max(1, m.pad.dummyWidth)
	for tokens > 0 {
		n := min(w, tokens)
		rows = append(rows, kernel.ReqShape{Scheduled: n, Computed: 1, PromptLen: 1})
		tokens -= n
	}
	return rows
}

var _ sim.DPStepPricer = (*Model)(nil)
