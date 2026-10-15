package kernelmodel

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/inference-sim/sim"
)

// graphResolver is the kernel's typed report of its CUDA-graph resolution
// (blis-latency-kernel CaptureTokens and GraphModes, kernel#54).
type graphResolver interface {
	CaptureTokens() int
	GraphModes() (decode, other string)
}

// captureAssumption and modeAssumption are the provenance entries in which
// blis-latency-kernel v0.2.0, which predates graphResolver, records the capture ceiling and
// an assumed graph mode ("<value>: <reason>").
const (
	captureAssumption = "max_cudagraph_capture_size"
	modeAssumption    = "cudagraph_mode"
)

// dpPadding is how a lockstep data-parallel step pads this pool's ranks (#1909), resolved once.
type dpPadding struct {
	// ceiling is the cudagraph capture ceiling in tokens; 0 when the layout runs eagerly.
	ceiling int
	// decodeOnly is whether only uniform decode batches are captured (FULL_DECODE_ONLY:
	// every other step runs with no graph, so a mixed group step is not padded).
	decodeOnly bool
	// dummyWidth is the dummy batch's token count: the uniform decode query length.
	dummyWidth int
}

// graphResolution is the kernel's resolved capture ceiling and graph modes: from its typed
// accessors when it has them, else from v0.2.0's provenance (the deployment's mode or the
// one the kernel assumed, replaced by an override where the kernel declined it).
func graphResolution(k kernel.Kernel) (ceiling int, decode, other string, err error) {
	if g, ok := k.(graphResolver); ok {
		decode, other = g.GraphModes()
		return g.CaptureTokens(), strings.ToUpper(decode), strings.ToUpper(other), nil
	}
	mode := strings.ToUpper(k.Deployment().Engine.CUDAGraphMode)
	if mode == "" {
		if v, ok := provenanceValue(k, modeAssumption); ok {
			mode = strings.ToUpper(v)
		}
	}
	for _, o := range k.Resolved().Overrides {
		if o.Field == modeAssumption {
			mode = strings.ToUpper(o.Resolved)
		}
	}
	// CUDAGraphMode.decode_mode / mixed_mode (vllm/config/compilation.py:62-69).
	switch mode {
	case "", "NONE":
		return 0, "NONE", "NONE", nil
	case "FULL_DECODE_ONLY":
		decode, other = "FULL", "NONE"
	case "FULL_AND_PIECEWISE":
		decode, other = "FULL", "PIECEWISE"
	default:
		decode, other = mode, mode
	}
	v, _ := provenanceValue(k, captureAssumption)
	if ceiling, err = strconv.Atoi(v); err != nil {
		return 0, decode, other, fmt.Errorf("provenance %s = %q: %w", captureAssumption, v, err)
	}
	return ceiling, decode, other, nil
}

func provenanceValue(k kernel.Kernel, name string) (string, bool) {
	for _, o := range k.Provenance() {
		if o.Name == name {
			v, _, _ := strings.Cut(o.Scope, ":")
			return strings.TrimSpace(v), true
		}
	}
	return "", false
}

// dpPaddingOf resolves the padding from the kernel. It fails closed: a layout that runs
// graphs but whose capture ceiling cannot be read would otherwise price every lockstep step
// unpadded with nothing reporting it.
func dpPaddingOf(k kernel.Kernel, specTokens int) (dpPadding, error) {
	p := dpPadding{dummyWidth: 1 + specTokens}
	ceiling, decode, other, err := graphResolution(k)
	graphs := decode != "NONE" || other != "NONE"
	if !graphs {
		return p, nil
	}
	if err != nil || ceiling <= 0 {
		return p, fmt.Errorf("kernelmodel: the kernel runs CUDA graphs (decode %s, other %s) but reports "+
			"no capture ceiling (%d, %v); lockstep data-parallel padding cannot be priced", decode, other, ceiling, err)
	}
	p.ceiling = ceiling
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
			kbs[i] = m.withPadding(kbs[i], m.pad.dummyWidth)
		}
		maxTokens = max(maxTokens, kbs[i].Tokens())
		allDecode = allDecode && kbs[i].UniformDecode(m.pad.dummyWidth)
	}
	padded := m.pad.ceiling > 0 && maxTokens <= m.pad.ceiling && (!m.pad.decodeOnly || allDecode)
	out := make([]int64, len(batches))
	for i := range kbs {
		if padded {
			kbs[i] = m.withPadding(kbs[i], maxTokens-kbs[i].Tokens())
		}
		out[i] = max(1, ticks(m.k.StepTime(kbs[i]).NoOverlap))
	}
	return out
}

// withPadding adds tokens of DP padding to a rank's batch.
//
// blis-latency-kernel v0.2.0 has no input for padding, so the slots are priced as decode
// rows of the dummy width on a one-token context: they run every GEMM and the MoE dispatch
// at the padded count and read no KV, but also pay the LM head and sampler, which vLLM skips
// for padded tokens (logits are taken only for the unpadded rows). That over-prices padding by
// the logits rows. The kernel branch dsv41/h100-capacity-fidelity is adding a typed
// pad-token input to kernel.Batch; this function, and the dummy batch below, is the one
// place to switch to it.
func (m *Model) withPadding(b kernel.Batch, tokens int) kernel.Batch {
	w := max(1, m.pad.dummyWidth)
	for tokens > 0 {
		n := min(w, tokens)
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: n, Computed: 1, PromptLen: 1})
		tokens -= n
	}
	return b
}

var _ sim.DPStepPricer = (*Model)(nil)
