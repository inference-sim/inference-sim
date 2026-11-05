package workload

import (
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"github.com/inference-sim/inference-sim/sim/check"
)

// SLODimTargets holds per-dimension SLO thresholds for one class.
// A field with value 0 means that dimension is not gated for this class.
// Used by WorkloadSpec.GoodputSLOTargets and TraceHeader.GoodputSLOTargets,
// and consumed by sim/cluster.SLOAttainmentMultiDim (issue #1409).
//
// Field semantics:
//   - TTFTMs: time to first token threshold, in milliseconds.
//   - ITLMs:  mean inter-token latency (TPOT) threshold, in milliseconds.
//     Compared against per-request mean ITL stored in
//     sim.Metrics.RequestITLs, computed in sim/simulator.go as
//     itlSum / (output_tokens - 1) — the vLLM TPOT convention,
//     which excludes the first generated token. This is NOT a
//     per-token cap; a request with one slow inter-token gap can
//     still pass if its mean is within the threshold.
//   - E2EMs:  end-to-end latency threshold, in milliseconds.
//
// R9 exemption: bare float64 (not *float64) is intentional. R9 requires
// pointer types when "zero is a valid runtime value the user may explicitly
// set". Here zero is the disabled-sentinel for a threshold; gating on a 0 ms
// latency target is meaningless (no completion has zero latency), so there
// is no user value distinct from "absent" and pointer types add no signal.
//
// It is contracts/spec/trace's SLODimTargets, which the trace header and the workload
// specification share.
type SLODimTargets = trace.SLODimTargets

// validateGoodputTargets refuses a stated latency target that is not a finite number of
// milliseconds above zero (check.LatencyTargetMs, the validator --slo-ttft, --slo-itl and
// --slo-e2e meet). A workload specification's and a trace header's goodput_slo_targets both call
// it (SIM-11). A target of 0 is the encoding of an absent one, so it is not checked. Classes are
// visited in sorted order, so the first refusal is deterministic (R2).
func validateGoodputTargets(targets map[string]SLODimTargets) error {
	classes := make([]string, 0, len(targets))
	for cls := range targets {
		classes = append(classes, cls)
	}
	sort.Strings(classes)
	for _, cls := range classes {
		t := targets[cls]
		for _, f := range []struct {
			name string
			v    float64
		}{{"ttft_ms", t.TTFTMs}, {"itl_ms", t.ITLMs}, {"e2e_ms", t.E2EMs}} {
			if f.v == 0 {
				continue
			}
			if err := check.LatencyTargetMs("goodput_slo_targets."+cls+"."+f.name, f.v); err != nil {
				return err
			}
		}
	}
	return nil
}
