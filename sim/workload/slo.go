package workload

// SLODimTargets holds per-dimension SLO thresholds for one class.
// A field with value 0 means that dimension is not gated for this class.
// Used by WorkloadSpec.GoodputSLOTargets and TraceHeader.GoodputSLOTargets,
// and consumed by sim/cluster.SLOAttainmentMultiDim (issue #1409).
//
// R9 exemption: bare float64 (not *float64) is intentional. R9 requires
// pointer types when "zero is a valid runtime value the user may explicitly
// set". Here zero is the disabled-sentinel for a threshold; gating on a 0 ms
// latency target is meaningless (no completion has zero latency), so there
// is no user value distinct from "absent" and pointer types add no signal.
type SLODimTargets struct {
	TTFTMs float64 `yaml:"ttft_ms,omitempty"`
	ITLMs  float64 `yaml:"itl_ms,omitempty"`
	E2EMs  float64 `yaml:"e2e_ms,omitempty"`
}
