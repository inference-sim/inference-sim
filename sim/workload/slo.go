package workload

// SLODimTargets holds per-dimension SLO thresholds for one class.
// A field with value 0 means that dimension is not gated for this class.
// Used by WorkloadSpec.GoodputSLOTargets and TraceHeader.GoodputSLOTargets,
// and consumed by sim/cluster.SLOAttainmentMultiDim (issue #1409).
type SLODimTargets struct {
	TTFTMs float64 `yaml:"ttft_ms,omitempty"`
	ITLMs  float64 `yaml:"itl_ms,omitempty"`
	E2EMs  float64 `yaml:"e2e_ms,omitempty"`
}
