package latency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// NewKernelLookupModel creates a kernel-lookup latency model from a kernel profile.
// TODO(task3): implement full kernel lookup logic.
func NewKernelLookupModel(_ sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	return nil, fmt.Errorf("kernel-lookup backend not yet implemented (profile=%s)", hw.KernelProfilePath)
}
