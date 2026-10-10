// backends.go names the estimators an arm can score with.
//
// The comparison once scored BLIS's own analytic backends (roofline and trained-physics)
// beside the kernel. Those backends were deleted when blis-latency-kernel became BLIS's only
// latency model (#1851), so the kernel is the only estimator this package can construct. The
// names of the retired arms are kept so the tools that still list them compile; asking for one
// is an error rather than a silent substitution of the kernel.
package harness

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// Estimator names the latency model an arm scores with.
type Estimator string

const (
	// EstimatorKernel is blis-latency-kernel, the subject of this experiment.
	EstimatorKernel Estimator = "kernel"
	// EstimatorRoofline and EstimatorTrainedPhysics name BLIS's retired analytic backends.
	// They can no longer be constructed; see altModel.
	EstimatorRoofline       Estimator = "roofline"
	EstimatorTrainedPhysics Estimator = "trained-physics"
)

// AllEstimators is the comparison order used in reports: the estimators this package can
// actually construct.
var AllEstimators = []Estimator{EstimatorKernel}

// BackendPaths located the inputs the retired analytic backends read. It is kept only so the
// callers that still populate it compile; nothing reads it.
type BackendPaths struct {
	Catalog  string
	HWConfig string
	Defaults string
}

// hostCosts carried the kernel's host per-token cost to the analytic arms.
type hostCosts struct {
	perOutputTokenUs int64
}

// altModel refuses every estimator other than the kernel: the analytic backends it once
// built no longer exist in BLIS.
func altModel(e Estimator, _ kernelmodel.Deployment, _ BackendPaths, _ hostCosts) (sim.LatencyModel, error) {
	return nil, fmt.Errorf("estimator %q is not available: BLIS's analytic latency backends were "+
		"removed when blis-latency-kernel became the only latency model (#1851)", e)
}
