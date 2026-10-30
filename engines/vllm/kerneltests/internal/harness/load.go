// Moved from blis-latency-kernel/internal/harness/load.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package harness

import (
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"

	blisschemas "github.com/inference-sim/blis-schemas"
)

// LoadBundle forwards to blis-schemas' LoadBundle, which is where the reader now lives (the
// kernel's own LoadBundle is a deprecated forward to it).
//
// It moved out with Open, for the same reason: a consumer in another module could not reach
// it here, and reading the scenario+deployment pair is part of answering "which files does
// this scenario imply" rather than part of this harness. The forward keeps every existing
// caller compiling.
func LoadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	return blisschemas.LoadBundle(path)
}
