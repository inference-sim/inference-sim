// Original to blis-vllm.

package build

import (
	"fmt"
	"os"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// TestMain checks that the kernel adapter's default data (the pinned substrate's embedded
// catalog and registry) opens, as compat/kernelmodel's own tests do, so the policy and knob
// tests can open real pools.
func TestMain(m *testing.M) {
	if err := kernelmodel.RequireRepos(); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
