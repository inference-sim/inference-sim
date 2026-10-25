// Original to blis-vllm.

package build

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// TestMain points the kernel adapter at this module's vendored catalog and registry
// (testdata/), as compat/kernelmodel's own tests do, so the policy and knob tests can open
// real pools.
func TestMain(m *testing.M) {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	kernelmodel.SetVendoredRoot(filepath.Dir(filepath.Dir(wd)))
	if err := kernelmodel.RequireRepos(kernelmodel.DefaultRepos()); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
