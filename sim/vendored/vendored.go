// Package vendored names this repository as the vendoring root of the latency kernel's
// adapter (blis-vllm compat/kernelmodel): its default catalog and registry are this
// repository's testdata/catalog and testdata/registry, and it locates the pinned
// blis-latency-kernel module through this repository's go.mod. The adapter used to find this
// repository from its own source path while it lived here (sim/kernelmodel). Now it lives in
// blis-vllm, so this repository names itself.
//
// Import it (for its side effect) in every package that opens a kernel through the adapter's
// defaults. It reads no environment and no configuration: BLIS_CATALOG and BLIS_REGISTRY still
// override the defaults, inside the adapter, as before.
package vendored

import (
	"path/filepath"
	"runtime"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

func init() { kernelmodel.SetVendoredRoot(Root()) }

// Root is this repository's root. self is <repo>/sim/vendored/vendored.go. In a binary built
// without file information the working directory stands in, as the adapter's own fallback did.
func Root() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}
