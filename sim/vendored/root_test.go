package vendored

import "github.com/inference-sim/blis-vllm/compat/kernelmodel"

// The lock test reads the adapter's defaults, which are this repository's once it is named
// the vendoring root.
func init() { kernelmodel.SetVendoredRoot(Root()) }
