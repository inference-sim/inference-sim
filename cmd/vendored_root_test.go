package cmd

import (
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/vendored"
)

// The kernel adapter's defaults are this repository's vendored catalog, registry and pinned
// kernel (sim/vendored). Only the test files open kernels through those defaults: a run opens
// its kernel at the build site, from the catalog, registry and scenario it is given.
func init() { kernelmodel.SetVendoredRoot(vendored.Root()) }
