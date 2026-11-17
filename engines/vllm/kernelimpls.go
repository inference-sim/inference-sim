// Original to blis-vllm.

package blisvllm

import (
	_ "embed"
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
)

// KernelImplsJSON is kernel_impls.json, the kernel-implementation table this build publishes
// (design §6.2, VAL-8): every attention kernel the vLLM release this build models (Upstream) can
// emit on the architectures the table covers, with the code hash the coefficient registry keys
// its entries by, and the gaps it does not yet list. Its type is blis-substrate's
// contracts/spec/kernelimpls; kernels/regen.py writes it from vLLM's own selection code, and CI
// checks it is current.
//
//go:embed kernel_impls.json
var KernelImplsJSON []byte

// table is the embedded table, parsed once when the package is initialised.
var table, tableErr = kernelimpls.Parse("kernel_impls.json", KernelImplsJSON)

// KernelImpls returns the table, loaded through the substrate's strict loader, which recomputes
// every row's code hash.
func KernelImpls() (*kernelimpls.Table, error) { return table, tableErr }

// ArchOf names the architecture of a device of compute capability major.minor as the table
// does: 9.0 is sm_90a, 10.0 sm_100a, 10.3 sm_103a.
func ArchOf(major, minor int) string { return fmt.Sprintf("sm_%d%da", major, minor) }

// AttentionSelection is the table's name for the selection of an attention backend, by its
// AttentionBackendEnum member: "attention backend FLASHINFER".
func AttentionSelection(backend string) string { return "attention backend " + backend }

// Implementation states the kernel implementation a lowered operation runs: the build of the
// library's kernel (variant) at the version this release pins, that the upstream selection runs
// for the primitive on a device of compute capability major.minor. It is what the lowering puts in
// plan.LoweredLayer.KernelImpl, so the cost model's lookup matches a coefficient only for the
// build that runs (VAL-8).
//
// The table answers through kernelimpls.Table.Resolve. A listed build comes back with its code
// hash and toolkit. A build the table does not know (an uncovered primitive or architecture, or a
// gap that covers the selection or the kernel) comes back with no code hash, which matches no
// keyed coefficient. A build the table knows the release cannot run, such as FA3 on SM100 or a
// misspelt library, is an error: the lowering that asked for it is wrong.
func Implementation(primitive plan.Primitive, selection, library, variant string, major, minor int) (plan.KernelImpl, error) {
	unidentified := plan.KernelImpl{Library: library, Variant: variant}
	t, err := KernelImpls()
	if err != nil {
		return unidentified, err
	}
	arch := ArchOf(major, minor)
	version, _ := t.PinnedVersion(library)
	r, res := t.Resolve(string(primitive), selection, library, version, variant, arch)
	switch res {
	case kernelimpls.Found:
		return plan.KernelImpl{Library: r.Library, Variant: r.Variant, CodeHash: r.CodeHash, Toolkit: r.Toolkit}, nil
	case kernelimpls.Unknown:
		return unidentified, nil
	}
	return unidentified, fmt.Errorf("vLLM %s cannot run %s %s for %s under %s on %s: kernel_impls.json covers them and lists no such build",
		Upstream.Version, library, variant, primitive, selection, arch)
}
