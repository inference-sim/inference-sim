// Original to blis-vllm.

package blisvllm

import (
	_ "embed"
	"fmt"
	"sync"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
)

// KernelImplsJSON is kernel_impls.json, the kernel-implementation table this build publishes
// (design §6.2, VAL-8): every attention kernel vLLM v0.32.0 can emit on the architectures it
// covers, with the code hash the coefficient registry keys its entries by. Its type is
// blis-substrate's contracts/spec/kernelimpls; kernels/gen_kernel_impls.py writes it from vLLM's
// own selection code, and CI checks it is current for the pinned release.
//
//go:embed kernel_impls.json
var KernelImplsJSON []byte

var (
	tableOnce sync.Once
	table     *kernelimpls.Table
	tableErr  error
)

// KernelImpls returns the table, loaded through the substrate's strict loader, which recomputes
// every row's code hash.
func KernelImpls() (*kernelimpls.Table, error) {
	tableOnce.Do(func() { table, tableErr = kernelimpls.Parse("kernel_impls.json", KernelImplsJSON) })
	return table, tableErr
}

// ArchOf names the architecture of a device of compute capability major.minor as the table
// does: 9.0 is sm_90a, 10.0 sm_100a, 10.3 sm_103a.
func ArchOf(major, minor int) string { return fmt.Sprintf("sm_%d%da", major, minor) }

// Implementation states the kernel implementation a lowered operation runs: the build of the
// library's kernel (variant) that this release pins, for the primitive, on a device of compute
// capability major.minor. It is what the lowering puts in plan.LoweredLayer.KernelImpl, so the
// cost model's lookup matches a coefficient only for the build that runs (VAL-8).
//
// When the table has no row for the build and does not cover the primitive or the architecture,
// or names a gap under the primitive, the build is not identified: the implementation carries the
// library and variant with no code hash, which matches no keyed coefficient, the honest answer.
// When the table covers both, names no gap under the primitive and lists no such build, the
// release cannot emit it, and that is an error in the lowering that asked for it.
func Implementation(primitive plan.Primitive, library, variant string, major, minor int) (plan.KernelImpl, error) {
	unidentified := plan.KernelImpl{Library: library, Variant: variant}
	t, err := KernelImpls()
	if err != nil {
		return unidentified, err
	}
	arch := ArchOf(major, minor)
	if !t.Covers(string(primitive)) || !coversArch(t, arch) {
		return unidentified, nil
	}
	version, _ := t.PinnedVersion(library)
	r, ok := t.Find(string(primitive), library, version, variant, arch)
	if !ok && len(t.GapsFor(string(primitive))) > 0 {
		return unidentified, nil
	}
	if !ok {
		return unidentified, fmt.Errorf("vLLM %s cannot run %s %s %s for %s on %s: kernel_impls.json lists no such build",
			Upstream.Version, library, version, variant, primitive, arch)
	}
	return plan.KernelImpl{Library: r.Library, Variant: r.Variant, CodeHash: r.CodeHash, Toolkit: r.Toolkit}, nil
}

func coversArch(t *kernelimpls.Table, arch string) bool {
	for _, a := range t.Archs {
		if a == arch {
			return true
		}
	}
	return false
}
