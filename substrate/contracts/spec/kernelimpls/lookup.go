package kernelimpls

import (
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
)

// Covers reports whether the table lists every implementation of the primitive.
func (t *Table) Covers(primitive string) bool {
	for _, w := range t.Primitives {
		if w == primitive {
			return true
		}
	}
	return false
}

// GapsFor returns the selections under the primitive whose kernels the table does not yet list.
func (t *Table) GapsFor(primitive string) []Gap {
	var out []Gap
	for _, g := range t.Gaps {
		if g.Primitive == primitive {
			out = append(out, g)
		}
	}
	return out
}

// PinnedVersion returns the version of the library the release pins, and false when the table
// names no such library.
func (t *Table) PinnedVersion(library string) (string, bool) {
	for _, l := range t.Libraries {
		if l.Name == library && l.Pinned {
			return l.Version, true
		}
	}
	return "", false
}

// Resolution says what a table knows about a build a lowering asks for.
type Resolution string

const (
	// Found: the table lists the build.
	Found Resolution = "found"
	// CannotEmit: the table covers the primitive, the architecture and the selection, names no
	// gap that covers the kernel, and lists no such build, so the release cannot run it. A lowering
	// that asks for it is wrong.
	CannotEmit Resolution = "cannot_emit"
	// Unknown: the table does not cover the primitive or the architecture, or a gap covers the
	// selection or the kernel, so a missing row says nothing.
	Unknown Resolution = "unknown"
)

// Resolve finds the build a lowering states: the kernel (library and variant at a library
// version) that the upstream selection runs for the primitive on a device of the architecture.
// It returns the row when the table lists it, and otherwise says whether the table knows the
// release cannot run it (CannotEmit) or does not know (Unknown).
func (t *Table) Resolve(primitive, selection, library, version, variant, arch string) (Implementation, Resolution) {
	for _, r := range t.Implementations {
		if r.Primitive == primitive && r.Selection == selection && r.Library == library &&
			r.LibraryVersion == version && r.Variant == variant && r.TargetArch == arch {
			return r, Found
		}
	}
	if !t.Covers(primitive) || !slices.Contains(t.Archs, arch) {
		return Implementation{}, Unknown
	}
	for _, g := range t.Gaps {
		if g.Primitive == primitive && g.Selection == selection &&
			(g.Library == "" || (g.Library == library && g.Variant == variant)) {
			return Implementation{}, Unknown
		}
	}
	return Implementation{}, CannotEmit
}

// Find returns the row of a kernel build: the primitive it executes, its library at a version,
// its variant and its target architecture. An engine's lowering calls it with the pinned
// version (PinnedVersion) to state the implementation of a lowered operation; a scoring
// harness calls it with the version a run recorded, under any selection. It returns false when
// the table has no such row; Resolve says whether that means the release cannot emit it.
func (t *Table) Find(primitive, library, version, variant, arch string) (Implementation, bool) {
	for _, r := range t.Implementations {
		if r.Primitive == primitive && r.Library == library && r.LibraryVersion == version &&
			r.Variant == variant && r.TargetArch == arch {
			return r, true
		}
	}
	return Implementation{}, false
}

// Impl is the row as the coefficient registry names an implementation: library, variant and
// code hash.
func (r Implementation) Impl() coefficient.KernelImpl {
	return coefficient.KernelImpl{Library: r.Library, Variant: r.Variant, CodeHash: r.CodeHash}
}

// Lists reports whether any row is the implementation k, by library, variant and code hash
// together (design VAL-8).
func (t *Table) Lists(k coefficient.KernelImpl) bool {
	for _, r := range t.Implementations {
		if r.Impl() == k {
			return true
		}
	}
	return false
}
