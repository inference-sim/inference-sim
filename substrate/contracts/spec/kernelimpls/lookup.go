package kernelimpls

import "github.com/inference-sim/blis-substrate/contracts/spec/coefficient"

// Covers reports whether the table lists every implementation of the primitive.
func (t *Table) Covers(primitive string) bool {
	for _, w := range t.Primitives {
		if w == primitive {
			return true
		}
	}
	return false
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

// Find returns the row of a kernel build: the primitive it executes, its library at a version,
// its variant and its target architecture. An engine's lowering calls it with the pinned
// version (PinnedVersion) to state the implementation of a lowered operation; a scoring
// harness calls it with the version a run recorded. It returns false when the table has no
// such row, which for a covered primitive means the release cannot emit that build.
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
