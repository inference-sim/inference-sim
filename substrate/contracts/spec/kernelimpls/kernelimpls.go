// Package kernelimpls describes an engine release's kernel-implementation table:
// kernel_impls.json, which lists every kernel implementation one release of an engine can
// emit, with the code hash the coefficient registry keys its entries by (design §6.2, D5,
// VAL-8).
//
// The engine repository generates the table from its upstream's own kernel-selection code and
// publishes it at its root, so a tag of the engine release carries it. Three readers use it:
// the engine's lowering, which states the implementation of each lowered operation from it; the
// registry's fitters, which key a fitted entry by the implementation its data exercised; and the
// scoring harness, which maps the library versions a run recorded to the hashes BLIS priced
// (design §11.3, G1). This package is the table's one definition: its Go type, its validator,
// its strict loader and the hash rule every engine uses.
//
// The yaml tags repeat the json tags. The table is read as JSON; Parse decodes it through the
// strict YAML decoder every BLIS document uses (JSON is YAML), and the yaml tags let the
// documentation test check the reference page's key tables against these types.
package kernelimpls

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
)

// Kind is the value of Table.Kind.
const Kind = "KernelImplTable"

// Table is one engine release's kernel-implementation table.
type Table struct {
	Kind string `json:"kind" yaml:"kind"`
	// Engine names the engine release whose kernels the table lists.
	Engine Engine `json:"engine" yaml:"engine"`
	// Primitives are the lowered primitives (plan.Primitive words) whose implementations the
	// table lists: every kernel the release can emit for one of them is a row, except where Gaps
	// names a selection the table does not yet list. A primitive not named here is not covered,
	// and a reader learns nothing from a missing row for it. Sorted, each once.
	Primitives []string `json:"primitives" yaml:"primitives"`
	// Archs are the device architectures the table lists builds for, as the compiler names them
	// (sm_90a, sm_100a), sorted, each once. A build for a device of another architecture is not
	// covered, and a missing row for it says nothing.
	Archs []string `json:"archs" yaml:"archs"`
	// Gaps are the upstream selections, or single kernels inside a selection, under a covered
	// primitive, that the table does not yet list, each with the issue that adds them. A reader
	// that finds no row for a build a gap covers learns nothing from it (Resolve). Sorted by
	// primitive, selection, library and variant.
	Gaps []Gap `json:"gaps,omitempty" yaml:"gaps,omitempty"`
	// Libraries are the kernel libraries the rows name, at each version the table covers: the
	// release's pinned version, and any other version runs of the release were seen with.
	// Sorted by name and then version, in byte order.
	Libraries []Library `json:"libraries" yaml:"libraries"`
	// Implementations has one row per (primitive, library, library version, variant, target
	// architecture), sorted in that order, in byte order.
	Implementations []Implementation `json:"implementations" yaml:"implementations"`
}

// Gap is an upstream selection whose kernels a table does not yet list.
type Gap struct {
	// Primitive is one of Table.Primitives.
	Primitive string `json:"primitive" yaml:"primitive"`
	// Selection names the upstream choice, in the upstream's own words: a backend, a kernel
	// path or a setting's value.
	Selection string `json:"selection" yaml:"selection"`
	// Library and Variant, when stated, narrow the gap to one kernel the selection runs and the
	// table does not list; both or neither. Neither means every kernel of the selection.
	Library string `json:"library,omitempty" yaml:"library,omitempty"`
	Variant string `json:"variant,omitempty" yaml:"variant,omitempty"`
	// Issue tracks the work that lists its kernels, as <owner>/<repo>#<n>.
	Issue string `json:"issue" yaml:"issue"`
}

// Engine names an engine release.
type Engine struct {
	// Kind is the engine kind the repository registers.
	Kind string `json:"kind" yaml:"kind"`
	// Version is the upstream release.
	Version string `json:"version" yaml:"version"`
	// Commit is the upstream release's commit, 40 lower-case hexadecimal digits.
	Commit string `json:"commit" yaml:"commit"`
}

// Library is one version of a kernel library.
type Library struct {
	// Name is the library's name as the registry's kernel_impls name it.
	Name string `json:"name" yaml:"name"`
	// Version is the library's own version, or the commit the engine builds it from.
	Version string `json:"version" yaml:"version"`
	// Pinned is true for the version the engine release pins. Each library has exactly one
	// pinned version.
	Pinned bool `json:"pinned" yaml:"pinned"`
	// Pin cites where the version is stated: for a pinned version, the engine release's lock or
	// build file as path:line@commit8, where commit8 is the release's commit; for the engine's own
	// kernels, whose version is the release's commit, any statement; for a version seen in runs,
	// the run record that shows it.
	Pin string `json:"pin" yaml:"pin"`
	// Origin is the immutable identity of what was hashed: a repository URL and commit, or a
	// distribution file and its SHA-256.
	Origin string `json:"origin" yaml:"origin"`
}

// Implementation is one kernel implementation the release can emit.
type Implementation struct {
	// Primitive is the lowered primitive the kernel executes, one of Table.Primitives.
	Primitive string `json:"primitive" yaml:"primitive"`
	// Selection names the upstream choice that runs the kernel, in the upstream's own words, as
	// a gap's Selection does: a backend, a kernel path or a setting's value. A lowering knows the
	// choice it made, and Resolve reads it.
	Selection string `json:"selection" yaml:"selection"`
	// Library and LibraryVersion name a row of Table.Libraries.
	Library        string `json:"library" yaml:"library"`
	LibraryVersion string `json:"library_version" yaml:"library_version"`
	// Variant is the library's own exported kernel name.
	Variant string `json:"variant" yaml:"variant"`
	// TargetArch is the architecture of the device the build runs on, as the compiler names it
	// (sm_90a, sm_100a), one of Table.Archs. A lowering finds it from the chip's compute
	// capability.
	TargetArch string `json:"target_arch" yaml:"target_arch"`
	// HashBasis says what the code hash covers.
	HashBasis HashBasis `json:"hash_basis" yaml:"hash_basis"`
	// HashInputs are the content identities the hash covers, sorted, each once: for a source
	// build, each source path with its git object id; for a binary, each shipped file or
	// manifest with its SHA-256.
	HashInputs []string `json:"hash_inputs" yaml:"hash_inputs"`
	// CodeHash is CodeHash(HashBasis, Library, Variant, TargetArch, HashInputs).
	CodeHash string `json:"code_hash" yaml:"code_hash"`
	// Toolkit is the compiler or CUDA toolkit version that builds the kernel, as
	// plan.KernelImpl.Toolkit is: nvcc's for CUDA source, the Triton or CuTe DSL compiler's for a
	// kernel those compile.
	Toolkit string `json:"toolkit" yaml:"toolkit"`
	// SelectedBy cites the upstream code that can select the kernel, as path:line@commit8 with
	// the release's commit, sorted, each once.
	SelectedBy []string `json:"selected_by" yaml:"selected_by"`
}

// HashBasis is what a code hash covers (design §6.2).
type HashBasis string

const (
	// HashSource: the kernel is built from source, and the hash covers its source files and
	// the target architecture.
	HashSource HashBasis = "source"
	// HashBinary: the kernel ships only as a binary, and the hash covers the shipped file's
	// SHA-256 and the target architecture.
	HashBinary HashBasis = "binary"
)

// hashVersion opens the text CodeHash digests. A change to the rule is a new version, and every
// hash changes with it.
const hashVersion = "blis kernel hash v1"

// CodeHash is the rule every engine uses to identify a kernel build: the SHA-256, in lower-case
// hexadecimal, of the lines
//
//	blis kernel hash v1
//	basis <basis>
//	library <library>
//	variant <variant>
//	arch <target arch>
//	input <input>        (one line per input, in byte order)
//
// each ending in a newline. No value may hold a line break, which would make two rows hash the
// same text; the validator refuses one. The library's version is not hashed: a version bump that leaves a
// kernel's inputs unchanged leaves its hash unchanged, so a coefficient keyed by it stays valid
// with no registry change (design §6.2). Inputs are sorted here, so their order does not
// matter to the caller.
func CodeHash(basis HashBasis, library, variant, arch string, inputs []string) string {
	in := append([]string(nil), inputs...)
	sort.Strings(in)
	var b strings.Builder
	b.WriteString(hashVersion + "\n")
	b.WriteString("basis " + string(basis) + "\n")
	b.WriteString("library " + library + "\n")
	b.WriteString("variant " + variant + "\n")
	b.WriteString("arch " + arch + "\n")
	for _, s := range in {
		b.WriteString("input " + s + "\n")
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:])
}
