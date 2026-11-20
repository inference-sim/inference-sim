// Moved from blis-latency-kernel/internal/artifacttest/require.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package artifacttest

import (
	"os"
	"testing"

	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// Pinned reports whether a root is the pinned blis-substrate release's own data: its
// catalog/ or registry/ in the module cache (internal/substratedata).
//
// The pinned data is never optional: if it fails to load, this repository and the substrate
// release it pins are incompatible, and that must fail rather than skip. An override pointing
// at a candidate checkout can genuinely be missing, and may skip.
func Pinned(root string) bool { return substratedata.InModule(root) }

// RequireArtifact reports a failure to load a catalog or registry artifact as either a
// skip or a test failure, depending on which of the two it actually is.
//
// An override that is NOT THERE skips: a test cannot read what is absent. An
// artifact that IS there and still failed to load FAILS, because that is an
// incompatibility between this repository and that one — and a blanket skip over it is
// how one hid for the whole life of a dependency pin.
//
// The history is specific. blis-catalog had already moved to blis-schemas v0.2.0 field
// names (read_bandwidth_mb_s) that the pseudo-version this repo was pinned to rejected, so
// every test building a kernel from a committed fixture skipped on "catalog unavailable"
// -- 41 of them -- and three commands exited 1 against the real catalog. A skip reads as a
// pass, so the suite was green by not running. The fixture
// testdata/minimax-m25-h200-ep16.yaml records a PRIOR occurrence of the same pathology, so
// it is this repository's recurring failure mode rather than a one-off.
//
// In an ordinary file of this leaf package rather than a _test.go one, so other packages'
// tests can call it; it is compiled into any binary that imports artifacttest, which is
// why it takes a testing.TB rather than reaching for os.Exit.
//
// Lives here, exported, rather than in each test package, because the first fix of this
// applied the rule to one package of three and left two command packages masking the
// same corruption under an "ok".
//
// root is the catalog or registry root being read; path is the specific artifact whose
// load failed, which is what gets stat'ed -- a root that exists tells you nothing about a
// file inside it that does not.
func RequireArtifact(t testing.TB, root, path, what string, err error) {
	t.Helper()
	if err == nil {
		return
	}
	// The pinned data is never optional: a failure is a real incompatibility.
	if Pinned(root) {
		t.Fatalf("the pinned blis-substrate %s failed to load %s, which is an "+
			"incompatibility rather than a missing checkout: %v", what, path, err)
	}
	// An override pointing at a live checkout. Skip only if the artifact genuinely is not
	// there; anything else -- a permission error, a path that is a file where a directory
	// belongs, a partially populated checkout -- is a real failure.
	if _, statErr := os.Stat(path); os.IsNotExist(statErr) {
		t.Skipf("%s artifact %s is absent", what, path)
	}
	t.Fatalf("%s artifact %s is present but unreadable, which is an incompatibility "+
		"rather than a missing checkout: %v", what, path, err)
}
