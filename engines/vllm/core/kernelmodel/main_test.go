// Moved from inference-sim/sim/kernelmodel/main_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"
	"os"
	"testing"
)

// TestMain fails the package, naming each missing root and the command that provides it,
// when an artifact root does not resolve. See roots.go for why it is never a skip.
//
// Not the shared internal/artifacts gate, which imports this package and so cannot be
// imported by its internal tests.
func TestMain(m *testing.M) {
	// The catalog and registry are the pinned substrate's embedded copy, or the directories
	// BLIS_CATALOG and BLIS_REGISTRY name (DefaultData).
	if err := RequireRepos(); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
