// Moved from inference-sim/sim/kernelmodel/main_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestMain fails the package, naming each missing root and the command that provides it,
// when an artifact root does not resolve. See roots.go for why it is never a skip.
//
// Not the shared internal/artifacts gate, which imports this package and so cannot be
// imported by its internal tests.
func TestMain(m *testing.M) {
	// This module vendors the catalog and registry the tests read under its own testdata/
	// (testdata/README.md): the adapter's tests name that root, as a simulator names its own.
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	SetVendoredRoot(filepath.Dir(filepath.Dir(wd)))
	if err := RequireRepos(DefaultRepos()); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
