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
	if err := RequireRepos(DefaultRepos()); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
