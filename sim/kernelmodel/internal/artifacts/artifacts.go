// Package artifacts gates a test package on the kernel artifact roots.
//
// A failed read used to become a SKIP, and a skip reads as a pass, so a wrong root was
// indistinguishable from a passing suite. Main fails the whole package instead, naming
// each missing root and the command that provides it.
//
// The measurement corpora are the exception, because they are third-party publications
// neither this repository nor blis-latency-kernel redistributes: the tests that read them
// run under `-tags scoring`, as blis-latency-kernel's own do. Without the tag
// Measurement skips naming what it needs; with it, a missing corpus is a failure.
package artifacts

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// Main runs a package's tests only when every artifact root resolves.
func Main(m *testing.M) {
	if err := kernelmodel.RequireRepos(kernelmodel.DefaultRepos()); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}

// Measurement is the path of one measurement corpus.
func Measurement(t testing.TB, name string) string {
	t.Helper()
	dir := kernelmodel.DefaultMeasurements()
	if err := kernelmodel.RequireMeasurements(dir); err != nil {
		if !scoring {
			t.Skipf("reads the measurement corpora: run with -tags scoring and "+
				"BLIS_MEASUREMENTS set (%s)", name)
		}
		t.Fatal(err)
	}
	return filepath.Join(dir, name)
}
