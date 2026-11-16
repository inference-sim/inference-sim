package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/internal/kernelfixtures"
	"github.com/inference-sim/inference-sim/tools/harness/corpora"
)

// This package's tests are gated on the kernel artifact roots. Moved from
// tools/harness/internal/artifacts and tools/harness/corpora, whose only callers were these tests
// (design LIF-7).
//
// A failed read used to become a SKIP, and a skip reads as a pass, so a wrong root was
// indistinguishable from a passing suite. artifactsMain fails the whole package instead, naming
// each missing root and the command that provides it.
//
// The measurement corpora are the exception, because they are third-party publications
// neither this repository nor blis-latency-kernel redistributes: the tests that read them
// run under `-tags scoring`, as blis-latency-kernel's own do. Without the tag
// measurementPath skips naming what it needs; with it, a missing corpus is a failure.

// artifactsMain runs the package's tests only when every artifact root resolves.
func artifactsMain(m *testing.M) {
	if _, err := kernelmodel.OpenData(load.Options{}); err != nil {
		fmt.Fprintf(os.Stderr, "kernel artifact roots are not available:\n%v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}

// measurementPath is the path of one measurement corpus.
func measurementPath(t testing.TB, name string) string {
	t.Helper()
	dir := corpora.DefaultMeasurements()
	if err := requireMeasurements(dir); err != nil {
		if !scoring {
			t.Skipf("reads the measurement corpora: run with -tags scoring and "+
				"BLIS_MEASUREMENTS set (%s)", name)
		}
		t.Fatal(err)
	}
	return filepath.Join(dir, name)
}

// requireMeasurements checks that the measurement-corpus directory is set and exists.
func requireMeasurements(dir string) error {
	const how = "export BLIS_MEASUREMENTS=<directory holding the extracted corpora>"
	if dir == "" {
		return fmt.Errorf("measurements root is not set; to set it:\n  %s", how)
	}
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("measurements root %q is unreadable (%v); to set it:\n  %s", dir, err, how)
	}
	return nil
}

// defaultRepos is the scenario fixtures and the embedded catalog and registry. It panics when
// the data does not open.
func defaultRepos() kernelmodel.Repos {
	d, err := kernelmodel.OpenData(load.Options{})
	if err != nil {
		panic(err)
	}
	return kernelmodel.Repos{Scenarios: kernelfixtures.AISimulate(), Data: d}
}
