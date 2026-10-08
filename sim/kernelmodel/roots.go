package kernelmodel

import (
	"os"
	"path/filepath"
	"runtime"
)

// The artifact roots a kernel-path command or test reads: the scenario+deployment
// fixtures, the catalog, the coefficient registry, and the InferenceX measurement
// corpora.
//
// These were twenty-six absolute paths under one developer's home directory, spread over
// eleven files. That makes the suite unrunnable for anyone else and -- because every
// affected test turns a failed read into a SKIP -- silent about it. Twenty-three such
// skip-on-missing sites exist in this package alone.
//
// That is not a hypothetical. blis-latency-kernel hit it: under the pseudo-version it was
// pinned to, the sibling catalog had already moved to blis-schemas v0.2.0 field names the
// pinned schema rejected, so 41 tests passed by not running. Its testdata/VENDORED.md
// records that. This side hit the mirror image: the sibling kernel checkout carries
// pre-v0.2.0 scenario fixtures (`hardware` and `pools` at the scenario top level, before
// the Scenario/Deployment split), which every scenario read under schemas v0.2.0 rejects.
//
// So the default is a pinned vendored copy under testdata/blis rather than a sibling
// checkout. A skip becomes a test failure, and the suite is hermetic: it depends on no
// path outside this repository. Each root stays independently overridable, so a working
// copy can still be scored against live upstream artifacts -- which is what
// cmd/metricscore is for -- without editing anything.
const (
	catalogEnv      = "BLIS_CATALOG"
	registryEnv     = "BLIS_REGISTRY"
	scenariosEnv    = "BLIS_SCENARIOS"
	measurementsEnv = "BLIS_MEASUREMENTS"
)

// vendorRoot is testdata/blis in this repository, located from this file's own compiled-in
// path rather than from the working directory.
//
// runtime.Caller is used because these roots are read from tests in several packages and
// from commands under cmd/, each of which runs with a different working directory. A
// relative path would resolve differently for each caller; the source location does not.
func vendorRoot() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		// Only reachable if the binary was built without file information. Fall back to
		// the working directory, which is correct when run from the repository root.
		return filepath.Join("testdata", "blis")
	}
	// self is <repo>/sim/kernelmodel/roots.go
	return filepath.Join(filepath.Dir(filepath.Dir(filepath.Dir(self))), "testdata", "blis")
}

func root(env, name string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	return filepath.Join(vendorRoot(), name)
}

// DefaultCatalog is the blis-catalog root: chip descriptors, fabrics, storage devices and
// derived model graphs.
func DefaultCatalog() string { return root(catalogEnv, "catalog") }

// DefaultRegistry is the blis-registry root, holding the fitted coefficient sets.
func DefaultRegistry() string { return root(registryEnv, "registry") }

// DefaultScenarios is the directory holding the scenario+deployment fixtures.
func DefaultScenarios() string { return root(scenariosEnv, "scenarios") }

// DefaultMeasurements is the directory holding the InferenceX and AISimulate corpora a
// scoring run reads.
func DefaultMeasurements() string { return root(measurementsEnv, "measurements") }

// DefaultRepos is the three artifact roots together, which is how Open takes them.
func DefaultRepos() Repos {
	return Repos{
		Scenarios: DefaultScenarios(),
		Catalog:   DefaultCatalog(),
		Registry:  DefaultRegistry(),
	}
}
