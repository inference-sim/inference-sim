// Moved from inference-sim/sim/kernelmodel/roots.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// The artifact roots a kernel-path command or test reads: the scenario+deployment
// fixtures, the catalog, the coefficient registry, and the measurement corpora. None is read
// from a path outside this repository, or outside the repository that names itself with
// SetVendoredRoot:
//
//   - Scenarios are the cost model's fixtures under kerneltests/testdata/aisimulate, which
//     this package's tests price (SetScenarioRoot names another directory).
//   - The catalog and the registry are verbatim subsets of blis-substrate's catalog/ and
//     registry/ at the release go.mod pins, vendored at testdata/catalog and
//     testdata/registry. They go once blis-substrate's loader is released, and the adapter
//     reads the substrate's embedded data instead.
//   - The measurement corpora are third-party publications this repository does not
//     redistribute. BLIS_MEASUREMENTS names them.
//
// Every root is overridable by its variable. An unresolvable root is never a skip: these were
// once absolute paths under one developer's home directory, and every affected test turned a
// failed read into a SKIP, so a wrong root was indistinguishable from a passing suite.
const (
	catalogEnv      = "BLIS_CATALOG"
	registryEnv     = "BLIS_REGISTRY"
	scenariosEnv    = "BLIS_SCENARIOS"
	measurementsEnv = "BLIS_MEASUREMENTS"
)

const measurementsHow = "export BLIS_MEASUREMENTS=<directory holding the AISimulate and " +
	"InferenceX corpora>"

// DefaultCatalog is the catalog root: BLIS_CATALOG when set, else the vendored
// testdata/catalog.
func DefaultCatalog() string { return vendored(catalogEnv, "catalog") }

// DefaultRegistry is the registry root: BLIS_REGISTRY when set, else the vendored
// testdata/registry.
func DefaultRegistry() string { return vendored(registryEnv, "registry") }

// vendored resolves a root to its variable, else to testdata/<name> in the vendoring
// repository, located from this file's compiled-in path rather than the working directory:
// tests in several packages and commands under cmd/ each run with a different one.
func vendored(env, name string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	return filepath.Join(repoRoot(), "testdata", name)
}

// vendoredRoot is the repository that vendors the catalog and registry (SetVendoredRoot);
// empty means this file's own repository.
var vendoredRoot string

// SetVendoredRoot names the repository whose testdata/catalog and testdata/registry are the
// defaults. A caller that builds this adapter from another module (the simulator) sets it to
// its own root, once, before opening a kernel.
func SetVendoredRoot(dir string) { vendoredRoot = dir }

// repoRoot is the vendoring repository: SetVendoredRoot's, else this file's own. self is
// <repo>/compat/kernelmodel/roots.go.
func repoRoot() string {
	if vendoredRoot != "" {
		return vendoredRoot
	}
	return ownRoot()
}

// ownRoot is this file's own repository.
func ownRoot() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		// Only reachable in a binary built without file information. The working directory
		// is right when run from the repository root, and RequireRepos reports it otherwise.
		return "."
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// scenarioRoot is the fixture directory SetScenarioRoot named; empty means this repository's
// kerneltests/testdata/aisimulate.
var scenarioRoot string

// SetScenarioRoot names the directory DefaultScenarios returns when BLIS_SCENARIOS is unset.
// A caller that holds the cost model's fixtures elsewhere (the simulator) sets it once.
func SetScenarioRoot(dir string) { scenarioRoot = dir }

// DefaultMeasurements is the measurement-corpus directory, or "" when BLIS_MEASUREMENTS
// is unset.
func DefaultMeasurements() string { return os.Getenv(measurementsEnv) }

// DefaultScenarios is the directory holding the scenario+deployment fixtures:
// BLIS_SCENARIOS when set, else SetScenarioRoot's directory, else this repository's
// kerneltests/testdata/aisimulate.
func DefaultScenarios() string {
	if p := os.Getenv(scenariosEnv); p != "" {
		return p
	}
	if scenarioRoot != "" {
		return scenarioRoot
	}
	return filepath.Join(ownRoot(), "kerneltests", "testdata", "aisimulate")
}

// DefaultRepos is the three artifact roots together, with the engine rules, which is how
// Open takes them.
func DefaultRepos() Repos {
	return withEngineRules(Repos{
		Scenarios: DefaultScenarios(),
		Catalog:   DefaultCatalog(),
		Registry:  DefaultRegistry(),
	})
}

// RequireRepos checks that every root in r is set and exists, and otherwise returns one
// error naming each missing root with the command that provides it. With no variable set
// every root is in this repository, so in practice this reports a bad override.
func RequireRepos(r Repos) error {
	var errs []error
	check := func(name, path, how string) {
		if path == "" {
			errs = append(errs, fmt.Errorf("%s root is not set; to set it:\n  %s", name, how))
			return
		}
		if _, err := os.Stat(path); err != nil {
			errs = append(errs, fmt.Errorf("%s root %q is unreadable (%v); to set it:\n  %s",
				name, path, err, how))
		}
	}
	scenariosHow := "export " + scenariosEnv + "=<directory>"
	check("scenarios", r.Scenarios, scenariosHow)
	check("catalog", r.Catalog, "export "+catalogEnv+"=<a blis-substrate catalog/ directory>")
	check("registry", r.Registry, "export "+registryEnv+"=<a blis-substrate registry/ directory>")
	return errors.Join(errs...)
}

// RequireMeasurements checks that the measurement-corpus directory is set and exists.
func RequireMeasurements(dir string) error {
	if dir == "" {
		return fmt.Errorf("measurements root is not set; to set it:\n  %s", measurementsHow)
	}
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("measurements root %q is unreadable (%v); to set it:\n  %s",
			dir, err, measurementsHow)
	}
	return nil
}

// MeasurementPath is the path of one measurement corpus under BLIS_MEASUREMENTS, or ""
// when that variable is unset, so a command's corpus flag has no default that points at
// nothing.
func MeasurementPath(name string) string {
	dir := DefaultMeasurements()
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, name)
}

// RequireCorpora checks that each named corpus flag holds a path. A scoring command calls
// it after flag parsing, so an unset corpus fails naming the flag and how to provide the
// corpora rather than failing later on a relative filename.
func RequireCorpora(flags map[string]string) error {
	names := make([]string, 0, len(flags))
	for name, path := range flags {
		if path == "" {
			names = append(names, "-"+name)
		}
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)
	return fmt.Errorf("%s not set, and %s is unset, so no default exists; either pass "+
		"the paths or:\n  %s", strings.Join(names, ", "), measurementsEnv, measurementsHow)
}
