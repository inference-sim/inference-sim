// Moved from inference-sim/sim/kernelmodel/roots.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/inference-sim/blis-substrate/load"
)

// What a kernel-path test or tool reads: the scenario+deployment fixtures, and the catalog and
// coefficient registry.
//
//   - Scenarios are the cost model's fixtures under kerneltests/testdata/aisimulate, which
//     this package's tests price (SetScenarioRoot names another directory).
//   - The catalog and the registry are the data of the blis-substrate release go.mod pins,
//     read through its loader (blis-substrate load): the embedded copy, or the directories
//     BLIS_CATALOG and BLIS_REGISTRY name. Either way the data is checked when it is opened.
//
// An unresolvable root is never a skip: these were once absolute paths under one developer's
// home directory, and every affected test turned a failed read into a SKIP, so a wrong root
// was indistinguishable from a passing suite.
const (
	catalogEnv   = "BLIS_CATALOG"
	registryEnv  = "BLIS_REGISTRY"
	scenariosEnv = "BLIS_SCENARIOS"
)

// DataOptions are the loader options BLIS_CATALOG and BLIS_REGISTRY state: each set variable
// names a directory that replaces the embedded catalog or registry (design §3.3).
func DataOptions() load.Options {
	return load.Options{Catalog: os.Getenv(catalogEnv), Registry: os.Getenv(registryEnv)}
}

var (
	dataMu    sync.Mutex
	dataCache = map[string]*load.Data{}
)

// OpenData opens the catalog and registry o names with blis-substrate's loader, once per
// distinct o, and returns the same Data to every later caller with the same options. A run's
// simulator and its engine therefore read one Data, and its selected-entry report covers both.
// Opening checks the data in full, so a malformed candidate is refused here rather than priced.
func OpenData(o load.Options) (*load.Data, error) {
	key := strings.Join([]string{o.Catalog, o.Registry,
		strings.Join(o.CatalogOverlays, "\x00"), strings.Join(o.RegistryOverlays, "\x00")}, "\x01")
	dataMu.Lock()
	defer dataMu.Unlock()
	if d, ok := dataCache[key]; ok {
		return d, nil
	}
	d, err := load.Open(o)
	if err != nil {
		return nil, err
	}
	dataCache[key] = d
	return d, nil
}

// DefaultData is the data DataOptions name (OpenData): the embedded copy when neither
// BLIS_CATALOG nor BLIS_REGISTRY is set.
func DefaultData() (*load.Data, error) { return OpenData(DataOptions()) }

// ownRoot is this file's own repository. self is <repo>/compat/kernelmodel/roots.go.
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

// DefaultRepos is the scenario root and the default data together, with the engine rules,
// which is how Open takes them. It panics when the data does not open, as a test's or tool's
// defaults cannot report an error; RequireRepos reports the same failure as an error.
func DefaultRepos() Repos {
	d, err := DefaultData()
	if err != nil {
		panic(fmt.Sprintf("kernelmodel: opening the catalog and registry: %v", err))
	}
	return withEngineRules(Repos{Scenarios: DefaultScenarios(), Data: d})
}

// RequireRepos checks that the default scenario root exists and the default data opens, and
// otherwise returns one error naming each failure and how to fix it.
func RequireRepos() error {
	var errs []error
	if p := DefaultScenarios(); p == "" {
		errs = append(errs, fmt.Errorf("scenarios root is not set; to set it:\n  export %s=<directory>", scenariosEnv))
	} else if _, err := os.Stat(p); err != nil {
		errs = append(errs, fmt.Errorf("scenarios root %q is unreadable (%v); to set it:\n  export %s=<directory>",
			p, err, scenariosEnv))
	}
	if _, err := DefaultData(); err != nil {
		errs = append(errs, fmt.Errorf("catalog and registry (%s=%q, %s=%q; unset means the embedded copy): %w",
			catalogEnv, os.Getenv(catalogEnv), registryEnv, os.Getenv(registryEnv), err))
	}
	return errors.Join(errs...)
}
