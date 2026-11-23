// Original to blis-vllm.

// Package substratedata locates the substrate/catalog and substrate/registry directories of this
// repository, for the kernel tests that load single documents by path and edit them, which a
// shared loader Data cannot hand out. BLIS_CATALOG and BLIS_REGISTRY override each directory.
//
// Runs and most tests read the data through the substrate's loader (load.Embedded, or load.Open
// with these directories).
package substratedata

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Module is the import path of the substrate packages, whose directory holds the data.
const Module = "github.com/inference-sim/inference-sim/substrate"

const (
	catalogEnv  = "BLIS_CATALOG"
	registryEnv = "BLIS_REGISTRY"

	simulatorModule = "module github.com/inference-sim/inference-sim"
)

var (
	once   sync.Once
	dir    string
	dirErr error
)

// Dir is the substrate/ directory of the repository the caller runs in. A test runs in its
// package directory, inside the one module, so it is found by walking up to the go.mod that
// declares the module.
func Dir() (string, error) {
	once.Do(func() { dir, dirErr = locate("") })
	return dir, dirErr
}

// locate is Dir for the module found from wd (the working directory when empty).
func locate(wd string) (string, error) {
	if wd == "" {
		var err error
		if wd, err = os.Getwd(); err != nil {
			return "", err
		}
	}
	for d := wd; ; d = filepath.Dir(d) {
		if b, err := os.ReadFile(filepath.Join(d, "go.mod")); err == nil && strings.HasPrefix(strings.TrimSpace(string(b)), simulatorModule) {
			return filepath.Join(d, "substrate"), nil
		}
		if filepath.Dir(d) == d {
			return "", fmt.Errorf("no go.mod declaring %s above %s", strings.TrimPrefix(simulatorModule, "module "), wd)
		}
	}
}

// Catalog is the catalog root: BLIS_CATALOG when set, else substrate/catalog.
func Catalog() (string, error) { return root(catalogEnv, "catalog") }

// Registry is the registry root: BLIS_REGISTRY when set, else substrate/registry.
func Registry() (string, error) { return root(registryEnv, "registry") }

// InModule reports whether path lies under the substrate directory: the data the repository
// carries, which is never optional, as an override may be.
func InModule(path string) bool {
	d, err := Dir()
	if err != nil {
		return false
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	return abs == d || strings.HasPrefix(abs, d+string(filepath.Separator))
}

func root(env, name string) (string, error) {
	if p := os.Getenv(env); p != "" {
		return p, nil
	}
	d, err := Dir()
	if err != nil {
		return "", fmt.Errorf("%s root: %w (or set %s)", name, err, env)
	}
	return filepath.Join(d, name), nil
}
