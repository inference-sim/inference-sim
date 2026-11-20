// Package substratedata locates the catalog and registry that this module's tests and offline
// tools read: the substrate/catalog and substrate/registry directories of this repository.
//
// A run does not use this package. It reads the data a run names with --catalog and
// --registry, or the copy embedded in the binary; the tests pass these directories as those
// flags.
package substratedata

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Module is the import path of the substrate packages in this repository.
const Module = "github.com/inference-sim/inference-sim/substrate"

const simulatorModule = "module github.com/inference-sim/inference-sim"

var (
	once   sync.Once
	dir    string
	dirErr error
)

// Dir is the substrate/ directory of the repository the caller runs in. A test runs in its
// package directory, inside this module, so it is found by walking up to the go.mod that
// declares the simulator's module.
func Dir() (string, error) {
	once.Do(func() { dir, dirErr = find() })
	return dir, dirErr
}

func find() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
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

// Catalog is the substrate/catalog directory.
func Catalog() (string, error) { return sub("catalog") }

// Registry is the substrate/registry directory.
func Registry() (string, error) { return sub("registry") }

func sub(name string) (string, error) {
	d, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, name), nil
}

// MustCatalog is Catalog for a flag default or a test helper, where there is no caller to
// return an error to. It panics with the lookup's error.
func MustCatalog() string { return must(Catalog()) }

// MustRegistry is Registry, panicking as MustCatalog does.
func MustRegistry() string { return must(Registry()) }

func must(p string, err error) string {
	if err != nil {
		panic(err)
	}
	return p
}
