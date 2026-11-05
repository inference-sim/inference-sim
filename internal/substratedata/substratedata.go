// Package substratedata locates the catalog and registry that this module's tests and offline
// tools read: the catalog/ and registry/ directories of the blis-substrate module that this
// build selects, as the go command resolves it (go list -m), in the module cache and never in
// a sibling checkout.
//
// It replaces the copies this repository vendored under testdata/catalog and
// testdata/registry, and the lock files that pinned them. The substrate release carries its
// own data (design §3.3), so go.mod's substrate requirement is the one place that says which
// data the tests read.
//
// A run does not use this package. It reads the data a run names with --catalog and
// --registry; the tests pass these directories as those flags.
package substratedata

import (
	"bytes"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// Module is the module path of the substrate whose data this package locates.
const Module = "github.com/inference-sim/blis-substrate"

var (
	once   sync.Once
	dir    string
	dirErr error
)

// Dir is the on-disk directory of the blis-substrate module this build selects. A test runs
// in its package directory, inside this module, so the answer is this module's selection.
func Dir() (string, error) {
	once.Do(func() {
		cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", Module)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			dirErr = fmt.Errorf("go list -m %s: %v: %s", Module, err, strings.TrimSpace(stderr.String()))
			return
		}
		if dir = strings.TrimSpace(string(out)); dir == "" {
			dirErr = fmt.Errorf("go list -m %s reports no directory; run `go mod download`", Module)
		}
	})
	return dir, dirErr
}

// Catalog is the selected substrate's catalog/ directory.
func Catalog() (string, error) { return sub("catalog") }

// Registry is the selected substrate's registry/ directory.
func Registry() (string, error) { return sub("registry") }

func sub(name string) (string, error) {
	d, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, name), nil
}

// MustCatalog is Catalog for a flag default or a test helper, where there is no caller to
// return an error to. It panics with the go command's error.
func MustCatalog() string { return must(Catalog()) }

// MustRegistry is Registry, panicking as MustCatalog does.
func MustRegistry() string { return must(Registry()) }

func must(p string, err error) string {
	if err != nil {
		panic(err)
	}
	return p
}
