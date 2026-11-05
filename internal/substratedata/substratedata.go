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
	"encoding/json"
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

// Dir is the on-disk directory of the blis-substrate module this build selects. It runs
// `go mod download -json` on the selected version, which downloads the module when the cache
// does not hold it, so a test needs no earlier `go mod download` (X25: the module go.mod
// selects is the test's declared input). A test runs in its package directory, inside this
// module, so the version is this module's selection.
func Dir() (string, error) {
	once.Do(func() { dir, dirErr = download() })
	return dir, dirErr
}

func download() (string, error) {
	version, err := goOutput("list", "-m", "-f", "{{.Version}}", Module)
	if err != nil {
		return "", err
	}
	if version == "" {
		return "", fmt.Errorf("go list -m %s reports no version: this module does not require it", Module)
	}
	out, err := goOutput("mod", "download", "-json", Module+"@"+version)
	if err != nil {
		return "", err
	}
	var d struct{ Dir, Error string }
	if err := json.Unmarshal([]byte(out), &d); err != nil {
		return "", fmt.Errorf("go mod download -json %s@%s: %w", Module, version, err)
	}
	if d.Error != "" || d.Dir == "" {
		return "", fmt.Errorf("go mod download %s@%s: no directory (%s)", Module, version, d.Error)
	}
	return d.Dir, nil
}

func goOutput(args ...string) (string, error) {
	cmd := exec.Command("go", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("go %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(string(out)), nil
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
