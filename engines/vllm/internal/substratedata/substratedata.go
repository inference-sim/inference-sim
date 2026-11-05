// Original to blis-vllm.

// Package substratedata locates the catalog/ and registry/ directories of the blis-substrate
// module go.mod pins, as the go command resolves it (go list -m), in the module cache and never
// in a sibling checkout. BLIS_CATALOG and BLIS_REGISTRY override each directory.
//
// Runs and most tests read the data through blis-substrate's loader (load.Embedded, or
// load.Open with these directories). This package is for the kernel tests that load single
// documents by path and edit them, which a shared loader Data cannot hand out.
package substratedata

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// Module is the module path of the substrate whose data this package locates.
const Module = "github.com/inference-sim/blis-substrate"

const (
	catalogEnv  = "BLIS_CATALOG"
	registryEnv = "BLIS_REGISTRY"
)

var (
	once   sync.Once
	dir    string
	dirErr error
)

// Dir is the on-disk directory of the blis-substrate module the current module selects, as
// `go list -m` reports it from the working directory. A test runs in its package directory,
// so this is the version the module under test selects: blis-vllm's own pin when blis-vllm's
// tests run, and the simulator's selection when the simulator's tests run.
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

// Catalog is the catalog root: BLIS_CATALOG when set, else the pinned substrate's catalog/.
func Catalog() (string, error) { return root(catalogEnv, "catalog") }

// Registry is the registry root: BLIS_REGISTRY when set, else the pinned substrate's
// registry/.
func Registry() (string, error) { return root(registryEnv, "registry") }

// InModule reports whether path lies under the selected substrate module's directory: the
// data a release carries, which is never optional, as an override may be.
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
