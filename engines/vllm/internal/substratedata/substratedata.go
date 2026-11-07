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
	"encoding/json"
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
	once.Do(func() { dir, dirErr = locate("") })
	return dir, dirErr
}

// goEnv is the environment the go command runs in here: the caller's, with the workspace off
// and the module graph read from go.mod alone (X25). A go.work above the checkout, GOWORK or
// GOFLAGS would otherwise change which blis-substrate go list reports, and a sibling checkout
// would pass for the pinned release.
func goEnv() []string {
	return append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod")
}

// goOutput runs the go command in wd (the working directory when empty) under goEnv.
func goOutput(wd string, args ...string) (string, error) {
	cmd := exec.Command("go", args...)
	cmd.Dir = wd
	cmd.Env = goEnv()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("go %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(string(out)), nil
}

// locate is Dir for the module found from wd. It refuses a result whose version is not the one
// go.mod requires, or whose directory is outside the module cache.
func locate(wd string) (string, error) {
	listed, err := goOutput(wd, "list", "-m", "-f", "{{.Version}}\t{{.Dir}}", Module)
	if err != nil {
		return "", err
	}
	version, d, _ := strings.Cut(listed, "\t")
	if d == "" {
		return "", fmt.Errorf("go list -m %s reports no directory; run `go mod download`", Module)
	}
	edit, err := goOutput(wd, "mod", "edit", "-json")
	if err != nil {
		return "", err
	}
	var mod struct {
		Require []struct{ Path, Version string }
	}
	if err := json.Unmarshal([]byte(edit), &mod); err != nil {
		return "", fmt.Errorf("reading go.mod: %w", err)
	}
	want := ""
	for _, r := range mod.Require {
		if r.Path == Module {
			want = r.Version
		}
	}
	if want == "" || version != want {
		return "", fmt.Errorf("go list -m %s reports version %q, but go.mod requires %q", Module, version, want)
	}
	cache, err := goOutput(wd, "env", "GOMODCACHE")
	if err != nil {
		return "", err
	}
	real := func(p string) string {
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return p
	}
	if rel, err := filepath.Rel(real(cache), real(d)); err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("go list -m %s reports %s, which is outside the module cache %s: a replace directive or a sibling checkout is not the pinned release", Module, d, cache)
	}
	return d, nil
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
