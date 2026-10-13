package kernelmodel

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// The artifact roots a kernel-path command or test reads: the scenario+deployment
// fixtures, the catalog, the coefficient registry, and the measurement corpora. Each is
// pinned to the tagged release of the repository that owns it, and none is read from a
// path outside this repository or the Go module cache:
//
//   - Scenarios are blis-latency-kernel's own testdata/aisimulate, read from the module
//     go.mod pins. The fixtures and the kernel that prices them therefore come from one
//     revision by construction: bumping the kernel in go.mod moves both.
//   - The catalog and the registry are verbatim subsets of their tagged releases, vendored
//     at testdata/catalog and testdata/registry (blis-schemas vendors the catalog the same
//     way). WHICH releases is not decided here: blis-latency-kernel's testdata/upstream.lock
//     pins the catalog and registry its fixtures resolve against, it ships in the module,
//     and TestTheVendoredCatalogAndRegistryAreTheLockedReleases holds the vendored copies to
//     it. So the go.mod pin decides all three.
//   - The measurement corpora are third-party publications blis-latency-kernel does not
//     redistribute, so neither does this repository. BLIS_MEASUREMENTS names them, and the
//     tests that need them run under the `scoring` build tag.
//
// Every root is overridable by its variable, so a working copy can be scored against live
// upstream checkouts. An unresolvable root is never a skip: these were once absolute paths
// under one developer's home directory, and every affected test turned a failed read into
// a SKIP, so a wrong root was indistinguishable from a passing suite.
const (
	catalogEnv      = "BLIS_CATALOG"
	registryEnv     = "BLIS_REGISTRY"
	scenariosEnv    = "BLIS_SCENARIOS"
	measurementsEnv = "BLIS_MEASUREMENTS"

	kernelModule = "github.com/inference-sim/blis-latency-kernel"
)

const measurementsHow = "export BLIS_MEASUREMENTS=<directory holding the AISimulate and " +
	"InferenceX corpora>\n  (blis-latency-kernel's testdata/README.md says how they are extracted)"

// Upstream is one release blis-latency-kernel's testdata/upstream.lock pins.
type Upstream struct {
	Repository string // clone URL
	Tag        string // release tag, for the reader
	Commit     string // the commit the tag names, which is what is verified
}

// LockedUpstreams reads blis-latency-kernel's testdata/upstream.lock from the module
// go.mod pins, keyed by its destination name ("catalog", "registry").
func LockedUpstreams() (map[string]Upstream, error) {
	dir, err := kernelModuleDir()
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "testdata", "upstream.lock")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	locked := map[string]Upstream{}
	for _, line := range strings.Split(string(raw), "\n") {
		f := strings.Fields(line)
		if len(f) == 0 || strings.HasPrefix(f[0], "#") {
			continue
		}
		if len(f) < 4 {
			return nil, fmt.Errorf("%s: malformed line %q", path, line)
		}
		locked[f[0]] = Upstream{Repository: f[1], Tag: f[2], Commit: f[3]}
	}
	return locked, nil
}

// cloneHow is the command that provides a locked upstream's root as a live checkout.
func cloneHow(dest, env string) string {
	locked, err := LockedUpstreams()
	if err != nil {
		return fmt.Sprintf("export %s=<clone root> (reading blis-latency-kernel's "+
			"upstream.lock failed: %v)", env, err)
	}
	u, ok := locked[dest]
	if !ok {
		return fmt.Sprintf("export %s=<clone root> (blis-latency-kernel's upstream.lock "+
			"pins no %s)", env, dest)
	}
	name := strings.TrimSuffix(filepath.Base(u.Repository), ".git")
	return fmt.Sprintf("git clone --branch %s --depth 1 %s\n  export %s=$PWD/%s",
		u.Tag, u.Repository, env, name)
}

// DefaultCatalog is the blis-catalog root: BLIS_CATALOG when set, else the vendored
// testdata/catalog.
func DefaultCatalog() string { return vendored(catalogEnv, "catalog") }

// DefaultRegistry is the blis-registry root: BLIS_REGISTRY when set, else the vendored
// testdata/registry.
func DefaultRegistry() string { return vendored(registryEnv, "registry") }

// vendored resolves a root to its variable, else to testdata/<name> in this repository,
// located from this file's compiled-in path rather than the working directory: tests in
// several packages and commands under cmd/ each run with a different one.
func vendored(env, name string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	return filepath.Join(repoRoot(), "testdata", name)
}

// repoRoot is this repository's root. self is <repo>/sim/kernelmodel/roots.go.
func repoRoot() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		// Only reachable in a binary built without file information. The working directory
		// is right when run from the repository root, and RequireRepos reports it otherwise.
		return "."
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// VendoredCommit is the upstream commit a vendored root was copied from, as recorded in its
// .upstream-commit stamp.
func VendoredCommit(root string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(root, ".upstream-commit"))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(raw)), nil
}

// DefaultMeasurements is the measurement-corpus directory, or "" when BLIS_MEASUREMENTS
// is unset.
func DefaultMeasurements() string { return os.Getenv(measurementsEnv) }

// DefaultScenarios is the directory holding the scenario+deployment fixtures:
// BLIS_SCENARIOS when set, else testdata/aisimulate inside the blis-latency-kernel module
// this build pins, or "" when that module cannot be located.
func DefaultScenarios() string {
	if p := os.Getenv(scenariosEnv); p != "" {
		return p
	}
	dir, err := kernelModuleDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "testdata", "aisimulate")
}

// DefaultRepos is the three artifact roots together, which is how Open takes them.
func DefaultRepos() Repos {
	return Repos{
		Scenarios: DefaultScenarios(),
		Catalog:   DefaultCatalog(),
		Registry:  DefaultRegistry(),
	}
}

// RequireRepos checks that every root in r is set and exists, and otherwise returns one
// error naming each missing root with the commands that provide it. With no variable set
// only the scenario root can fail -- the catalog and registry are vendored -- so in practice
// this reports a bad override or a module that has not been downloaded.
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
	scenariosHow := "the " + kernelModule + " module pinned in go.mod provides them " +
		"(run `go mod download`), or export " + scenariosEnv + "=<directory>"
	if _, err := kernelModuleDir(); err != nil && os.Getenv(scenariosEnv) == "" {
		scenariosHow += fmt.Sprintf("\n  (locating the module failed: %v)", err)
	}
	check("scenarios", r.Scenarios, scenariosHow)
	check("catalog", r.Catalog, cloneHow("catalog", catalogEnv))
	check("registry", r.Registry, cloneHow("registry", registryEnv))
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

var (
	moduleDirOnce sync.Once
	moduleDir     string
	moduleDirErr  error
)

// kernelModuleDir is the on-disk directory of the blis-latency-kernel module this build
// pins, as `go list -m` reports it -- a path in the Go module cache, never a sibling
// checkout. It runs from this repository's root, so the answer is the go.mod pin whatever
// the caller's working directory.
func kernelModuleDir() (string, error) {
	moduleDirOnce.Do(func() {
		cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", kernelModule)
		cmd.Dir = repoRoot()
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			moduleDirErr = fmt.Errorf("go list -m %s: %v: %s", kernelModule, err,
				strings.TrimSpace(stderr.String()))
			return
		}
		if moduleDir = strings.TrimSpace(string(out)); moduleDir == "" {
			moduleDirErr = fmt.Errorf("go list -m %s reports no directory; run `go mod download`",
				kernelModule)
		}
	})
	return moduleDir, moduleDirErr
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
