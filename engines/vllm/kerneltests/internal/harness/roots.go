// Moved from blis-latency-kernel/internal/harness/roots.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package harness

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// The roots a command or test reads by default: the committed scenario fixtures under
// kerneltests/testdata, and blis-substrate's catalog/ and registry/ at the release go.mod
// pins (internal/substratedata), never a checkout in anyone's home directory.
//
// They used to be absolute paths under one developer's home. That made the suite
// unrunnable for anyone else and, because a failed read became a SKIP, silent about it:
// under the pseudo-version this repo was pinned to, the sibling catalog had already moved
// to v0.2.0 field names the pinned schema rejected, so every test that built a kernel from
// a committed fixture skipped and 41 of them passed by not running.
//
// A caller that wants a candidate passes -catalog/-registry, or sets
// BLIS_CATALOG/BLIS_REGISTRY. The pinned release is only the default.
const (
	scenariosEnv    = "BLIS_SCENARIOS"
	measurementsEnv = "BLIS_MEASUREMENTS"
)

// repoRoot returns the kernel tests' root: blis-vllm's kerneltests/, which holds the
// testdata/ and docs/ these commands and tests read.
//
// Not a relative path from the working directory: a test runs with its own package
// directory as the working directory while a command runs from wherever it was invoked, so
// a bare "testdata" names two different places. Two strategies, in order:
//
// Walk up from the working directory looking for the go.mod that declares blis-vllm, and
// take its kerneltests/. That is correct wherever the tree actually is, including a copy
// someone moved.
//
// Failing that, fall back to this file's own compile-time path. That is right for `go
// test` and `go run` from a source checkout, and it is the only thing available to a
// binary invoked from outside the tree — though for a binary COPIED to another machine the
// path will not exist, which is why the caller treats a missing root as a missing checkout
// rather than a crash.
func repoRoot() string {
	const module = "module github.com/inference-sim/blis-vllm"
	if dir, err := os.Getwd(); err == nil {
		for {
			if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil &&
				strings.HasPrefix(string(b), module+"\n") {
				return filepath.Join(dir, "kerneltests")
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if _, thisFile, _, ok := runtime.Caller(0); ok {
		// this file is <kerneltests>/internal/harness/roots.go
		return filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
	}
	return ""
}

// underRoot joins a path onto the repository root, refusing to produce a relative one.
//
// The refusal is the point. filepath.Join("", "testdata") is NOT an obviously broken
// absolute path — it is the relative path "testdata", which resolves against whatever
// working directory the process happens to have. That is precisely the working-directory
// dependence repoRoot exists to remove, so an unresolvable root would silently restore it.
//
// A panic rather than an error because these are flag defaults, evaluated before a command
// can report anything, and because the condition is not recoverable. The message names the
// way out.
func underRoot(parts ...string) string {
	root := repoRoot()
	if root == "" {
		panic("cannot locate the blis-vllm checkout: no go.mod for this module above the " +
			"working directory, and no usable compile-time path. Set " + scenariosEnv + ".")
	}
	return filepath.Join(append([]string{root}, parts...)...)
}

// DefaultCatalog is the catalog root a command uses unless told otherwise: BLIS_CATALOG, or
// the pinned blis-substrate's catalog/. It panics when neither resolves, for the reason
// underRoot does.
func DefaultCatalog() string { return must(substratedata.Catalog()) }

// DefaultRegistry is the registry root a command uses unless told otherwise: BLIS_REGISTRY,
// or the pinned blis-substrate's registry/.
func DefaultRegistry() string { return must(substratedata.Registry()) }

func must(path string, err error) string {
	if err != nil {
		panic(err)
	}
	return path
}

// DefaultScenarios is the directory holding the committed scenario+deployment fixtures.
func DefaultScenarios() string {
	if p := os.Getenv(scenariosEnv); p != "" {
		return p
	}
	return underRoot("testdata")
}

// DefaultRepos is the three roots together, with the vLLM rules packs, which is how every
// caller uses them.
func DefaultRepos() Repos {
	return WithVLLMRules(Repos{
		Scenarios: DefaultScenarios(),
		Catalog:   DefaultCatalog(),
		Registry:  DefaultRegistry(),
	})
}

// DefaultMeasurements is the directory holding the measurement corpora cmd/score and
// cmd/shape score against: BLIS_MEASUREMENTS, or testdata/measurements.
//
// The corpora are NOT part of this repository. They are measured latencies extracted from
// third-party publications and are not redistributed here.
func DefaultMeasurements() string {
	if p := os.Getenv(measurementsEnv); p != "" {
		return p
	}
	return underRoot("testdata", "measurements")
}
