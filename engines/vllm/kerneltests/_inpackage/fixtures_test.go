// Moved from blis-latency-kernel/fixtures_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Every committed fixture is a Scenario and the Deployment applied to it, two YAML
// documents in one file. These tests hold that shape to its contract.
//
// The contract is worth a test rather than a convention because the failure it prevents is
// silent. A fixture whose second document went missing, or whose pools landed in the
// scenario half, does not fail to parse — strict decoding would accept a scenario and then
// a kernel would fail far away, on a pool index that has no pool, with nothing naming the
// file. There are 685 of these and 678 are generated, so one generator bug is 678 bad
// fixtures.

// fixturePaths returns every committed scenario+deployment file.
func fixturePaths(t *testing.T) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir("testdata", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Pruned by PATH rather than by base name, so a directory that happens to
			// share one of these names deeper in the tree is still walked.
			switch path {
			// The corpora the fixtures are scored against, not fixtures.
			case filepath.Join("testdata", "measurements"):
				return filepath.SkipDir
			// The pinned upstream copies: chips, fabrics, model graphs and coefficient
			// sets. They are catalog and registry documents with their own schemas, so
			// they are not scenario+deployment pairs and must not be held to that shape.
			case filepath.Join("testdata", "catalog"), filepath.Join("testdata", "registry"):
				return filepath.SkipDir
			}
		}
		// Both spellings: a fixture written as .yml would otherwise be invisible to every
		// contract below, which is the silent kind of gap this file exists to close.
		if !d.IsDir() && (strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml")) {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking testdata: %v", err)
	}
	// A floor, not just non-empty. 678 of these are generated, so a glob or generator
	// regression that emits one fixture instead of 678 would satisfy every assertion below
	// while testing almost nothing — the "zero tests ran" failure in a subtler form. The
	// bound is deliberately loose: it catches a collapse without breaking on the ordinary
	// addition or removal of a handful of scenarios.
	const floor = 600
	if len(out) < floor {
		t.Fatalf("found %d fixtures, want at least %d: the corpus is 685 files, of which "+
			"678 are generated, so a count this low means a glob or the generator dropped "+
			"most of them rather than that scenarios were deleted", len(out), floor)
	}
	return out
}
