package validate_test

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/validate/registry"
)

// registryRoot is this repository's registry. A test runs in its package directory, so the
// registry is one level up.
const registryRoot = "../registry"

// TestEveryCoefficientSetLoadsAndValidates loads every committed coefficient set and runs its
// own checks, including the duplicate (name, scope) check, against the file as committed.
// Sets are discovered with registry.SetPaths, the predicate the validator command uses, so a
// set that the command would read cannot be skipped here.
func TestEveryCoefficientSetLoadsAndValidates(t *testing.T) {
	sets, err := registry.SetPaths(registryRoot)
	if err != nil {
		t.Fatalf("registry.SetPaths(%s): %v", registryRoot, err)
	}
	// A floor rather than "not empty": eight sets are committed, so a discovery bug that
	// found one would otherwise pass while testing almost nothing.
	const floor = 8
	if len(sets) < floor {
		t.Fatalf("found %d coefficient sets under %s, want at least %d", len(sets), registryRoot, floor)
	}
	for _, path := range sets {
		s, err := contracts.LoadCoefficientSet(path)
		if err != nil {
			t.Errorf("LoadCoefficientSet(%s): %v", filepath.Base(path), err)
			continue
		}
		if len(s.Coefficients) == 0 {
			t.Errorf("%s: parsed no entries", filepath.Base(path))
			continue
		}
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}
}

// TestEveryAcceptanceSetLoadsAndValidates does the same for acceptance profile sets.
func TestEveryAcceptanceSetLoadsAndValidates(t *testing.T) {
	paths, err := registry.AcceptancePaths(registryRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatalf("found no acceptance sets under %s", registryRoot)
	}
	for _, path := range paths {
		s, err := contracts.LoadAcceptanceProfileSet(path)
		if err != nil {
			t.Errorf("%s: %v", filepath.Base(path), err)
			continue
		}
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}
}
