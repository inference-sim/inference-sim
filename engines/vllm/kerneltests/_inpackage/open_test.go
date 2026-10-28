// Moved from blis-latency-kernel/open_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"path/filepath"
	"strings"
	"testing"
)

// repos points Open at the fixtures and the pinned catalog and registry, which is the
// configuration every other test in this package reads.
func repos() Repos {
	return withRules(Repos{Scenarios: "testdata", Catalog: catalogRoot, Registry: registryRoot})
}

// TestOpenAndNewAgreeOnTheSameScenario is the property that makes two constructors safe.
//
// Open is New with the loading done for the caller, so for one scenario the two must
// produce the same kernel. Compared on a PRICED STEP rather than on the Inputs they
// assembled: a divergence that does not change a step time is not one a caller can observe,
// and a divergence that does is exactly what this must catch. If Open ever grows a default
// New does not apply — or stops passing something along — this fails.
func TestOpenAndNewAgreeOnTheSameScenario(t *testing.T) {
	const fixtureName = "minimax-m25-h200-ep8.yaml"

	viaOpen, err := Open(fixtureName, repos())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	viaNew := fixture(t, fixtureName)

	b := decodeBatch(64, 1, 4096)
	a, c := viaOpen.StepTime(b), viaNew.StepTime(b)
	if a.Overlap != c.Overlap || a.NoOverlap != c.NoOverlap {
		t.Errorf("Open priced %v/%v where New priced %v/%v; the two constructors must "+
			"resolve one scenario identically",
			a.Overlap, a.NoOverlap, c.Overlap, c.NoOverlap)
	}
	if a.Bottleneck != c.Bottleneck {
		t.Errorf("Open reports bottleneck %v, New %v", a.Bottleneck, c.Bottleneck)
	}
}

// TestOpenNamesWhatItCouldNotRead: each failure says which artifact was missing.
//
// A constructor that reads six kinds of file from three roots has six ways to fail, and
// "no such file or directory" alone sends a reader to check the wrong one. These are the
// errors a consumer in another module will actually hit while wiring up its roots, so the
// message is the feature.
func TestOpenNamesWhatItCouldNotRead(t *testing.T) {
	for _, tc := range []struct {
		what     string
		scenario string
		r        Repos
		wantIn   []string
	}{
		{
			what:     "a scenario that is not there",
			scenario: "no-such-scenario.yaml",
			r:        repos(),
			wantIn:   []string{"no-such-scenario.yaml"},
		},
		{
			what:     "a catalog root that holds no models",
			scenario: "minimax-m25-h200-ep8.yaml",
			r:        withRules(Repos{Scenarios: "testdata", Catalog: t.TempDir(), Registry: registryRoot}),
			wantIn:   []string{"model", "minimax-m2.5"},
		},
		{
			what:     "a registry root that holds no coefficients",
			scenario: "minimax-m25-h200-ep8.yaml",
			r:        withRules(Repos{Scenarios: "testdata", Catalog: catalogRoot, Registry: t.TempDir()}),
			wantIn:   []string{"coefficient set"},
		},
	} {
		t.Run(tc.what, func(t *testing.T) {
			_, err := Open(tc.scenario, tc.r)
			if err == nil {
				t.Fatal("Open succeeded against an incomplete set of roots")
			}
			for _, want := range tc.wantIn {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// TestEveryCommittedFixturePassesFieldValidation turns a one-off probe into a standing
// guard.
//
// Adding validation to New is only safe because every committed artifact passes the field
// layer — measured once over 687 bundles and 32 catalog graphs. Asserting it here means a
// fixture or catalog update that breaks the field layer fails on the fixture, naming it,
// rather than surfacing as an unexplained New() error in whichever test happens to use it.
//
// Field layer only, deliberately. The rules layer rejects 213 of these for a speculative
// method the 0.29.0 pack does not accept, which is a catalog or rules-pack question rather
// than a defect in a constructor. See validateDocuments.
func TestEveryCommittedFixturePassesFieldValidation(t *testing.T) {
	for _, path := range fixturePaths(t) {
		t.Run(path, func(t *testing.T) {
			// Relative to this package's directory, which is what fixtureInputs wants.
			rel, err := filepath.Rel("testdata", path)
			if err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			if err := validateDocuments(fixtureInputs(t, rel)); err != nil {
				t.Errorf("field validation: %v", err)
			}
		})
	}
}
