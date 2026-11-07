// Moved from inference-sim/sim/kernelmodel/roots_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/load"
)

// With neither BLIS_CATALOG nor BLIS_REGISTRY set, the data is the pinned blis-substrate's
// embedded copy, and it holds what this package's tests read.
func TestTheDefaultDataIsTheEmbeddedCopy(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	d, err := DefaultData()
	if err != nil {
		t.Fatal(err)
	}
	if d.Catalog.Source != load.SourceEmbedded || d.Registry.Source != load.SourceEmbedded {
		t.Errorf("catalog %s, registry %s; want both embedded", d.Catalog.Source, d.Registry.Source)
	}
	if _, err := d.Model("glm-5"); err != nil {
		t.Errorf("the embedded catalog lacks a model this package reads: %v", err)
	}
}

// Each variable replaces its half of the data on its own: a candidate registry is tested
// against the embedded catalog, and a directory that holds no data is refused, never ignored.
func TestEachVariableReplacesItsHalfOfTheData(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "/elsewhere/registry")
	if o := DataOptions(); o.Catalog != "" || o.Registry != "/elsewhere/registry" {
		t.Errorf("DataOptions() = %+v with only %s set", o, registryEnv)
	}
	empty := t.TempDir()
	t.Setenv(registryEnv, empty)
	if _, err := DefaultData(); err == nil {
		t.Errorf("%s=%s, an empty directory, was accepted as the registry", registryEnv, empty)
	}
}

// OpenData returns a fresh Data on every call, so a run's selected-entry report never carries
// an earlier run's entries (VAL-6): record an entry, open again, and the new report is empty.
func TestOpenDataIsFreshPerCallAndItsReportStartsEmpty(t *testing.T) {
	a, err := OpenData(load.Options{})
	if err != nil {
		t.Fatal(err)
	}
	a.RecordCoefficient("some-set", "some-entry", coefficient.Scope{}, "exact")
	if got := len(a.Report().Selected); got != 1 {
		t.Fatalf("the first run's report holds %d entries, want the 1 it recorded", got)
	}
	b, err := OpenData(load.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if a == b {
		t.Error("two opens returned one Data: a second run would inherit the first run's report")
	}
	if got := b.Report().Selected; len(got) != 0 {
		t.Errorf("a new run's report holds %d entries, want none: %v", len(got), got)
	}
}

// An unreadable scenario root or data fails naming itself and how to set it -- never a skip.
func TestRequireReposNamesEachFailureAndHowToFixIt(t *testing.T) {
	t.Setenv(scenariosEnv, filepath.Join(t.TempDir(), "absent"))
	t.Setenv(registryEnv, t.TempDir())
	err := RequireRepos()
	if err == nil {
		t.Fatal("an absent scenario root and an empty registry were accepted")
	}
	for _, want := range []string{"scenarios root", scenariosEnv, "catalog and registry", registryEnv} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q:\n%v", want, err)
		}
	}
}

// The scenario fixtures default to this repository's kerneltests/testdata/aisimulate, follow
// SetScenarioRoot, and yield to BLIS_SCENARIOS.
func TestDefaultScenariosAreThisRepositorysFixtures(t *testing.T) {
	t.Setenv(scenariosEnv, "")
	got := DefaultScenarios()
	if want := filepath.Join(ownRoot(), "kerneltests", "testdata", "aisimulate"); got != want {
		t.Errorf("scenario root %q, want %q", got, want)
	}
	if _, err := os.Stat(filepath.Join(got, "glm-5-h200-fp8-sglang-tp8.yaml")); err != nil {
		t.Errorf("scenario root %q lacks a fixture this package reads: %v", got, err)
	}
	dir := t.TempDir()
	SetScenarioRoot(dir)
	defer SetScenarioRoot("")
	if got := DefaultScenarios(); got != dir {
		t.Errorf("DefaultScenarios() = %q under SetScenarioRoot(%q)", got, dir)
	}
	t.Setenv(scenariosEnv, "/from/env")
	if got := DefaultScenarios(); got != "/from/env" {
		t.Errorf("DefaultScenarios() = %q with %s set", got, scenariosEnv)
	}
}

// DefaultRepos agrees with the individual defaults; it is the form Open takes. Its Data is its
// own, opened for that call.
func TestDefaultReposMatchesTheDefaults(t *testing.T) {
	r := DefaultRepos()
	if r.Scenarios != DefaultScenarios() || r.Data == nil || r.Rules == nil || r.EngineSettings == nil {
		t.Errorf("DefaultRepos() = %+v, want the default scenarios, data, rules and engine settings", r)
	}
}
