// Moved from inference-sim/sim/kernelmodel/roots_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The catalog and registry default to the copies vendored in this repository, which exist
// and are the trees this package reads.
func TestTheCatalogAndRegistryDefaultToTheVendoredCopies(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	repo, err := filepath.Abs(repoRoot())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ name, got, probe string }{
		{"catalog", DefaultCatalog(), filepath.Join("models", "glm-5", "graph.yaml")},
		{"registry", DefaultRegistry(), "coefficients"},
	} {
		if want := filepath.Join(repo, "testdata", c.name); c.got != want {
			t.Errorf("%s root is %q, want the vendored %q", c.name, c.got, want)
		}
		if _, err := os.Stat(filepath.Join(c.got, c.probe)); err != nil {
			t.Errorf("%s root %q lacks %s: %v", c.name, c.got, c.probe, err)
		}
	}
}

// An unset root fails naming itself and the command that provides it -- never a skip, and
// never a bare "no such file" that leaves the reader to work out which repository is
// meant.
func TestRequireReposNamesEachMissingRootAndHowToSetIt(t *testing.T) {
	err := RequireRepos(Repos{Scenarios: t.TempDir()})
	if err == nil {
		t.Fatal("unset catalog and registry roots were accepted")
	}
	for _, want := range []string{
		"catalog root is not set", catalogEnv,
		"registry root is not set", registryEnv,
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q:\n%v", want, err)
		}
	}
	if strings.Contains(err.Error(), "scenarios") {
		t.Errorf("an existing scenario root was reported missing:\n%v", err)
	}
}

// Every root must stay independently overridable, so a working copy can be scored against
// a candidate catalog or registry.
func TestEachRootIsIndependentlyOverridable(t *testing.T) {
	for _, c := range []struct {
		env string
		get func() string
	}{
		{catalogEnv, DefaultCatalog},
		{registryEnv, DefaultRegistry},
		{scenariosEnv, DefaultScenarios},
	} {
		t.Run(c.env, func(t *testing.T) {
			want := filepath.Join(t.TempDir(), "elsewhere")
			t.Setenv(c.env, want)
			if got := c.get(); got != want {
				t.Errorf("with %s=%q the root resolved to %q", c.env, want, got)
			}
		})
	}
}

// Overriding one root must not move the others. They come from separate upstream
// repositories on separate releases; a shared prefix would make it impossible to score a
// live registry against the pinned catalog.
func TestOverridingOneRootLeavesTheOthersAlone(t *testing.T) {
	before := []string{DefaultRegistry(), DefaultScenarios()}
	t.Setenv(catalogEnv, filepath.Join(t.TempDir(), "only-the-catalog"))
	after := []string{DefaultRegistry(), DefaultScenarios()}
	for i := range before {
		if before[i] != after[i] {
			t.Errorf("overriding %s moved another root: %q became %q",
				catalogEnv, before[i], after[i])
		}
	}
}

// The scenario default must not depend on the working directory. Tests in several packages
// and commands under cmd/ each run with a different one.
func TestTheScenarioRootDoesNotDependOnTheWorkingDirectory(t *testing.T) {
	t.Setenv(scenariosEnv, "")
	from := DefaultScenarios()
	t.Chdir(t.TempDir())
	if got := DefaultScenarios(); got != from {
		t.Errorf("after chdir the scenario root moved from %q to %q", from, got)
	}
}

// DefaultRepos must agree with the individual resolvers. It is the form Open takes, and a
// divergence would mean a test and a command disagreeing about which artifacts they read
// while both appearing to use the defaults.
func TestDefaultReposMatchesTheIndividualRoots(t *testing.T) {
	r := DefaultRepos()
	for _, c := range []struct{ name, got, want string }{
		{"Scenarios", r.Scenarios, DefaultScenarios()},
		{"Catalog", r.Catalog, DefaultCatalog()},
		{"Registry", r.Registry, DefaultRegistry()},
	} {
		if c.got != c.want {
			t.Errorf("DefaultRepos().%s is %q, but the resolver says %q",
				c.name, c.got, c.want)
		}
	}
}

// The vendored defaults come from the repository the caller names (SetVendoredRoot), not from
// wherever this adapter's source lives: built from another module, its source sits in the Go
// module cache, which vendors no catalog. With no variable set, DefaultCatalog and
// DefaultRegistry are testdata/catalog and testdata/registry under that root; with none set,
// they are under this file's own repository.
func TestTheVendoredDefaultsComeFromTheCallersRoot(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	own := DefaultCatalog()
	root := t.TempDir()
	SetVendoredRoot(root)
	defer SetVendoredRoot("")
	if got, want := DefaultCatalog(), filepath.Join(root, "testdata", "catalog"); got != want {
		t.Errorf("DefaultCatalog() = %q under SetVendoredRoot(%q), want %q", got, root, want)
	}
	if got, want := DefaultRegistry(), filepath.Join(root, "testdata", "registry"); got != want {
		t.Errorf("DefaultRegistry() = %q under SetVendoredRoot(%q), want %q", got, root, want)
	}
	SetVendoredRoot("")
	if got := DefaultCatalog(); got != own {
		t.Errorf("DefaultCatalog() = %q after SetVendoredRoot(\"\"), want this repository's %q", got, own)
	}
}

// The scenario fixtures default to this repository's kerneltests/testdata/aisimulate, and
// follow SetScenarioRoot.
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
}
