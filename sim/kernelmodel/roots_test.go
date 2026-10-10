package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The scenario fixtures default to blis-latency-kernel's own testdata/aisimulate, in the
// module go.mod pins -- not a copy here and not a sibling checkout. That is what makes the
// fixtures and the kernel pricing them one revision.
func TestDefaultScenariosAreThePinnedKernelModulesFixtures(t *testing.T) {
	t.Setenv(scenariosEnv, "")
	got := DefaultScenarios()
	if got == "" {
		_, err := kernelModuleDir()
		t.Fatalf("the scenario root did not resolve: %v", err)
	}
	if !strings.Contains(got, kernelModule+"@") {
		t.Errorf("scenario root %q is not inside the module cache's %s@<version>", got,
			kernelModule)
	}
	if _, err := os.Stat(filepath.Join(got, "glm-5-h200-fp8-sglang-tp8.yaml")); err != nil {
		t.Errorf("scenario root %q lacks a fixture this package reads: %v", got, err)
	}
}

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
	if got := DefaultMeasurements(); os.Getenv(measurementsEnv) == "" && got != "" {
		t.Errorf("with %s unset the measurements root resolved to %q", measurementsEnv, got)
	}
}

// An unset root fails naming itself and the command that provides it -- never a skip, and
// never a bare "no such file" that leaves the reader to work out which repository is
// meant. The clone command quotes the release blis-latency-kernel's upstream.lock pins.
func TestRequireReposNamesEachMissingRootAndHowToSetIt(t *testing.T) {
	locked, err := LockedUpstreams()
	if err != nil {
		t.Fatal(err)
	}
	err = RequireRepos(Repos{Scenarios: t.TempDir()})
	if err == nil {
		t.Fatal("unset catalog and registry roots were accepted")
	}
	for _, want := range []string{
		"catalog root is not set", "--branch " + locked["catalog"].Tag, catalogEnv,
		"registry root is not set", "--branch " + locked["registry"].Tag, registryEnv,
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q:\n%v", want, err)
		}
	}
	if strings.Contains(err.Error(), "scenarios") {
		t.Errorf("an existing scenario root was reported missing:\n%v", err)
	}
	if err := RequireMeasurements(""); err == nil ||
		!strings.Contains(err.Error(), measurementsEnv) {
		t.Errorf("an unset measurements root was not reported naming %s: %v",
			measurementsEnv, err)
	}
}

// The vendored catalog and registry must be the releases blis-latency-kernel's lock pins.
// A copy of another release builds kernels too, and a renamed chip or a refitted
// coefficient would then show up as a score that moved for no stated reason. Bumping the
// kernel in go.mod moves the lock, and this fails until the copies are re-vendored.
func TestTheVendoredCatalogAndRegistryAreTheLockedReleases(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	locked, err := LockedUpstreams()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ dest, root string }{
		{"catalog", DefaultCatalog()},
		{"registry", DefaultRegistry()},
	} {
		u, ok := locked[c.dest]
		if !ok {
			t.Fatalf("blis-latency-kernel's upstream.lock pins no %s", c.dest)
		}
		got, err := VendoredCommit(c.root)
		if err != nil {
			t.Fatalf("%s: %v", c.dest, err)
		}
		if got != u.Commit {
			t.Errorf("vendored %s is %s; blis-latency-kernel's lock pins %s (%s) -- re-vendor "+
				"it from that release (testdata/README.md says how)", c.dest, got, u.Tag, u.Commit)
		}
	}
}

// Every root must stay independently overridable, so a working copy can be scored against
// live upstream artifacts -- which is what cmd/metricscore is for.
func TestEachRootIsIndependentlyOverridable(t *testing.T) {
	for _, c := range []struct {
		env string
		get func() string
	}{
		{catalogEnv, DefaultCatalog},
		{registryEnv, DefaultRegistry},
		{scenariosEnv, DefaultScenarios},
		{measurementsEnv, DefaultMeasurements},
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
	before := []string{DefaultRegistry(), DefaultScenarios(), DefaultMeasurements()}
	t.Setenv(catalogEnv, filepath.Join(t.TempDir(), "only-the-catalog"))
	after := []string{DefaultRegistry(), DefaultScenarios(), DefaultMeasurements()}
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
