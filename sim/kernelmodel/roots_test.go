package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The artifact roots must resolve to something that EXISTS by default, with no environment
// set. This is the property that turns a missing artifact from a silent skip into a
// failure: twenty-three tests in this package skip when a read fails, and a skip reads as
// a pass, so "the default root is wrong" was previously indistinguishable from "the suite
// is fine".
func TestDefaultRootsResolveToVendoredArtifactsThatExist(t *testing.T) {
	for _, c := range []struct {
		name  string
		got   string
		probe string // a file that must be present, proving it is the right tree
	}{
		{"catalog", DefaultCatalog(), filepath.Join("models", "glm-5", "graph.yaml")},
		{"registry", DefaultRegistry(), "coefficients"},
		{"scenarios", DefaultScenarios(), "glm-5-h200-fp8-sglang-tp8.yaml"},
		{"measurements", DefaultMeasurements(), "aisimulate_e2e.json"},
	} {
		if _, err := os.Stat(c.got); err != nil {
			t.Errorf("%s root %q does not exist: %v", c.name, c.got, err)
			continue
		}
		if _, err := os.Stat(filepath.Join(c.got, c.probe)); err != nil {
			t.Errorf("%s root %q exists but does not contain %s, so it is not the tree "+
				"this package expects: %v", c.name, c.got, c.probe, err)
		}
	}
}

// No default root may name a path outside this repository. The whole point of vendoring is
// that the suite is hermetic; a default pointing at a sibling checkout or a home directory
// is what made it unrunnable for anyone else, and made cross-repository schema drift show
// up as this repository's tests failing.
//
// Checked as a property of the resolved strings rather than by grepping the source, so it
// also catches a root that reaches outside via "..".
func TestNoDefaultRootEscapesTheRepository(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("locating the repository root: %v", err)
	}
	for name, got := range map[string]string{
		"catalog":      DefaultCatalog(),
		"registry":     DefaultRegistry(),
		"scenarios":    DefaultScenarios(),
		"measurements": DefaultMeasurements(),
	} {
		abs, err := filepath.Abs(got)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if rel, err := filepath.Rel(repo, abs); err != nil || strings.HasPrefix(rel, "..") {
			t.Errorf("%s root resolves to %q, outside the repository at %q; defaults must "+
				"be vendored so the suite does not depend on another checkout",
				name, abs, repo)
		}
		// The specific regression: an absolute path under a developer's home directory.
		if strings.Contains(abs, "/Users/") || strings.Contains(abs, "/home/") {
			t.Errorf("%s root %q names a home directory", name, abs)
		}
	}
}

// Every root must stay independently overridable, so a working copy can be scored against
// live upstream artifacts. If an override were ignored, cmd/metricscore could not do the
// one thing it exists for -- compare against whatever upstream currently says -- and the
// vendored copy would become a cage rather than a default.
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

// Overriding one root must not move the others. They are separate because the trees come
// from separate upstream repositories on separate commits; a shared prefix would make it
// impossible to score a live registry against the pinned catalog.
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

// The resolvers must not depend on the working directory. Tests in several packages and
// commands under cmd/ each run with a different one, so a relative default would resolve
// differently per caller -- which is exactly the bug a naive `filepath.Join("testdata",
// ...)` would introduce, and it would show up only as skips in some packages.
func TestRootsDoNotDependOnTheWorkingDirectory(t *testing.T) {
	from := DefaultCatalog()
	dir := t.TempDir()
	t.Chdir(dir)
	got := DefaultCatalog()
	if got != from {
		t.Errorf("after chdir to %q the catalog root moved from %q to %q", dir, from, got)
	}
	if _, err := os.Stat(got); err != nil {
		t.Errorf("catalog root unreadable from a different working directory: %v", err)
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
