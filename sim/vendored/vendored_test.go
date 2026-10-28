package vendored

import (
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// This repository's vendored catalog and registry must be the releases blis-latency-kernel's
// lock pins (testdata/upstream.lock in the kernel module go.mod pins). A copy of another release
// builds kernels too, and a renamed chip or a refitted coefficient would then show up as a score
// that moved for no stated reason. Bumping the kernel in go.mod moves the lock, and this fails
// until testdata/catalog and testdata/registry are re-vendored (docs/contributing/cross-repo-changes.md).
//
// The test moved to blis-vllm with the kernel adapter (compat/kernelmodel), where it checks
// blis-vllm's own copies. This one checks this repository's, which this package's test init
// (root_test.go) names as the adapter's vendoring root.
//
// An entry in testdata/upstream-prerelease.lock stands in for the kernel's while a pinned kernel
// pre-release lags the unreleased upstreams it was developed against; it must name the same
// repository and a different commit, so it cannot outlive the kernel's catching up.
func TestThisRepositorysVendoredCatalogAndRegistryAreTheLockedReleases(t *testing.T) {
	// The defaults, not an override: BLIS_CATALOG and BLIS_REGISTRY point a run elsewhere.
	t.Setenv("BLIS_CATALOG", "")
	t.Setenv("BLIS_REGISTRY", "")
	locked, err := kernelmodel.LockedUpstreams()
	if err != nil {
		t.Fatal(err)
	}
	pre, err := kernelmodel.PrereleaseUpstreams()
	if err != nil {
		t.Fatal(err)
	}
	for dest, p := range pre {
		k, ok := locked[dest]
		switch {
		case !ok:
			t.Errorf("testdata/upstream-prerelease.lock names %s, which blis-latency-kernel's lock does not", dest)
		case p.Repository != k.Repository:
			t.Errorf("testdata/upstream-prerelease.lock: %s is %s, the kernel locks %s", dest, p.Repository, k.Repository)
		case p.Commit == k.Commit:
			t.Errorf("TEMPORARY testdata/upstream-prerelease.lock restates the kernel's %s lock (%s); delete the line", dest, k.Commit)
		}
		locked[dest] = p
	}
	for _, c := range []struct{ dest, root string }{
		{"catalog", kernelmodel.DefaultCatalog()},
		{"registry", kernelmodel.DefaultRegistry()},
	} {
		if want := Root() + "/testdata/" + c.dest; c.root != want {
			t.Fatalf("the adapter's default %s is %s, not this repository's %s", c.dest, c.root, want)
		}
		u, ok := locked[c.dest]
		if !ok {
			t.Fatalf("blis-latency-kernel's upstream.lock pins no %s", c.dest)
		}
		got, err := kernelmodel.VendoredCommit(c.root)
		if err != nil {
			t.Fatalf("%s: %v", c.dest, err)
		}
		if got != u.Commit {
			t.Errorf("vendored %s is %s; the lock pins %s (%s) -- re-vendor testdata/%s from that release "+
				"(docs/contributing/cross-repo-changes.md says how)", c.dest, got, u.Tag, u.Commit, c.dest)
		}
	}
}
