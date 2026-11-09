package cmd

import (
	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"

	"github.com/inference-sim/inference-sim/internal/substratedata"
)

// The tests' data: the catalog/ and registry/ directories of the blis-substrate module this
// build selects (internal/substratedata), which hold the embedded copy's files. A CLI test
// passes the directories as --catalog and --registry; a library test opens a kernel from the
// Data the loader opens on them, the one a run opens on the same paths.

// testCatalogDir and testRegistryDir are the directories.
func testCatalogDir() string  { return substratedata.MustCatalog() }
func testRegistryDir() string { return substratedata.MustRegistry() }

// testData is the loader's Data on them.
func testData() *load.Data {
	d, err := kernelmodel.OpenData(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		panic(err)
	}
	return d
}

// reposOf is the kernel roots for a scenario directory and a catalog and registry directory, the
// loader's Data on the two. A directory that does not load is a fatal test error in the caller's
// next step, so it panics with the loader's message.
func reposOf(scenarios, catalog, registry string) kernelmodel.Repos {
	d, err := kernelmodel.OpenData(load.Options{Catalog: catalog, Registry: registry})
	if err != nil {
		panic(err)
	}
	return kernelmodel.Repos{Scenarios: scenarios, Data: d}
}

// dataWithCatalogOverlay is the embedded data with dir as a catalog overlay: the way a test
// adds or replaces a few catalog files (a preset, the storage-device table) while every other
// entry stays the release's. The loader checks the combined catalog in full, so an overlay the
// release would refuse is refused here; the error is returned for the test to assert on.
func dataWithCatalogOverlay(dir string) (*load.Data, error) {
	return load.Open(load.Options{CatalogOverlays: []string{dir}})
}

// dataFromCatalog opens the embedded registry with the catalog directory dir, which replaces the
// embedded catalog whole.
func dataFromCatalog(dir string) (*load.Data, error) {
	return load.Open(load.Options{Catalog: dir})
}
