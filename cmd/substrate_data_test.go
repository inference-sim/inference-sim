package cmd

import (
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/internal/kernelfixtures"
	"github.com/inference-sim/inference-sim/internal/substratedata"
)

// defaultKernelRepos is the scenario fixtures, catalog and registry the kernel tests open:
// the cost model's fixtures and the catalog/ and registry/ of the blis-substrate module this
// build selects (internal/substratedata). A run opens its kernel at the build site, from the
// catalog, registry and scenario it is given.
func defaultKernelRepos() kernelmodel.Repos {
	return kernelmodel.Repos{
		Scenarios: kernelfixtures.AISimulate(),
		Catalog:   substratedata.MustCatalog(),
		Registry:  substratedata.MustRegistry(),
	}
}
