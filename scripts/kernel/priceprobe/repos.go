package main

import (
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// reposFor opens catalog and registry through the substrate's loader, as a run does. An empty
// directory means the embedded copy.
func reposFor(scenarios, catalog, registry string) (kernelmodel.Repos, error) {
	d, err := kernelmodel.OpenData(load.Options{Catalog: catalog, Registry: registry})
	if err != nil {
		return kernelmodel.Repos{}, err
	}
	return kernelmodel.Repos{Scenarios: scenarios, Data: d}, nil
}
