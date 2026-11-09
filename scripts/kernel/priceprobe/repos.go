package main

import (
	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// reposFor opens catalog and registry through blis-substrate's loader, as a run does. An empty
// directory means the embedded copy.
func reposFor(scenarios, catalog, registry string) (kernelmodel.Repos, error) {
	d, err := kernelmodel.OpenData(load.Options{Catalog: catalog, Registry: registry})
	if err != nil {
		return kernelmodel.Repos{}, err
	}
	return kernelmodel.Repos{Scenarios: scenarios, Data: d}, nil
}
