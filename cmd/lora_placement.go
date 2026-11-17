package cmd

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// loraPlacementOf is the policy bundle's lora_placement block as the cluster's config. It refuses,
// before the run, a block that does not resolve and an assignment that names an adapter the run
// does not declare; the cluster checks the instances and their capacity when it starts.
func loraPlacementOf(bundle *sim.PolicyBundle, registry workload.AdapterRegistry) cluster.LoRAPlacementConfig {
	if bundle == nil { // no policy bundle: the baseline places nothing
		return cluster.LoRAPlacementConfig{}
	}
	b := bundle.LoRAPlacement
	cfg := cluster.LoRAPlacementConfig{Bundle: b.Bundle, Creation: b.Creation, Assignment: b.Assignment,
		PeriodicIntervalUs: b.PeriodicIntervalUs}
	if _, err := cfg.Resolve(); err != nil {
		logrus.Fatalf("policy bundle: %v", err)
	}
	for idx, names := range cfg.Assignment {
		for _, n := range names {
			if registry == nil || !registry.Has(n) {
				logrus.Fatalf("policy bundle lora_placement.assignment puts adapter %q on instance %d, but the run "+
					"declares no such adapter (--lora-config adapters)", n, idx)
			}
		}
	}
	return cfg
}
