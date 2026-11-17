package cmd

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// clusterAutoscalerOf is the policy bundle's cluster_autoscaler block as the cluster's config,
// refused before the run when it cannot run (a policy that does not exist, a negative number).
func clusterAutoscalerOf(b sim.ClusterAutoscalerBundleConfig) cluster.ClusterAutoscalerConfig {
	cfg := cluster.ClusterAutoscalerConfig{Policy: b.Policy,
		PendingDrivenConfig: cluster.PendingDrivenConfig{PendingThresholdUs: b.PendingThresholdUs, MaxNodesPerPool: b.MaxNodesPerPool},
		ScaleDown:           b.ScaleDown,
		BinPackConfig:       cluster.BinPackConfig{LowWaterMark: b.LowWaterMark, MinNodesPerPool: b.MinNodesPerPool}}
	if err := cfg.Validate(); err != nil {
		logrus.Fatalf("policy bundle: %v", err)
	}
	return cfg
}
