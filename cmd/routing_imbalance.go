package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// Expert-parallel routing imbalance (blis-schemas routing.Resolve): the command line is the
// highest layer, above the scenario's pin, the workload's value and the registry's default
// sets. The resolution itself runs where each pool opens (kernelmodel.resolveRouting) and is
// recorded in the results file's kernel provenance.

var (
	routingTokenImbalance  float64 // --routing-token-imbalance; 0 = not stated
	routingExpertImbalance float64 // --routing-expert-imbalance; 0 = not stated
)

// registerRoutingFlags registers the routing flags on run and replay alike (INV-13). The
// names are literals, as the physics-literal guard reads registrations (#1819).
func registerRoutingFlags(cmd *cobra.Command) {
	cmd.Flags().Float64Var(&routingTokenImbalance, "routing-token-imbalance", 0, "Expert parallelism: the busiest rank's routed tokens over the mean rank's, at least 1, flat in the step size. Overrides every other routing layer (registry profile, workload, scenario). Unset, those decide, else ranks are priced as evenly loaded.")
	cmd.Flags().Float64Var(&routingExpertImbalance, "routing-expert-imbalance", 0, "Expert parallelism: the busiest rank's distinct experts touched over the uniform-routing count, positive. Only with --routing-token-imbalance; unset, 1.")
}

// applyRoutingOverrides hands the command line's routing layer to the pools about to open
// (build.RoutingOverridesOf). CLI boundary: an invalid value is fatal.
func applyRoutingOverrides(cmd *cobra.Command) {
	o, err := routingOverridesOf(routingTokenImbalance, cmd.Flags().Changed("routing-token-imbalance"),
		routingExpertImbalance, cmd.Flags().Changed("routing-expert-imbalance"))
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	kernelmodel.SetRoutingOverrides(o)
}

func currentRoutingOverrides() kernelmodel.RoutingOverrides {
	return kernelmodel.CurrentRoutingOverrides()
}
