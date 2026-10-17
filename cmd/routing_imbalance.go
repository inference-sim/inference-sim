package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-schemas/spec/routing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
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

// applyRoutingOverrides hands the command line's routing layer to the pools about to open.
// The workload layer is the scenario's own binding: the pools open before a catalog preset
// or an agentic descriptor is read, so their category reaches routing through the
// scenario's pin (routing.profiles plus fallback_category), not here. CLI boundary: an
// invalid value is fatal.
func applyRoutingOverrides(cmd *cobra.Command) {
	o := kernelmodel.RoutingOverrides{}
	tokSet := cmd.Flags().Changed("routing-token-imbalance")
	if cmd.Flags().Changed("routing-expert-imbalance") && !tokSet {
		logrus.Fatalf("--routing-expert-imbalance needs --routing-token-imbalance")
	}
	if tokSet {
		v := &routing.Value{TokenImbalance: routingTokenImbalance, Source: "--routing-token-imbalance"}
		if cmd.Flags().Changed("routing-expert-imbalance") {
			e := routingExpertImbalance
			v.ExpertImbalance = &e
		}
		if p := v.Validate(); !p.OK() {
			logrus.Fatalf("--routing-token-imbalance: %s", p.Error())
		}
		o.CLI = v
	}
	kernelmodel.SetRoutingOverrides(o)
}

func currentRoutingOverrides() kernelmodel.RoutingOverrides {
	return kernelmodel.CurrentRoutingOverrides()
}
