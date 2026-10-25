package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// longPrefillFromFlag and dpLoadBalancingFromFlag record that --long-prefill-token-threshold
// and --dp-load-balancing were given, so the flag's value applies to every pool
// (poolEngineKnobs); otherwise each pool runs what its scenario pool states.
var longPrefillFromFlag, dpLoadBalancingFromFlag bool

// adoptScenarioEngineKnobs makes the scenario the source of each pool's
// engine.long_prefill_token_threshold and parallel.data_parallel_load_balancing; the flags
// may only restate them (build.RunEngineKnobs).
func adoptScenarioEngineKnobs(cmd *cobra.Command, shape scenarioShapeT) {
	longPrefillFromFlag = cmd.Flags().Changed("long-prefill-token-threshold")
	dpLoadBalancingFromFlag = cmd.Flags().Changed("dp-load-balancing")
	threshold, mode, err := runEngineKnobs(shape, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	longPrefillTokenThreshold, dpLoadBalancing = threshold, mode
}

// poolEngineKnobs sets a P/D pool's own threshold and balancing mode (build.PoolEngineKnobs).
func poolEngineKnobs(m *engineModel, o *cluster.PoolOverrides) {
	threshold, mode, err := poolEngineKnobsOf(m, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	o.LongPrefillTokenThreshold, o.DPLoadBalancing = &threshold, mode
}
