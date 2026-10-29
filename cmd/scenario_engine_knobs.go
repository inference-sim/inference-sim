package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

// dpLoadBalancingFromFlag records that --dp-load-balancing was given, so the flag's value
// applies to every pool (poolEngineKnobs); otherwise each pool runs what its scenario pool
// states. engine.long_prefill_token_threshold has no flag: it is the scenario's (#2021 B6).
var dpLoadBalancingFromFlag bool

// adoptScenarioEngineKnobs makes the scenario the source of each pool's
// engine.long_prefill_token_threshold and parallel.data_parallel_load_balancing; the
// --dp-load-balancing flag may only restate the latter (build.RunEngineKnobs).
func adoptScenarioEngineKnobs(cmd *cobra.Command, shape scenarioShapeT) {
	dpLoadBalancingFromFlag = cmd.Flags().Changed("dp-load-balancing")
	threshold, mode, err := runEngineKnobs(shape, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	longPrefillTokenThreshold, dpLoadBalancing = threshold, mode
}
