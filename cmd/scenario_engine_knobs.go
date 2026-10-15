package cmd

import (
	"fmt"
	"strconv"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// adoptScenarioEngineKnobs makes the scenario the source of engine.long_prefill_token_threshold
// (blis-schemas#75) and parallel.data_parallel_load_balancing (#76), as #1910 made it the
// source of the offload tiers: a value the scenario states is the run's, and
// --long-prefill-token-threshold / --dp-load-balancing may only restate it. A scenario that
// states neither leaves the flags in charge. BLIS applies one value of each to the whole run,
// so pools stating different values are refused.
func adoptScenarioEngineKnobs(cmd *cobra.Command, shape kernelmodel.Shape) {
	threshold, err := onePoolValue(shape.LongPrefillTokenThresholds, 0, "engine.long_prefill_token_threshold")
	if err == nil && threshold != 0 {
		err = restateOnly(cmd, "long-prefill-token-threshold", strconv.FormatInt(longPrefillTokenThreshold, 10),
			strconv.Itoa(threshold), "engine.long_prefill_token_threshold")
		longPrefillTokenThreshold = int64(threshold)
	}
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	mode, err := onePoolValue(shape.DPLoadBalancings, "", "parallel.data_parallel_load_balancing")
	if err == nil && mode != "" {
		err = restateOnly(cmd, "dp-load-balancing", dpLoadBalancing, string(mode.Mode()),
			"parallel.data_parallel_load_balancing")
		dpLoadBalancing = string(mode.Mode())
	}
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
}

// onePoolValue is the one value the stating pools agree on, or unstated when none states it.
func onePoolValue[T comparable](perPool []T, unstated T, field string) (T, error) {
	v := unstated
	for i, p := range perPool {
		switch {
		case p == unstated:
		case v == unstated:
			v = p
		case p != v:
			return unstated, fmt.Errorf("pools state different %s (%v and %v, pool %d); BLIS applies one "+
				"value to the whole run", field, v, p, i)
		}
	}
	return v, nil
}

// restateOnly refuses a flag that was set to a value other than the scenario's.
func restateOnly(cmd *cobra.Command, flag, flagValue, stated, field string) error {
	if !cmd.Flags().Changed(flag) || flagValue == stated {
		return nil
	}
	return fmt.Errorf("--%s %s disagrees with the scenario's %s %s; the scenario is the deployment, so the "+
		"flag may only restate it -- drop the flag, or change the scenario", flag, flagValue, field, stated)
}
