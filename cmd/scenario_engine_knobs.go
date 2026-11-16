package cmd

import (
	"github.com/sirupsen/logrus"
)

// adoptScenarioEngineKnobs makes the scenario the source of each pool's
// engine.long_prefill_token_threshold and parallel.data_parallel_load_balancing
// (build.RunEngineKnobs). No flag restates either (#2021 B6, #2037).
func adoptScenarioEngineKnobs(shape scenarioShapeT) {
	threshold, mode, err := runEngineKnobs(shape, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	longPrefillTokenThreshold, dpLoadBalancing = threshold, mode
}
