package cmd

import (
	"fmt"
	"strconv"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// longPrefillFromFlag and dpLoadBalancingFromFlag record that --long-prefill-token-threshold
// and --dp-load-balancing were given, so the flag's value applies to every pool
// (poolEngineKnobs); otherwise each pool runs what its scenario pool states.
var longPrefillFromFlag, dpLoadBalancingFromFlag bool

// adoptScenarioEngineKnobs makes the scenario the source of each pool's
// engine.long_prefill_token_threshold (blis-schemas#75) and
// parallel.data_parallel_load_balancing (#76), as #1910 made it the source of the offload
// tiers. vLLM sets both per engine, so each pool runs its own value. The flags are run-wide
// overrides: given, they apply to every pool and must agree with every value a pool states,
// or the run is refused. A colocated run's one pool sets the run-wide value.
func adoptScenarioEngineKnobs(cmd *cobra.Command, shape kernelmodel.Shape) {
	longPrefillFromFlag = cmd.Flags().Changed("long-prefill-token-threshold")
	dpLoadBalancingFromFlag = cmd.Flags().Changed("dp-load-balancing")
	for i, v := range shape.LongPrefillTokenThresholds {
		if v == 0 {
			continue
		}
		if longPrefillFromFlag && int64(v) != longPrefillTokenThreshold {
			refuseRestatement("long-prefill-token-threshold", strconv.FormatInt(longPrefillTokenThreshold, 10),
				poolField(shape, i, "engine.long_prefill_token_threshold"), strconv.Itoa(v))
		}
	}
	for i, m := range shape.DPLoadBalancings {
		if m == "" {
			continue
		}
		if dpLoadBalancingFromFlag && string(m.Mode()) != dpLoadBalancing {
			refuseRestatement("dp-load-balancing", dpLoadBalancing,
				poolField(shape, i, "parallel.data_parallel_load_balancing"), string(m.Mode()))
		}
	}
	if len(shape.Roles) != 1 {
		return
	}
	if v := shape.LongPrefillTokenThresholds[0]; v != 0 {
		longPrefillTokenThreshold = int64(v)
	}
	if m := shape.DPLoadBalancings[0]; m != "" {
		dpLoadBalancing = string(m.Mode())
	}
}

// poolEngineKnobs sets a P/D pool's own threshold and balancing mode: the flag's when given
// (adoptScenarioEngineKnobs checked it agrees), else what the pool states, else vLLM's
// defaults (no threshold, internal).
func poolEngineKnobs(m *kernelmodel.Model, o *cluster.PoolOverrides) {
	eng, err := m.Engine()
	if err != nil {
		logrus.Fatalf("scenario %q: reading the %s pool's engine: %v", kernelScenario, m.Role(), err)
	}
	threshold := int64(eng.LongPrefillTokenThreshold)
	if longPrefillFromFlag {
		threshold = longPrefillTokenThreshold
	}
	mode := string(m.DPLoadBalancing().Mode())
	if dpLoadBalancingFromFlag {
		mode = dpLoadBalancing
	}
	o.LongPrefillTokenThreshold, o.DPLoadBalancing = &threshold, mode
}

func poolField(shape kernelmodel.Shape, i int, field string) string {
	return fmt.Sprintf("%s pool's %s", shape.Roles[i], field)
}

// refuseRestatement refuses a run-wide flag that disagrees with a value the scenario states.
func refuseRestatement(flag, flagValue, field, stated string) {
	logrus.Fatalf("scenario %q: --%s %s disagrees with the scenario's %s %s; the scenario is the "+
		"deployment, so the flag may only restate it -- drop the flag, or change the scenario",
		kernelScenario, flag, flagValue, field, stated)
}
