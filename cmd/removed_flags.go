package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

// The scenario is the only configuration input (inference-sim #1602, #2037; design X1, SUB-19).
// A flag that restated a scenario field is gone from run and replay, and so are eager workload
// generation and --lazy-generation (#1451): v0.16.0 deprecated them and flagsRemovedIn removed
// them (LIF-7). The command line keeps paths, and the options the Result records (SIM-10).
const flagsRemovedIn = "v0.18.0"

// removedFlags maps each flag removed from run and replay to what states its setting now. A
// command line that still passes one is refused with that text, not with pflag's bare
// "unknown flag".
var removedFlags = map[string]string{
	"scheduler":            "state engine.scheduling_policy (fcfs or priority, vLLM's) in each pool's engine block of the scenario",
	"preemption-policy":    "state engine.scheduling_policy (fcfs or priority, vLLM's) in each pool's engine block of the scenario; it sets the preemption victim too",
	"dp-load-balancing":    "state parallel.data_parallel_load_balancing in each pool of the scenario",
	"swa-bounded-replay":   "state engine.swa_bounded_replay in each pool's engine block of the scenario",
	"kv-offload-config":    "state the offload block (tiers, connector, spec, eviction_policy) in the scenario's deployment",
	"kv-cpu-blocks":        "state a cpu_dram tier in the offload block of the scenario's deployment",
	"kv-offload-threshold": "omit it: the engine never read it, and the offload block of the scenario's deployment states the tiers",
	"lazy-generation":      "omit it: requests are always streamed from the workload generator",
}

// removedFlagError is the refusal of a removed flag, naming what replaces it.
func removedFlagError(name string) error {
	return fmt.Errorf("--%s was removed in %s: %s", name, flagsRemovedIn, removedFlags[name])
}

// refuseRemovedFlags makes cmd refuse a removed flag with removedFlagError. Any other flag
// error is cmd's own.
func refuseRemovedFlags(cmd *cobra.Command) {
	cmd.SetFlagErrorFunc(func(_ *cobra.Command, err error) error {
		if name, ok := strings.CutPrefix(err.Error(), "unknown flag: --"); ok {
			if _, removed := removedFlags[name]; removed {
				return removedFlagError(name)
			}
		}
		return err
	})
}
