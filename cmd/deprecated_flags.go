package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// The releases that deprecate and remove the flags below. The release plan relaxes LIF-7 until
// M4 for these features (§3.3): v0.16.0 deprecates them, they keep working, and v0.19.0, the M4
// release, removes them.
const (
	flagsDeprecatedIn = "v0.16.0"
	flagsRemovedIn    = "v0.19.0"
)

// deprecatedFlags maps each deprecated flag of run and replay to what replaces it. An engine
// setting is stated only in the scenario, in the upstream's own field (X1, inference-sim
// #2037). Streaming workload generation is the default, and the eager generator goes with
// --lazy-generation (#1450, #1451).
var deprecatedFlags = map[string]string{
	"scheduler":          "state engine.scheduling_policy (fcfs or priority, vLLM's) in the scenario's pool engine block; sjf and reverse-priority have no vLLM equivalent (#2037)",
	"preemption-policy":  "state engine.scheduling_policy in the scenario's pool engine block (#2037)",
	"dp-load-balancing":  "state parallel.data_parallel_load_balancing in the scenario's pool (#2037)",
	"swa-bounded-replay": "state engine.swa_bounded_replay in the scenario's pool engine block (#2037)",
	"kv-offload-config":  "state the offload block in the scenario (#2037)",
	"lazy-generation":    "omit it: requests are streamed by default, and the eager generator that --lazy-generation=false selects is removed with the flag (#1450, #1451)",
}

// replayDeprecatedFlags replaces a deprecatedFlags entry on replay, where --lazy-generation
// never had an effect: replay reads its requests from a trace.
var replayDeprecatedFlags = map[string]string{
	"lazy-generation": "omit it: it has no effect on replay (#1451)",
}

// markDeprecatedFlags records each deprecated flag cmd registers as deprecated (LIF-7), with an
// entry of overrides, keyed by flag name, in place of deprecatedFlags' for that flag. It sets
// pflag's Deprecated field and leaves the flag visible, so --help shows the notice beside the
// flag and cobra prints it on standard error whenever the flag is given. The flag still works,
// and standard output is unchanged.
func markDeprecatedFlags(cmd *cobra.Command, overrides map[string]string) {
	for name, instead := range deprecatedFlags {
		if o, ok := overrides[name]; ok {
			instead = o
		}
		if f := cmd.Flags().Lookup(name); f != nil {
			f.Deprecated = deprecationNotice(instead)
		}
	}
}

// deprecationNotice is the notice of a flag deprecated in flagsDeprecatedIn. pflag prints it on
// standard error after "Flag --<name> has been deprecated, ", and --help after "DEPRECATED: ".
func deprecationNotice(instead string) string {
	return fmt.Sprintf("since %s and is removed in %s. Instead, %s", flagsDeprecatedIn, flagsRemovedIn, instead)
}
