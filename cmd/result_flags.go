package cmd

// Every command-line flag that can change what a run simulates is recorded in its Result
// (#1871 review, B1): the flags with a dedicated home in blis-schemas v0.9.1's input (load,
// num_requests, seed, horizon_s, workload, policy, goodput targets, the scenario and the
// deployment) are written there; every other flag the user set to a value other than its
// default becomes an `overrides` row, so two documents with the same input and overrides
// describe the same simulation (INV-6). A flag that only selects what is reported (an output
// path, a view, a log level) changes nothing simulated and is not recorded.
//
// TestResultFlags_EveryFlagIsRecordedOrExempt walks both commands' flag sets, so a flag added
// later is recorded unless it is listed here as exempt, with the reason.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	sim "github.com/inference-sim/inference-sim/sim"
)

// resultFlagsExempt are the flags not recorded as overrides, by command, each with where its
// effect is recorded or why it has none.
var resultFlagsExempt = map[string]map[string]string{
	"run": {
		"help":                     "no effect",
		"format":                   "output encoding only",
		"output":                   "output path only",
		"view":                     "stderr rendering only",
		"log":                      "stderr verbosity only",
		"metrics-path":             "output path only (the file is an artifact)",
		"trace-output":             "output path only (the files are artifacts)",
		"detectors":                "selects the saturation section; simulates nothing",
		"saturation-config":        "tunes the saturation section; simulates nothing",
		"saturation-report":        "output path only",
		"saturation-final-window":  "tunes the saturation section; simulates nothing",
		"server-series-interval-s": "bins the engine server series; simulates nothing",
		"trace-level":              "records routing decisions; simulates nothing",
		"counterfactual-k":         "records routing decisions; simulates nothing",
		"summarize-trace":          "stderr rendering only",
		"fitness-weights":          "stderr rendering only",
		"lazy-generation":          "generates the same requests (INV-6 eager == lazy)",
		"scenario":                 "input.scenario and input.deployment",
		"scenarios":                "input.scenario and input.deployment",
		"catalog":                  "revisions.catalog",
		"registry":                 "revisions.registry",
		"policy-config":            "input.policy",
		"rate":                     "input.run.load",
		"concurrency":              "input.run.load",
		"num-requests":             "input.run.num_requests",
		"seed":                     "input.run.seed",
		"horizon":                  "input.run.horizon_s",
		"workload":                 "input.run.workload (a catalog shape)",
		"slo-ttft":                 "goodput.targets",
		"slo-itl":                  "goodput.targets",
		"slo-e2e":                  "goodput.targets",
		"scheduler":                "overrides, against the scenario's stated policy (recordPolicyOverride)",
		"preemption-policy":        "overrides, against the scenario's stated policy (recordPolicyOverride)",
	},
	"replay": {
		"help":                     "no effect",
		"format":                   "output encoding only",
		"output":                   "output path only",
		"view":                     "stderr rendering only",
		"log":                      "stderr verbosity only",
		"metrics-path":             "output path only (the file is an artifact)",
		"results-path":             "output path only (the file is an artifact)",
		"trace-output":             "output path only",
		"detectors":                "selects the saturation section; simulates nothing",
		"saturation-config":        "tunes the saturation section; simulates nothing",
		"saturation-report":        "output path only",
		"saturation-final-window":  "tunes the saturation section; simulates nothing",
		"server-series-interval-s": "bins the engine server series; simulates nothing",
		"trace-level":              "records routing decisions; simulates nothing",
		"counterfactual-k":         "records routing decisions; simulates nothing",
		"summarize-trace":          "stderr rendering only",
		"fitness-weights":          "has no effect in replay",
		"lazy-generation":          "accepted for symmetry with run; has no effect",
		"corpus-cache":             "a parse cache; output is identical with or without it",
		"scenario":                 "input.scenario and input.deployment",
		"scenarios":                "input.scenario and input.deployment",
		"catalog":                  "revisions.catalog",
		"registry":                 "revisions.registry",
		"policy-config":            "input.policy",
		"seed":                     "input.run.seed",
		"horizon":                  "input.run.horizon_s",
		"trace-header":             "input.run.workload.trace",
		"trace-data":               "input.run.workload.trace (by sha256)",
		"slo-ttft":                 "goodput.targets",
		"slo-itl":                  "goodput.targets",
		"slo-e2e":                  "goodput.targets",
		"agentic-scenario":         "input.run.workload.replay",
		"workload-spec":            "input.run.workload.replay",
		"corpus":                   "input.run.workload.replay.corpus (by sha256)",
		"corpus-sha256":            "input.run.workload.replay.corpus",
		"concurrency":              "input.run.workload.replay.concurrency and input.run.load",
		"benchmark-duration":       "input.run.workload.replay",
		"grace-period":             "input.run.workload.replay",
		"max-context-length":       "input.run.workload.replay",
		"scheduler":                "overrides, against the scenario's stated policy (recordPolicyOverride)",
		"preemption-policy":        "overrides, against the scenario's stated policy (recordPolicyOverride)",
	},
}

// resultFlagFiles are the flags whose value is a file the run reads: they are recorded by the
// file's sha256, never its path (a fact of the host).
var resultFlagFiles = map[string]bool{
	"workload-spec": true, "lora-config": true, "kv-offload-config": true, "defaults-filepath": true,
}

// resultFlagOverrides are the overrides the command line made, recorded when the command starts,
// before any code resolves a flag's variable to a derived value. Nil before a command runs.
var resultFlagOverrides []sim.ScenarioOverride

// snapshotResultFlags records the explicitly set flags of command ("run" or "replay") for its
// Result. A file the run will read that cannot be read is fatal here, naming the flag.
func snapshotResultFlags(command string, cmd *cobra.Command) {
	o, err := flagOverrides(command, cmd.Flags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	resultFlagOverrides = o
}

// flagOverrides is one override per flag the user set to a value other than its default, that is
// not exempt for the command, in flag-name order.
func flagOverrides(command string, fs *pflag.FlagSet) ([]sim.ScenarioOverride, error) {
	exempt, ok := resultFlagsExempt[command]
	if !ok {
		return nil, fmt.Errorf("recording the run's inputs: unknown command %q", command)
	}
	var out []sim.ScenarioOverride
	var errs []string
	fs.Visit(func(f *pflag.Flag) {
		if _, ok := exempt[f.Name]; ok {
			return
		}
		v := f.Value.String()
		if v == f.DefValue {
			return
		}
		if resultFlagFiles[f.Name] && v != "" {
			d, err := fileDigest(v)
			if err != nil {
				errs = append(errs, fmt.Sprintf("--%s %s: %v", f.Name, v, err))
				return
			}
			v = "sha256:" + d
		}
		// Input states no command-line flag, so the row's input value is empty; the flag's default
		// is the documented value it replaced.
		out = append(out, sim.ScenarioOverride{Setting: f.Name, Run: v, Source: "--" + f.Name})
	})
	if len(errs) > 0 {
		return nil, fmt.Errorf("recording the run's inputs: %s", strings.Join(errs, "; "))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Setting < out[j].Setting })
	return out, nil
}

// runOverrides are the overrides a Result records: the settings the run took from somewhere
// other than its scenario (scenarioOverrides), then every flag override not already one of them.
func runOverrides(extra ...sim.ScenarioOverride) []sim.ScenarioOverride {
	out := append([]sim.ScenarioOverride(nil), scenarioOverrides...)
	bySource := map[string]bool{}
	for _, o := range out {
		bySource[o.Source] = true
	}
	for _, o := range resultFlagOverrides {
		if !bySource[o.Source] {
			out = append(out, o)
		}
	}
	return append(out, extra...)
}
