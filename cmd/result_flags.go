package cmd

// Every command-line flag that can change a byte of the Result is recorded in it (SIM-10;
// #1871 review B1, #2062): the flags with a dedicated home in blis-schemas v0.10.0's input
// (load, num_requests, seed, horizon_s, workload, policy, goodput targets, the scenario and
// the deployment) are written there; every other flag the user set to a value other than its
// default becomes an `overrides` row, so two documents with the same input and overrides are
// the same document outside runtime (INV-6). That includes the flags that change only a
// reported section and simulate nothing (--detectors and the saturation section,
// --server-series-interval-s and the engine server series, --fitness-weights and
// --summarize-trace and the router section). A flag that changes no byte of the Result (an
// output path, a view, a log level) is not recorded.
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

	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	sim "github.com/inference-sim/inference-sim/sim"
)

// resultFlagsExempt are the flags not recorded as overrides, by command, each with where its
// effect is recorded or why it has none.
var resultFlagsExempt = map[string]map[string]string{
	"run": {
		"help":              "no effect",
		"format":            "output encoding only",
		"output":            "output path only",
		"view":              "stderr rendering only",
		"log":               "stderr verbosity only",
		"from-result":       "the reproduction source; recorded as rerun_of",
		"trace-output":      "output path only (the files are artifacts)",
		"saturation-report": "output path only",
		"counterfactual-k":  "adds candidate rows to the in-memory decision trace; no Result byte reads them",
		"lazy-generation":   "generates the same requests (INV-6 eager == lazy)",
		"scenario":          "input.scenario and input.deployment",
		"scenarios":         "input.scenario and input.deployment",
		"catalog":           "revisions.catalog",
		"registry":          "revisions.registry",
		"policy-config":     "input.policy",
		"rate":              "input.run.load",
		"concurrency":       "input.run.load",
		"num-requests":      "input.run.num_requests",
		"seed":              "input.run.seed",
		"horizon":           "input.run.horizon_s",
		"workload":          "input.run.workload (a catalog shape)",
		"slo-ttft":          "goodput.targets",
		"slo-itl":           "goodput.targets",
		"slo-e2e":           "goodput.targets",
		"scheduler":         "overrides, against the scenario's stated policy (recordPolicyOverride)",
		"preemption-policy": "overrides, against the scenario's stated policy (recordPolicyOverride)",
	},
	"replay": {
		"help":               "no effect",
		"format":             "output encoding only",
		"output":             "output path only",
		"view":               "stderr rendering only",
		"log":                "stderr verbosity only",
		"from-result":        "the reproduction source; recorded as rerun_of",
		"trace-output":       "output path only",
		"saturation-report":  "output path only",
		"counterfactual-k":   "adds candidate rows to the in-memory decision trace; no Result byte reads them",
		"fitness-weights":    "has no effect in replay",
		"lazy-generation":    "accepted for symmetry with run; has no effect",
		"corpus-cache":       "a parse cache; output is identical with or without it",
		"scenario":           "input.scenario and input.deployment",
		"scenarios":          "input.scenario and input.deployment",
		"catalog":            "revisions.catalog",
		"registry":           "revisions.registry",
		"policy-config":      "input.policy",
		"seed":               "input.run.seed",
		"horizon":            "input.run.horizon_s",
		"trace-header":       "input.run.workload.trace",
		"trace-data":         "input.run.workload.trace (by sha256)",
		"slo-ttft":           "goodput.targets",
		"slo-itl":            "goodput.targets",
		"slo-e2e":            "goodput.targets",
		"agentic-scenario":   "input.run.workload.replay",
		"workload-spec":      "input.run.workload.replay",
		"corpus":             "input.run.workload.replay.corpus (by sha256)",
		"corpus-sha256":      "input.run.workload.replay.corpus",
		"concurrency":        "input.run.workload.replay.concurrency and input.run.load",
		"benchmark-duration": "input.run.workload.replay",
		"grace-period":       "input.run.workload.replay",
		"max-context-length": "input.run.workload.replay",
		"scheduler":          "overrides, against the scenario's stated policy (recordPolicyOverride)",
		"preemption-policy":  "overrides, against the scenario's stated policy (recordPolicyOverride)",
	},
}

// resultFlagFiles are the flags whose value is a file the run reads: they are recorded by the
// file's sha256, never its path (a fact of the host).
var resultFlagFiles = map[string]bool{
	"workload-spec": true, "lora-config": true, "kv-offload-config": true, "defaults-filepath": true,
	"saturation-config": true,
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
	opts, err := inputOptions(command, cmd.Flags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	resultInputOptions = opts
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

// resultInputOptions are the options the command line gave, recorded in input.options.
var resultInputOptions []result.Option

// optionsExcluded are the flags input.options does not record, each with why: they only
// choose where output goes without adding a reference to the Result, or what the Result records elsewhere in full (the scenario and the
// deployment in input, the policy bundle in input.policy, the catalog and registry in data).
// --trace-output is recorded: it adds an artifacts reference.
// --from-result needs the flags the Result records here, the embedded scenario, deployment
// and policy, and the data and file flags the caller supplies again.
var optionsExcluded = map[string]string{
	"help": "no effect", "format": "output encoding only", "output": "output path only",
	"view": "stderr rendering only", "log": "stderr verbosity only",
	"saturation-report": "output path only; the report is not referenced by the Result",
	"counterfactual-k":  "adds in-memory decision-trace rows; no Result byte reads them",
	"lazy-generation":   "generates the same requests (INV-6 eager == lazy)",
	"corpus-cache":      "a parse cache; output is identical with or without it",
	"from-result":       "the reproduction source; recorded as rerun_of",
	"scenario":          "input.scenario and input.deployment", "scenarios": "input.scenario and input.deployment",
	"policy-config": "input.policy",
	"catalog":       "data.catalog", "registry": "data.registry",
	"catalog-overlay": "data.catalog.overlays", "registry-overlay": "data.registry.overlays",
}

// optionFileFlags are the flags whose value is a file the run reads: an option records the
// file's sha256 and never its path (a fact of the host), and --from-result takes the file again
// from the command line and checks it against the digest.
var optionFileFlags = map[string]bool{
	"workload-spec": true, "lora-config": true, "kv-offload-config": true, "defaults-filepath": true,
	"saturation-config": true, "trace-header": true, "trace-data": true, "corpus": true,
}

// inputOptions is one option per flag the command line set that is not excluded, sorted by name
// (SIM-10: every option that changes a byte of the Result is recorded in its inputs).
func inputOptions(command string, fs *pflag.FlagSet) ([]result.Option, error) {
	var out []result.Option
	var errs []string
	fs.Visit(func(f *pflag.Flag) {
		if _, ok := optionsExcluded[f.Name]; ok {
			return
		}
		v := f.Value.String()
		if optionFileFlags[f.Name] && v != "" {
			d, err := fileDigest(v)
			if err != nil {
				errs = append(errs, fmt.Sprintf("--%s %s: %v", f.Name, v, err))
				return
			}
			v = "sha256:" + d
		}
		out = append(out, result.Option{Name: f.Name, Value: v, Source: result.OptionCommandLine})
	})
	if len(errs) > 0 {
		return nil, fmt.Errorf("recording the run's inputs: %s", strings.Join(errs, "; "))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
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
