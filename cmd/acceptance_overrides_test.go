package cmd

import (
	"testing"

	"github.com/spf13/pflag"

	"github.com/inference-sim/blis-schemas/spec/acceptance"

	"github.com/inference-sim/blis-vllm/compat/build"
)

// The kernel sizes an adaptive verify budget from the run's acceptance
// (blis-latency-kernel#89), so the command line's and the workload's acceptance layers reach it
// (setAcceptanceOverrides), as they reach the simulator's own resolution
// (build.ResolveAcceptance): the CLI value, the workload's terms with the CLI's category and
// thinking mode applied, and the fallback category.
func TestAcceptanceOverridesReachTheKernel(t *testing.T) {
	defer func(l, r float64, th, c, fb string, w *acceptance.Workload) {
		speculativeAcceptanceLength, speculativeAcceptance = l, r
		speculativeThinking, speculativeCategory, speculativeFallbackCategory, workloadAcceptance = th, c, fb, w
	}(speculativeAcceptanceLength, speculativeAcceptance, speculativeThinking, speculativeCategory,
		speculativeFallbackCategory, workloadAcceptance)
	fl := pflag.NewFlagSet("t", pflag.ContinueOnError)
	for _, n := range []string{build.AcceptanceLengthFlag, build.AcceptanceRateFlag, build.AcceptanceThinkingFlag,
		build.AcceptanceCategoryFlag, build.AcceptanceFallbackFlag} {
		fl.String(n, "", "")
	}
	_ = fl.Set(build.AcceptanceLengthFlag, "3.5")
	_ = fl.Set(build.AcceptanceCategoryFlag, "coding")
	speculativeAcceptanceLength, speculativeCategory, speculativeFallbackCategory = 3.5, "coding", "chat"
	workloadAcceptance = &acceptance.Workload{Category: "chat", Thinking: acceptance.ThinkingOff}

	o := acceptanceOverridesOf(fl, 7)
	if o.CLI == nil || o.CLI.AcceptanceLength == nil || *o.CLI.AcceptanceLength != 3.5 || o.CLI.NumSpecTokens != 7 {
		t.Fatalf("CLI layer %+v, want --speculative-acceptance-length 3.5 at 7 tokens", o.CLI)
	}
	if o.Workload == nil || o.Workload.Category != "coding" || o.Workload.Thinking != acceptance.ThinkingOff {
		t.Fatalf("workload layer %+v, want the workload's thinking with the CLI's category", o.Workload)
	}
	if o.FallbackCategory != "chat" {
		t.Fatalf("fallback category %q, want chat", o.FallbackCategory)
	}
	// Nothing stated: no layer beyond the scenario file.
	workloadAcceptance, speculativeFallbackCategory = nil, ""
	none := acceptanceOverridesOf(pflag.NewFlagSet("n", pflag.ContinueOnError), 7)
	if none.CLI != nil || none.Workload != nil || none.FallbackCategory != "" {
		t.Fatalf("no flags and no workload terms gave %+v", none)
	}
}
