// Original to blis-vllm.

package build

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// The simulator's own acceptance resolution carries the target's architecture, as the kernel's does:
// deepseek-v4-pro-nvfp4 has no profile of its own and borrows the mtp curve of deepseek-v4-pro, which
// records the same architecture. Without the architecture the borrowing step is skipped.
func TestResolveAcceptanceBorrowsByTheTargetsArchitecture(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOff}
	got, ok, err := ResolveAcceptance(AcceptanceInputs{
		Scenario: "s", Shape: kernelmodel.Shape{Model: "deepseek-v4-pro-nvfp4"}, Data: data, Workload: &terms,
	}, 3, "mtp")
	if err != nil || !ok {
		t.Fatalf("resolved %+v, %v, %v; want the borrowed mtp profile", got, ok, err)
	}
	if got.Res.Step != acceptance.StepOtherTarget || got.Res.Profile == nil || got.Res.Profile.Model != "deepseek-v4-pro" {
		t.Errorf("resolved %+v, want deepseek-v4-pro's mtp profile borrowed by architecture", got.Res)
	}
	if a := targetArchitecture(data, "deepseek-v4-pro-nvfp4"); a != "DeepseekV4ForCausalLM" {
		t.Errorf("architecture %q, want DeepseekV4ForCausalLM", a)
	}
	if a := targetArchitecture(data, "no-such-model"); a != "" {
		t.Errorf("architecture %q for a model the catalog does not have, want none", a)
	}
}

// A separate drafter is part of the lookup key: a run that names one does not match a profile
// measured with the model's own speculator, and the request says which drafter it names.
func TestResolveAcceptanceNamesTheDrafter(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOff}
	in := AcceptanceInputs{Scenario: "s", Shape: kernelmodel.Shape{Model: "deepseek-v4-pro"}, Data: data, Workload: &terms}
	if _, ok, err := ResolveAcceptance(in, 3, "mtp"); err != nil || !ok {
		t.Fatalf("the model's own speculator did not resolve: %v, %v", ok, err)
	}
	in.Drafter = "a-separate-drafter"
	if got, ok, err := ResolveAcceptance(in, 3, "mtp"); err == nil && ok && got.Res.Profile != nil && got.Res.Profile.Drafter == "" && got.Res.Step == acceptance.StepExact {
		t.Errorf("a run naming a separate drafter resolved to the exact profile of the model's own speculator: %+v", got.Res)
	}
}
