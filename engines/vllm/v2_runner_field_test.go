// Original to blis-vllm.

package blisvllm

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// VLLM_USE_V2_MODEL_RUNNER is no longer in vLLM's envs.py: the release removed Model Runner V1, and the
// platform's environment check (vllm/platforms/interface.py validate_environ) accepts the value 1 with a
// warning that it is redundant and refuses every other value. Stated as 1 it changes nothing the engine
// decides, so it is proven inert with that one value; a recipe that states it must not be refused as a
// field the release does not list.
func TestTheV2RunnerEnvKeyIsProvenInertAtOne(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	var f *fields.Field
	for i := range l.Fields {
		if l.Fields[i].Name == "VLLM_USE_V2_MODEL_RUNNER" && l.Fields[i].Source == fields.SourceEnv {
			f = &l.Fields[i]
		}
	}
	if f == nil {
		t.Fatal("VLLM_USE_V2_MODEL_RUNNER: not in the list, so a recipe that states it is refused as a field the release does not have")
	}
	if f.Status != fields.StatusProvenInert {
		t.Errorf("status %s, want proven_inert", f.Status)
	}
	if f.ValueClasses == nil || !slices.Equal(f.ValueClasses.Values, []string{"1"}) {
		t.Errorf("value classes %+v, want the one value 1: vLLM refuses every other", f.ValueClasses)
	}
}
