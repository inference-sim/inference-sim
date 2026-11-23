// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// A recipe that states VLLM_USE_FLASHINFER_MOE_FP4 (an NVFP4 MoE recipe of an earlier release) must not be
// refused as a field the release does not have. The pinned release reads no such variable, and an unlisted VLLM_
// variable only draws a warning, so a stated value reaches no code: the field list holds it as proven inert.
func TestTheFlashInferMoeFp4VariableIsProvenInert(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for i := range l.Fields {
		f := l.Fields[i]
		if f.Name == "VLLM_USE_FLASHINFER_MOE_FP4" && f.Source == fields.SourceEnv {
			if f.Status != fields.StatusProvenInert {
				t.Errorf("status %s, want proven_inert", f.Status)
			}
			return
		}
	}
	t.Fatal("VLLM_USE_FLASHINFER_MOE_FP4: not in the list, so a recipe that states it is refused as a field the release does not have")
}
