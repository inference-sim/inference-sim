// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// A recipe that states VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD (every AgentX Kimi-K3 recipe states 0) must
// not be refused as a field the release does not list. No file of the pinned vLLM release names it, so a stated
// value reaches no code: the field list holds it as proven inert.
func TestTheRoutedDownProjThresholdIsProvenInert(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for i := range l.Fields {
		f := l.Fields[i]
		if f.Name == "VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD" && f.Source == fields.SourceEnv {
			if f.Status != fields.StatusProvenInert {
				t.Errorf("status %s, want proven_inert", f.Status)
			}
			return
		}
	}
	t.Fatal("VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD: not in the list, so a recipe that states it is refused as a field the release does not have")
}
