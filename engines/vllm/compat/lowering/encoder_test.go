// Original to blis-vllm.

package lowering

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

func towerWithHeads(heads int) *model.Graph {
	return &model.Graph{Encoder: &model.Encoder{Towers: []model.Tower{{
		Modality: model.TowerImage,
		LayerKinds: []model.EncoderLayerKind{{ID: "b", Nodes: []model.EncoderNode{
			{Op: model.EncoderAttention, NumHeads: heads, HeadDim: 64}}}},
	}}}}
}

func stating(v bool) vllmsettings.Settings { return vllmsettings.Settings{LanguageModelOnly: &v} }

// language_model_only true builds no tower; unset and false (vLLM's default) keep it; a text-only
// graph states nothing about a tower.
func TestEncoderRunFollowsLanguageModelOnly(t *testing.T) {
	g := towerWithHeads(16)
	for name, eng := range map[string]vllmsettings.Settings{"unset": {}, "false": stating(false)} {
		got := encoderRun(eng, g, 4, 1)
		if got == nil || !got.Resident || got.Mode != plan.EncoderWeightsSharded {
			t.Errorf("%s: %+v, want a resident weights-sharded tower", name, got)
		}
	}
	if got := encoderRun(stating(true), g, 4, 1); got == nil || got.Resident {
		t.Errorf("true: %+v, want a tower that is not resident", got)
	}
	if got := encoderRun(vllmsettings.Settings{}, &model.Graph{}, 4, 1); got != nil {
		t.Errorf("a text-only graph was given %+v", got)
	}
}

// vLLM runs the tower data-parallel when its head count does not divide the tensor-parallel width
// (model_executor/models/vision.py:149-155@v0.33.0): 12 heads at tp 8, but 16 at tp 8 split by weights.
func TestEncoderRunSplitsTheBatchWhereTheHeadsDoNotDivideTP(t *testing.T) {
	if got := encoderRun(vllmsettings.Settings{}, towerWithHeads(12), 8, 1); got.Mode != plan.EncoderBatchSplit {
		t.Errorf("12 heads at tp 8: %+v, want batch split", got)
	}
	if got := encoderRun(vllmsettings.Settings{}, towerWithHeads(16), 8, 1); got.Mode != plan.EncoderWeightsSharded {
		t.Errorf("16 heads at tp 8: %+v, want weights sharded", got)
	}
	if got := encoderRun(vllmsettings.Settings{}, towerWithHeads(12), 1, 1); got.Mode != plan.EncoderWeightsSharded {
		t.Errorf("12 heads at tp 1: %+v, want weights sharded (nothing to divide)", got)
	}
}

// The tower sits on the first pipeline stage, and the kernel prices a resident tower at pipeline width
// 1 only. At pp 2 the lowering therefore states nothing about a tower unless language_model_only is
// true (no tower is built, whatever the stage count): it never says Resident, which the kernel refuses,
// and never says not-resident for a tower vLLM builds.
func TestEncoderRunStatesNoTowerUnderPipelineParallelism(t *testing.T) {
	g := towerWithHeads(16)
	if got := encoderRun(vllmsettings.Settings{}, g, 4, 2); got != nil {
		t.Errorf("unset at pp 2: %+v, want nothing stated", got)
	}
	if got := encoderRun(stating(false), g, 4, 8); got != nil {
		t.Errorf("false at pp 8: %+v, want nothing stated", got)
	}
	if got := encoderRun(stating(true), g, 4, 2); got == nil || got.Resident {
		t.Errorf("true at pp 2: %+v, want a tower that is not resident", got)
	}
}
