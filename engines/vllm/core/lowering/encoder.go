// Original to blis-vllm.

package lowering

import (
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/core/vllmsettings"
)

// encoderRun is what the engine does with a graph's vision tower, as the plan states it; nil for a
// graph that prices none.
//
// language_model_only true makes every modality's limit 0 (vllm/config/multimodal.py:696-700@v0.33.0),
// so _mark_tower_model builds the tower as a StageMissingLayer with no weights and the scheduler gives
// the encoder no budget (model_executor/models/interfaces.py:339-396@v0.33.0): the tower is not
// resident. Unset or false is vLLM's default, which keeps it.
//
// A resident tower divides by its weights over the tensor-parallel ranks (mm_encoder_tp_mode's default,
// "weights", config/multimodal.py:385@v0.33.0), except where the head count does not divide the width:
// is_vit_use_data_parallel forces the data mode then (model_executor/models/vision.py:149-155@v0.33.0),
// every rank holding the whole tower and encoding its share of the images. The deployment cannot state
// mm_encoder_tp_mode today, so the default is what runs.
//
// Under pipeline parallelism (pp above 1) a resident tower sits on the first stage only, and the
// kernel prices a resident tower at pp 1 only (it refuses the other, kernel/new.go). The lowering
// then states nothing about the tower rather than a falsehood: not Resident: true, which the kernel
// refuses, and not Resident: false, which says no tower is built. The kernel records that it prices
// none, and core/kernelmodel names the language_model_only substitution
// (languageModelOnlySubstitution), so such a deployment is priced as before, with the gap disclosed.
func encoderRun(eng vllmsettings.Settings, g *model.Graph, tp, pp int) *plan.EncoderRun {
	t, ok := g.ImageTower()
	if !ok {
		return nil
	}
	if eng.LanguageModelOnly != nil && *eng.LanguageModelOnly {
		return &plan.EncoderRun{Resident: false}
	}
	if pp > 1 {
		return nil
	}
	mode := plan.EncoderWeightsSharded
	for _, k := range t.LayerKinds {
		for _, n := range k.Nodes {
			if n.Op == model.EncoderAttention && tp > 1 && n.NumHeads%tp != 0 {
				mode = plan.EncoderBatchSplit
			}
		}
	}
	return &plan.EncoderRun{Resident: true, Mode: mode}
}
