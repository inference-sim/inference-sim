// Original to blis-vllm.

package lowering

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-vllm/core/vllmsettings"
)

// A pre-quantized checkpoint is loaded with its own method, and vLLM raises when another is stated
// (ModelConfig._verify_quantization, vllm/config/model.py:1383-1462@v0.33.0). Stating that method
// therefore serves the checkpoint exactly as stating nothing does: its BF16 attention and head stay
// BF16 beside the quantized experts. It lowered to a four-bit served format instead, and the
// kernel priced the step's unshaped work at that format's compute peak.
func TestStatingACheckpointsOwnMethodServesItAsStored(t *testing.T) {
	for _, c := range []struct {
		at     where
		method string
	}{
		{where{"gpt-oss-120b", "h200", "tp: 2, pp: 1, dp: 1"}, "mxfp4"},
		{where{"gpt-oss-120b", "b200", "tp: 1, pp: 1, dp: 1"}, "mxfp4"},
		{where{"kimi-k2.5-nvfp4", "b200", "tp: 4, pp: 1, dp: 1"}, "modelopt_fp4"},
		{where{"minimax-m2.5-nvfp4", "b200", "tp: 1, pp: 1, dp: 1"}, "modelopt_fp4"},
	} {
		t.Run(c.at.model+"/"+c.at.chip, func(t *testing.T) {
			stated := lowered(t, c.at, vllmsettings.Settings{Quantization: c.method})
			unstated := lowered(t, c.at, vllmsettings.Settings{})
			if stated.Plan.WeightFormat != unstated.Plan.WeightFormat {
				t.Errorf("quantization %s serves %q, the checkpoint as stored %q", c.method,
					stated.Plan.WeightFormat, unstated.Plan.WeightFormat)
			}
			if !reflect.DeepEqual(stated.Plan, unstated.Plan) {
				t.Errorf("stating the checkpoint's own method %s changes the plan:\n%+v\n%+v",
					c.method, stated.Plan, unstated.Plan)
			}
		})
	}
}

// The other direction: a checkpoint stored at 16 bits throughout is served in the format the value
// names (vLLM quantizes it online), so the stated format still governs.
func TestAQuantizationAnUnquantizedCheckpointDoesNotStoreServesIt(t *testing.T) {
	l := lowered(t, where{"qwen3-14b", "h100", "tp: 1, pp: 1, dp: 1"}, vllmsettings.Settings{Quantization: "fp8"})
	if l.Plan.WeightFormat != "fp8" {
		t.Errorf("WeightFormat %q, want fp8", l.Plan.WeightFormat)
	}
}
