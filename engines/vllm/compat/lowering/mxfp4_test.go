package lowering

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// vLLM v0.33.0's MXFP4 methods quantize the routed experts and leave every linear layer
// unquantized (Mxfp4Config.get_quant_method, quantization/mxfp4.py:91-107@4ba787ec). A GPT-OSS
// checkpoint stores its experts in MXFP4 and its linears in bf16, and vLLM names its method
// gpt_oss_mxfp4 (GptOssMxfp4Config, :116-149): stated either way, the linears are served in bf16,
// as they are when the quantization is left unstated.
func TestAnMXFP4MethodOnGPTOSSServesTheLinearsAtTheCheckpointsWidth(t *testing.T) {
	at := where{"gpt-oss-120b", "h100", "tp: 2, pp: 1, dp: 1"}
	base, err := tryLower(t, at, vllmsettings.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if base.Plan.WeightFormat != "bf16" {
		t.Fatalf("unstated: WeightFormat %q, want the checkpoint's bf16", base.Plan.WeightFormat)
	}
	for _, q := range []string{"gpt_oss_mxfp4", "mxfp4"} {
		l, err := tryLower(t, at, vllmsettings.Settings{Quantization: q})
		if err != nil {
			t.Errorf("quantization %q: %v", q, err)
			continue
		}
		if l.Plan.WeightFormat != "bf16" {
			t.Errorf("quantization %q: WeightFormat %q, want bf16 (vLLM leaves the linears unquantized)", q, l.Plan.WeightFormat)
		}
	}
}

// gpt_oss_mxfp4 is the method vLLM gives a GPT-OSS checkpoint only (override_quantization_method
// requires model_type gpt_oss, mxfp4.py:129-146@4ba787ec), and an MXFP4 method loads only a
// checkpoint whose routed experts are stored in MXFP4. Kimi-K3 stores MXFP4 experts and is not
// GPT-OSS; MiniMax-M3's NVFP4 checkpoint stores no MXFP4 experts.
func TestAnMXFP4MethodTheCheckpointCannotLoadIsRefused(t *testing.T) {
	cases := []struct {
		at   where
		q    string
		want string
	}{
		{where{"kimi-k3", "b200", "tp: 8, pp: 1, dp: 1"}, "gpt_oss_mxfp4", "model_type gpt_oss"},
		{where{"minimax-m3-nvfp4", "b200", "tp: 4, pp: 1, dp: 1"}, "gpt_oss_mxfp4", "no routed experts in MXFP4"},
		{where{"minimax-m3-nvfp4", "b200", "tp: 4, pp: 1, dp: 1"}, "mxfp4", "no routed experts in MXFP4"},
	}
	for _, c := range cases {
		_, err := tryLower(t, c.at, vllmsettings.Settings{Quantization: c.q})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s with quantization %q: %v, want a refusal naming %q", c.at.model, c.q, err, c.want)
		}
	}
	// Kimi-K3's MXFP4 experts load under the generic method.
	if _, err := tryLower(t, where{"kimi-k3", "b200", "tp: 8, pp: 1, dp: 1"}, vllmsettings.Settings{Quantization: "mxfp4"}); err != nil {
		t.Errorf("kimi-k3 with quantization mxfp4: %v", err)
	}
}
