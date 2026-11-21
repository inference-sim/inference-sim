// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// Three groups of settings had no decision: moe_backend, mm_encoder_tp_mode and the environment
// variables of the DeepSeek-V4 serving path. Each is decided here, for vLLM v0.33.0, as "refused
// by name": the engine neither prices the setting nor claims it inert, so a run that states it away
// from its default is not scored as if it had not (VLM-3). moe_backend alone is also approximated
// where the engine prices the backend another way (fields_test.go,
// TestAStatedMoEBackendIsApproximatedWhereTheEnginePricesIt). The decision is the list entry; this
// test holds it, so a change to a setting's status is a change to this table.
func TestMoEBackendMMEncoderAndDeepSeekV4EnvAreRefusedByName(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	byKey := map[string]fields.Field{}
	for _, f := range l.Fields {
		byKey[string(f.Source)+":"+f.Name] = f
	}
	decided := []string{
		"config:moe_backend",
		"config:kernel_config.moe_backend",
		"config:speculative_config.moe_backend",
		"config:mm_encoder_tp_mode",
		// The DeepSeek-V4 serving path's environment: the DSV4-named switches, the DeepGEMM
		// family it selects its grouped GEMMs with, the mega-MoE AOT artifact and the Engram
		// table's ROCm mapping.
		"env:VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION",
		"env:VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4",
		"env:VLLM_USE_DEEP_GEMM",
		"env:VLLM_USE_DEEP_GEMM_E8M0",
		"env:VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
		"env:VLLM_DEEP_GEMM_WARMUP",
		"env:VLLM_MOE_USE_DEEP_GEMM",
		"env:VLLM_USE_MEGA_AOT_ARTIFACT",
		"env:VLLM_ROCM_ENGRAM_HIP_VMM",
	}
	for _, key := range decided {
		f, ok := byKey[key]
		if !ok {
			t.Errorf("%s is not in the surface list", key)
			continue
		}
		if f.Status != fields.StatusNotModelled {
			t.Errorf("%s is %s; the decision is not_modelled, refused by name", key, f.Status)
		}
		if f.Evidence.Issue == "" {
			t.Errorf("%s is refused and cites no issue that tracks it", key)
		}
		// A value that is no backend the engine prices another way is neither exact nor
		// approximated: the point carrying it is refused by the setting's name.
		facts := fields.Facts{Value: "1", ComputeCapability: "9.0"}
		if fields.Exact(f.Estimation, facts) {
			t.Errorf("%s is scored as exact, but the engine does not model it", key)
		}
		if _, covered := fields.Approximate(f.Estimation, facts); covered {
			t.Errorf("%s is approximated, but the decision is to refuse it", key)
		}
	}
}
