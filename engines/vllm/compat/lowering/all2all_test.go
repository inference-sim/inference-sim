package lowering

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// The plan states the all-to-all the engine runs, not the name it was given (blis-vllm#75): a
// deprecated alias is the backend it runs as, an unstated backend beside a MoE backend that
// dispatches itself is passthrough, and passthrough runs no framework all-to-all, so the cost
// model prices no dispatch for it. Any other backend is named by itself, for the cost model to
// price from its own coefficients or to say it has none.
func TestThePlanStatesTheAll2AllBackendTheEngineRuns(t *testing.T) {
	rulesVersion = "0.33.0"
	t.Cleanup(func() { rulesVersion = "0.32.0" })
	for _, c := range []struct {
		name     string
		s        vllmsettings.Settings
		dispatch plan.KernelImpl
		noted    string // the all2all_backend note, "" for none
	}{
		{"an unstated backend is the gather-and-scatter dispatch", vllmsettings.Settings{}, plan.KernelImpl{}, ""},
		{"a removed backend runs as the gather-and-scatter dispatch", vllmsettings.Settings{All2AllBackend: "pplx"}, plan.KernelImpl{}, "allgather_reducescatter"},
		{"DeepEP low-latency is itself", vllmsettings.Settings{All2AllBackend: "deepep_low_latency"}, plan.DispatchDeepEPLowLatency, ""},
		{"the deprecated FlashInfer alias runs as the two-sided backend",
			vllmsettings.Settings{All2AllBackend: "flashinfer_all2allv"},
			plan.KernelImpl{Library: "flashinfer_nvlink_two_sided"}, "flashinfer_nvlink_two_sided"},
		{"a backend with no alias is named by itself", vllmsettings.Settings{All2AllBackend: "mori_high_throughput"},
			plan.KernelImpl{Library: "mori_high_throughput"}, ""},
		{"a self-communicating MoE backend binds the default backend to passthrough, which runs no all-to-all",
			vllmsettings.Settings{MoEBackend: "flashinfer_moe_ep_cutedsl"}, plan.KernelImpl{}, "passthrough"},
		{"passthrough stated beside it is the same", vllmsettings.Settings{All2AllBackend: "passthrough", MoEBackend: "flashinfer_moe_ep_cutedsl"},
			plan.KernelImpl{}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			l := lowered(t, moe, c.s)
			if l.Plan.MoE.Dispatch != c.dispatch {
				t.Errorf("MoE.Dispatch %+v, want %+v", l.Plan.MoE.Dispatch, c.dispatch)
			}
			v, ok := noted(l, "all2all_backend")
			if c.noted == "" && ok || c.noted != "" && (!ok || v != c.noted) {
				t.Errorf("all2all_backend note %q (%t), want %q", v, ok, c.noted)
			}
		})
	}
}

// vLLM refuses what bind_passthrough_all2all_backend refuses (vllm/config/kernel.py:206-227): a
// FlashInfer MoE-EP backend beside an explicit all2all_backend, and passthrough beside any other MoE
// backend. The lowering refuses both by name, citing the open issue, and builds the pairs vLLM
// accepts; the note for a bound passthrough says the communication is inside the MoE kernel.
func TestTheLoweringRefusesWhatVLLMsPassthroughBindingRefuses(t *testing.T) {
	rulesVersion = "0.33.0"
	t.Cleanup(func() { rulesVersion = "0.32.0" })
	for _, c := range []struct {
		name    string
		s       vllmsettings.Settings
		refused bool
	}{
		{"an explicit DeepEP backend beside a self-communicating MoE backend",
			vllmsettings.Settings{All2AllBackend: "deepep_low_latency", MoEBackend: "flashinfer_moe_ep_cutedsl"}, true},
		{"an explicit backend beside the other self-communicating MoE backend",
			vllmsettings.Settings{All2AllBackend: "mori_high_throughput", MoEBackend: "flashinfer_moe_ep_mega_deep_gemm"}, true},
		{"passthrough beside Triton", vllmsettings.Settings{All2AllBackend: "passthrough", MoEBackend: "triton"}, true},
		{"passthrough beside an unstated MoE backend", vllmsettings.Settings{All2AllBackend: "passthrough"}, true},
		{"the default backend stated beside a self-communicating MoE backend",
			vllmsettings.Settings{All2AllBackend: "allgather_reducescatter", MoEBackend: "flashinfer_moe_ep_cutedsl"}, false},
		{"a removed backend (runs as the default) beside a self-communicating MoE backend",
			vllmsettings.Settings{All2AllBackend: "pplx", MoEBackend: "flashinfer_moe_ep_cutedsl"}, false},
		{"passthrough beside a self-communicating MoE backend",
			vllmsettings.Settings{All2AllBackend: "passthrough", MoEBackend: "flashinfer_moe_ep_cutedsl"}, false},
		{"an explicit DeepEP backend beside Triton", vllmsettings.Settings{All2AllBackend: "deepep_low_latency", MoEBackend: "triton"}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := tryLower(t, moe, c.s)
			if !c.refused {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "issues/271") {
				t.Fatalf("err %v, want a refusal naming the combination and issues/271", err)
			}
		})
	}
	l := lowered(t, moe, vllmsettings.Settings{MoEBackend: "flashinfer_moe_ep_cutedsl"})
	if v, ok := noted(l, "all2all_backend"); !ok || v != "passthrough" {
		t.Fatalf("all2all_backend note %q (%t), want passthrough", v, ok)
	}
	for _, n := range l.Notes {
		if n.Name == "all2all_backend" && !strings.Contains(n.Reason, "not priced as a separate collective") {
			t.Errorf("the passthrough note does not say the communication is inside the MoE kernel: %s", n.Reason)
		}
	}
}
