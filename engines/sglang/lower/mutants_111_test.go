// Original to blis-sglang.

package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

func strp(s string) *string { return &s }
func intp(i int) *int       { return &i }

// attnDP is an attention-DP setting built in the test: tp 8, attn_dp_size 2, dp_size 1, so two DP
// ranks and an MLP TP group of 8/2 = 4 per rank.
func attnDP(backend string, dense *int, dpLmHead bool) *config.ServerArgs {
	a := config.Defaults()
	a.TpSize, a.AttnDpSize, a.DpSize = 8, 2, 1
	a.MoeA2aBackend = backend
	a.MoeDenseTpSize = dense
	a.EnableDpLmHead = dpLmHead
	return &a
}

// A split attention-backend field overrides attention_backend for its phase; unset or empty, the
// phase keeps attention_backend (attention_backends, runtime_context.py:2010-2022).
func TestAttentionBackendsSplitFieldOverridesTheBase(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		prefill, decode      *string
		wantPrefill, wantDec string
	}{
		{"unset", nil, nil, "fa3", "fa3"},
		{"empty", strp(""), strp(""), "fa3", "fa3"},
		{"both set", strp("flashinfer"), strp("triton"), "flashinfer", "triton"},
		{"prefill only", strp("flashinfer"), nil, "flashinfer", "fa3"},
		{"decode only", nil, strp("triton"), "fa3", "triton"},
	} {
		a := config.Defaults()
		a.AttentionBackend, a.PrefillAttentionBackend, a.DecodeAttentionBackend = strp("fa3"), tc.prefill, tc.decode
		if p, d := AttentionBackends(&a); p != tc.wantPrefill || d != tc.wantDec {
			t.Errorf("%s: (%q, %q), want (%q, %q)", tc.name, p, d, tc.wantPrefill, tc.wantDec)
		}
	}
}

// A resolved ServerArgs with no decode capture list is refused with an error, never a panic.
func TestCaptureBatchSizesRefusesAMissingCaptureList(t *testing.T) {
	a := config.Defaults()
	for name, cfg := range map[string]*config.CudaGraphConfig{
		"no cuda_graph_config": nil,
		"empty decode list":    {},
	} {
		a.CudaGraphConfig = cfg
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panicked (%v), want an error", name, r)
				}
			}()
			if bs, err := CaptureBatchSizes(&a, 16, 1); err == nil {
				t.Errorf("%s: captured %v, want an error", name, bs)
			}
		}()
	}
}

// An EP scale-up joiner enables attention DP even with attn_dp_size 1 (attn_dp_enabled_of); any
// other join mode does not.
func TestScaleUpJoinerEnablesAttentionDP(t *testing.T) {
	a := config.Defaults()
	if ParallelOf(&a).AttnDPEnabled {
		t.Error("defaults enable attention DP")
	}
	a.EpJoinMode = strp("scale")
	if !ParallelOf(&a).AttnDPEnabled {
		t.Error("an EP scale-up joiner does not enable attention DP")
	}
	a.EpJoinMode = strp("other")
	if ParallelOf(&a).AttnDPEnabled {
		t.Error("a join mode other than scale enables attention DP")
	}
	a.EpJoinMode, a.AttnDpSize, a.TpSize = nil, 2, 2
	if !ParallelOf(&a).AttnDPEnabled {
		t.Error("attn_dp_size 2 does not enable attention DP")
	}
}

// With attention DP on, a single DP rank is refused (SGLang asserts num_dp_ranks > 1); the
// scale-up joiner with dp_size 1 and attn_dp_size 1 is that case.
func TestAttentionDPNeedsMoreThanOneRank(t *testing.T) {
	a := config.Defaults()
	a.EpJoinMode = strp("scale")
	if g, err := RequireMLPTPGather(&a); err == nil {
		t.Errorf("one DP rank with attention DP lowered (gather %v), want an error", g)
	}
	a.EpJoinMode = nil
	if g, err := RequireMLPTPGather(&a); err != nil || g {
		t.Errorf("no attention DP: (%v, %v), want (false, nil)", g, err)
	}
}

// require_mlp_tp_gather's chain for attention DP, case by case (common.py:4067-4121).
func TestMLPTPGatherRuleChain(t *testing.T) {
	for _, tc := range []struct {
		name string
		args *config.ServerArgs
		want bool
	}{
		{"no dense TP size", attnDP("deepep", nil, true), true},
		{"no dense TP size, no a2a", attnDP("", nil, true), true},
		{"dp lm head off", attnDP("deepep", intp(1), false), true},
		{"no a2a backend", attnDP("", intp(1), true), true},
		{"explicit none", attnDP("none", intp(1), true), true},
		{"flashinfer", attnDP("flashinfer", intp(1), true), true},
		{"flashinfer_megamoe", attnDP("flashinfer_megamoe", intp(1), true), true},
		// The default branch: gather only when the dense TP group spans more than one DP rank's
		// share of the TP group (8 / 2 ranks = 4).
		{"other backend, dense below the share", attnDP("deepep", intp(1), true), false},
		{"other backend, dense at the share", attnDP("deepep", intp(4), true), false},
		{"other backend, dense above the share", attnDP("deepep", intp(5), true), true},
	} {
		got, err := RequireMLPTPGather(tc.args)
		if err != nil || got != tc.want {
			t.Errorf("%s: (%v, %v), want (%v, nil)", tc.name, got, err, tc.want)
		}
	}
}
