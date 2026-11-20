// SPDX-License-Identifier: Apache-2.0

package lower

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
)

const qwen35 = "qwen3.5-397b-a17b-fp8"

// A hybrid (Gated DeltaNet plus attention) model with its radix cache off keeps one recurrent
// state a request, in the width mamba_ssm_dtype names: the model configuration's float32 unless
// the flag overrides it (mamba_utils.py mamba2_state_dtype@297e54fd).
func TestHybridPoolLowersWithTheRadixCacheOff(t *testing.T) {
	base := "mem_fraction_static: 0.8\ndisable_radix_cache: true\n"
	for _, tc := range []struct {
		extra string
		bytes float64
	}{
		{"", 4},
		{"mamba_ssm_dtype: float32\n", 4},
		{"mamba_ssm_dtype: bfloat16\n", 2},
		{"mamba_ssm_dtype: float16\n", 2},
	} {
		low, err := lowerPool(t, qwen35, 8, base+tc.extra)
		if err != nil {
			t.Fatalf("%q: %v", tc.extra, err)
		}
		c := low.Plan.Cache
		if c.StateElementBytes != tc.bytes || c.Recurrent != plan.RecurrentLatest {
			t.Errorf("%q: state element %v bytes, recurrent %q; want %v bytes and the latest state only", tc.extra, c.StateElementBytes, c.Recurrent, tc.bytes)
		}
	}
}

// With the radix cache on, SGLang keeps the cached states of a prefix in the mamba pool by the
// strategy it resolves; the plan cannot state that yet, so the pool is refused by name, never
// priced as a latest-state-only cache.
func TestHybridPoolWithTheRadixCacheIsRefusedByName(t *testing.T) {
	_, err := lowerPool(t, qwen35, 8, "mem_fraction_static: 0.8\n")
	var nl *NotLowered
	if !errors.As(err, &nl) || !strings.Contains(nl.Name, "mamba radix cache") || !strings.Contains(nl.Name, "extra_buffer") {
		t.Fatalf("got %v, want a refusal naming the mamba radix cache strategy", err)
	}
}

// mm_attention_backend selects the vision tower's attention kernel. A text-only scenario never
// runs the tower, so no value of it reaches a decision: the lowered plan and every other
// resolved field are the ones with the field unset.
func TestMMAttentionBackendChangesNoDecision(t *testing.T) {
	base := "mem_fraction_static: 0.8\ndisable_radix_cache: true\n"
	want, err := lowerPool(t, qwen35, 8, base)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"sdpa", "fa3", "triton_attn", "flashinfer_cudnn"} {
		got, err := lowerPool(t, qwen35, 8, base+"mm_attention_backend: "+v+"\n")
		if err != nil {
			t.Fatalf("%s: %v", v, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("mm_attention_backend=%s changes the lowered plan", v)
		}
	}
}
