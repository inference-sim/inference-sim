// SPDX-License-Identifier: Apache-2.0

package lower

import (
	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
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

// With the radix cache on, SGLang keeps the cached states of prefixes in the mamba pool by the
// strategy it resolves. The extra buffer gives each request tracking slots beside its working slot
// (get_mamba_tracking_slots: 2 with the overlap schedule, 1 without it or on the lazy buffer), which
// the plan states as the tracked word. (no_buffer needs the overlap schedule off, which Resolve does
// not yet accept: inference-sim/inference-sim#2451.)
func TestHybridPoolWithTheRadixCacheStatesItsTrackingSlots(t *testing.T) {
	base := "mem_fraction_static: 0.8\n"
	for _, tc := range []struct {
		name, extra string
		word        plan.RecurrentStates
		slots       int
	}{
		{"the default strategy with overlap", "", plan.RecurrentTracked, 2},
		{"extra_buffer stated", "mamba_radix_cache_strategy: extra_buffer\n", plan.RecurrentTracked, 2},
		{"extra_buffer_lazy", "mamba_radix_cache_strategy: extra_buffer_lazy\n", plan.RecurrentTracked, 1},
	} {
		low, err := lowerPool(t, qwen35, 8, base+tc.extra)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if c := low.Plan.Cache; c.Recurrent != tc.word || c.TrackingSlots != tc.slots {
			t.Errorf("%s: recurrent %q with %d tracking slots, want %q with %d", tc.name, c.Recurrent, c.TrackingSlots, tc.word, tc.slots)
		}
	}
}

// A state the engine cannot hold is refused by name, never priced as another.
func TestHybridPoolWithInt8CheckpointsIsRefused(t *testing.T) {
	_, err := lowerPool(t, qwen35, 8, "mem_fraction_static: 0.8\nenable_int8_mamba_checkpoint: true\n")
	if err == nil || !strings.Contains(err.Error(), "enable_int8_mamba_checkpoint") {
		t.Fatalf("got %v, want a refusal naming enable_int8_mamba_checkpoint", err)
	}
}

// mm_attention_backend selects the vision tower's attention kernel. A text-only scenario never
// runs the tower, so no value of it reaches a decision here: the lowered plan and every other
// resolved field are the ones with the field unset. The engine's flashinfer_cudnn also allocates
// a workspace when the tower is built, which this lowering does not subtract (see
// TestMMAttentionBackendIsUnsetEquivalentOnlyWhereTheEngineIgnoresIt).
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

// The field list (fields.json) says what the port does with a field. These laws hold it to the
// lowering for the three fields the Qwen3.5 pool reads: a field whose value the lowering turns into
// a decision is listed modelled, a field the lowering refuses by name is listed under the issue the
// refusal cites, and a value the lowering ignores is listed as unset-equivalent only where the
// engine ignores it too.
func listed(t *testing.T, name string) fields.Field {
	t.Helper()
	l, err := fields.Load("../fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range l.Fields {
		if f.Source == "config" && f.Name == name {
			return f
		}
	}
	t.Fatalf("fields.json lists no config field %s", name)
	return fields.Field{}
}

func TestMambaSsmDtypeIsListedModelledBecauseTheLoweringReadsIt(t *testing.T) {
	base := "mem_fraction_static: 0.8\ndisable_radix_cache: true\n"
	f32, err := lowerPool(t, qwen35, 8, base+"mamba_ssm_dtype: float32\n")
	if err != nil {
		t.Fatal(err)
	}
	b16, err := lowerPool(t, qwen35, 8, base+"mamba_ssm_dtype: bfloat16\n")
	if err != nil {
		t.Fatal(err)
	}
	if f32.Plan.Cache.StateElementBytes == b16.Plan.Cache.StateElementBytes {
		t.Fatal("the lowering no longer turns mamba_ssm_dtype into a state width; this law needs a field it reads")
	}
	if f := listed(t, "mamba_ssm_dtype"); f.Status != fields.StatusModelled {
		t.Errorf("mamba_ssm_dtype sets the recurrent state's width and is listed %s, want modelled", f.Status)
	}
}

func TestMambaRadixCacheStrategyIsListedModelledBecauseTheLoweringReadsIt(t *testing.T) {
	eb, err := lowerPool(t, qwen35, 8, "mem_fraction_static: 0.8\nmamba_radix_cache_strategy: extra_buffer\n")
	if err != nil {
		t.Fatal(err)
	}
	lazy, err := lowerPool(t, qwen35, 8, "mem_fraction_static: 0.8\nmamba_radix_cache_strategy: extra_buffer_lazy\n")
	if err != nil {
		t.Fatal(err)
	}
	if eb.Plan.Cache.TrackingSlots == lazy.Plan.Cache.TrackingSlots {
		t.Fatal("the lowering no longer turns the strategy into tracking slots; this law needs a field it reads")
	}
	if f := listed(t, "mamba_radix_cache_strategy"); f.Status != fields.StatusModelled {
		t.Errorf("mamba_radix_cache_strategy sets the plan's recurrent word and is listed %s, want modelled", f.Status)
	}
}

// flashinfer_cudnn is the one accepted value of mm_attention_backend the engine acts on beyond the
// vision tower's kernel: it allocates a 128 MiB workspace when the tower is built
// (python/sglang/srt/models/qwen3_vl.py:430-438, FLASHINFER_WORKSPACE_SIZE_BYTES at
// layers/attention/vision.py:110@297e54fd), which the port does not take from the KV pool.
func TestMMAttentionBackendIsUnsetEquivalentOnlyWhereTheEngineIgnoresIt(t *testing.T) {
	f := listed(t, "mm_attention_backend")
	if f.Status != fields.StatusNotModelled || f.Estimation.Rule != fields.RuleLeaveUnset || f.Estimation.Preserves == "" {
		t.Fatalf("mm_attention_backend is listed %s with rule %q and preserves %q; want not_modelled, leave_unset, with the claim stated", f.Status, f.Estimation.Rule, f.Estimation.Preserves)
	}
	var in []string
	for _, c := range f.Estimation.ValidWhen {
		in = append(in, c.ValueIn...)
	}
	sort.Strings(in)
	if want := []string{"fa3", "fa4", "sdpa", "triton_attn"}; !reflect.DeepEqual(in, want) {
		t.Errorf("mm_attention_backend stands in as unset for %v, want %v (flashinfer_cudnn allocates a workspace the port does not)", in, want)
	}
	base := "mem_fraction_static: 0.8\ndisable_radix_cache: true\n"
	unset, err := lowerPool(t, qwen35, 8, base)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range in {
		got, err := lowerPool(t, qwen35, 8, base+"mm_attention_backend: "+v+"\n")
		if err != nil || !reflect.DeepEqual(got, unset) {
			t.Errorf("mm_attention_backend=%s: lowered plan differs from the unset one (error %v)", v, err)
		}
	}
}

// The bytes of one request's recurrent state, taken from the catalog graph's stated dimensions, equal
// what SGLang's own state shape gives for the model's configuration (core/cache's port of
// Mamba2CacheParams, whose sizes the pool oracle checks against SGLang's function): 60 layers, one in
// four full attention, 16 key heads of 128, 64 value heads of 128, a convolution kernel of 4.
func TestStatePerRequestBytesEqualsSGLangsStateShape(t *testing.T) {
	g := graph(t, qwen35)
	cfg := cache.LinearAttnConfig{NumHiddenLayers: 60, FullAttentionInterval: 4, KeyHeadDim: 128, NumKeyHeads: 16,
		ValueHeadDim: 128, NumValueHeads: 64, ConvKernel: 4}
	bf16, f16 := "bfloat16", "float16"
	for _, tc := range []struct {
		tp   int
		flag *string
		temp cache.DType
	}{{1, nil, cache.Float32}, {2, nil, cache.Float32}, {4, nil, cache.Float32}, {8, nil, cache.Float32}, {8, &bf16, cache.BFloat16}, {4, &f16, cache.Float16}} {
		want, err := cache.NewLinearStateParams(cfg, tc.tp, cache.BFloat16, tc.temp)
		if err != nil {
			t.Fatal(err)
		}
		got, err := StatePerRequestBytes(g, tc.tp, tc.flag)
		if err != nil {
			t.Fatal(err)
		}
		if got != want.PerReqBytes() {
			t.Errorf("tp %d, flag %v: %d bytes, SGLang's state shape gives %d", tc.tp, tc.flag, got, want.PerReqBytes())
		}
	}
	if _, err := StatePerRequestBytes(g, 32, nil); err == nil {
		t.Error("a width above the 16 key groups pads them, which the graph does not state: it must be refused")
	}
}
