// Original to blis-vllm.

package vllm

import (
	"fmt"
	"strings"
	"testing"
)

// mamba_cache_mode is compared with the mode that RUNS, not with the field's initializer.
// CacheConfig's initializer is "none" (vllm/config/cache.py:189@5df63a72). For a Mamba or hybrid
// model, MambaModelConfig.verify_and_update_config sets "align" when prefix caching is on and
// the stated mode is "none", and "none" when prefix caching is off and the stated mode is not
// (vllm/model_executor/models/config.py:685-706@5df63a72). Prefix caching defaults on (cache.py:141).
// So for such a model every stated value resolves to the one mode the hook picks, and none is
// away from the default; for any other model no hook runs and "align" is away from the default
// "none". The cases call the resolver on the engine block, so they do not depend on the
// factory's earlier refusal of a Mamba model's KV groups (vllmengine, blis-vllm#7).
func TestMambaCacheModeIsComparedWithTheResolvedDefault(t *testing.T) {
	on, off := "enable_prefix_caching: true\n", "enable_prefix_caching: false\n"
	for _, c := range []struct {
		name   string
		mamba  bool
		prefix string // "" leaves prefix caching unstated, which is on
		mode   string
		runsAs string
		refuse bool
	}{
		// A Mamba or hybrid model: the hook decides, and every value resolves to the default.
		{"Mamba, prefix caching unstated, mode unstated", true, "", "", "align", false},
		{"Mamba, prefix caching unstated, stated none", true, "", "none", "align", false},
		{"Mamba, prefix caching unstated, stated align", true, "", "align", "align", false},
		{"Mamba, prefix caching on, mode unstated", true, on, "", "align", false},
		{"Mamba, prefix caching on, stated none", true, on, "none", "align", false},
		{"Mamba, prefix caching on, stated align", true, on, "align", "align", false},
		{"Mamba, prefix caching off, mode unstated", true, off, "", "none", false},
		{"Mamba, prefix caching off, stated none", true, off, "none", "none", false},
		{"Mamba, prefix caching off, stated align", true, off, "align", "none", false},
		// Any other model: the stated value stands and the initializer is the default.
		{"no Mamba, prefix caching on, mode unstated", false, on, "", "none", false},
		{"no Mamba, prefix caching on, stated none", false, on, "none", "none", false},
		{"no Mamba, prefix caching on, stated align", false, on, "align", "align", true},
		{"no Mamba, prefix caching off, mode unstated", false, off, "", "none", false},
		{"no Mamba, prefix caching off, stated none", false, off, "none", "none", false},
		{"no Mamba, prefix caching off, stated align", false, off, "align", "align", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			caching := c.prefix != off
			if got := effectiveMambaCacheMode(c.mode, caching, c.mamba); got != c.runsAs {
				t.Errorf("mode %q runs as %q, want %q", c.mode, got, c.runsAs)
			}
			extra := c.prefix
			if c.mode != "" {
				extra += "mamba_cache_mode: " + c.mode + "\n"
			}
			cfg, err := decodeConfig(block(t, extra))
			if err != nil {
				t.Fatal(err)
			}
			cfg.MambaModel = c.mamba
			err = checkMambaCache(cfg, caching)
			if c.refuse != (err != nil) {
				t.Fatalf("checkMambaCache = %v, want refused %t", err, c.refuse)
			}
			if err != nil && !strings.Contains(err.Error(), "issues/7") {
				t.Errorf("the refusal does not cite #7: %v", err)
			}
		})
	}
}

// resolve reaches the same answer from a whole engine block: a Mamba model that states none with
// prefix caching on builds, where the initializer-based predicate it replaces would have accepted it
// only by accident, and a Mamba model that states align with prefix caching off builds too.
func TestResolveAppliesTheMambaCheckToAMambaModel(t *testing.T) {
	for _, c := range []struct {
		mamba bool
		extra string
		ok    bool
	}{
		{true, "mamba_cache_mode: none\n", true},
		{true, "mamba_cache_mode: align\n", true},
		{true, "mamba_cache_mode: align\nenable_prefix_caching: false\n", true},
		{false, "mamba_cache_mode: align\n", false},
	} {
		extra := strings.ReplaceAll(c.extra, "\n", "\n")
		cfg, err := decodeConfig(block(t, extra))
		if err != nil {
			t.Fatal(err)
		}
		cfg.MambaModel = c.mamba
		_, err = resolve(cfg)
		if c.ok != (err == nil) {
			t.Errorf("mamba %t, %q: resolve = %v, want ok %t", c.mamba, c.extra, err, c.ok)
		}
	}
}

// mamba_ssm_cache_dtype: the initializer is "auto" (cache.py:185). NemotronH and Qwen3.5 replace a
// stated "auto" with the checkpoint's value and keep any other stated value
// (config.py:716-732, 850-866@5df63a72), so a stated value equal to that checkpoint value is the
// default; another is away from it. A model with no such hook keeps what is stated.
func TestMambaSSMCacheDTypeIsComparedWithTheResolvedDefault(t *testing.T) {
	for _, c := range []struct {
		name, stated, modelDefault string
		refuse                     bool
	}{
		{"unstated", "", "", false},
		{"auto", "auto", "", false},
		{"auto, model default float32", "auto", "float32", false},
		{"stated float32 equal to the model's", "float32", "float32", false},
		{"stated float16 away from the model's float32", "float16", "float32", true},
		{"stated float32, no hook", "float32", "", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg, err := decodeConfig(block(t, ""))
			if err != nil {
				t.Fatal(err)
			}
			cfg.MambaSSMCacheDType, cfg.MambaSSMCacheDTypeDefault = c.stated, c.modelDefault
			err = checkMambaCache(cfg, true)
			if c.refuse != (err != nil) {
				t.Fatalf("checkMambaCache = %v, want refused %t", err, c.refuse)
			}
			if err != nil && !strings.Contains(err.Error(), "issues/83") {
				t.Errorf("the refusal does not cite #83: %v", err)
			}
		})
	}
}

// swa_bounded_replay is the layers' to carry: a stated false builds on either runner (it is the
// effective value on the V1 runner and a layout on the V2 runner, where each kv_layers run states
// no bounded replay), a stated true is the default, and false beside a run that states a bounded
// replay contradicts itself and is refused.
func TestSWABoundedReplayFalseBuildsWhereTheLayersStateNone(t *testing.T) {
	const layers = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window_mla, sliding_window: 128, num_kv_heads: 1, head_size: 512, block_size: 32, state_bytes: 584%s}\n" +
		"- {count: 1, spec: mla, num_kv_heads: 1, head_size: 512, block_size: 128, tokens_per_state: 4, state_bytes: 584}\n"
	for _, c := range []struct {
		extra  string
		refuse bool
	}{
		{"swa_bounded_replay: true\nuse_v2_model_runner: false\n", false},
		{"swa_bounded_replay: false\nuse_v2_model_runner: false\n", false},
		{"swa_bounded_replay: true\nuse_v2_model_runner: true\n", false},
		{"swa_bounded_replay: false\nuse_v2_model_runner: true\n", false},
		{"swa_bounded_replay: false\nuse_v2_model_runner: true\n" + fmt.Sprintf(layers, ", bounded_replay: true"), true},
	} {
		cfg, err := decodeConfig(block(t, c.extra))
		if err != nil {
			t.Fatal(err)
		}
		_, err = resolve(cfg)
		if c.refuse != (err != nil) {
			t.Errorf("%q: resolve = %v, want refused %t", strings.ReplaceAll(c.extra, "\n", " "), err, c.refuse)
		}
	}
}
