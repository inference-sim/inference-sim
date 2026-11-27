// Original to blis-sglang.

package config

import (
	"encoding/json"
	"errors"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

// answers is config/testdata/serverargs.json: SGLang's own resolution of each case of
// oracle/serverargs/cases.json (oracle/run_serverargs.py), with each case's inputs.
type answers struct {
	Cases []struct {
		Name       string            `json:"name"`
		Args       map[string]any    `json:"args"`
		Chip       Chip              `json:"chip"`
		ModelFacts ModelFacts        `json:"model_facts"`
		Resolved   map[string]any    `json:"resolved"`
		DeclaredBy map[string]string `json:"declared_by"`
		Refused    *struct {
			Message string `json:"message"`
		} `json:"refused"`
		RefusedAt string `json:"refused_at"`
	} `json:"cases"`
}

func readAnswers(t *testing.T) answers {
	t.Helper()
	b, err := testgz.ReadFile("testdata/serverargs.json")
	if err != nil {
		t.Fatal(err)
	}
	var a answers
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Cases) == 0 {
		t.Fatal("testdata/serverargs.json holds no case")
	}
	return a
}

func canon(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// TestResolveAgainstSGLang: every case resolves as SGLang v0.5.23's own pipeline resolves it,
// field by field over every field SGLang records, with the step that decided each changed field;
// a case SGLang refuses is refused with SGLang's message, raised in the same function.
func TestResolveAgainstSGLang(t *testing.T) {
	for _, c := range readAnswers(t).Cases {
		t.Run(c.Name, func(t *testing.T) {
			args := map[string]any{"model_path": "<model>"}
			for k, v := range c.Args {
				args[k] = v
			}
			got, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
			if c.Refused != nil {
				var up *UpstreamRefusal
				if !errors.As(err, &up) {
					t.Fatalf("SGLang refuses this case (%q at %s); Resolve returned %v", c.Refused.Message, c.RefusedAt, err)
				}
				if up.Message != c.Refused.Message || up.Site != c.RefusedAt {
					t.Errorf("refusal %q at %s, SGLang's is %q at %s", up.Message, up.Site, c.Refused.Message, c.RefusedAt)
				}
				return
			}
			if err != nil {
				t.Fatalf("SGLang accepts this case; Resolve refused it: %v", err)
			}
			if len(c.Resolved) != len(FieldNames())+4 {
				t.Errorf("SGLang records %d fields, ServerArgs in Go has %d and 4 internal ones", len(c.Resolved), len(FieldNames()))
			}
			names := make([]string, 0, len(c.Resolved))
			for n := range c.Resolved {
				names = append(names, n)
			}
			sort.Strings(names)
			for _, name := range names {
				want := c.Resolved[name]
				v, ok := got.Value(name)
				if !ok {
					t.Errorf("%s: SGLang records it; ServerArgs in Go has no such field", name)
					continue
				}
				if name == "random_seed" && want == "<random>" {
					if !got.RandomSeedDrawn || v != nil {
						t.Errorf("random_seed: SGLang draws it; Resolve gives %v, drawn=%v", v, got.RandomSeedDrawn)
					}
					continue
				}
				if canon(v) != canon(want) {
					t.Errorf("%s = %s, SGLang resolves %s", name, canon(v), canon(want))
				}
			}
			for name, src := range c.DeclaredBy {
				if strings.HasPrefix(name, "_") || name == "random_seed" {
					continue
				}
				if got.DeclaredBy[name] != src {
					t.Errorf("%s is decided by %q, SGLang's step is %q", name, got.DeclaredBy[name], src)
				}
			}
			for name, src := range got.DeclaredBy {
				if _, ok := c.DeclaredBy[name]; !ok {
					t.Errorf("%s is decided by %q, which SGLang's resolution leaves at its input", name, src)
				}
			}
		})
	}
}

// TestResolveRefusesAnUnknownFieldByItsName: a name that is not a ServerArgs field is refused,
// and the refusal names it (X1). The nearest accepted input resolves.
func TestResolveRefusesAnUnknownFieldByItsName(t *testing.T) {
	c := readAnswers(t).Cases[0]
	in := Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts}
	if _, err := Resolve(in); err != nil {
		t.Fatalf("the nearest accepted input: %v", err)
	}
	in.Args["max_num_seqs"] = 8 // vLLM's name for SGLang's max_running_requests
	_, err := Resolve(in)
	var u *UnknownField
	if !errors.As(err, &u) || u.Name != "max_num_seqs" || !strings.Contains(err.Error(), "max_num_seqs") {
		t.Fatalf("got %v, want an UnknownField naming max_num_seqs", err)
	}
}

// TestResolveRefusesWhatItDoesNotModelByName: a ServerArgs field outside the modelled input
// space, set away from its default, is refused naming the field and an open issue; the same
// field at its default resolves.
func TestResolveRefusesWhatItDoesNotModelByName(t *testing.T) {
	c := readAnswers(t).Cases[0]
	for _, tc := range []struct {
		field string
		value any
		issue string
	}{
		{"attn_cp_size", 2, "inference-sim/inference-sim#2448"},
		{"pp_size", 2, "inference-sim/inference-sim#2358"},
		{"disaggregation_transfer_backend", "mori", "inference-sim/inference-sim#2353"},
	} {
		in := Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts}
		def := Defaults()
		in.Args[tc.field] = def.value(tc.field)
		if _, err := Resolve(in); err != nil {
			t.Errorf("%s at its default: %v", tc.field, err)
		}
		in.Args[tc.field] = tc.value
		_, err := Resolve(in)
		var nm *NotModelled
		if !errors.As(err, &nm) || nm.Name != tc.field || nm.Issue != tc.issue {
			t.Errorf("%s=%v: got %v, want NotModelled naming it and %s", tc.field, tc.value, err, tc.issue)
		}
	}
	// P/D over the default transfer backend, mooncake, is refused citing its own issue.
	in := Input{Args: map[string]any{"model_path": "<model>", "disaggregation_mode": "prefill"}, Chip: c.Chip, Model: c.ModelFacts}
	_, err := Resolve(in)
	var nm *NotModelled
	if !errors.As(err, &nm) || nm.Name != "disaggregation_transfer_backend" || nm.Issue != "inference-sim/inference-sim#2446" {
		t.Errorf("P/D over mooncake: got %v, want NotModelled naming disaggregation_transfer_backend and #2446", err)
	}
}

// A setting of a feature the engine does not model is refused naming the open issue that models that
// feature, not the general one: the stage issue where one exists (X1, X28). The values are the ones
// SGLang's help text and choices name (python/sglang/srt/arg_groups/fields/*.py@297e54fd).
func TestResolveRefusalsNameTheIssueOfTheFeature(t *testing.T) {
	c := readAnswers(t).Cases[0]
	const repo = "inference-sim/inference-sim#"
	for _, tc := range []struct {
		field string
		value any
		issue string
	}{
		{"disaggregation_decode_allocation_policy", "prefill_complete", "2351"},
		{"disaggregation_decode_polling_interval", 4, "2351"},
		{"disaggregation_decode_retraction_backup", "host_pool", "2352"},
		{"disaggregation_decode_enable_offload_kvcache", true, "2352"},
		{"disaggregation_ib_device", "mlx5_0", "2353"},
		{"hicache_write_policy", "write_back", "2341"},
		{"hicache_storage_prefetch_policy", "wait_complete", "2342"},
		{"hicache_storage_backend", "nixl", "2342"},
		{"hicache_storage_backend", "mooncake", "2443"},
		{"hicache_storage_backend", "aibrix", "2445"},
		{"hicache_storage_backend", "tensorcast", "2445"},
		{"hicache_storage_backend_extra_config", "@storage.json", "2459"},
		{"load_balance_method", "minimum_tokens", "2356"},
		{"enable_dp_lm_head", true, "2356"},
		{"deepep_mode", "low_latency", "2357"},
		{"enable_eplb", true, "2357"},
		{"enable_two_batch_overlap", true, "2357"},
		{"moe_a2a_backend", "mooncake", "2450"},
		{"moe_a2a_backend", "mori", "2357"},
		{"pp_max_micro_batch_size", 2, "2358"},
		{"swa_full_tokens_ratio", 0.5, "2359"},
		{"mamba_track_interval", 128, "2366"},
		{"speculative_algorithm", "STANDALONE", "2460"},
		{"speculative_algorithm", "NGRAM", "2460"},
		{"speculative_algorithm", "DFLASH", "2460"},
		{"speculative_algorithm", "UNO", "2460"},
		{"speculative_algorithm", "FROZEN_KV_MTP", "2460"},
		{"speculative_algorithm", "DSPARK", "2454"},
		{"speculative_token_map", "map.json", "2355"},
		{"speculative_dspark_block_size", 8, "2454"},
		{"disaggregation_transfer_backend", "mooncake_tcp", "2446"},
	} {
		in := Input{Args: map[string]any{"model_path": "<model>", tc.field: tc.value}, Chip: c.Chip, Model: c.ModelFacts}
		_, err := Resolve(in)
		var nm *NotModelled
		if !errors.As(err, &nm) || nm.Name != tc.field || nm.Issue != repo+tc.issue {
			t.Errorf("%s=%v: got %v, want NotModelled naming it and %s%s", tc.field, tc.value, err, repo, tc.issue)
		}
	}
}

// The scheduler's queue, priority, truncation and retraction settings resolve as stated: no resolution
// step of SGLang v0.5.23 reads them (grep of python/sglang/srt/arg_groups@297e54fd finds them only in
// validation_hook.py:281-305, which CheckServerArgs ports, and kv_cache_hook.py:368, a decode-server rule).
func TestTheSchedulersSettingsResolveAsStated(t *testing.T) {
	c := readAnswers(t).Cases[0]
	base, err := Resolve(Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts})
	if err != nil {
		t.Fatal(err)
	}
	args := map[string]any{
		"model_path": "<model>", "enable_priority_scheduling": true,
		"priority_scheduling_preemption_threshold": 3, "schedule_low_priority_values_first": true,
		"abort_on_priority_when_disabled": true, "disable_priority_preemption": true, "max_queued_requests": 7, "prefill_max_requests": 2,
		"allow_auto_truncate": true, "retraction_policy": "priority",
	}
	got, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
	if err != nil {
		t.Fatal(err)
	}
	a := got.Args
	if !a.EnablePriorityScheduling || a.PrioritySchedulingPreemptionThreshold != 3 ||
		!a.ScheduleLowPriorityValuesFirst || !a.AbortOnPriorityWhenDisabled || !a.DisablePriorityPreemption || a.MaxQueuedRequests == nil || *a.MaxQueuedRequests != 7 ||
		a.PrefillMaxRequests == nil || *a.PrefillMaxRequests != 2 || !a.AllowAutoTruncate || a.RetractionPolicy != "priority" {
		t.Errorf("a stated setting did not resolve as stated: %+v", a)
	}
	if len(got.DeclaredBy) != len(base.DeclaredBy) {
		t.Errorf("stating the settings changes %d fields of the resolution, the unset one changes %d", len(got.DeclaredBy), len(base.DeclaredBy))
	}
	// The retraction policy has two choices (fields/schedule.py:136); another is refused by name.
	args["retraction_policy"] = "random"
	_, err = Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
	var nm *NotModelled
	if !errors.As(err, &nm) || nm.Name != "retraction_policy" {
		t.Errorf("retraction_policy random: got %v, want NotModelled naming it", err)
	}
}

// CheckServerArgs ports the two rules of check_server_args that read the priority settings
// (validation_hook.py:281-305@297e54fd), with SGLang's messages.
func TestCheckServerArgsRefusesWhatSGLangRefusesOfThePrioritySettings(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*ServerArgs)
		want string
	}{
		{"defaults", func(*ServerArgs) {}, ""},
		{"priority scheduling over fcfs", func(a *ServerArgs) { a.EnablePriorityScheduling = true }, ""},
		{"priority scheduling over lof", func(a *ServerArgs) { a.EnablePriorityScheduling, a.SchedulePolicy = true, "lof" }, ""},
		{"priority retraction with priority scheduling", func(a *ServerArgs) { a.EnablePriorityScheduling, a.RetractionPolicy = true, "priority" }, ""},
		{"disabled preemption without priority scheduling", func(a *ServerArgs) { a.DisablePriorityPreemption = true }, ""},
		{"priority scheduling over lpm", func(a *ServerArgs) { a.EnablePriorityScheduling, a.SchedulePolicy = true, "lpm" },
			"To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. 'lpm' is not supported."},
		{"priority retraction without priority scheduling", func(a *ServerArgs) { a.RetractionPolicy = "priority" },
			"--retraction-policy priority requires --enable-priority-scheduling"},
	} {
		a := Defaults()
		tc.set(&a)
		err := CheckServerArgs(&a)
		if tc.want == "" {
			if err != nil {
				t.Errorf("%s: %v", tc.name, err)
			}
			continue
		}
		var up *UpstreamRefusal
		if !errors.As(err, &up) || up.Message != tc.want || !strings.HasSuffix(up.Site, "validation_hook.py:check_server_args") {
			t.Errorf("%s: got %v, want SGLang's refusal %q", tc.name, err, tc.want)
		}
	}
}

// The pair --radix-eviction-policy tlru with enable_flexkv is one SGLang refuses at tree-cache
// creation (mem_cache/registry.py:261-271@297e54fd): FlexKVRadixCache is not a UnifiedRadixCache.
// Each member alone, and the pair where FlexKV is not the tree selected (the radix cache disabled,
// LMCache on, the unified cache's external linker on), is not that refusal: Resolve refuses the
// members as not modelled instead.
func TestResolveRefusesTlruWithFlexKVAsSGLangDoes(t *testing.T) {
	c := readAnswers(t).Cases[0]
	const want = "--radix-eviction-policy tlru requires UnifiedRadixCache, but tree_cache is FlexKVRadixCache. " +
		"Drop the flag or the option that selected another tree cache for this model."
	resolve := func(extra map[string]any) error {
		args := map[string]any{"model_path": "<model>"}
		for k, v := range extra {
			args[k] = v
		}
		_, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
		return err
	}
	var up *UpstreamRefusal
	err := resolve(map[string]any{"radix_eviction_policy": "tlru", "enable_flexkv": true})
	if !errors.As(err, &up) || up.Message != want || !strings.HasSuffix(up.Site, "registry.py:create_tree_cache") {
		t.Errorf("tlru with flexkv: got %v, want SGLang's refusal %q", err, want)
	}
	for name, extra := range map[string]map[string]any{
		"tlru alone":                   {"radix_eviction_policy": "tlru"},
		"flexkv alone":                 {"enable_flexkv": true},
		"the radix cache disabled":     {"radix_eviction_policy": "tlru", "enable_flexkv": true, "disable_radix_cache": true},
		"lmcache selected":             {"radix_eviction_policy": "tlru", "enable_flexkv": true, "enable_lmcache": true},
		"the external linker selected": {"radix_eviction_policy": "tlru", "enable_flexkv": true, "enable_unified_cache_external_linker": true},
	} {
		var nm *NotModelled
		if err := resolve(extra); errors.As(err, &up) || !errors.As(err, &nm) {
			t.Errorf("%s: got %v, want a NotModelled refusal, not SGLang's pair refusal", name, err)
		}
	}
}

// A server's record of its resolved arguments states custom_weight_loader as an empty list; SGLang's
// resolution gives an unset field the same value, so it is not a setting and resolves like the unset
// one. A real loader is still refused.
func TestAnEmptyCustomWeightLoaderIsTheUnsetOne(t *testing.T) {
	c := readAnswers(t).Cases[0]
	in := func(extra map[string]any) Input {
		args := map[string]any{"model_path": "<model>"}
		for k, v := range extra {
			args[k] = v
		}
		return Input{Args: args, Chip: c.Chip, Model: c.ModelFacts}
	}
	want, err := Resolve(in(nil))
	if err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(in(map[string]any{"custom_weight_loader": []any{}}))
	if err != nil {
		t.Fatalf("an empty custom_weight_loader was refused: %v", err)
	}
	if len(got.Args.CustomWeightLoader) != len(want.Args.CustomWeightLoader) || len(got.Args.CustomWeightLoader) != 0 {
		t.Errorf("resolved %v, the unset field resolves %v", got.Args.CustomWeightLoader, want.Args.CustomWeightLoader)
	}
	if _, err := Resolve(in(map[string]any{"custom_weight_loader": []any{"x.y"}})); err == nil {
		t.Error("a real custom weight loader was accepted")
	}
}

// attention_backend nsa was removed in SGLang 1.0 and is refused with the name of its successor
// dsa (validation_hook.py:991-1016@c4af57ff); a DSA model that states dsa resolves as one that
// states neither, apart from the field itself.
func TestResolveRefusesNSAAndTakesDSAAsTheDefaultBackend(t *testing.T) {
	for _, c := range readAnswers(t).Cases {
		if c.Name != "glm5-h200-tp8-recipe" {
			continue
		}
		resolve := func(backend any) *Resolved {
			args := map[string]any{"model_path": "<model>"}
			for k, v := range c.Args {
				args[k] = v
			}
			if backend != nil {
				args["attention_backend"] = backend
			}
			got, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
			if err != nil {
				t.Fatalf("attention_backend %v: %v", backend, err)
			}
			return got
		}
		def := resolve(nil)
		args := map[string]any{"model_path": "<model>", "attention_backend": "nsa"}
		for k, v := range c.Args {
			args[k] = v
		}
		args["attention_backend"] = "nsa"
		var up *UpstreamRefusal
		if _, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts}); !errors.As(err, &up) || !strings.Contains(up.Message, "use dsa") {
			t.Errorf("attention_backend nsa: got %v, want SGLang's refusal naming dsa", err)
		}
		for _, backend := range []string{"dsa"} {
			got := resolve(backend)
			if got.Args.DsaPrefillBackend == nil || def.Args.DsaPrefillBackend == nil ||
				*got.Args.DsaPrefillBackend != *def.Args.DsaPrefillBackend ||
				*got.Args.DsaDecodeBackend != *def.Args.DsaDecodeBackend ||
				got.Args.KvCacheDtype != def.Args.KvCacheDtype ||
				*got.Args.PageSize != *def.Args.PageSize {
				t.Errorf("attention_backend %s resolves the DSA kernels, cache dtype or page differently from the default", backend)
			}
			if got.Args.AttentionBackend == nil || *got.Args.AttentionBackend != backend {
				t.Errorf("attention_backend %s is kept as stated, got %v", backend, got.Args.AttentionBackend)
			}
		}
		return
	}
	t.Fatal("the glm5-h200-tp8-recipe case is not in the answers")
}

// A case name stands for one case: the answers are keyed by it, so a repeated name would be recorded
// once and counted twice.
func TestNoCaseNameOfTheOracleIsRepeated(t *testing.T) {
	b, err := os.ReadFile("../oracle/serverargs/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Name string `json:"name"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, c := range f.Cases {
		if seen[c.Name] {
			t.Errorf("the case %s is listed twice", c.Name)
		}
		seen[c.Name] = true
	}
}
