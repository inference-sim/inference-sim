// Original to blis-sglang.

package sglangengine

import (
	"encoding/json"
	"errors"
	"os"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

type surfaceItem struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Name     string `json:"name"`
	Status   string `json:"status"`
	Issue    string `json:"issue"`
}

func surfaceItems(t *testing.T) []surfaceItem {
	t.Helper()
	raw, err := os.ReadFile("../surface.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Items []surfaceItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	return doc.Items
}

// startUp builds an engine from a ServerArgs block and an environment map, the way a scenario's
// engine block does, and returns the start-up error.
func startUp(t *testing.T, args map[string]any, env map[string]any) error {
	t.Helper()
	blk := map[string]any{"model_path": "m"}
	for k, v := range args {
		blk[k] = v
	}
	if env != nil {
		blk["env"] = env
	}
	b, err := yaml.Marshal(blk)
	if err != nil {
		t.Fatal(err)
	}
	_, err = FactoryFor(newHost()).New(specOf(t, string(b)))
	return err
}

var issueRef = regexp.MustCompile(`inference-sim/(?:blis-sglang|inference-sim)(#|/issues/)[0-9]+`)

// A start-up refusal is SGLang's own (an UpstreamRefusal) or names the open issue that models the
// item (SGL-6, X28).
func isRefusal(err error) bool {
	var up *config.UpstreamRefusal
	return err != nil && (errors.As(err, &up) || strings.Contains(err.Error(), "SGLang refuses this configuration") ||
		issueRef.MatchString(err.Error()))
}

// stateItem returns the ServerArgs settings and environment that select a surface item, or false
// for a category no single setting selects (a class SGLang's code picks by the model, a metric, a
// scheduler loop, a stage of the pipeline).
func stateItem(it surfaceItem) (args map[string]any, env map[string]any, ok bool) {
	args = map[string]any{}
	switch it.Category {
	case "transfer_backend":
		args["disaggregation_mode"], args["disaggregation_transfer_backend"] = "prefill", it.Name
	case "hicache_storage":
		args["enable_hierarchical_cache"], args["hicache_storage_backend"] = true, it.Name
	case "hicache_option":
		f, v, _ := strings.Cut(it.Name, "=")
		args["enable_hierarchical_cache"], args[f] = true, v
	case "moe_a2a_backend":
		args["moe_a2a_backend"] = it.Name
	case "pd_allocation_policy":
		args["disaggregation_decode_allocation_policy"] = it.Name
	case "spec_algo":
		args["speculative_algorithm"], args["speculative_num_steps"] = it.Name, 3
		args["speculative_eagle_topk"], args["speculative_num_draft_tokens"] = 1, 4
	case "eviction_policy":
		args["radix_eviction_policy"] = it.Name
	case "schedule_policy":
		args["schedule_policy"] = it.Name
	case "tree_core":
		env = map[string]any{"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": it.Name}
	case "nixl_plugin":
		env = map[string]any{"SGLANG_HICACHE_NIXL_BACKEND_PLUGIN": it.Name}
	case "unified_cache_linker":
		args["enable_unified_cache_external_linker"], args["unified_cache_external_linker_backend"] = true, it.Name
	case "tree_cache_backend":
		args["radix_cache_backend"] = it.Name
	default:
		return nil, nil, false
	}
	return args, env, true
}

// The surface items the list calls refused but whose stated value is the value SGLang's own
// default or a value Resolve models, so a scenario that states it starts. The list cannot tell
// them from a refusal until the dispatch from a surface item to the value that selects it is
// generated (inference-sim/inference-sim#2348); until then this list is the ratchet: an item that starts being
// refused must leave it, and an item that joins it is a regression.
var listedRefusedButAccepted = map[string]string{
	"eviction_policy:lru":        "radix_eviction_policy lru is the default",
	"moe_a2a_backend:none":       "moe_a2a_backend none is the default",
	"moe_a2a_backend:deepep":     "Resolve models deepep",
	"pd_allocation_policy:early": "disaggregation_decode_allocation_policy early is the default",
}

// Every refused surface item that one setting or environment variable selects is refused at
// start-up, through the engine factory, by SGLang's own message or the open issue that models it,
// and none starts: a test that fails if the refusal is removed (SGL-6, inference-sim/inference-sim#2349).
func TestEverySettingSelectedSurfaceItemIsRefusedAtStartUp(t *testing.T) {
	checked, accepted := 0, map[string]bool{}
	for _, it := range surfaceItems(t) {
		if it.Status != "refused" {
			continue
		}
		args, env, ok := stateItem(it)
		if !ok {
			continue
		}
		err := startUp(t, args, env)
		if err == nil {
			accepted[it.ID] = true
			continue
		}
		checked++
		if !isRefusal(err) {
			t.Errorf("%s: refused without SGLang's message or an issue: %v", it.ID, err)
		}
	}
	for id, why := range listedRefusedButAccepted {
		if !accepted[id] {
			t.Errorf("%s (%s) is now refused or no longer listed: take it off listedRefusedButAccepted", id, why)
		}
		delete(accepted, id)
	}
	for id := range accepted {
		t.Errorf("%s: the list says refused, but a scenario that states it starts", id)
	}
	if checked < 70 {
		t.Fatalf("only %d surface items were held to a start-up refusal", checked)
	}
}

// Items an externally loaded class, file or registration selects are refused, never loaded: the
// engine runs no Python, so a value that only exists after an extension registers it is not one
// of the values it models (SGL-6, inference-sim/inference-sim#2460).
func TestExternallyLoadedItemsAreRefused(t *testing.T) {
	const ext = "plugin-registered-elsewhere"
	for _, tc := range []struct {
		name string
		args map[string]any
		env  map[string]any
	}{
		{"dynamic hicache storage", map[string]any{"enable_hierarchical_cache": true, "hicache_storage_backend": "dynamic",
			"hicache_storage_backend_extra_config": `{"backend_name":"x","module_path":"m","class_name":"C"}`}, nil},
		{"dynamic unified cache linker", map[string]any{"enable_unified_cache_external_linker": true,
			"unified_cache_external_linker_backend": "dynamic"}, nil},
		{"customized a2a dispatcher", map[string]any{"moe_a2a_backend": "customized"}, nil},
		{"registered tree cache", map[string]any{"radix_cache_backend": ext}, nil},
		{"registered tree core", nil, map[string]any{"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": ext}},
		{"registered speculative algorithm", map[string]any{"speculative_algorithm": ext, "speculative_num_steps": 3,
			"speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4}, nil},
		{"registered kv-events publisher", map[string]any{"kv_events_config": `{"publisher":"` + ext + `"}`}, nil},
		{"registered load format", map[string]any{"load_format": ext}, nil},
		{"registered quantization method", map[string]any{"quantization": ext}, nil},
		{"registered attention backend", map[string]any{"attention_backend": ext}, nil},
		{"registered draft attention backend", map[string]any{"speculative_draft_attention_backend": ext}, nil},
		{"registered grammar backend", map[string]any{"grammar_backend": ext}, nil},
		{"registered moe runner backend", map[string]any{"moe_runner_backend": ext}, nil},
		{"registered fp8 gemm runner backend", map[string]any{"fp8_gemm_runner_backend": ext}, nil},
		{"registered fp4 gemm runner backend", map[string]any{"fp4_gemm_runner_backend": ext}, nil},
		{"registered radix eviction policy", map[string]any{"radix_eviction_policy": ext}, nil},
		{"registered rl on-policy target", map[string]any{"rl_on_policy_target": ext}, nil},
		{"registered linear attention backend", map[string]any{"linear_attn_backend": ext}, nil},
		{"registered disaggregation transfer backend", map[string]any{"disaggregation_mode": "prefill",
			"disaggregation_transfer_backend": ext}, nil},
		{"a custom weight loader", map[string]any{"custom_weight_loader": []string{"pkg.loader"}}, nil},
		{"plugins loaded by entry point", nil, map[string]any{"SGLANG_PLUGINS": ext}},
		{"an external model package", nil, map[string]any{"SGLANG_EXTERNAL_MODEL_PACKAGE": ext}},
		{"an external multimodal model architecture", nil, map[string]any{"SGLANG_EXTERNAL_MM_MODEL_ARCH": ext}},
		{"an external multimodal processor package", nil, map[string]any{"SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": ext}},
		{"a custom sigquit handler", map[string]any{"custom_sigquit_handler": ext}, nil},
	} {
		err := startUp(t, tc.args, tc.env)
		if err == nil || !isRefusal(err) {
			t.Errorf("%s: got %v, want a refusal by SGLang's message or an issue", tc.name, err)
		}
	}
}

// The partner of the table above: the same engine with none of them set starts, so the refusals are
// the loaded items', not the start-up path's.
func TestEngineStartsWithNoExternallyLoadedItem(t *testing.T) {
	if err := startUp(t, nil, nil); err != nil {
		t.Fatalf("start-up refused with nothing external set: %v", err)
	}
}

// SGLang's own refusals, quoted from the source at the pin, are raised at start-up with SGLang's
// message; each has an accepted partner (SIM-9). The quotes are checked against the source by
// oracle/check_refusal_quotes.py.
func TestSGLangsOwnRefusalsAreRaisedWithItsMessage(t *testing.T) {
	cases := []struct {
		name    string
		args    map[string]any
		want    string
		partner map[string]any
	}{
		{"an unknown policy (schedule_policy.py:340-344)", map[string]any{"schedule_policy": "nope"},
			"Unknown schedule_policy: policy='nope'", map[string]any{"schedule_policy": "lpm"}},
		{"shortest-prefill-first without a prefix cache (schedule_policy.py:332-336, caught at :340, raised at :344)",
			map[string]any{"schedule_policy": "shortest-prefill-first", "disable_radix_cache": true},
			"Unknown schedule_policy: policy='shortest-prefill-first'",
			map[string]any{"schedule_policy": "shortest-prefill-first"}},
		{"tlru off the unified tree (registry.py:261-271)",
			map[string]any{"radix_eviction_policy": "tlru", "enable_flexkv": true},
			"--radix-eviction-policy tlru requires UnifiedRadixCache, but tree_cache is FlexKVRadixCache. " +
				"Drop the flag or the option that selected another tree cache for this model.",
			map[string]any{"radix_eviction_policy": "tlru"}},
		{"priority scheduling over lpm (validation_hook.py:check_server_args)",
			map[string]any{"enable_priority_scheduling": true, "schedule_policy": "lpm"},
			"To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. 'lpm' is not supported.",
			map[string]any{"enable_priority_scheduling": true, "schedule_policy": "lof"}},
		{"priority retraction without priority scheduling (validation_hook.py:check_server_args)",
			map[string]any{"retraction_policy": "priority"},
			"--retraction-policy priority requires --enable-priority-scheduling",
			map[string]any{"retraction_policy": "priority", "enable_priority_scheduling": true}},
	}
	for _, tc := range cases {
		err := startUp(t, tc.args, nil)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want SGLang's message %q", tc.name, err, tc.want)
		}
		if err := startUp(t, tc.partner, nil); err != nil && strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: the partner %v is refused with the same message: %v", tc.name, tc.partner, err)
		}
	}
}

// tlru is a policy SGLang accepts on the unified tree, so alone it is not SGLang's refusal but the
// port's not modelling it: the message differs from the pair's, which is SGLang's.
func TestTlruAloneIsNotTheUpstreamRefusal(t *testing.T) {
	err := startUp(t, map[string]any{"radix_eviction_policy": "tlru"}, nil)
	var nm *config.NotModelled
	if !errors.As(err, &nm) || nm.Name != "radix_eviction_policy" {
		t.Fatalf("got %v, want a NotModelled naming radix_eviction_policy", err)
	}
	err = startUp(t, map[string]any{"radix_eviction_policy": "tlru", "enable_flexkv": true, "disable_radix_cache": true}, nil)
	var up *config.UpstreamRefusal
	if errors.As(err, &up) {
		t.Errorf("with the radix cache disabled FlexKV is not selected (registry.py:91-99), yet %v", err)
	}
}
