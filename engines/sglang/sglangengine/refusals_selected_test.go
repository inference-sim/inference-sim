// Original to blis-sglang.

package sglangengine

import (
	"sort"
	"strings"
	"testing"
)

// selector is what a scenario states to select one refused surface item: ServerArgs settings, an
// environment, or the architecture the model resolves to. why cites the SGLang source (at
// 297e54fd) that makes the setting the one that selects the item.
type selector struct {
	args, env map[string]any
	arch      string
	// archs, when set, are the architectures that each select the item: every one is refused.
	archs []string
	// names are the settings (or architecture) the refusal must name: a refusal for another
	// reason does not hold the item. Empty means every key of args and env, or arch.
	names []string
	why   string
}

func (s selector) archList() []string {
	if len(s.archs) > 0 {
		return s.archs
	}
	return []string{s.arch}
}

func (s selector) mentions(arch string) []string {
	if len(s.names) > 0 {
		return s.names
	}
	var out []string
	for k := range s.args {
		out = append(out, k)
	}
	for k := range s.env {
		out = append(out, k)
	}
	if arch != "" {
		out = append(out, arch)
	}
	return out
}

func kvEvents(cfg string) selector {
	return selector{args: map[string]any{"kv_events_config": cfg},
		why: "scheduler.py:582-589, kv_events_publisher.py:60: enable_kv_cache_events is bool(kv_events_config)"}
}

func specAlgo(name string, more map[string]any) selector {
	a := map[string]any{"speculative_algorithm": name, "speculative_num_steps": 3, "speculative_eagle_topk": 1,
		"speculative_num_draft_tokens": 4}
	for k, v := range more {
		a[k] = v
	}
	return selector{args: a, names: []string{"speculative_algorithm"},
		why: "spec_info.py: the worker class is chosen by the algorithm (the item's condition)"}
}

func setting(why, name string, v any) selector {
	return selector{args: map[string]any{name: v}, why: why}
}

func envVar(why, name, v string) selector {
	return selector{env: map[string]any{name: v}, why: why}
}

// The scheduler's event loop is chosen by the disaggregation mode, pp_size and the overlap switch
// (scheduler.py:6225-6247, the conditions the surface list records); overlap is on unless
// disable_overlap_schedule is set (scheduler.py:500).
var schedulerLoopSelectors = map[string]selector{
	"event_loop_pdmux":                  setting("scheduler.py:6225", "enable_pdmux", true),
	"event_loop_pp":                     setting("scheduler.py:6227", "pp_size", 2),
	"event_loop_normal":                 setting("scheduler.py:6233: not enable_overlap", "disable_overlap_schedule", true),
	"event_loop_pp_disagg_prefill":      disagg("prefill", map[string]any{"pp_size": 2}, "pp_size"),
	"event_loop_overlap_disagg_prefill": disagg("prefill", nil, "disaggregation"),
	"event_loop_normal_disagg_prefill":  disagg("prefill", map[string]any{"disable_overlap_schedule": true}, "disable_overlap_schedule"),
	"event_loop_pp_disagg_decode":       disagg("decode", map[string]any{"pp_size": 2}, "pp_size"),
	"event_loop_overlap_disagg_decode":  disagg("decode", nil, "disaggregation"),
	"event_loop_normal_disagg_decode":   disagg("decode", map[string]any{"disable_overlap_schedule": true}, "disable_overlap_schedule"),
}

func disagg(mode string, more map[string]any, names ...string) selector {
	a := map[string]any{"disaggregation_mode": mode}
	for k, v := range more {
		a[k] = v
	}
	return selector{args: a, names: names, why: "scheduler.py:6236-6247: the loop is chosen by disaggregation_mode"}
}

// The speculative worker class, and the max_running_requests default each handler declares, are
// chosen by the algorithm (spec_info.py:324-369; speculative_hook.py:439-1295).
var specAlgorithms = map[string]string{
	"DFlashWorkerV2": "DFLASH", "UnoWorkerV2": "UNO", "DSparkWorkerV2": "DSPARK",
	"FrozenKVMTPWorkerV2": "FROZEN_KV_MTP", "EAGLEWorkerV2": "EAGLE", "StandaloneWorkerV2": "STANDALONE",
	"NGRAMWorker":    "NGRAM",
	"_handle_dflash": "DFLASH", "_handle_dspark": "DSPARK", "_handle_frozen_kv_mtp": "FROZEN_KV_MTP",
	"_handle_eagle_family": "EAGLE", "_handle_ngram": "NGRAM",
}

// An extension point is how an external package adds a value to a setting's choices, or registers a
// class: its item is selected by a value only that package could have registered (the value below),
// stated in the setting the point extends (choices.py:215-283; the registries named by the site).
const extValue = "plugin-registered-elsewhere"

var extensionSelectors = map[string]selector{
	"add_load_format_choices":             setting("choices.py:215", "load_format", extValue),
	"add_quantization_method_choices":     setting("choices.py:221", "quantization", extValue),
	"add_attention_backend_choices":       setting("choices.py:225", "attention_backend", extValue),
	"add_draft_attention_backend_choices": setting("choices.py:227", "speculative_draft_attention_backend", extValue),
	"add_disagg_transfer_backend_choices": {args: map[string]any{"disaggregation_mode": "prefill",
		"disaggregation_transfer_backend": extValue}, names: []string{"disaggregation_transfer_backend"}, why: "choices.py:256"},
	"add_grammar_backend_choices":            setting("choices.py:260", "grammar_backend", extValue),
	"add_moe_runner_backend_choices":         setting("choices.py:265", "moe_runner_backend", extValue),
	"add_fp8_gemm_runner_backend_choices":    setting("choices.py:269", "fp8_gemm_runner_backend", extValue),
	"add_fp4_gemm_runner_backend_choices":    setting("choices.py:271", "fp4_gemm_runner_backend", extValue),
	"add_radix_eviction_policy_choices":      setting("choices.py:275", "radix_eviction_policy", extValue),
	"add_rl_on_policy_target_choices":        setting("choices.py:279", "rl_on_policy_target", extValue),
	"add_linear_attn_kernel_backend_choices": setting("choices.py:283", "linear_attn_backend", extValue),
	"register_radix_cache_backend":           setting("registry.py:58: --radix-cache-backend names the registered factory", "radix_cache_backend", extValue),
	"register_tree_core_backend":             envVar("tree_core_registry.py:134; environ.py:741", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", extValue),
	"register_algorithm":                     specAlgo(extValue, nil),
	"register_publisher":                     kvEvents(`{"publisher":"` + extValue + `"}`),
	// A resolution hook and a platform class are registered by a plugin the entry-point loader or
	// SGLANG_PLATFORM names (plugins/__init__.py:10,51; platforms/__init__.py:108).
	"register_resolution_hook":                     envVar("plugins/__init__.py:51", "SGLANG_PLUGINS", extValue),
	"SRTPlatform.get_graph_runner_cls":             envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_full_graph_backend_cls":       envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_kv_pool_cls":                  envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_paged_allocator_cls":          envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_piecewise_backend_cls":        envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_speculative_worker_cls":       envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
	"SRTPlatform.get_speculative_graph_runner_cls": envVar("platforms/__init__.py:108", "SGLANG_PLATFORM", extValue),
}

// The kv-event hashes the events path computes (mem_cache/events.py:30-36,175-296); the storage
// hashes of the other names also serve HiCache, and are not selected by kv_events_config alone.
var kvEventHashes = map[string]bool{"compute_node_event_hash_values": true, "kv_event_namespace_seed": true,
	"namespaced_block_hash": true, "_NAMESPACE_SEED_TAG": true, "hash_str_to_int64": true}

// The metric families the scheduler and the tokenizer collect. SchedulerMetricsCollector.init_new
// builds its collector only if enable_metrics (metrics_collector.py:1375) and the tokenizer manager
// builds its own under `if self.enable_metrics` (tokenizer_manager.py:837-841). The two classes are
// the line ranges of metrics_collector.py at the pin (334-1821, 1822-2196).
func metricsGated(it surfaceItem) bool {
	_, ln, ok := strings.Cut(it.Site, "metrics_collector.py:")
	if !ok {
		return false
	}
	n := 0
	for _, c := range ln {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
	}
	return n >= 334 && n < 2197
}

// selectedBy returns the selector of a refused surface item that stateItem does not cover and that
// a ServerArgs setting, an environment variable or the model's architecture selects.
func selectedBy(it surfaceItem) (selector, bool) {
	switch it.Category {
	case "kv_event_type", "kv_event_medium", "kv_event_merge", "kv_event_granularity":
		return kvEvents(`{"publisher":"zmq"}`), true
	case "kv_event_publisher":
		return kvEvents(`{"publisher":"` + it.Name + `"}`), true
	case "kv_event_format":
		if it.Name == "dynamo" {
			return kvEvents(`{"publisher":"zmq","format":"dynamo"}`), true
		}
		return kvEvents(`{"publisher":"zmq","format":"default"}`), true
	case "kv_event_hash":
		return kvEvents(`{"publisher":"zmq"}`), kvEventHashes[it.Name]
	case "scheduler_loop":
		s, ok := schedulerLoopSelectors[it.Name]
		return s, ok
	case "spec_worker", "spec_running_requests":
		if a, ok := specAlgorithms[it.Name]; ok {
			return specAlgo(a, nil), true
		}
		if it.Name == "MultiLayerEagleWorkerV2" {
			return specAlgo("EAGLE", map[string]any{"enable_multi_layer_eagle": true}), true
		}
	case "extension_point":
		s, ok := extensionSelectors[it.Name]
		return s, ok
	case "metric_family":
		return setting("metrics_collector.py:1375; tokenizer_manager.py:837", "enable_metrics", true), metricsGated(it)
	case "tree_cache":
		switch it.Name {
		case "FlexKVRadixCache":
			return setting("registry.py:128-135", "enable_flexkv", true), true
		case "LMCacheUnifiedRadixCache":
			return setting("registry.py:101-116", "enable_lmcache", true), true
		}
	case "tree_attachment":
		if it.Name == "init_hicache" {
			s := setting("registry.py:196", "enable_hierarchical_cache", true)
			s.names = []string{"HiCache"}
			return s, true
		}
		return selector{args: map[string]any{"enable_unified_cache_external_linker": true,
			"unified_cache_external_linker_backend": extValue}, why: "registry.py:206"}, true
	case "framework":
		switch it.Name {
		case "HiCacheController", "LayerDoneCounter", "L2TransferEngine":
			s := setting("cache_controller.py:359,379: built by the HiCache controller", "enable_hierarchical_cache", true)
			s.names = []string{"HiCache"}
			return s, true
		}
	case "host_pool":
		switch it.Name {
		case "HiSparseMHATokenToKVPoolHost":
			return setting("hisparse_coordinator.py:603", "enable_hisparse", true), true
		}
	case "engine_setting":
		switch it.Name {
		case "schedule_conservativeness":
			return setting("environ/schedule: the setting itself", it.Name, 0.5), true
		case "disable_overlap_schedule":
			return setting("the setting itself", it.Name, true), true
		case "kv_events_config":
			return kvEvents(`{"publisher":"zmq"}`), true
		case "SGLANG_MAX_NEW_TOKENS_LIMIT":
			return envVar("the variable itself", it.Name, "100"), true
		case "SGLANG_DISABLE_OVERLAP_AFTER_PREFILL":
			return envVar("scheduler.py:2327: the condition's variable", it.Name, "1"), true
		case "SGLANG_HICACHE_NIXL_BACKEND_PLUGIN":
			return envVar("the variable itself", it.Name, "UCX"), true
		}
	case "overlap_disabler":
		if it.Name == "process_after_prefill_early" {
			return envVar("scheduler.py:2327: the condition's variable", "SGLANG_DISABLE_OVERLAP_AFTER_PREFILL", "1"), true
		}
	case "kv_pool":
		switch it.Name {
		case "MHATokenToKVPoolFP4", "MLATokenToKVPoolFP4":
			return setting("the pool class of kv_cache_dtype fp4_e2m1", "kv_cache_dtype", "fp4_e2m1"), true
		case "MHATokenToKVPoolMXFP8":
			return setting("the pool class of kv_cache_dtype mxfp8", "kv_cache_dtype", "mxfp8"), true
		case "HiSparseMHAMainPool", "HiSparseC4DevicePool", "HiSparseDSATokenToKVPool":
			return setting("the HiSparse pools", "enable_hisparse", true), true
		}
	case "allocator":
		if it.Name == "HiSparseTokenToKVPoolAllocator" {
			return setting("the HiSparse allocator", "enable_hisparse", true), true
		}
	case "transformers_fallback":
		a := it.Name
		if a == "auto" { // the route of an architecture SGLang has no class for
			a = "UnregisteredForCausalLM"
		}
		return selector{arch: a, why: "the architecture SGLang's config translation yields, read by acceptedModels"}, true
	case "drafter_arch":
		if _, ok := runDrafters[it.Name]; ok || len(it.TargetArchs) == 0 {
			return selector{}, false
		}
		algo := "NEXTN"
		if strings.HasSuffix(it.Name, "DSpark") {
			algo = "DSPARK"
		}
		s := specAlgo(algo, nil)
		s.archs, s.names = it.TargetArchs, nil
		s.why = "model_config.py: the drafter is chosen by the target architecture under speculative decoding"
		return s, true
	}
	return selector{}, false
}

// The drafters whose target architecture is accepted and whose MTP draft ships in the target's
// checkpoint: a scenario with the NEXTN algorithm starts, so no refusal holds them.
var runDrafters = map[string]string{
	"DeepseekV3ForCausalLMNextN": "DeepseekV3ForCausalLM",
	"GlmMoeDsaForCausalLMNextN":  "GlmMoeDsaForCausalLM",
	// Qwen3.5's NextN draft layer ships in the target checkpoint and the draft path stays unset
	// (config/speculative.go eagleDraftPathless); a pool that runs it states disable_radix_cache.
	"Qwen3_5ForCausalLMMTP": "Qwen3_5MoeForConditionalGeneration",
}

// deferral records a refused surface item that cannot be refused at start-up by a one-setting
// scenario, and where it is refused or accepted instead. The maintainer decides each (the list is
// DEFERRED.md's); accepted, when set, is a scenario that starts, the proof that the engine runs it.
type deferral struct {
	kind     string // accepted, pool-class, model, unselected, gated
	where    string
	accepted *selector
}

func deferredBy(it surfaceItem) (deferral, bool) {
	switch it.Category {
	case "kv_pool", "allocator", "req_pool", "host_pool":
		return deferral{"pool-class", "core/cache/refuse.go CheckPoolSelection refuses the class in the library; start-up does not call it (DeepSeekV4 and Qwen3.5 are accepted architectures that select pool classes this list refuses)", nil}, true
	}
	if it.Name == "event_loop_overlap_mlx" || it.Name == "enable_overlap_mlx" {
		return deferral{"unselected", "selected by use_mlx(), the platform, not a ServerArgs setting (enable_overlap_mlx is not a ServerArgs field)", nil}, true
	}
	switch it.Category {
	case "drafter_arch":
		if arch, ok := runDrafters[it.Name]; ok {
			return deferral{"accepted", arch + " is an accepted architecture whose MTP draft ships in its own checkpoint (config/speculative.go eagleDraftInTarget): Resolve models its speculative decoding",
				&selector{arch: arch, args: specAlgo("NEXTN", nil).args}}, true
		}
		return deferral{"unselected", "selected by a list of architectures (MIMO_V2_MODEL_ARCHS), not one the allowlist names", nil}, true
	case "scheduler_loop", "overlap_switch", "overlap_depth":
		return deferral{"accepted", "overlap is the engine's default scheduling loop (scheduler.py:500,6231)", &selector{}}, true
	case "memory_budget":
		return deferral{"accepted", "the engine budgets memory (kv_cache_configurator.py:3007-3060) with mem_fraction_static",
			&selector{args: map[string]any{"mem_fraction_static": 0.8}}}, true
	case "tree_cache":
		return deferral{"accepted", "RadixCache is the default tree (registry.py); the other classes are abstract or chosen by the model", &selector{}}, true
	case "engine_setting":
		var sel *selector
		switch it.Name {
		case "context_length":
			sel = &selector{args: map[string]any{"context_length": 4096}}
		case "mem_fraction_static":
			sel = &selector{args: map[string]any{"mem_fraction_static": 0.8}}
		case "page_size":
			sel = &selector{args: map[string]any{"page_size": 16}}
		case "SGLANG_INIT_NEW_TOKEN_RATIO", "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", "SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS",
			"SGLANG_RETRACT_DECODE_STEPS", "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION":
			sel = &selector{env: map[string]any{it.Name: "5"}}
		case "SGLANG_HICACHE_NIXL_BACKEND_PLUGIN":
			return deferral{}, false
		default:
			return deferral{}, false
		}
		return deferral{"accepted", "Resolve and the engine model this setting; a value that starts is the proof", sel}, true
	case "tree_component", "tree_core_divergence", "server_args_group", "framework", "overlap_disabler",
		"spec_worker", "extension_point", "tree_core", "kv_event_hash", "metric_family":
		return deferral{"unselected", "no single ServerArgs setting or environment variable selects the item in the source: " + it.Category, nil}, true
	}
	return deferral{}, false
}

// Every refused surface item is held to a start-up refusal, or deferred with the place it is refused
// or accepted instead (SGL-6, inference-sim/inference-sim#2349). A new refused item that neither a
// selector nor a deferral names fails here.
func TestEveryRefusedSurfaceItemIsHeldOrDeferred(t *testing.T) {
	held, deferred := map[string]int{}, map[string]int{}
	var unnamed []string
	for _, it := range surfaceItems(t) {
		if it.Status != "refused" {
			continue
		}
		if _, _, ok := stateItem(it); ok {
			held["setting"]++
			continue
		}
		if _, ok := selectedBy(it); ok {
			held["selector"]++
			continue
		}
		if d, ok := deferredBy(it); ok {
			deferred[d.kind]++
			continue
		}
		unnamed = append(unnamed, it.ID)
	}
	if len(unnamed) > 0 {
		sort.Strings(unnamed)
		t.Errorf("%d refused surface items are neither held by a selector nor deferred: %v", len(unnamed), unnamed)
	}
	t.Logf("held %v, deferred %v", held, deferred)
}

// A refused surface item that a selector names is refused at start-up by SGLang's message or the
// open issue that models it, the refusal names the selecting setting, and the same scenario without
// it starts (SIM-9): a test that fails when the refusal is removed.
func TestSelectedSurfaceItemsAreRefusedAtStartUp(t *testing.T) {
	checked := 0
	var accepted []string
	for _, it := range surfaceItems(t) {
		if it.Status != "refused" {
			continue
		}
		if _, _, ok := stateItem(it); ok {
			continue
		}
		s, ok := selectedBy(it)
		if !ok {
			continue
		}
		checked++
		for _, arch := range s.archList() {
			err := startUpArch(t, arch, s.args, s.env)
			if err == nil {
				accepted = append(accepted, it.ID+"@"+arch)
				continue
			}
			if !isRefusal(err) {
				t.Errorf("%s: refused without SGLang's message or an issue: %v", it.ID, err)
			}
			named := false
			for _, n := range s.mentions(arch) {
				named = named || strings.Contains(err.Error(), n)
			}
			if !named {
				t.Errorf("%s: the refusal does not name %v (%s): %v", it.ID, s.mentions(arch), s.why, err)
			}
		}
	}
	if len(accepted) > 0 {
		t.Errorf("a scenario that selects these refused items starts: %v", accepted)
	}
	if checked < 150 {
		t.Fatalf("only %d selected surface items were held to a start-up refusal", checked)
	}
}

// The deferred items the engine runs start: that is the proof they cannot be refused at start-up.
func TestDeferredAcceptedItemsStart(t *testing.T) {
	n := 0
	for _, it := range surfaceItems(t) {
		if it.Status != "refused" {
			continue
		}
		if _, _, ok := stateItem(it); ok {
			continue
		}
		if _, ok := selectedBy(it); ok {
			continue
		}
		d, ok := deferredBy(it)
		if !ok || d.accepted == nil {
			continue
		}
		n++
		if err := startUpArch(t, d.accepted.arch, d.accepted.args, d.accepted.env); err != nil {
			t.Errorf("%s is deferred as accepted (%s) but a scenario that states it is refused: %v", it.ID, d.where, err)
		}
	}
	if n == 0 {
		t.Fatal("no deferred accepted item was checked")
	}
}
