// Moved from inference-sim/blis-schemas/spec/deployment/fieldnames_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"reflect"
	"strings"
	"testing"
)

// The Engine struct's YAML names are the contract a deployment author writes
// against, so each must be the engine's own name for the setting rather than an
// invented one. An earlier draft carried three that no engine accepts — an
// allreduce_backend enum where the engine has a boolean, a cache-dtype set with four
// of seventeen members, and scheduler_policy where the flag is scheduling-policy.
// All three would have rejected valid deployments.
//
// This test cannot reach into an engine checkout, so it pins the names instead.
// Verification against a live tree belongs in whatever CI has that checkout; what
// this guards is silent renaming during a refactor here.
func TestEngineFieldNames(t *testing.T) {
	// Names taken from the engine's configuration surface. Where the CLI flag and
	// the internal field differ, the CLI name is used, because a deployment is written
	// by whoever launched the engine.
	want := map[string]string{
		"all2all_backend":              "VLLM_ALL2ALL_BACKEND / --all2all-backend",
		"disable_custom_all_reduce":    "--disable-custom-all-reduce",
		"allreduce_backend":            "resolved override, not an engine field",
		"cudagraph_mode":               "-O/--compilation-config cudagraph_mode",
		"max_cudagraph_capture_size":   "-O/--compilation-config max_cudagraph_capture_size",
		"cudagraph_capture_sizes":      "-O/--compilation-config cudagraph_capture_sizes",
		"async_scheduling":             "--async-scheduling",
		"frontend":                     "VLLM_USE_RUST_FRONTEND (environment, not a flag)",
		"disable_cascade_attn":         "--disable-cascade-attn",
		"enable_prefix_caching":        "--enable-prefix-caching / --no-enable-prefix-caching",
		"swa_bounded_replay":           "--swa-bounded-replay / --no-swa-bounded-replay",
		"block_size":                   "--block-size",
		"max_num_batched_tokens":       "--max-num-batched-tokens",
		"max_num_seqs":                 "--max-num-seqs",
		"long_prefill_token_threshold": "--long-prefill-token-threshold",
		"max_model_len":                "--max-model-len",
		"quantization":                 "--quantization",
		"cache_dtype":                  "--kv-cache-dtype",
		"mamba_cache_dtype":            "--mamba-cache-dtype",
		"mamba_ssm_cache_dtype":        "--mamba-ssm-cache-dtype",
		"mamba_cache_mode":             "--mamba-cache-mode",
		"gpu_memory_utilization":       "--gpu-memory-utilization",
		"scheduling_policy":            "--scheduling-policy",
		"dbo":                          "--enable-dbo and its thresholds",
		"eplb":                         "--enable-eplb and --eplb-config",
		"speculative":                  "--speculative-config",
		"engram_config":                "--engram-config (cpu_offload, dp_shared_memory, use_thp)",
		"moe_backend":                  "--moe-backend",
		"dcp_comm_backend":             "--dcp-comm-backend",
		"dcp_q_replicate":              "--dcp-q-replicate / --no-dcp-q-replicate",
		"cp_kv_cache_interleave_size":  "--cp-kv-cache-interleave-size",
		// inference-sim#2021 B9 and accuracy-dashboard#33, all at v0.32.0.
		"long_prefill_token_threshold_adaptive":  "--long-prefill-token-threshold-adaptive",
		"max_num_scheduled_tokens":               "--max-num-scheduled-tokens",
		"max_num_active_seqs":                    "--max-num-active-seqs",
		"watermark":                              "--watermark",
		"prefill_schedule_interval":              "--prefill-schedule-interval",
		"stream_interval":                        "--stream-interval",
		"disable_hybrid_kv_cache_manager":        "--disable-hybrid-kv-cache-manager / --no-disable-hybrid-kv-cache-manager",
		"kv_cache_memory_bytes":                  "--kv-cache-memory-bytes",
		"prefix_match_unit":                      "--prefix-match-unit",
		"prefix_cache_retention_interval":        "--prefix-cache-retention-interval",
		"kv_events":                              "--kv-events-config (enable_kv_cache_events, publisher, topic)",
		"max_loras":                              "--max-loras",
		"max_cpu_loras":                          "--max-cpu-loras",
		"decoder_replay_cudagraph_capture_sizes": "-O/--compilation-config decoder_replay_cudagraph_capture_sizes",
		"custom_ops":                             "-O/--compilation-config custom_ops",
		"use_trtllm_attention":                   "--attention-config use_trtllm_attention",
		"mla_prefill_backend":                    "--attention-config mla_prefill_backend",
		"indexer_kv_dtype":                       "--attention-config indexer_kv_dtype",
		"linear_backend":                         "--kernel-config linear_backend",
		"env":                                    "the launch environment (VLLM_* variables), not a flag",
	}
	got := yamlNames(reflect.TypeOf(Engine{}))
	if len(got) != len(want) {
		t.Errorf("Engine has %d yaml fields, the mapping covers %d: %v",
			len(got), len(want), got)
	}
	for _, name := range got {
		if _, ok := want[name]; !ok {
			t.Errorf("Engine field %q has no engine counterpart recorded; either it is invented or the mapping needs updating", name)
		}
	}
	for name := range want {
		if !contains(got, name) {
			t.Errorf("the mapping records %q but Engine no longer declares it", name)
		}
	}
}

func yamlNames(t reflect.Type) []string {
	var out []string
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "" || tag == "-" {
			continue
		}
		out = append(out, strings.Split(tag, ",")[0])
	}
	return out
}

func contains(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}
