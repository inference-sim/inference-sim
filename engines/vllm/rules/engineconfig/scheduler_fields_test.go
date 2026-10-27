// Moved from inference-sim/blis-schemas/spec/deployment/scheduler_fields_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The scheduler, cache and LoRA knobs inference-sim#2021 B9 and accuracy-dashboard#33 need.
// Each takes the engine's bound, and an omitted one leaves the engine's default; a stated
// value outside the bound is refused here, before an engine would refuse it at launch.
func TestSchedulerAndCacheKnobs(t *testing.T) {
	for _, c := range []struct {
		name string
		e    Engine
		bad  string // the key a finding must name, or "" for valid
	}{
		{"all unset", Engine{}, ""},
		{"prefill_schedule_interval 16", Engine{PrefillScheduleInterval: 16}, ""},
		{"prefill_schedule_interval negative", Engine{PrefillScheduleInterval: -1}, "prefill_schedule_interval"},
		{"stream_interval 10", Engine{StreamInterval: 10}, ""},
		{"stream_interval negative", Engine{StreamInterval: -2}, "stream_interval"},
		{"max_num_active_seqs within max_num_seqs", Engine{MaxNumSeqs: 64, MaxNumActiveSeqs: 32}, ""},
		{"max_num_active_seqs above max_num_seqs", Engine{MaxNumSeqs: 64, MaxNumActiveSeqs: 65}, "max_num_active_seqs"},
		{"max_num_active_seqs negative", Engine{MaxNumActiveSeqs: -1}, "max_num_active_seqs"},
		{"max_num_scheduled_tokens", Engine{MaxNumScheduledTokens: 8192}, ""},
		{"max_num_scheduled_tokens negative", Engine{MaxNumScheduledTokens: -1}, "max_num_scheduled_tokens"},
		{"watermark 0 stated", Engine{Watermark: ptrFloat(0)}, ""},
		{"watermark 0.05", Engine{Watermark: ptrFloat(0.05)}, ""},
		{"watermark 1", Engine{Watermark: ptrFloat(1)}, "watermark"},
		{"watermark negative", Engine{Watermark: ptrFloat(-0.1)}, "watermark"},
		{"watermark NaN", Engine{Watermark: ptrFloat(math.NaN())}, "watermark"},
		{"kv_cache_memory_bytes", Engine{KVCacheMemoryBytes: ptrInt64(53687091200)}, ""},
		{"kv_cache_memory_bytes zero", Engine{KVCacheMemoryBytes: ptrInt64(0)}, "kv_cache_memory_bytes"},
		{"prefix_match_unit", Engine{PrefixMatchUnit: 128}, ""},
		{"prefix_match_unit negative", Engine{PrefixMatchUnit: -1}, "prefix_match_unit"},
		{"prefix_cache_retention_interval 0", Engine{PrefixCacheRetentionInterval: ptrInt(0)}, ""},
		{"prefix_cache_retention_interval negative", Engine{PrefixCacheRetentionInterval: ptrInt(-1)}, "prefix_cache_retention_interval"},
		{"max_loras 1", Engine{MaxLoRAs: 1}, ""},
		{"max_loras negative", Engine{MaxLoRAs: -1}, "max_loras"},
		{"max_cpu_loras below max_loras", Engine{MaxLoRAs: 4, MaxCPULoRAs: 2}, "max_cpu_loras"},
		{"max_cpu_loras at max_loras", Engine{MaxLoRAs: 4, MaxCPULoRAs: 4}, ""},
		// 0 is the engine's "unset" too (vllm/engine/arg_utils.py:2663-2665@5df63a72).
		{"max_cpu_loras 0 is max_loras", Engine{MaxLoRAs: 4, MaxCPULoRAs: 0}, ""},
		// A negative value has two engine readings (the CLI's None, LoRAConfig's refusal), so
		// it is refused rather than read as either.
		{"max_cpu_loras negative", Engine{MaxLoRAs: 4, MaxCPULoRAs: -1}, "max_cpu_loras"},
		{"kv_events zmq", Engine{KVEvents: &KVEvents{EnableKVCacheEvents: true, Publisher: "zmq"}}, ""},
		{"kv_events unknown publisher", Engine{KVEvents: &KVEvents{EnableKVCacheEvents: true, Publisher: "kafka"}}, "kv_events.publisher"},
		{"custom_ops", Engine{CustomOps: []string{"all"}}, ""},
		{"custom_ops empty entry", Engine{CustomOps: []string{""}}, "custom_ops"},
		{"decoder_replay sizes", Engine{DecoderReplayCUDAGraphCaptureSizes: []int{6, 12}}, ""},
		{"decoder_replay size zero", Engine{DecoderReplayCUDAGraphCaptureSizes: []int{0}}, "decoder_replay_cudagraph_capture_sizes"},
		{"env", Engine{Env: map[string]string{"VLLM_MARLIN_USE_ATOMIC_ADD": "1"}}, ""},
		{"env bad name", Engine{Env: map[string]string{"not a name": "1"}}, "env"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := engineOnly(c.e).Validate()
			if c.bad == "" {
				if !p.OK() {
					t.Fatalf("refused:\n%s", p.Error())
				}
				return
			}
			if p.OK() {
				t.Fatalf("accepted; want a finding on %s", c.bad)
			}
			if !strings.Contains(p.Error(), ".engine."+c.bad) {
				t.Fatalf("the finding does not name engine.%s:\n%s", c.bad, p.Error())
			}
		})
	}
}

// Each new key round-trips under its vLLM name, and an omitted one is not written.
func TestSchedulerAndCacheKnobsRoundTrip(t *testing.T) {
	e := Engine{
		PrefillScheduleInterval: 8, DisableHybridKVCacheManager: ptrBool(true),
		KVCacheMemoryBytes: ptrInt64(1 << 35), StreamInterval: 20, MaxNumActiveSeqs: 16,
		MaxNumScheduledTokens: 4096, Watermark: ptrFloat(0), LongPrefillTokenThresholdAdaptive: ptrBool(true),
		PrefixMatchUnit: 128, PrefixCacheRetentionInterval: ptrInt(32768),
		KVEvents:  &KVEvents{EnableKVCacheEvents: true, Publisher: "zmq", Topic: "kv-events"},
		CustomOps: []string{"all"}, DecoderReplayCUDAGraphCaptureSizes: []int{6, 12},
		UseTRTLLMAttention: ptrBool(true), MLAPrefillBackend: "TRTLLM_RAGGED", IndexerKVDType: "fp8", LinearBackend: "flashinfer_cutlass",
		Env: map[string]string{"VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm"}, MaxLoRAs: 2, MaxCPULoRAs: 8,
	}
	out, err := yaml.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{
		"prefill_schedule_interval", "disable_hybrid_kv_cache_manager", "kv_cache_memory_bytes",
		"stream_interval", "max_num_active_seqs", "max_num_scheduled_tokens", "watermark",
		"long_prefill_token_threshold_adaptive", "prefix_match_unit", "prefix_cache_retention_interval",
		"kv_events", "custom_ops", "decoder_replay_cudagraph_capture_sizes", "use_trtllm_attention",
		"mla_prefill_backend", "indexer_kv_dtype", "linear_backend", "env", "max_loras", "max_cpu_loras",
	} {
		if !containsKey(string(out), key) {
			t.Errorf("%s is not written:\n%s", key, out)
		}
	}
	var back Engine
	dec := yaml.NewDecoder(strings.NewReader(string(out)))
	dec.KnownFields(true)
	if err := dec.Decode(&back); err != nil {
		t.Fatal(err)
	}
	again, _ := yaml.Marshal(back)
	if string(again) != string(out) {
		t.Errorf("round trip changed the document:\n%s\nbecame\n%s", out, again)
	}
	// watermark: 0 is a stated value, so it is written; an omitted one is not.
	if !containsKey(string(out), "watermark") {
		t.Error("a stated watermark of 0 was dropped")
	}
	empty, _ := yaml.Marshal(Engine{})
	if strings.TrimSpace(string(empty)) != "{}" {
		t.Errorf("an empty engine writes keys:\n%s", empty)
	}
}
