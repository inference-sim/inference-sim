// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/vllm.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/config/scheduler.py, vllm/config/cache.py and vllm/config/lora.py (the defaults a
// launch inherits).

package vllm

import (
	"bytes"
	"fmt"
	"slices"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// Config is the engine block of one pool, as the simulator hands it over in engine.Spec.Config.
//
// The pool's engine settings are blis-schemas' deployment.Engine, inline, which is what a
// scenario's engine: block holds. Two facts the scheduler needs are not engine settings and sit
// beside them:
//
//   - NumGPUBlocks is the KV cache's block count. vLLM derives it from the GPU memory left after
//     loading the model (vllm/v1/core/kv_cache_utils.py:1003-1035@5df63a72); that is a cost
//     model's answer, not this engine's, so the caller states it (#2021 B5).
//   - PipelineParallelSize is parallel.pp, which sets max_concurrent_batches with async
//     scheduling (vllm/config/vllm.py:577-588@5df63a72).
//
// UseV2ModelRunner is whether the V2 model runner runs (VllmConfig.use_v2_model_runner,
// vllm/config/vllm.py:703-739@5df63a72). vLLM resolves an unset VLLM_USE_V2_MODEL_RUNNER from
// whether Triton is installed and which features are on, so it is a fact of the serving image
// the caller states, here or as the env key; see resolveV2Runner.
type Config struct {
	deployment.Engine    `yaml:",inline"`
	NumGPUBlocks         int   `yaml:"num_gpu_blocks"`
	PipelineParallelSize int   `yaml:"pipeline_parallel_size,omitempty"`
	UseV2ModelRunner     *bool `yaml:"use_v2_model_runner,omitempty"`
}

// decodeConfig reads an engine block strictly: a key that is not a field is an error, as it
// is for every blis-schemas loader, so a misspelt setting can never be silently dropped.
func decodeConfig(node yaml.Node) (Config, error) {
	var c Config
	if node.Kind == 0 {
		return c, fmt.Errorf("vllm: the engine block is empty")
	}
	raw, err := yaml.Marshal(&node)
	if err != nil {
		return c, fmt.Errorf("vllm: engine block: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return c, fmt.Errorf("vllm: engine block: %w", err)
	}
	return c, nil
}

// settings are a Config resolved into the scheduler's and engine core's inputs.
type settings struct {
	sched     sched.Config
	queueSize int
	v2Runner  bool
}

// resolve turns an engine block into the scheduler configuration vLLM v0.32.0 would run, or
// refuses it. A setting that changes scheduling and that this engine does not model is refused
// rather than ignored (refuseUnmodelled): a run that drops one would still produce numbers, and
// nothing would say they describe another engine. Settings that change only step time or
// memory (backends, graph modes, quantization, memory utilization, cache dtypes) are the
// caller's Pricer's and pass through unread.
func resolve(c Config) (settings, error) {
	var s settings
	e := c.Engine
	if err := refuseUnmodelled(e); err != nil {
		return s, err
	}
	if c.NumGPUBlocks < 2 {
		// One block is the null block (vllm/v1/core/block_pool.py:187-188@5df63a72); with
		// none left vLLM refuses to start (kv_cache_utils.py:1009-1017@5df63a72).
		return s, fmt.Errorf("vllm: num_gpu_blocks %d leaves no block for KV: the caller sizes the cache, "+
			"as vLLM sizes it from the memory left after the model loads", c.NumGPUBlocks)
	}
	pp := max(c.PipelineParallelSize, 1)

	blockSize := e.BlockSize
	if blockSize <= 0 {
		// CacheConfig.block_size defaults to None, which the platform resolves from the
		// attention backends (vllm/config/cache.py:74-76@5df63a72;
		// vllm/platforms/interface.py:703-781@5df63a72): a fact of the model and the GPU, not
		// of the deployment, so the caller states it.
		return s, fmt.Errorf("vllm: block_size must be stated; vLLM resolves an unstated one from the " +
			"attention backends of the model and GPU, which the caller knows")
	}
	maxLen, ok := e.MaxModelLen.Stated()
	if !ok {
		// An unstated or auto window is the model's own, or one fitted to memory
		// (vllm/config/model.py@5df63a72); either is the caller's to resolve.
		return s, fmt.Errorf("vllm: max_model_len must be stated; an unstated or auto window is the model's " +
			"or the memory's, which the caller resolves")
	}
	// vLLM refuses to start when the KV cache cannot hold one max_model_len sequence: a
	// full-attention group needs cdiv(max_model_len, block_size) blocks
	// (FullAttentionSpec.max_memory_usage_bytes, vllm/v1/kv_cache_interface.py:602-607@5df63a72),
	// checked against the blocks left beside the null block
	// (vllm/v1/core/kv_cache_utils.py:2627-2647, 1003-1040@5df63a72). A request the cache could
	// never hold would otherwise wait for ever.
	if need := (maxLen + blockSize - 1) / blockSize; need > c.NumGPUBlocks-1 {
		return s, fmt.Errorf("vllm: num_gpu_blocks %d cannot hold one max_model_len %d sequence (%d blocks of %d "+
			"beside the null block), which vLLM refuses at startup (vllm/v1/core/kv_cache_utils.py:2638-2647@5df63a72)",
			c.NumGPUBlocks, maxLen, need, blockSize)
	}
	seqs, budget := e.MaxNumSeqs, e.MaxNumBatchedTokens
	if seqs == 0 || budget == 0 {
		// SchedulerConfig's own defaults, 128 and 2048, are "mainly for convenience when testing"
		// (vllm/config/scheduler.py:47-66@5df63a72); vllm serve resolves unstated values from the
		// usage context and the GPU's memory (vllm/engine/arg_utils.py:2894-2990, 3097-3186@5df63a72),
		// which the caller knows, so they must be stated rather than guessed.
		return s, fmt.Errorf("vllm: max_num_seqs and max_num_batched_tokens must be stated (got %d and %d); vllm "+
			"serve resolves unstated ones from the usage context and the GPU (vllm/engine/arg_utils.py:3097-3186@5df63a72)",
			seqs, budget)
	}
	// The scheduler fields' own bounds (vllm/config/scheduler.py:48,55,62,69,79@5df63a72; ge=1, or
	// ge=0 for the long-prefill threshold), and max_loras's (vllm/config/lora.py:35@5df63a72). Zero
	// means unset where the field is optional.
	for _, f := range []struct {
		name  string
		value int
	}{{"max_num_seqs", seqs}, {"max_num_batched_tokens", budget}, {"max_num_scheduled_tokens", e.MaxNumScheduledTokens},
		{"max_num_active_seqs", e.MaxNumActiveSeqs}, {"long_prefill_token_threshold", e.LongPrefillTokenThreshold},
		{"max_loras", e.MaxLoRAs}} {
		if f.value < 0 {
			return s, fmt.Errorf("vllm: %s %d is negative, which vLLM refuses (vllm/config/scheduler.py:48-79@5df63a72, "+
				"vllm/config/lora.py:35@5df63a72)", f.name, f.value)
		}
	}
	if budget < seqs {
		return s, fmt.Errorf("vllm: max_num_batched_tokens %d is below max_num_seqs %d, which vLLM refuses "+
			"(vllm/config/scheduler.py:328-332@5df63a72)", budget, seqs)
	}
	policy := sched.FCFS
	switch e.SchedulingPolicy {
	case "", "fcfs":
	case "priority":
		policy = sched.Priority
	default:
		return s, fmt.Errorf("vllm: scheduling_policy %q; vLLM v0.32.0 accepts fcfs and priority "+
			"(vllm/config/scheduler.py:21@5df63a72)", e.SchedulingPolicy)
	}
	watermark := 0.0
	if e.Watermark != nil {
		watermark = *e.Watermark
	}
	if !(watermark >= 0 && watermark < 1) { // also refuses NaN, which fails every comparison
		return s, fmt.Errorf("vllm: watermark %v is outside [0, 1) (vllm/config/scheduler.py:196@5df63a72)", watermark)
	}
	if e.MaxNumActiveSeqs > seqs {
		return s, fmt.Errorf("vllm: max_num_active_seqs %d is above max_num_seqs %d, which vLLM refuses "+
			"(vllm/config/scheduler.py:69@5df63a72)", e.MaxNumActiveSeqs, seqs)
	}
	if e.LongPrefillTokenThreshold > maxLen {
		return s, fmt.Errorf("vllm: long_prefill_token_threshold %d is above max_model_len %d, which vLLM "+
			"refuses (vllm/config/scheduler.py verify_max_model_len@5df63a72)", e.LongPrefillTokenThreshold, maxLen)
	}

	spec := sched.Spec{}
	method := ""
	if sp := e.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		method = sp.Method
		if !specMethodModelled(method) {
			// draft_model's draft slots depend on parallel_drafting (PARD, K slots, or one;
			// vllm/config/speculative.py:1907-1926@5df63a72), and an MTP drafter's prefill
			// lookahead on its checkpoint's num_nextn_predict_layers (vllm/config/vllm.py:655-665@5df63a72):
			// neither is a deployment field, so neither method is built until it is.
			return s, fmt.Errorf("vllm: speculative method %q is not built by this factory; it builds eagle, "+
				"eagle3 and dspark, whose scheduler numbers the deployment fully states (docs/feature-matrix.md)", method)
		}
		if method == "dspark" && !boolOr(c.UseV2ModelRunner, false) && c.Env["VLLM_USE_V2_MODEL_RUNNER"] != "1" {
			// DSpark is implemented only by the V2 model runner (vllm/config/vllm.py:2976-2979@5df63a72).
			return s, fmt.Errorf("vllm: dspark speculative decoding runs only on the V2 model runner")
		}
		spec = sched.ResolveSpec(sched.SpecOptions{Method: method, NumTokens: sp.NumSpecTokens})
	}

	v2, err := resolveV2Runner(c)
	if err != nil {
		return s, err
	}
	async := resolveAsync(e.AsyncScheduling, method, pp, v2)
	if err := engine.CheckConfig(pp, async, v2); err != nil {
		return s, fmt.Errorf("vllm: %w", err)
	}
	queue := engine.MaxConcurrentBatches(pp, async, v2)
	if async && spec.NumTokens > 0 {
		// The AsyncScheduler's draft placeholders (async_scheduler.py:26-49@5df63a72) are not
		// diff-tested together with speculative decoding (docs/feature-matrix.md).
		return s, fmt.Errorf("vllm: async scheduling with speculative decoding is not modelled; " +
			"state async_scheduling: false to run the drafter synchronously")
	}
	if queue > 1 {
		// vLLM's batch queue decides batch membership from its own loop: an empty schedule
		// occupies a slot, a full queue blocks on the oldest batch, and new requests are read only
		// between steps (vllm/v1/engine/core.py:724-768, 1573-1601@5df63a72). Through engine.Engine
		// the simulator decides when Schedule runs, so the adapter would co-schedule requests vLLM
		// holds back. Async scheduling is vLLM's default, so an unstated one is refused too.
		return s, fmt.Errorf("vllm: max_concurrent_batches %d (async_scheduling %t, pipeline_parallel_size %d) is not "+
			"modelled through engine.Engine: vLLM's batch queue (vllm/v1/engine/core.py:724-768@5df63a72) is not "+
			"reproduced yet; state async_scheduling: false and pipeline_parallel_size 1 "+
			"(https://github.com/inference-sim/blis-vllm/issues/21)", queue, async, pp)
	}

	prefixCaching := e.EnablePrefixCaching == nil || *e.EnablePrefixCaching
	maxLoRAs := 0
	if e.MaxLoRAs > 0 {
		maxLoRAs = e.MaxLoRAs
	}
	s.sched = sched.Config{
		MaxNumSeqs:                seqs,
		MaxNumActiveSeqs:          e.MaxNumActiveSeqs,
		MaxNumBatchedTokens:       budget,
		MaxNumScheduledTokens:     e.MaxNumScheduledTokens,
		LongPrefillTokenThreshold: e.LongPrefillTokenThreshold,
		LongPrefillAdaptive:       e.LongPrefillTokenThresholdAdaptive != nil && *e.LongPrefillTokenThresholdAdaptive,
		// enable_chunked_prefill defaults to true (vllm/config/scheduler.py:134@5df63a72) and the
		// deployment schema has no field to turn it off.
		EnableChunkedPrefill: true,
		Policy:               policy,
		MaxModelLen:          maxLen,
		BlockSize:            blockSize,
		NumBlocks:            c.NumGPUBlocks,
		EnablePrefixCaching:  prefixCaching,
		Watermark:            watermark,
		// scheduler_reserve_full_isl defaults to true (vllm/config/scheduler.py:190@5df63a72).
		ReserveFullSequence:  true,
		Spec:                 spec,
		MaxLoRAs:             maxLoRAs,
		KVEvents:             e.KVEvents != nil && e.KVEvents.EnableKVCacheEvents,
		MaxConcurrentBatches: queue,
		Async:                async,
		DecodeStagger:        pp,
		UseV2ModelRunner:     v2,
	}
	s.queueSize, s.v2Runner = queue, v2
	return s, nil
}

// specMethodModelled reports whether this factory builds a draft method: one tier 2 diff-tests
// (docs/feature-matrix.md, "Speculative decoding") whose scheduler numbers depend only on the
// method and num_speculative_tokens.
func specMethodModelled(method string) bool {
	return slices.Contains([]string{"eagle", "eagle3", "dspark"}, method)
}

func boolOr(b *bool, def bool) bool {
	if b == nil {
		return def
	}
	return *b
}

// resolveV2Runner is VllmConfig.use_v2_model_runner (vllm/config/vllm.py:703-739@5df63a72) for
// what a deployment states: use_v2_model_runner, else the VLLM_USE_V2_MODEL_RUNNER env key,
// read as vLLM reads it (bool(int(value)), vllm/envs.py:358-361@5df63a72). Unstated, vLLM
// decides from whether Triton is installed (:724-728) and the features on (:730-737), which
// this engine cannot see, so it refuses rather than guess.
func resolveV2Runner(c Config) (bool, error) {
	if c.UseV2ModelRunner != nil {
		return *c.UseV2ModelRunner, nil
	}
	if v, ok := c.Env["VLLM_USE_V2_MODEL_RUNNER"]; ok {
		switch v {
		case "0":
			return false, nil
		case "1":
			return true, nil
		}
		return false, fmt.Errorf("vllm: env VLLM_USE_V2_MODEL_RUNNER=%q is not 0 or 1, which vLLM reads as int "+
			"(vllm/envs.py:358-361@5df63a72)", v)
	}
	return false, fmt.Errorf("vllm: use_v2_model_runner must be stated (or env VLLM_USE_V2_MODEL_RUNNER): " +
		"vLLM resolves it from whether the image has Triton (vllm/config/vllm.py:720-737@5df63a72)")
}

// resolveAsync is SchedulerConfig.async_scheduling as VllmConfig resolves it
// (vllm/config/vllm.py:1553-1644@5df63a72): a stated value stands (vLLM refuses an
// incompatible one, which CheckConfig and resolve's own checks catch), and an unstated one is
// on unless the draft method is not an EAGLE-family, draft_model or DSpark method, or PP runs
// on the V1 runner. Pooling models, executor backends without async support and ROCm DeepEP
// DBO are not deployments this engine models.
func resolveAsync(stated *bool, method string, pp int, v2 bool) bool {
	if stated != nil {
		return *stated
	}
	switch method {
	case "", "eagle", "eagle3", "mtp", "dflash", "dspark", "draft_model", "ngram_gpu":
	default:
		return false
	}
	if pp > 1 && !v2 {
		return false
	}
	return true
}

// refuseUnmodelled refuses engine settings whose effect on scheduling this engine does not
// model, each with the reason. Settings that only change step time (backends, graph modes,
// quantization, memory utilization) are the cost model's and pass through untouched.
func refuseUnmodelled(e deployment.Engine) error {
	switch {
	case e.PrefixMatchUnit != 0:
		return fmt.Errorf("vllm: prefix_match_unit is not modelled (https://github.com/inference-sim/blis-vllm/issues/7)")
	case e.PrefixCacheRetentionInterval != nil:
		return fmt.Errorf("vllm: prefix_cache_retention_interval is not modelled (https://github.com/inference-sim/blis-vllm/issues/7)")
	case e.PrefillScheduleInterval > 1:
		// _should_throttle_prefills acts across data-parallel ranks (vllm/v1/engine/core.py:2318-2325@5df63a72);
		// one engine here is one rank.
		return fmt.Errorf("vllm: prefill_schedule_interval throttles across data-parallel ranks; one engine is one rank")
	case e.MaxCPULoRAs != 0:
		return fmt.Errorf("vllm: max_cpu_loras is not modelled (https://github.com/inference-sim/blis-vllm/issues/5)")
	case e.MambaCacheMode != "":
		// mamba_cache_mode "align" splits prefill chunks on block boundaries for a model with
		// Mamba layers (vllm/v1/core/sched/scheduler.py:375-379@5df63a72). This factory builds
		// the full-attention KV cache only (sched.Config.KVGroups unset), so a recurrent
		// model's groups, and with them this setting, are not modelled.
		return fmt.Errorf("vllm: mamba_cache_mode is not modelled: this factory builds full-attention KV " +
			"groups only (https://github.com/inference-sim/blis-vllm/issues/7, " +
			"https://github.com/inference-sim/blis-vllm/issues/17)")
	case e.SWABoundedReplay != nil:
		// swa_bounded_replay keeps DeepSeek-V4.1's sliding-window KV out of prefix caching
		// (vllm/config/cache.py:250@5df63a72; vllm/models/deepseek_v41/attention.py:529-547@5df63a72),
		// a sliding-window group this factory does not build.
		return fmt.Errorf("vllm: swa_bounded_replay is not modelled: this factory builds full-attention KV " +
			"groups only (https://github.com/inference-sim/blis-vllm/issues/7, " +
			"https://github.com/inference-sim/blis-vllm/issues/17)")
	}
	return nil
}
