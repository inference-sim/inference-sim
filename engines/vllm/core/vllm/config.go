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
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/sched"
	"github.com/inference-sim/blis-vllm/rules/v0_32"
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
//   - PipelineParallelSize is the pool's parallel.pp, which sets max_concurrent_batches with
//     async scheduling (vllm/config/vllm.py:577-588@5df63a72). The parallel block is not part of
//     the engine block a Spec carries, so the caller restates it here, and it must be stated: an
//     unstated one would build a one-batch engine for a pool that pipelines.
//   - MTPNextnPredictLayers is an MTP drafter checkpoint's num_nextn_predict_layers, which sets
//     the prefill lookahead the scheduler reserves: num_speculative_tokens for multi-module MTP
//     (min(layers, K) > 1), else one (VllmConfig.num_prefill_lookahead_tokens,
//     vllm/config/vllm.py:644-664; use_multi_module_mtp, vllm/config/speculative.py:1982-1988
//     @5df63a72). It is the model's, not the deployment's, so the caller states it from the
//     checkpoint's config.json; an MTP drafter without it is refused.
//
// UseV2ModelRunner is whether the V2 model runner runs (VllmConfig.use_v2_model_runner,
// vllm/config/vllm.py:703-739@5df63a72). vLLM resolves an unset VLLM_USE_V2_MODEL_RUNNER from
// whether Triton is installed and which features are on, so it is a fact of the serving image
// the caller states, here or as the env key; see resolveV2Runner.
type Config struct {
	deployment.Engine     `yaml:",inline"`
	NumGPUBlocks          int   `yaml:"num_gpu_blocks"`
	PipelineParallelSize  *int  `yaml:"pipeline_parallel_size"`
	UseV2ModelRunner      *bool `yaml:"use_v2_model_runner,omitempty"`
	MTPNextnPredictLayers int   `yaml:"mtp_nextn_predict_layers,omitempty"`
	// DataParallelSize is the pool's parallel.dp, restated as pipeline_parallel_size is: above 1
	// the factory builds a data-parallel group of that many ranks behind one endpoint (DPGroup).
	// Unstated or 1 is one engine.
	DataParallelSize int `yaml:"data_parallel_size,omitempty"`
	// KVLayers are the model's attention layers by KV cache spec, which vLLM groups into the
	// scheduler's KV cache groups (KVGroups). Unstated is one full-attention group.
	KVLayers []KVLayerRun `yaml:"kv_layers,omitempty"`
	// DPSyncInterval is parallel_config.dp_sync_interval (vllm/config/parallel.py:162@5df63a72,
	// default 16): the steps between a DP group's unfinished-request all-reduces.
	DPSyncInterval int `yaml:"dp_sync_interval,omitempty"`
	// DPStatsIntervalUs is how often the DP coordinator publishes the ranks' request counts to
	// the front end's load balancer: min_stats_update_interval_ms = 100
	// (vllm/v1/engine/coordinator.py:154-162, 267-269@5df63a72), 100000 µs unstated. A
	// behavioural test states it as a step count's worth of time.
	DPStatsIntervalUs int64 `yaml:"dp_stats_interval_us,omitempty"`
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
	// dp is the data-parallel group: ranks (1 for one engine), the sync interval, the
	// coordinator's publish interval and the prefill throttle interval.
	dp dpSettings
}

// dpSettings are a data-parallel group's settings.
type dpSettings struct {
	ranks, syncInterval, prefillInterval int
	statsIntervalUs                      int64
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
	if err := refuseUnmodelled(e, c.DataParallelSize > 1); err != nil {
		return s, err
	}
	if c.NumGPUBlocks < 2 {
		// One block is the null block (vllm/v1/core/block_pool.py:187-188@5df63a72); with
		// none left vLLM refuses to start (kv_cache_utils.py:1009-1017@5df63a72).
		return s, fmt.Errorf("vllm: num_gpu_blocks %d leaves no block for KV: the caller sizes the cache, "+
			"as vLLM sizes it from the memory left after the model loads", c.NumGPUBlocks)
	}
	if c.PipelineParallelSize == nil || *c.PipelineParallelSize < 1 {
		// The pool's parallel.pp is not in the engine block, and it changes how many batches
		// vLLM keeps in flight (vllm/config/vllm.py:577-588@5df63a72), so it is not assumed.
		return s, fmt.Errorf("vllm: pipeline_parallel_size must be stated, at least 1: it is the pool's parallel.pp, " +
			"which is not in the engine block, and it sets max_concurrent_batches (vllm/config/vllm.py:577-588@5df63a72)")
	}
	pp := *c.PipelineParallelSize

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
			// vllm/config/speculative.py:1907-1926@5df63a72), which is not a deployment field.
			return s, fmt.Errorf("vllm: speculative method %q is not built by this factory; it builds eagle, "+
				"eagle3, mtp and dspark, whose scheduler numbers the engine block fully states (docs/feature-matrix.md)", method)
		}
		if method == "mtp" && c.MTPNextnPredictLayers < 1 {
			// The prefill lookahead depends on the drafter checkpoint's layer count
			// (vllm/config/vllm.py:644-664@5df63a72), which only its config.json holds.
			return s, fmt.Errorf("vllm: an mtp drafter needs mtp_nextn_predict_layers, its checkpoint's " +
				"num_nextn_predict_layers, which sets the prefill lookahead (vllm/config/vllm.py:644-664@5df63a72)")
		}
		spec = sched.ResolveSpec(sched.SpecOptions{Method: method, NumTokens: sp.NumSpecTokens,
			NextnPredictLayers: c.MTPNextnPredictLayers})
	}

	v2, err := resolveV2Runner(c)
	if err != nil {
		return s, err
	}
	if method == "dspark" && !v2 {
		// DSpark is implemented only by the V2 model runner (vllm/config/vllm.py:2976-2979@5df63a72),
		// which is the runner as resolved, not either statement of it alone.
		return s, fmt.Errorf("vllm: dspark speculative decoding runs only on the V2 model runner")
	}
	if err := checkAdaptiveVerification(e, pp, v2); err != nil {
		return s, err
	}
	async := resolveAsync(e.AsyncScheduling, method, pp, v2)
	if err := engine.CheckConfig(pp, async, v2); err != nil {
		return s, fmt.Errorf("vllm: %w", err)
	}
	queue := engine.MaxConcurrentBatches(pp, async, v2)
	// Async scheduling with a drafter runs the AsyncScheduler's placeholder drafts
	// (async_scheduler.py:25-58@5df63a72), diff-tested with MTP and DSpark against vLLM's own
	// EngineCore through the batch queue (core/engine/testdata/golden/t3-async-mtp*.json,
	// t3-async-dspark.json).
	// max_concurrent_batches > 1 runs vLLM's batch queue: the adapter keeps step_with_batch_queue's
	// loop state (an empty schedule occupies a slot, a full queue blocks on the oldest batch, and a
	// blocked step schedules nothing until that batch completes; vllm/v1/engine/core.py:698-810@5df63a72).

	prefixCaching := e.EnablePrefixCaching == nil || *e.EnablePrefixCaching
	maxLoRAs := 0
	if e.MaxLoRAs > 0 {
		maxLoRAs = e.MaxLoRAs
	}
	groups, err := KVGroups(c.KVLayers, e.DisableHybridKVCacheManager != nil && *e.DisableHybridKVCacheManager,
		trailingDrafter(method, c.KVLayers))
	if err != nil {
		return settings{}, err
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
		KVGroups:             groups,
	}
	s.queueSize, s.v2Runner = queue, v2
	s.dp = dpSettings{ranks: max(c.DataParallelSize, 1), syncInterval: c.DPSyncInterval,
		prefillInterval: max(e.PrefillScheduleInterval, 1), statsIntervalUs: c.DPStatsIntervalUs}
	if s.dp.syncInterval == 0 {
		s.dp.syncInterval = 16
	}
	if s.dp.statsIntervalUs == 0 {
		s.dp.statsIntervalUs = 100_000
	}
	if s.dp.ranks > 1 && pp > 1 {
		// A DP rank's batch queue is diff-tested inside vLLM's DP loop with async scheduling on one
		// pipeline stage (oracle/run_dp.py, core/engine/testdata/golden-dp-async); pipeline
		// parallelism inside a DP rank is not.
		return settings{}, fmt.Errorf("vllm: data parallelism with pipeline parallelism (dp %d, pp %d) is not built "+
			"yet: the DP group is diff-tested on one pipeline stage per rank (oracle/run_dp.py)", s.dp.ranks, pp)
	}
	// The release's own accepted values and refusals, for what the checks above do not cover.
	if err := checkRules(e, pp); err != nil {
		return settings{}, err
	}
	return s, nil
}

// specMethodModelled reports whether this factory builds a draft method: one tier 2 diff-tests
// (docs/feature-matrix.md, "Speculative decoding") whose scheduler numbers depend only on the
// method and num_speculative_tokens.
func specMethodModelled(method string) bool {
	return slices.Contains([]string{"eagle", "eagle3", "mtp", "dspark"}, method)
}

// resolveV2Runner is VllmConfig.use_v2_model_runner (vllm/config/vllm.py:703-739@5df63a72) for
// what a deployment states: use_v2_model_runner, else the VLLM_USE_V2_MODEL_RUNNER env key,
// read as vLLM reads it (bool(int(value)), vllm/envs.py:358-361@5df63a72). Unstated, vLLM
// decides from whether Triton is installed (:724-728) and the features on (:730-737), which
// this engine cannot see, so it refuses rather than guess.
//
// vLLM itself reads only the env key (vllm/config/vllm.py:720-722, vllm/envs.py:2151-2152@5df63a72):
// use_v2_model_runner is this engine block's restatement of it, so when both are stated they must
// agree, and a disagreement is refused rather than one silently winning.
func resolveV2Runner(c Config) (bool, error) {
	var fromEnv *bool
	if v, ok := c.Env["VLLM_USE_V2_MODEL_RUNNER"]; ok {
		var b bool
		switch v {
		case "0":
		case "1":
			b = true
		default:
			return false, fmt.Errorf("vllm: env VLLM_USE_V2_MODEL_RUNNER=%q is not 0 or 1, which vLLM reads as int "+
				"(vllm/envs.py:358-361@5df63a72)", v)
		}
		fromEnv = &b
	}
	switch {
	case c.UseV2ModelRunner != nil && fromEnv != nil && *c.UseV2ModelRunner != *fromEnv:
		return false, fmt.Errorf("vllm: use_v2_model_runner %t and env VLLM_USE_V2_MODEL_RUNNER=%s disagree; vLLM reads "+
			"only the env key (vllm/config/vllm.py:720-722@5df63a72), so state one, or both alike",
			*c.UseV2ModelRunner, c.Env["VLLM_USE_V2_MODEL_RUNNER"])
	case c.UseV2ModelRunner != nil:
		return *c.UseV2ModelRunner, nil
	case fromEnv != nil:
		return *fromEnv, nil
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
func refuseUnmodelled(e deployment.Engine, dataParallel bool) error {
	if keys := unreadEnv(e.Env); len(keys) > 0 {
		// blis-schemas' Env rule: a consumer refuses, or discloses, every key it cannot honour,
		// and never ignores one (spec/deployment/deployment.go Engine.Env).
		return fmt.Errorf("vllm: env %s is not read by this engine, which reads only %s; the keys that "+
			"change step time are the latency kernel's to price (https://github.com/inference-sim/blis-latency-kernel/issues/85)",
			strings.Join(keys, ", "), strings.Join(readEnv, ", "))
	}
	switch {
	case e.PrefixMatchUnit != 0:
		return fmt.Errorf("vllm: prefix_match_unit is not modelled (https://github.com/inference-sim/blis-vllm/issues/7)")
	case e.PrefixCacheRetentionInterval != nil:
		return fmt.Errorf("vllm: prefix_cache_retention_interval is not modelled (https://github.com/inference-sim/blis-vllm/issues/7)")
	case e.PrefillScheduleInterval > 1 && !dataParallel:
		// _should_throttle_prefills is the DP engine core's (vllm/v1/engine/core.py:2318-2325@5df63a72):
		// EngineCore's never throttles (:652-655), so on one engine the setting has no effect vLLM
		// would show, and stating it is refused rather than dropped.
		return fmt.Errorf("vllm: prefill_schedule_interval throttles a data-parallel group's ranks; this engine " +
			"has data_parallel_size 1")
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

// readEnv are the engine environment variables this engine reads (resolveV2Runner).
var readEnv = []string{"VLLM_USE_V2_MODEL_RUNNER"}

// unreadEnv are env's keys this engine does not read, sorted.
func unreadEnv(env map[string]string) []string {
	var out []string
	for k := range env {
		if !slices.Contains(readEnv, k) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// checkAdaptiveVerification refuses speculative.enable_adaptive_verification where vLLM v0.32.0
// refuses it at startup, for what the engine block states: the V1 model runner does not support
// it (VllmConfig._get_v1_model_runner_unsupported_features, vllm/config/vllm.py:2980-2981@5df63a72),
// and _validate_adaptive_verification (:3001-3032@5df63a72) refuses it with LoRA, without full
// CUDA graphs and with pipeline parallelism. The verify width it trims is step time, which is
// the latency kernel's (docs/feature-matrix.md, "DSpark adaptive verification").
func checkAdaptiveVerification(e deployment.Engine, pp int, v2 bool) error {
	sp := e.Speculative
	if sp == nil || sp.EnableAdaptiveVerification == nil || !*sp.EnableAdaptiveVerification {
		return nil
	}
	switch {
	case !v2:
		return fmt.Errorf("vllm: speculative.enable_adaptive_verification: adaptive draft verification runs only " +
			"on the V2 model runner, and vLLM refuses it on V1 (vllm/config/vllm.py:2980-2981@5df63a72)")
	case e.MaxLoRAs > 0:
		// LoRAConfig exists whenever LoRA is enabled, which max_loras states here
		// (vllm/engine/arg_utils.py:2650-2669@5df63a72).
		return fmt.Errorf("vllm: speculative.enable_adaptive_verification: \"Adaptive verification is not currently " +
			"compatible with LoRA\" (vllm/config/vllm.py:3005-3010@5df63a72)")
	case !fullCUDAGraphs(e.CUDAGraphMode):
		return fmt.Errorf("vllm: speculative.enable_adaptive_verification: \"Adaptive verification requires full CUDA "+
			"graphs\", and cudagraph_mode %s captures none (vllm/config/vllm.py:3012-3017@5df63a72)", e.CUDAGraphMode)
	case pp > 1:
		return fmt.Errorf("vllm: speculative.enable_adaptive_verification: \"Adaptive verification is not currently " +
			"compatible with pipeline parallelism\" (vllm/config/vllm.py:3019-3028@5df63a72)")
	}
	return nil
}

// fullCUDAGraphs is CUDAGraphMode.has_full_cudagraphs for a stated mode: FULL, FULL_DECODE_ONLY
// and FULL_AND_PIECEWISE capture full graphs (vllm/config/compilation.py:59-91@5df63a72). An
// unstated mode is vLLM's default, FULL_AND_PIECEWISE (vllm/config/vllm.py:303@5df63a72).
func fullCUDAGraphs(mode string) bool {
	switch mode {
	case "", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE":
		return true
	}
	return false
}

// checkRules runs the v0.32.0 rules pack (rules/v0_32) over the engine block, the startup check
// #2021 B11 step 2 gives the engine: the values the release accepts and the refusals it makes.
// The block is one pool's engine, and it is all the engine is given: the model graph, the cluster
// and the pool's parallel widths other than pipeline_parallel_size are not in it. So a refusal
// stands only where the block decides it whatever those widths are: the pack runs once for each
// parallel shape in blockShapes, and only an error every run reports refuses. A clause that
// turns on a width the block does not state, such as engram-config-valid's "dp_shared_memory
// requires dp above 1" (vllm/config/engram.py@5df63a72), is left to the caller that validates
// the whole document set, as is every rule that needs the model graph, which checks nothing
// without one. Warnings are not refusals. pp is the pool's pipeline_parallel_size.
func checkRules(e deployment.Engine, pp int) error { return checkRulesWith(v0_32.Pack(), e, pp) }

// blockShapes are the tensor and data parallel widths checkRules runs the pack under: each at 1
// and at 2, so that a clause turning on either width, whichever way, is silent in one of them.
var blockShapes = []struct{ tp, dp int }{{1, 1}, {1, 2}, {2, 1}, {2, 2}}

// checkRulesWith is checkRules for a given rules pack.
func checkRulesWith(pack rules.EngineRules, e deployment.Engine, pp int) error {
	var first []rules.Problem       // the first shape's errors, in the order the pack reports them
	in := map[string]map[int]bool{} // an error's key: the shapes that reported it
	for i, shape := range blockShapes {
		doc := rules.Input{Deployment: &deployment.Deployment{Pools: []deployment.Pool{{
			Role:     deployment.RoleColocated,
			Parallel: deployment.Parallelism{TP: shape.tp, PP: max(pp, 1), DP: shape.dp},
			Engine:   e,
		}}}}
		for _, p := range pack.Check(doc).Errors() {
			key := p.Rule + "\x00" + p.Message
			if in[key] == nil {
				in[key] = map[int]bool{}
				if i == 0 {
					first = append(first, p)
				}
			}
			in[key][i] = true
		}
	}
	// The block is checked as the one pool of a document, so its findings are named from the
	// engine block, not from a pool index the caller never wrote.
	var lines []string
	for _, p := range first {
		if len(in[p.Rule+"\x00"+p.Message]) == len(blockShapes) {
			lines = append(lines, fmt.Sprintf("[%s] %s", p.Rule, strings.TrimPrefix(p.Message, "pools[0].")))
		}
	}
	if len(lines) == 0 {
		return nil
	}
	return fmt.Errorf("vllm: the engine block fails vLLM %s's rules: %s", pack.EngineVersion(),
		strings.Join(lines, "; "))
}
