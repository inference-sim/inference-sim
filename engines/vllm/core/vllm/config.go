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

	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"

	"github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/placement"
	"github.com/inference-sim/blis-vllm/core/sched"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
	"github.com/inference-sim/blis-vllm/rules/v0_32"
)

// Config is the engine block of one pool, as the simulator hands it over in engine.Spec.Config.
//
// The pool's engine settings are blis-vllm's engineconfig.Engine, inline, which is what a
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
	engineconfig.Engine   `yaml:",inline"`
	NumGPUBlocks          int   `yaml:"num_gpu_blocks"`
	PipelineParallelSize  *int  `yaml:"pipeline_parallel_size"`
	UseV2ModelRunner      *bool `yaml:"use_v2_model_runner,omitempty"`
	MTPNextnPredictLayers int   `yaml:"mtp_nextn_predict_layers,omitempty"`
	// DataParallelSize is the pool's parallel.dp, restated as pipeline_parallel_size is: above 1
	// the factory builds a data-parallel group of that many ranks behind one endpoint (DPGroup).
	// Unstated or 1 is one engine.
	DataParallelSize int `yaml:"data_parallel_size,omitempty"`
	// MambaModel is whether the model runs vLLM's MambaModelConfig hook: a Mamba or hybrid
	// attention/Mamba model (ModelConfig.is_hybrid, vllm/config/model.py:2023-2032@5df63a72, or a
	// Mamba architecture, vllm/model_executor/models/config.py:1073-1095@5df63a72). It is the model's,
	// not the deployment's, so the caller states it from the graph's recurrent layers. Unstated is a
	// model without recurrent layers. checkMambaCache reads it.
	MambaModel bool `yaml:"mamba_model,omitempty"`
	// MambaSSMCacheDTypeDefault is what the model's hook resolves mamba_ssm_cache_dtype "auto" to:
	// NemotronH's checkpoint value or float32 (config.py:716-732@5df63a72) and Qwen3.5's checkpoint
	// value (:850-866). It is the checkpoint's, so the caller states it; empty means the model's hook
	// rewrites nothing and the default stays "auto".
	MambaSSMCacheDTypeDefault string `yaml:"mamba_ssm_cache_dtype_default,omitempty"`
	// KVLayers are the model's attention layers by KV cache spec, which vLLM groups into the
	// scheduler's KV cache groups (KVGroups). Unstated is one full-attention group.
	KVLayers []KVLayerRun `yaml:"kv_layers,omitempty"`
	// DPStatsIntervalUs is how often the DP coordinator publishes the ranks' request counts to
	// the front end's load balancer: min_stats_update_interval_ms = 100
	// (vllm/v1/engine/coordinator.py:154-162, 267-269@5df63a72), 100000 µs unstated. A
	// behavioural test states it as a step count's worth of time.
	DPStatsIntervalUs int64 `yaml:"dp_stats_interval_us,omitempty"`
	// KVAvailableBytes is the memory one rank's KV cache was sized from, when the caller sized
	// num_gpu_blocks from memory: the start-up check then states vLLM's numbers in GiB
	// (kv_cache_utils.py:1031-1040@5df63a72). Unstated, it states them in blocks.
	KVAvailableBytes int64 `yaml:"kv_available_bytes,omitempty"`
	// Placement is where the pool sits: its chip, network, layout and the node this engine is on. The
	// engine states every transfer it starts against it (engine.Transfer.Move) and builds the plan
	// those transfers are priced against from it (Plan), so a pool that can start a transfer (a P/D
	// connector or a KV offload) must state it, and a refusal says so. A pool that starts none needs
	// no placement, and its Plan is empty.
	Placement *placement.Placement `yaml:"placement,omitempty"`
	// KVTransfer is the pool's kv_transfer_config for P/D: the connector and the pool's kv_role.
	// The deployment states the connector (pd_transfer.connector) and the pool's role; a caller
	// restates them here. Unstated is no connector.
	KVTransfer *connectorConfig `yaml:"kv_transfer,omitempty"`
	// KVOffload is the pool's KV offload connector and its CPU tier (offload.go). The deployment
	// states it (offload: connector, tiers, eviction_policy); a caller restates it here with the
	// bytes a block its memory model sizes. Unstated is no offload.
	KVOffload *offloadConfig `yaml:"kv_offload,omitempty"`

	// statedZero names the scheduler fields the block stated as 0 (max_num_seqs,
	// max_num_batched_tokens). The settings struct reads 0 as unstated, and a block that states 0
	// is refused for vLLM's own rule (ge=1) rather than told to state the value it did state.
	// decodeConfig fills it; a Config built in Go has none.
	statedZero []string
}

// decodeConfig reads an engine block strictly: a key that is not a field is an error, as it
// is for every blis-schemas loader, so a misspelt setting can never be silently dropped.
func decodeConfig(node yaml.Node) (Config, error) {
	var c Config
	if node.Kind == 0 {
		return c, fmt.Errorf("vllm: the engine block is empty")
	}
	// The block is written in vLLM's own setting names (blis-vllm#93); Config decodes the Go-shaped
	// block, with the factory's own extras beside it.
	inner, err := engineconfig.FromVLLM(&node)
	if err != nil {
		return c, fmt.Errorf("vllm: engine block: %w", err)
	}
	raw, err := yaml.Marshal(inner)
	if err != nil {
		return c, fmt.Errorf("vllm: engine block: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return c, fmt.Errorf("vllm: engine block: %w", err)
	}
	c.statedZero = statedZeroFields(node)
	return c, nil
}

// statedZeroFields lists the scheduler fields a block states as 0, which the decoded integer
// cannot tell from a field left out.
func statedZeroFields(node yaml.Node) []string {
	var zero []string
	if node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		key, val := node.Content[i].Value, node.Content[i+1]
		if key != "max_num_seqs" && key != "max_num_batched_tokens" {
			continue
		}
		var n int
		if val.Tag == "!!int" && val.Decode(&n) == nil && n == 0 {
			zero = append(zero, key)
		}
	}
	return zero
}

// settings are a Config resolved into the scheduler's and engine core's inputs.
type settings struct {
	sched     sched.Config
	queueSize int
	v2Runner  bool
	// connector is the pool's P/D connector, nil for none; pp is pipeline_parallel_size, which
	// vLLM's NIXL worker checks it against.
	connector *connectorConfig
	pp        int
	// offload is the pool's KV offload, nil for none; specMethod the drafter's method.
	offload    *offloadConfig
	specMethod string
	// placement is where the pool sits, nil when the caller states none (Config.Placement).
	placement *placement.Placement
	// dp is the data-parallel group: ranks (1 for one engine), the sync interval, the
	// coordinator's publish interval and the prefill throttle interval.
	dp dpSettings
	// hashSeed is PYTHONHASHSEED when the deployment sets it (nil: unset), which roots every
	// request's block-hash chain (resolve_none_hash_seed, vllm/v1/core/kv_cache_utils.py:131-146
	// @5df63a72). It changes the hashes a KV-event stream publishes, never which blocks hit.
	hashSeed *string
}

// dpSettings are a data-parallel group's settings.
type dpSettings struct {
	ranks, syncInterval, prefillInterval int
	statsIntervalUs                      int64
}

// maxModelLenIssue is the open issue an unstated max_model_len cites (design X28).
const maxModelLenIssue = "https://github.com/inference-sim/blis-vllm/issues/69"

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
		// (vllm/config/model.py@5df63a72); either is the caller's to resolve. The engine will
		// resolve an unstated one from the graph's context window, with the model values moved
		// into graph fields (https://github.com/inference-sim/blis-vllm/issues/69).
		return s, fmt.Errorf("vllm: max_model_len must be stated; an unstated or auto window is the model's "+
			"or the memory's, which the caller resolves (%s)", maxModelLenIssue)
	}
	// vLLM refuses to start when the KV cache cannot hold one max_model_len sequence: a
	// full-attention group needs cdiv(max_model_len, block_size) blocks
	// (FullAttentionSpec.max_memory_usage_bytes, vllm/v1/kv_cache_interface.py:602-607@5df63a72),
	// checked against the blocks left beside the null block
	// (vllm/v1/core/kv_cache_utils.py:2627-2647, 1003-1040@5df63a72). A request the cache could
	// never hold would otherwise wait for ever.
	// A model stating its KV layers is checked over its KV cache groups instead, with vLLM's own
	// message (checkKVLayersFit, check_enough_kv_cache_memory): a hybrid model's groups share the
	// pool, so one full-attention group's need is not the whole of it.
	if need := (maxLen + blockSize - 1) / blockSize; len(c.KVLayers) == 0 && need > c.NumGPUBlocks-1 {
		return s, fmt.Errorf("vllm: num_gpu_blocks %d cannot hold one max_model_len %d sequence (%d blocks of %d "+
			"beside the null block), which vLLM refuses at startup (vllm/v1/core/kv_cache_utils.py:2638-2647@5df63a72)",
			c.NumGPUBlocks, maxLen, need, blockSize)
	}
	seqs, budget := e.MaxNumSeqs, e.MaxNumBatchedTokens
	if len(c.statedZero) > 0 {
		// A stated 0 is not an unstated value: vLLM's SchedulerConfig refuses it (ge=1).
		return s, fmt.Errorf("vllm: %s is stated as 0, below the minimum of 1, which vLLM refuses "+
			"(vllm/config/scheduler.py:48,62@5df63a72, ge=1)", strings.Join(c.statedZero, " and "))
	}
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
				"eagle3, mtp, dspark and ngram, whose scheduler numbers the engine block fully states (docs/feature-matrix.md)", method)
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
	if e.SWABoundedReplay != nil && !*e.SWABoundedReplay {
		// CacheConfig.swa_bounded_replay: true (vllm/config/cache.py:250@5df63a72). False makes
		// DeepSeek-V4.1's sliding-window MLA specs prefix-cacheable with no replay window
		// (SlidingWindowMLASpec.prefix_cacheable and prefix_replay_tokens,
		// vllm/v1/kv_cache_interface.py:968-974), which is the layout the V1 runner and prefill
		// context parallelism force (vllm/models/deepseek_v41/attention.py:529-547). It is carried
		// by the layers: each kv_layers run states its own bounded_replay, which the host sets
		// from this flag, and the groups, the pool and the lookup follow the runs. A run that
		// still states a bounded replay contradicts the flag.
		for _, r := range c.KVLayers {
			if r.BoundedReplay {
				return s, fmt.Errorf("vllm: swa_bounded_replay is false but a %s run in kv_layers states "+
					"bounded_replay: the layers of a deployment without bounded replay state none", r.Spec)
			}
		}
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
	if async && e.AsyncScheduling != nil && !asyncSpecMethod(method) {
		// vLLM refuses a stated async_scheduling with a drafter that is not EAGLE-family, ngram_gpu,
		// draft_model, DSpark or DFlash (vllm/config/vllm.py:1566-1576@5df63a72); the unstated
		// default falls back to synchronous instead (resolveAsync).
		return s, fmt.Errorf("vllm: async_scheduling is only supported with EAGLE, MTP, draft_model, ngram_gpu, "+
			"DSpark or DFlash speculative decoding, and speculative method %q is not one of them "+
			"(vllm/config/vllm.py:1566-1576@5df63a72)", method)
	}
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
	if err := checkMambaCache(c, prefixCaching); err != nil {
		return settings{}, err
	}
	chunked := ResolveChunkedPrefill(e.EnableChunkedPrefill)
	maxLoRAs := 0
	if e.MaxLoRAs > 0 {
		maxLoRAs = e.MaxLoRAs
	}
	groups, err := KVGroups(c.KVLayers, e.DisableHybridKVCacheManager != nil && *e.DisableHybridKVCacheManager,
		trailingDrafter(method, c.KVLayers))
	if err != nil {
		return settings{}, err
	}
	if err := checkKVLayersFit(c, blockSize, maxLen, queue*budget, spec.PrefillLookahead, method); err != nil {
		return settings{}, err
	}
	if err := checkChunkedPrefill(c, chunked, chunkedInput{prefixCaching: prefixCaching, maxModelLen: maxLen,
		budget: budget, groups: groups, method: method, pp: pp}); err != nil {
		return settings{}, err
	}
	s.sched = sched.Config{
		MaxNumSeqs:                seqs,
		MaxNumActiveSeqs:          e.MaxNumActiveSeqs,
		MaxNumBatchedTokens:       budget,
		MaxNumScheduledTokens:     e.MaxNumScheduledTokens,
		LongPrefillTokenThreshold: e.LongPrefillTokenThreshold,
		LongPrefillAdaptive:       e.LongPrefillTokenThresholdAdaptive != nil && *e.LongPrefillTokenThresholdAdaptive,
		// enable_chunked_prefill defaults to true (vllm/config/scheduler.py:134@5df63a72); the
		// stated value stands (ResolveChunkedPrefill, checkChunkedPrefill).
		EnableChunkedPrefill: chunked,
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
	if v, ok := e.Env["PYTHONHASHSEED"]; ok {
		s.hashSeed = &v
	}
	s.connector, s.pp = c.KVTransfer, pp
	s.offload, s.specMethod = c.KVOffload, method
	if c.Placement != nil {
		if err := c.Placement.Validate(); err != nil {
			return settings{}, fmt.Errorf("vllm: placement: %w", err)
		}
	}
	s.placement = c.Placement
	if s.offload, err = checkConnectorPair(s.connector, s.offload); err != nil {
		return settings{}, err
	}
	if s.offload != nil && c.DataParallelSize > 1 {
		// Each rank would run its own offload connector and CPU tier; the group aggregates a P/D
		// connector's transfers (DPGroup) but not the offload's jobs, which it routes by request.
		return settings{}, fmt.Errorf("vllm: kv_offload with data parallelism (dp %d) is not built yet: the "+
			"data-parallel group does not carry an offload connector's jobs (%s)", c.DataParallelSize, "https://github.com/inference-sim/blis-vllm/issues/47")
	}
	if s.connector != nil && s.connector.Role.producer() && c.DataParallelSize > 1 && c.Placement != nil {
		// A handoff names no endpoint (engine.Handoff), so the decode side is told only the prefill
		// instance's first node (engine.Request.Peer). A prefill group whose ranks sit on several nodes
		// would price every pull from the first node's links, wrong for a request computed on another
		// rank. This is a limit of the contract (inference-sim/blis-substrate#48), not of vLLM, which
		// starts such a group; it is refused until Handoff carries the owning rank's endpoint. A decode
		// group has no such limit: each rank states its own node as the end of its pulls.
		if n := c.Placement.GroupNodes(c.DataParallelSize, rankWidth(*c.Placement)); n > 1 {
			return settings{}, fmt.Errorf("vllm: a %s group of %d data-parallel ranks spans %d nodes: the engine contract's handoff "+
				"names no endpoint, so a pull could not be priced from the node of the rank that holds the KV "+
				"(inference-sim/blis-substrate#48); vLLM itself starts such a group. Keep a prefill group on one node, "+
				"or wait for a handoff that carries its rank's endpoint", c.KVTransfer.Role, c.DataParallelSize, n)
		}
	}
	s.dp = dpSettings{ranks: max(c.DataParallelSize, 1), syncInterval: e.DPSyncInterval,
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
	if pp > 1 && !async {
		// Synchronous pipeline parallelism runs vLLM's batch queue at max_concurrent_batches = pp
		// without the async scheduler (vllm/config/vllm.py:577-588@5df63a72). The adapter's loop
		// matches step_with_batch_queue (core.py:698-810@5df63a72) with async scheduling on the V2
		// runner (TestBatchQueueAgainstVLLM: t3-async-v2-pp2, t3-v2-pp2-async-stagger), but not
		// without it (t3-pp2-concurrent and t3-pp4-sync diverge), so it is refused rather than run.
		return settings{}, fmt.Errorf("vllm: synchronous pipeline parallelism (pp %d, async_scheduling off) is not "+
			"built yet: the adapter's batch queue matches vLLM's step_with_batch_queue only with async scheduling "+
			"on the V2 runner (https://github.com/inference-sim/blis-vllm/issues/21)", pp)
	}
	// The release's own accepted values and refusals, for what the checks above do not cover.
	if err := checkRules(e, pp); err != nil {
		return settings{}, err
	}
	// After the release's own checks, so a value the release rejects is reported as it rejects it.
	if err := refuseUnreadSettings(e); err != nil {
		return settings{}, err
	}
	return s, nil
}

// effectiveMambaCacheMode is the mamba_cache_mode that runs for a stated one. CacheConfig's
// initializer is "none" (vllm/config/cache.py:189@5df63a72). For a model that runs
// MambaModelConfig.verify_and_update_config the hook sets "align" when prefix caching is on and
// the stated mode is "none", and sets "none" when prefix caching is off and the stated mode is not
// (vllm/model_executor/models/config.py:685-706@5df63a72). It cannot tell a stated "none" from the
// initializer, so a stated "none" with prefix caching on runs as "align", and a stated "align"
// with it off runs as "none". Prefix caching defaults on (cache.py:141). For any other model no hook
// runs and the stated value, or the initializer, stands: only the Mamba layers' specs and
// backends read it, and the config validators that assert on "align" (vllm/config/vllm.py:3277-3281).
func effectiveMambaCacheMode(stated string, prefixCaching, hook bool) string {
	if stated == "" {
		stated = "none"
	}
	if !hook {
		return stated
	}
	if prefixCaching {
		return "align"
	}
	return "none"
}

// checkMambaCache refuses mamba_cache_mode and mamba_ssm_cache_dtype only where no layer of the
// model reads them, or the value that runs is one this engine does not build: each is compared
// with what an unstated value resolves to for this model and this enable_prefix_caching, not
// with the field's initializer (X1).
//
// mamba_cache_mode: for a model with recurrent layers the hook resolves every stated value to the
// mode that prefix caching picks, "align" with it on and "none" with it off, and the KV cache
// groups carry that mode (kv.Group.MambaCacheMode), so nothing is refused. For another model the
// stated value stands, no layer reads it, and "align" is away from the default "none": it splits
// prefill chunks on block boundaries only for Mamba layers
// (vllm/v1/core/sched/scheduler.py:375-379@5df63a72), which this model lacks.
//
// mamba_ssm_cache_dtype: the initializer is "auto" (cache.py:185). NemotronH and Qwen3.5 replace a
// stated "auto" with the checkpoint's value (config.py:724-732, 864-866) and keep any other stated
// value. A model with recurrent layers sizes its state at the dtype that runs (the engine block's
// kv_layers state bytes), so any stated value is built; a model without them reads none, and a
// stated value away from the default would be dropped.
func checkMambaCache(c Config, prefixCaching bool) error {
	e := c.Engine
	stated := effectiveMambaCacheMode(e.MambaCacheMode, prefixCaching, c.MambaModel)
	if def := effectiveMambaCacheMode("", prefixCaching, c.MambaModel); stated != def {
		return fmt.Errorf("vllm: mamba_cache_mode %q runs as %q, away from the default %q for this model with enable_prefix_caching %t; "+
			"this model has no recurrent layer to read it, so a stated value would be dropped (%s)",
			e.MambaCacheMode, stated, def, prefixCaching, issueBase+"7")
	}
	if v := e.MambaSSMCacheDType; v != "" && v != "auto" && v != c.MambaSSMCacheDTypeDefault && !c.MambaModel {
		return fmt.Errorf("vllm: mamba_ssm_cache_dtype %q differs from the default %q, and this model has no recurrent layer "+
			"to read it, so a stated value would be dropped (%s)", v, defaultOr(c.MambaSSMCacheDTypeDefault, "auto"), issueBase+"83")
	}
	return nil
}

func defaultOr(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

// specMethodModelled reports whether this factory builds a draft method: one tier 2 diff-tests
// (docs/feature-matrix.md, "Speculative decoding") whose scheduler numbers depend only on the
// method and num_speculative_tokens.
func specMethodModelled(method string) bool {
	return slices.Contains([]string{"eagle", "eagle3", "mtp", "dspark", "ngram"}, method)
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

// asyncSpecMethod is whether async scheduling runs with a drafter of this method: none, an
// EAGLE-family method, ngram_gpu, draft_model, DSpark or DFlash (vllm/config/vllm.py:1566-1576).
func asyncSpecMethod(method string) bool {
	switch method {
	case "", "eagle", "eagle3", "mtp", "dflash", "dspark", "draft_model", "ngram_gpu":
		return true
	}
	return false
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
	if !asyncSpecMethod(method) {
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
func refuseUnmodelled(e engineconfig.Engine, dataParallel bool) error {
	if keys := unreadEnv(e.Env); len(keys) > 0 {
		// blis-schemas' Env rule: a consumer refuses, or discloses, every key it cannot honour,
		// and never ignores one (spec/deployment/deployment.go Engine.Env).
		return fmt.Errorf("vllm: env %s is not read by this engine, which reads only %s; the keys that "+
			"change step time are the latency kernel's to price (https://github.com/inference-sim/blis-vllm/issues/80)",
			strings.Join(keys, ", "), strings.Join(readEnv, ", "))
	}
	switch {
	case e.PrefixMatchUnit != 0:
		return fmt.Errorf("vllm: prefix_match_unit is not modelled (https://github.com/inference-sim/blis-vllm/issues/109)")
	case e.PrefixCacheRetentionInterval != nil && *e.PrefixCacheRetentionInterval != 0:
		// CacheConfig.prefix_cache_retention_interval: 0 (vllm/config/cache.py:159@5df63a72); a stated 0
		// is the default and keeps only the semantic checkpoints, which this engine builds.
		return fmt.Errorf("vllm: prefix_cache_retention_interval is not modelled (https://github.com/inference-sim/blis-vllm/issues/109)")
	case e.PrefillScheduleInterval > 1 && !dataParallel:
		// _should_throttle_prefills is the DP engine core's (vllm/v1/engine/core.py:2318-2325@5df63a72):
		// EngineCore's never throttles (:652-655), so on one engine the setting has no effect vLLM
		// would show, and stating it is refused rather than dropped.
		return fmt.Errorf("vllm: prefill_schedule_interval throttles a data-parallel group's ranks; this engine " +
			"has data_parallel_size 1")
	case e.MaxCPULoRAs != 0 && e.MaxCPULoRAs != max(e.MaxLoRAs, 1):
		// LoRAConfig.max_cpu_loras: None, which becomes max_loras (vllm/config/lora.py:42, 120-121@5df63a72),
		// and max_loras defaults to 1 (:35); a stated value equal to that is the default.
		return fmt.Errorf("vllm: max_cpu_loras is not modelled (https://github.com/inference-sim/blis-vllm/issues/5)")
	}
	// mamba_cache_mode and mamba_ssm_cache_dtype are checked on their resolved values (checkMambaCache),
	// and swa_bounded_replay after the model runner is resolved (resolve).
	return nil
}

// unreadSetting is an engine-block key that nothing in blis-vllm reads, neither this engine nor
// the latency kernel (enginesettings.Settings has no field for it), with the open issue that
// tracks modelling it. VLM-3 and X1: a field not modelled is refused when it is set away from
// vLLM's default, not when it is merely stated, so a user who hands BLIS the flags they gave vLLM,
// with a value equal to the default, is not refused. away reports whether e states the key at a
// value other than vLLM's default; each entry cites the line that sets the default.
type unreadSetting struct {
	key, field, issue string
	away              func(engineconfig.Engine) bool
}

const issueBase = "https://github.com/inference-sim/blis-vllm/issues/"

var unreadSettings = []unreadSetting{
	// CacheConfig.kv_cache_memory_bytes: None (vllm/config/cache.py:257@5df63a72).
	{"kv_cache_memory_bytes", "kv_cache_memory_bytes", issueBase + "81", func(e engineconfig.Engine) bool { return e.KVCacheMemoryBytes != nil }},
	// AttentionConfig.use_trtllm_attention: None, auto-detect (vllm/config/attention.py:74@5df63a72);
	// a stated true or false is a choice. Which prefill kernel runs is the engine's to resolve and
	// to report as the implementation of attention.prefill, which this engine does not do yet; the
	// kernel reads no such setting and would price a stated value as the default (VAL-8).
	{"use_trtllm_attention", "attention_config.use_trtllm_attention", issueBase + "80", func(e engineconfig.Engine) bool { return e.UseTRTLLMAttention != nil }},
	// AttentionConfig.mla_prefill_backend: None, selected automatically (vllm/config/attention.py:81@5df63a72).
	{"mla_prefill_backend", "attention_config.mla_prefill_backend", issueBase + "80", func(e engineconfig.Engine) bool { return e.MLAPrefillBackend != "" }},
	// KernelConfig.linear_backend: "auto" (vllm/config/kernel.py:350@5df63a72). A value is lower-cased
	// and its dashes become underscores before it is validated (kernel.py:393-397, arg_utils.py:1809),
	// so "AUTO" is the default too.
	{"linear_backend", "kernel_config.linear_backend", issueBase + "80",
		func(e engineconfig.Engine) bool {
			v := strings.ReplaceAll(strings.ToLower(e.LinearBackend), "-", "_")
			return v != "" && v != "auto"
		}},
	// CompilationConfig.custom_ops: [] (vllm/config/compilation.py:480@5df63a72).
	{"custom_ops", "compilation_config.custom_ops", issueBase + "80", func(e engineconfig.Engine) bool { return len(e.CustomOps) > 0 }},
	// CompilationConfig.decoder_replay_cudagraph_capture_sizes: [] (vllm/config/compilation.py:565@5df63a72).
	{"decoder_replay_cudagraph_capture_sizes", "compilation_config.decoder_replay_cudagraph_capture_sizes", issueBase + "80",
		func(e engineconfig.Engine) bool { return len(e.DecoderReplayCUDAGraphCaptureSizes) > 0 }},
	// SpeculativeConfig.kv_cache_dtype: None, the draft keeps the target's layout
	// (vllm/config/speculative.py:428@5df63a72). A stated "auto" is a value, not None: it overrides
	// the draft's cache_config.cache_dtype (:382), so it differs from the default.
	{"speculative.kv_cache_dtype", "speculative_config.kv_cache_dtype", issueBase + "80",
		func(e engineconfig.Engine) bool { return e.Speculative != nil && e.Speculative.KVCacheDType != "" }},
	// SpeculativeConfig.rejection_sample_method: "standard" (vllm/config/speculative.py:529@5df63a72).
	// Only "synthetic" fixes the acceptance length (:532-548). It is accepted with
	// synthetic_acceptance_length, the run's supplied acceptance profile, which vLLM resolves to the
	// rates the acceptance resolver derives; stated alone it is refused, as vLLM refuses it (:575).
	// "block", which recorded runs state, decides which drafts are accepted, and acceptance is an
	// input (#106).
	{"speculative.rejection_sample_method", "speculative_config.rejection_sample_method", issueBase + "106",
		func(e engineconfig.Engine) bool {
			return e.Speculative != nil && e.Speculative.RejectionSampleMethod == "synthetic" &&
				e.Speculative.SyntheticAcceptanceLength == nil
		}},
	// mamba_cache_mode and mamba_ssm_cache_dtype are not listed: a model hook rewrites them, so what
	// differs from the default is decided on the resolved value (checkMambaCache).
	// ParallelConfig.dbo_decode_token_threshold: 32 (vllm/config/parallel.py:262@5df63a72)..
	{"dbo.dbo_decode_token_threshold", "dbo_decode_token_threshold", issueBase + "74",
		func(e engineconfig.Engine) bool {
			return e.DBO != nil && e.DBO.DecodeTokenThreshold != nil && *e.DBO.DecodeTokenThreshold != 32
		}},
	// ParallelConfig.dbo_prefill_token_threshold: 512 (vllm/config/parallel.py:267@5df63a72).
	{"dbo.dbo_prefill_token_threshold", "dbo_prefill_token_threshold", issueBase + "74",
		func(e engineconfig.Engine) bool {
			return e.DBO != nil && e.DBO.PrefillTokenThreshold != nil && *e.DBO.PrefillTokenThreshold != 512
		}},
	// EPLBConfig.window_size: 1000 (vllm/config/parallel.py:66@5df63a72).
	{"eplb.window_size", "eplb_config.window_size", issueBase + "108",
		func(e engineconfig.Engine) bool {
			return e.EPLB != nil && e.EPLB.WindowSize != 0 && e.EPLB.WindowSize != 1000
		}},
	// EPLBConfig.step_interval: 3000 (vllm/config/parallel.py:68@5df63a72).
	{"eplb.step_interval", "eplb_config.step_interval", issueBase + "108",
		func(e engineconfig.Engine) bool {
			return e.EPLB != nil && e.EPLB.StepInterval != 0 && e.EPLB.StepInterval != 3000
		}},
	// KVEventsConfig.publisher: None, resolved in __post_init__ to "zmq" when enable_kv_cache_events
	// is set and "null" otherwise (vllm/config/kv_events.py:19, 51-52@5df63a72), so the default depends on
	// that setting.
	{"kv_events.publisher", "kv_events_config.publisher", issueBase + "60",
		func(e engineconfig.Engine) bool {
			if e.KVEvents == nil || e.KVEvents.Publisher == "" {
				return false
			}
			def := "null"
			if e.KVEvents.EnableKVCacheEvents {
				def = "zmq"
			}
			return e.KVEvents.Publisher != def
		}},
	// KVEventsConfig.topic: "" (vllm/config/kv_events.py:45@5df63a72).
	{"kv_events.topic", "kv_events_config.topic", issueBase + "60",
		func(e engineconfig.Engine) bool { return e.KVEvents != nil && e.KVEvents.Topic != "" }},
}

// RefuseUnreadSettings is refuseUnreadSettings for a host that opens the cost model first: the
// cost model's kernel also refuses a stated kernel-facing setting it cannot price, but by the
// registry's issue, and the engine refuses these with its own, so the host asks the engine first.
func RefuseUnreadSettings(e engineconfig.Engine) error { return refuseUnreadSettings(e) }

// refuseUnreadSettings refuses the first key of unreadSettings stated away from vLLM's default,
// naming the key, the field as vLLM names it and the issue.
func refuseUnreadSettings(e engineconfig.Engine) error {
	for _, u := range unreadSettings {
		if u.away(e) {
			return fmt.Errorf("vllm: %s (vLLM's %s) is not modelled: this engine and the cost model read no such setting, so a "+
				"value other than vLLM's default would be dropped (%s)", u.key, u.field, u.issue)
		}
	}
	return nil
}

// readEnv are the engine environment variables this engine reads (resolveV2Runner).
var readEnv = []string{"PYTHONHASHSEED", "VLLM_USE_V2_MODEL_RUNNER", "UCX_CUDA_IPC_ENABLE_MNNVL"}

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
func checkAdaptiveVerification(e engineconfig.Engine, pp int, v2 bool) error {
	sp := e.Speculative
	if sp == nil || sp.EnableAdaptiveVerification == nil || !*sp.EnableAdaptiveVerification {
		return nil
	}
	switch {
	case !v2:
		return fmt.Errorf("vllm: speculative_config.enable_adaptive_verification: adaptive draft verification runs only " +
			"on the V2 model runner, and vLLM refuses it on V1 (vllm/config/vllm.py:2980-2981@5df63a72)")
	case e.MaxLoRAs > 0:
		// LoRAConfig exists whenever LoRA is enabled, which max_loras states here
		// (vllm/engine/arg_utils.py:2650-2669@5df63a72).
		return fmt.Errorf("vllm: speculative_config.enable_adaptive_verification: \"Adaptive verification is not currently " +
			"compatible with LoRA\" (vllm/config/vllm.py:3005-3010@5df63a72)")
	case !fullCUDAGraphs(e.CUDAGraphMode):
		return fmt.Errorf("vllm: speculative_config.enable_adaptive_verification: \"Adaptive verification requires full CUDA "+
			"graphs\", and compilation_config.cudagraph_mode %s captures none (vllm/config/vllm.py:3012-3017@5df63a72)", e.CUDAGraphMode)
	case pp > 1:
		return fmt.Errorf("vllm: speculative_config.enable_adaptive_verification: \"Adaptive verification is not currently " +
			"compatible with pipeline parallelism\" (vllm/config/vllm.py:3019-3028@5df63a72)")
	}
	return nil
}

// fullCUDAGraphs is CUDAGraphMode.has_full_cudagraphs for a stated mode: FULL, FULL_DECODE_ONLY
// and FULL_AND_PIECEWISE capture full graphs (vllm/config/compilation.py:59-91@5df63a72). An
// unstated mode is vLLM's default, FULL_AND_PIECEWISE (vllm/config/vllm.py:303@5df63a72).
func fullCUDAGraphs(mode string) bool {
	// CompilationConfig upper-cases a string before it parses the mode
	// (vllm/config/compilation.py:873-879@5df63a72).
	switch strings.ToUpper(mode) {
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
// parallel shape in blockShapes (tensor and data widths at 1 and at 2, with and without expert
// parallelism), and only an error every run reports refuses. A clause that
// turns on a width the block does not state, such as engram-config-valid's "dp_shared_memory
// requires dp above 1" (vllm/config/engram.py@5df63a72), is left to the caller that validates
// the whole document set, as is every rule that needs the model graph, which checks nothing
// without one. Warnings are not refusals. pp is the pool's pipeline_parallel_size.
func checkRules(e engineconfig.Engine, pp int) error { return checkRulesWith(v0_32.Pack(), e, pp) }

// blockShapes are the parallel shapes checkRules runs the pack under: the tensor and data widths
// each at 1 and at 2, so that a clause turning on either width, whichever way, is silent in one of
// them.
var blockShapes = []struct {
	tp, dp         int
	expertParallel bool
}{
	{1, 1, false}, {1, 2, false}, {2, 1, false}, {2, 2, false},
	// Expert parallelism is a fact of the pool's parallel block too, and a rule that turns on it
	// (eplb-needs-expert-parallelism) must be silent in a shape that has it, so the shapes with
	// it are run beside the shapes without.
	{1, 1, true}, {1, 2, true}, {2, 1, true}, {2, 2, true},
}

// checkRulesWith is checkRules for a given rules pack.
func checkRulesWith(pack rules.EngineRules, e engineconfig.Engine, pp int) error {
	block, err := engineconfig.Encode(e)
	if err != nil {
		return fmt.Errorf("vllm: encoding the engine block: %w", err)
	}
	var first []rules.Problem       // the first shape's errors, in the order the pack reports them
	in := map[string]map[int]bool{} // an error's key: the shapes that reported it
	for i, shape := range blockShapes {
		doc := rules.Input{Deployment: &deployment.Deployment{Pools: []deployment.Pool{{
			Role: deployment.RoleColocated,
			Parallel: deployment.Parallelism{TP: shape.tp, PP: max(pp, 1), DP: shape.dp,
				EnableExpertParallel: shape.expertParallel},
			Engine: block,
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
