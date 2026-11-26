// Original to blis-sglang.

package sglang

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
)

// SchedArgs are the resolved ServerArgs the scheduler reads, by SGLang's names
// (arg_groups/fields/schedule.py, memory.py, disagg.py@297e54fd). The caller resolves them
// (config.Resolve) and hands the values over; this package reads no ServerArgs itself.
type SchedArgs struct {
	PageSize                   int
	SchedulePolicy             string
	MaxPrefillTokens           int
	ChunkedPrefillSize         *int
	EnableMixedChunk           bool
	DisableRadixCache          bool
	DisableOverlapSchedule     bool
	ScheduleConservativeness   float64
	PrefillDecodeInterval      *int
	PrefillDecodeIntervalMinBs int
	EnablePriorityScheduling   bool
	DisablePriorityPreemption  bool
	// SchedulePriorityThreshold is priority_scheduling_preemption_threshold.
	PriorityPreemptionThreshold    int
	ScheduleLowPriorityValuesFirst bool
	AbortOnPriorityWhenDisabled    bool
	MaxQueuedRequests              *int
	PrefillMaxRequests             *int
	RetractionPolicy               string
	AllowAutoTruncate              bool
	RandomSeed                     int
	DisaggregationMode             string
	PPSize                         int
	PPMaxMicroBatchSize            *int
	RadixEvictionPolicy            string
	EnableMetrics                  bool
	DecodeLogInterval              int
	// SpeculativeNumSteps and SpeculativeNumDraftTokens are speculative_num_steps and
	// speculative_num_draft_tokens of an EAGLE-family algorithm at top-k 1; zero is no speculation.
	SpeculativeNumSteps       int
	SpeculativeNumDraftTokens int
}

// Env is the environment variables the scheduler's settings read (environ.py@297e54fd).
type Env struct {
	InitNewTokenRatio, MinNewTokenRatioFactor float64
	NewTokenRatioDecaySteps                   float64
	RetractDecodeSteps                        int
	// ClipMaxNewTokens is SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION (schedule_policy.py:89).
	ClipMaxNewTokens                 int
	DisableConsecutivePrefillOverlap bool
}

// DefaultEnv is the environment of a process that sets none of them.
func DefaultEnv() Env {
	return Env{InitNewTokenRatio: 0.7, MinNewTokenRatioFactor: 0.14, NewTokenRatioDecaySteps: 600,
		RetractDecodeSteps: 20, ClipMaxNewTokens: 4096}
}

// Runtime is what SGLang's model runner and model config hand the scheduler: the pool sizes the
// model runner derived from the device (tp_worker.py get_worker_info, :550-588@297e54fd) and the
// model's own facts.
type Runtime struct {
	MaxTotalNumTokens  int
	ReqPoolSize        int
	MaxRunningRequests int
	ContextLen         int
	EOSTokenIDs        []int64
	VocabSize          int
	// BudgetBytes is the memory the KV pool may take: _profile_available_bytes, the device's free
	// memory after the weights less the static memory fraction's slack
	// (kv_cache_configurator.py:2990-3060@297e54fd). CellBytes is the pool's per-token cost, the
	// figure TokenCapacity divides the budget by. They are the sizing's own terms, so a report of
	// the pool states what this sizing used rather than recomputing it.
	BudgetBytes int64
	CellBytes   int64
}

// extraMaxContextLen is the request pool's row slack beyond the context length for a model with
// no speculative decoding (kv_cache_configurator.py, extra_max_context_len).
const extraMaxContextLen = 4

// requestRowLen is the request pool's row length: the context length plus the headroom of
// get_req_to_token_extra_context_len (allocation_sizing.py:96-116@297e54fd), 4 plus the draft tokens, and
// at a page size above one at least the double-buffered decode reserve plus the page's rounding.
func requestRowLen(a SchedArgs, contextLen int) int {
	extra := extraMaxContextLen
	if a.SpeculativeNumDraftTokens > 0 {
		alloc := max(a.SpeculativeNumSteps, a.SpeculativeNumDraftTokens)
		extra = extraMaxContextLen + a.SpeculativeNumDraftTokens
		if a.PageSize > 1 {
			extra = max(extra, 2*alloc+a.PageSize-1)
		}
	}
	return contextLen + extra
}

// SchedConfig is the settings SGLang's Scheduler.__init__ derives from resolved ServerArgs, the
// model runner's pool sizes and the environment (scheduler.py:480-735, 1195-1250, 1352-1380,
// 1465-1474@297e54fd), as the scheduler is configured from them.
func SchedConfig(a SchedArgs, rt Runtime, env Env) (sched.Config, error) {
	if a.PageSize < 1 {
		return sched.Config{}, fmt.Errorf("sglang: page_size %d", a.PageSize)
	}
	if a.PPSize < 1 {
		a.PPSize = 1
	}
	policy, err := sched.ResolvePolicy(a.SchedulePolicy, a.DisableRadixCache)
	if err != nil {
		return sched.Config{}, err
	}
	// scheduler.py:1360-1373: a chunk size that is not positive means chunked prefill is off.
	chunk := a.ChunkedPrefillSize
	if chunk != nil && *chunk <= 0 {
		chunk = nil
	}
	// tp_worker.py:568-572.
	maxReqLen := min(rt.ContextLen-1, rt.MaxTotalNumTokens-1)
	// scheduler.py:1237-1244.
	microBatch := max(rt.MaxRunningRequests/a.PPSize, 1)
	if a.PPMaxMicroBatchSize != nil && *a.PPMaxMicroBatchSize != 0 {
		microBatch = *a.PPMaxMicroBatchSize
	}
	interval := 0
	if a.PrefillDecodeInterval != nil {
		interval = *a.PrefillDecodeInterval
	}
	// NewTokenRatioTracker.from_config (new_token_ratio_tracker.py:21-31).
	initRatio := min(env.InitNewTokenRatio*a.ScheduleConservativeness, 1.0)
	minRatio := min(initRatio*env.MinNewTokenRatioFactor, 1.0)
	evict := a.RadixEvictionPolicy
	if evict == "" {
		evict = "lru"
	}
	retraction := a.RetractionPolicy
	if retraction == "" {
		retraction = "length"
	}
	allocator := "TokenToKVPoolAllocator"
	if a.PageSize != 1 {
		allocator = "PagedTokenToKVPoolAllocator"
	}
	mode := a.DisaggregationMode
	if mode == "" {
		mode = "null"
	}
	return sched.Config{
		PageSize:            a.PageSize,
		Policy:              policy,
		MaxTotalNumTokens:   rt.MaxTotalNumTokens,
		MaxPrefillTokens:    a.MaxPrefillTokens,
		ChunkedPrefillSize:  chunk,
		MaxRunningRequests:  rt.MaxRunningRequests,
		ReqPoolSize:         rt.ReqPoolSize,
		MaxContextLen:       requestRowLen(a, rt.ContextLen),
		PPMaxMicroBatchSize: microBatch,
		PPSize:              a.PPSize,
		MaxReqLen:           maxReqLen,
		MaxReqInputLen:      maxReqLen - 5,
		IsMixedChunk:        chunk != nil && a.EnableMixedChunk,
		Disable:             a.DisableRadixCache,
		// kv_cache_configurator.py:932,1025,1511,2805.
		NeedSort:    mode == "decode" || mode == "prefill",
		EOSTokenIDs: rt.EOSTokenIDs,
		VocabSize:   rt.VocabSize,
		NewTokenRatio: sched.NewTokenRatioConfig{Init: initRatio, Min: minRatio,
			Decay: (initRatio - minRatio) / env.NewTokenRatioDecaySteps},
		RetractDecodeSteps:                 env.RetractDecodeSteps,
		ClipMaxNewTokens:                   env.ClipMaxNewTokens,
		InBatchPrefixCheckThreshold:        32,
		InBatchPrefixDeprioritizeThreshold: 32,
		EnablePriorityScheduling:           a.EnablePriorityScheduling,
		PrefillDecodeInterval:              interval,
		PrefillDecodeIntervalMinBs:         a.PrefillDecodeIntervalMinBs,
		Allocator:                          allocator,
		TreeCache:                          "UnifiedRadixCache",
		DisaggregationMode:                 mode,
		EnableOverlap:                      !a.DisableOverlapSchedule,
		DisableConsecutivePrefillOverlap:   env.DisableConsecutivePrefillOverlap,
		EnableMetrics:                      a.EnableMetrics,
		AllowAutoTruncate:                  a.AllowAutoTruncate,
		ContextLen:                         rt.ContextLen,
		DecodeLogInterval:                  a.DecodeLogInterval,
		RadixEvictionPolicy:                evict,
		RetractionPolicy:                   retraction,
		RandomSeed:                         a.RandomSeed,
		ScheduleLowPriorityValuesFirst:     a.ScheduleLowPriorityValuesFirst,
		PriorityPreemptionThreshold:        a.PriorityPreemptionThreshold,
		// scheduler.py:1469-1472.
		EnablePriorityPreemption:    a.EnablePriorityScheduling && !a.DisablePriorityPreemption,
		AbortOnPriorityWhenDisabled: a.AbortOnPriorityWhenDisabled,
		MaxQueuedRequests:           a.MaxQueuedRequests,
		PrefillMaxRequests:          a.PrefillMaxRequests,
		SpecNumSteps:                a.SpeculativeNumSteps,
		SpecNumDraftTokens:          a.SpeculativeNumDraftTokens,
	}, nil
}

// PoolInput is what sizes the KV pool: the device, the settings that bound it, and the model's
// context length.
type PoolInput struct {
	// PreModelLoadGiB is the device's free memory before the weights load (get_available_gpu_memory
	// before load_model); the host states it, since a simulator has no device to ask.
	PreModelLoadGiB    float64
	MemFractionStatic  float64
	PageSize           int
	MaxTotalTokens     *int
	MaxRunningRequests *int
	ContextLen         int
	// AttnDPSize above one (DP attention) is refused until S35.
	AttnDPSize int
}

// PoolPlan sizes the pools as SGLang's KV cache configurator does from the memory the kernel's
// cost model holds resident: the weights leave the free memory the budget is read from
// (_profile_available_bytes), the budget over the per-token KV bytes is the token capacity
// (config_from_budget), and the request pool follows (resolve_max_num_reqs;
// kv_cache_configurator.py:2990-3060, :3119-3229@297e54fd). The returned Runtime carries the
// sizes only; the caller adds the model's facts.
func PoolPlan(k kernel.Kernel, in PoolInput) (Runtime, error) {
	if in.AttnDPSize > 1 {
		return Runtime{}, fmt.Errorf("sglang: pool sizing under DP attention is not modelled (inference-sim/inference-sim#2356)")
	}
	const gib = float64(1 << 30)
	weights := float64(k.FixedBytes().Weights) / gib
	budget, err := cache.KVBudgetBytes(cache.Profile{AvailableGiB: in.PreModelLoadGiB - weights,
		PreModelLoadGiB: in.PreModelLoadGiB, MemFractionStatic: in.MemFractionStatic})
	if err != nil {
		return Runtime{}, err
	}
	// The per-token cell: the kernel's variable bytes over a page-aligned span, so a model whose
	// bytes come in pages is not rounded to zero.
	const span = 4096
	cell := (k.SequenceVariableBytes(span) + span - 1) / span
	tokens, err := cache.TokenCapacity(budget, cell, in.PageSize, in.MaxTotalTokens, in.ContextLen)
	if err != nil {
		return Runtime{}, err
	}
	reqs := cache.MaxNumReqs(tokens, in.ContextLen, in.MaxRunningRequests, 1)
	return Runtime{MaxTotalNumTokens: tokens, ReqPoolSize: reqs, MaxRunningRequests: reqs, ContextLen: in.ContextLen,
		BudgetBytes: budget, CellBytes: cell}, nil
}
