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
	// MambaStrategy is the resolved mamba radix cache strategy (no_buffer, extra_buffer or
	// extra_buffer_lazy) of a hybrid model that keeps its states in the radix cache; empty for any
	// other model, whose state (if any) is the one latest state a request holds.
	MambaStrategy string
	// HybridCacheCheckpointsLevel, MambaTrackInterval and MambaMaxStatesPerPath are the settings of the
	// same names. MambaStateChunkSize is the model's mamba_chunk_size, 0 for the kernel's default (64).
	HybridCacheCheckpointsLevel int
	MambaTrackInterval          int
	MambaMaxStatesPerPath       *int
	MambaStateChunkSize         int
}

// Env is the environment variables the scheduler's settings read (environ.py@297e54fd).
type Env struct {
	InitNewTokenRatio, MinNewTokenRatioFactor float64
	NewTokenRatioDecaySteps                   float64
	RetractDecodeSteps                        int
	// ClipMaxNewTokens is SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION (schedule_policy.py:89).
	ClipMaxNewTokens           int
	DisableOverlapAfterPrefill bool
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
	// MambaSize is the state slots of a hybrid model's pool (max_mamba_cache_size after the solve of
	// _handle_max_mamba_cache); 0 is a model with none.
	MambaSize int
}

// requestRowLen is the request pool's row length: the context length plus the headroom of
// get_req_to_token_headroom (mem_cache/allocation_sizing.py:150-160@c4af57ff). A step without
// speculation writes one slot past the context, whatever the page size. A speculative step runs
// get_decode_reserve() ahead of the committed length, twice the larger of the steps and the draft
// tokens (a chain, top-k 1), and a paged allocator rounds that up by the page size less one.
func requestRowLen(a SchedArgs, contextLen int) int {
	if a.SpeculativeNumDraftTokens <= 0 {
		return contextLen + 1
	}
	reserve := 2 * max(a.SpeculativeNumSteps, a.SpeculativeNumDraftTokens)
	return contextLen + reserve + a.PageSize - 1
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
	mamba, err := mambaConfig(a, rt)
	if err != nil {
		return sched.Config{}, err
	}
	return sched.Config{
		Mamba:               mamba,
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
		DisableOverlapAfterPrefill:         env.DisableOverlapAfterPrefill,
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
	// Mamba is a hybrid model whose states live in the radix cache; nil is any other model.
	Mamba *MambaPoolInput
}

// MambaPoolInput is what sizes the state pool of a hybrid model (_handle_max_mamba_cache,
// kv_cache_configurator.py): the strategy, the two settings that bound the pool, and the cost
// model's bytes for the state a request holds.
type MambaPoolInput struct {
	Strategy       cache.MambaStrategy
	DisableOverlap bool
	// MaxMambaCacheSize is --max-mamba-cache-size, nil when unset; FullMemoryRatio is
	// --mamba-full-memory-ratio resolved (0.9 unset).
	MaxMambaCacheSize *int
	FullMemoryRatio   float64
	// StatePerReqBytes is mamba_cache_per_req on a rank: the convolution and the temporal state of every
	// recurrent layer. Zero takes it from the cost model's per-sequence bytes, which count the temporal
	// state alone.
	StatePerReqBytes int64
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
	if m := in.Mamba; m != nil {
		return hybridPoolPlan(k, in, *m, budget, cell)
	}
	tokens, err := cache.TokenCapacity(budget, cell, in.PageSize, in.MaxTotalTokens, in.ContextLen)
	if err != nil {
		return Runtime{}, err
	}
	reqs := cache.MaxNumReqs(tokens, in.ContextLen, in.MaxRunningRequests, 1)
	return Runtime{MaxTotalNumTokens: tokens, ReqPoolSize: reqs, MaxRunningRequests: reqs, ContextLen: in.ContextLen}, nil
}

// mambaConfig is the recurrent-state settings the scheduler reads for a hybrid model whose states
// live in the radix cache (get_exec().mamba, HybridReqToTokenPool.__init__, memory_pool.py:1380-1425,
// and mamba_path_state_cap_of, arg_groups/overrides.py:274-278@c4af57ff); nil for any other model.
func mambaConfig(a SchedArgs, rt Runtime) (*sched.MambaConfig, error) {
	if a.MambaStrategy == "" {
		return nil, nil
	}
	extra := a.MambaStrategy == string(cache.MambaExtraBuffer) || a.MambaStrategy == string(cache.MambaExtraBufferLazy)
	lazy := a.MambaStrategy == string(cache.MambaExtraBufferLazy)
	overlap := !a.DisableOverlapSchedule
	level := a.HybridCacheCheckpointsLevel
	if level == 0 {
		level = 2
	}
	cap := 1
	if level >= 3 {
		cap = -1
	}
	if a.MambaMaxStatesPerPath != nil && *a.MambaMaxStatesPerPath != cap {
		return nil, fmt.Errorf("sglang: mamba_max_states_per_path=%d is not modelled: the tree's cap is checked at 1 (checkpoint level 2) and none (level 3) (inference-sim/inference-sim#2366)", *a.MambaMaxStatesPerPath)
	}
	chunk := a.MambaStateChunkSize
	if chunk == 0 {
		chunk = 64
	}
	if hi, lo := max(chunk, a.PageSize), min(chunk, a.PageSize); hi%lo != 0 {
		return nil, fmt.Errorf("sglang: for SSM models, either chunk_size or page_size must be divisible by the other, got chunk_size=%d, page_size=%d", chunk, a.PageSize)
	}
	return &sched.MambaConfig{ExtraBuffer: extra, Lazy: lazy, PathStateCap: cap, CheckpointsLevel: level,
		MambaSize: rt.MambaSize, PingPongSize: cache.MambaTrackingSlots(true, overlap, false),
		InitialTrackingSlots: cache.MambaTrackingSlots(extra, overlap, lazy), TrackInterval: a.MambaTrackInterval,
		StateChunkSize: chunk, TreeSlotInterval: max(chunk, a.PageSize)}, nil
}

// hybridPoolPlan sizes the pools of a hybrid model that keeps its states in the radix cache: the
// state pool takes its share of the budget (MambaCacheSize), the full-attention KV pool the rest, and
// the request pool is capped by the state slots a running request is charged (MaxNumReqsMamba,
// resolve_max_num_reqs; kv_cache_configurator.py:3119-3175, 3229-3411@297e54fd). One state a request
// holds is the cost model's per-sequence fixed bytes over the states the plan keeps it in (its working
// state and the tracking slots).
func hybridPoolPlan(k kernel.Kernel, in PoolInput, m MambaPoolInput, budget int64, cell int64) (Runtime, error) {
	const gib = float64(1 << 30)
	held := int64(1)
	if m.Strategy == cache.MambaExtraBuffer || m.Strategy == cache.MambaExtraBufferLazy {
		held += int64(cache.MambaTrackingSlots(true, !m.DisableOverlap, m.Strategy == cache.MambaExtraBufferLazy))
	}
	perReq := m.StatePerReqBytes
	if perReq <= 0 {
		perReq = k.SequenceFixedBytes() / held
	}
	size, restGiB, err := cache.MambaCacheSize(cache.MambaBudget{TotalRestGiB: float64(budget) / gib, PerReqBytes: perReq,
		MaxRunningRequests: in.MaxRunningRequests, MaxMambaCacheSize: m.MaxMambaCacheSize, AttnDPSize: 1,
		FullMemoryRatio: m.FullMemoryRatio, Strategy: m.Strategy, DisableOverlap: m.DisableOverlap})
	if err != nil {
		return Runtime{}, err
	}
	tokens, err := cache.TokenCapacity(int64(restGiB*gib), cell, in.PageSize, in.MaxTotalTokens, in.ContextLen)
	if err != nil {
		return Runtime{}, err
	}
	ratio, err := cache.MambaRatio(false, m.Strategy, m.DisableOverlap)
	if err != nil {
		return Runtime{}, err
	}
	reqs, err := cache.MaxNumReqsMamba(tokens, in.ContextLen, in.MaxRunningRequests, 1, size, ratio)
	if err != nil {
		return Runtime{}, err
	}
	return Runtime{MaxTotalNumTokens: tokens, ReqPoolSize: reqs, MaxRunningRequests: reqs, ContextLen: in.ContextLen, MambaSize: size}, nil
}
