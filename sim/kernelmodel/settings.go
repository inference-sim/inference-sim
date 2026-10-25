package kernelmodel

import (
	"fmt"
	"math"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// Settings is everything a simulator sizes one engine from, every value answered by the
// kernel for the pool it prices. It exists so no consumer re-derives any of it: the KV
// budget comes from the kernel's memory methods, the admission caps and the engine knobs
// from the pool the kernel resolved, the data-parallel width from its resolution.
//
// Values are PER DATA-PARALLEL RANK -- one vLLM EngineCore. A simulator that runs each rank
// as its own replica uses them as they are; one that models the ranks as a single aggregate
// instance multiplies the caps by DataParallel and takes AggregateKVBlocks.
type Settings struct {
	// KVBlocks is one rank's usable KV blocks; AggregateKVBlocks is the budget of all ranks
	// together for an MoE model, and equal to KVBlocks for a dense one, as KVBudget.TotalBlocks
	// reports it (see KVBudget.DPScaled).
	KVBlocks          int64
	AggregateKVBlocks int64
	// KVBytesPerBlock is the kernel's per-sequence KV cost of one block, already
	// page-quantized and sharded by the parallel layout.
	KVBytesPerBlock int64
	// SequenceState is each resident sequence's bounded state in pool blocks, step by step
	// (#1941): Hold is the kernel's SequenceBoundedBytesAt in whole blocks, Gate the admission
	// gate's min(cdiv(tokens), cap). Zero value for a model with no bounded per-sequence state.
	// The byte and block figures beside it are the admission cap and the steady decode hold,
	// for the log and the capacity figure.
	SequenceState           sim.KVSequenceState
	SequenceAdmissionBlocks int64
	SequenceAdmissionBytes  int64
	SequenceSteadyBlocks    int64
	SequenceSteadyBytes     int64

	BlockSize           int
	MaxNumSeqs          int
	MaxNumBatchedTokens int
	// MaxModelLen is the window the engine serves (see resolvedMaxModelLen), or 0 when
	// neither the deployment nor the model graph states one.
	MaxModelLen int
	// MaxModelLenFrom says which of the three resolutions gave MaxModelLen.
	MaxModelLenFrom string
	DataParallel    int

	PrefixCachingDisabled bool

	// SpeculativeTokens and SpeculativeMethod are the pool's draft configuration; zero and
	// empty when it does not speculate. Acceptance is not here: it is a property of the
	// workload against the draft model, not of the deployment.
	SpeculativeTokens int
	SpeculativeMethod string
}

// Settings answers the simulator's sizing questions for this model's pool.
func (m *Model) Settings() (Settings, error) { return m.SettingsReserving(0) }

// SettingsReserving is Settings with reservedBytes of each rank's HBM set aside before the KV
// budget is sized (see KVBudgetReserving).
func (m *Model) SettingsReserving(reservedBytes int64) (Settings, error) {
	e, err := m.Engine()
	if err != nil {
		return Settings{}, err
	}
	b, err := m.KVBudgetReserving(reservedBytes)
	if err != nil {
		return Settings{}, err
	}
	d := m.Deployment()
	s := Settings{
		KVBlocks:                b.PerRankBlocks,
		AggregateKVBlocks:       b.TotalBlocks,
		KVBytesPerBlock:         b.PerBlockBytes,
		SequenceAdmissionBlocks: b.SequenceAdmissionBlocks,
		SequenceAdmissionBytes:  b.SequenceAdmissionBytes,
		SequenceSteadyBlocks:    b.SequenceSteadyBlocks,
		SequenceSteadyBytes:     b.SequenceSteadyBytes,
		SequenceState:           m.sequenceState(b),
		BlockSize:               e.BlockSize,
		MaxNumSeqs:              e.MaxNumSeqs,
		MaxNumBatchedTokens:     e.MaxNumBatchedTokens,
		DataParallel:            m.DataParallelWidth(),
		PrefixCachingDisabled:   d.PrefixCachingDisabled,
	}
	if sp := e.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		s.SpeculativeTokens, s.SpeculativeMethod = sp.NumSpecTokens, sp.Method
	}
	need := m.requestBlocks(s.SequenceState, b.PerBlockBytes)
	s.MaxModelLen, s.MaxModelLenFrom = m.resolvedMaxModelLen(e.MaxModelLen,
		estimateMaxModelLen(need, checkBlocks(b), m.id.defaultWindow))
	if e.MaxModelLen.Auto() && s.MaxModelLen == 0 && m.id.defaultWindow > 0 {
		return Settings{}, fmt.Errorf("kernelmodel: Cannot auto-fit max_model_len: not enough GPU " +
			"memory available to serve even a single token. Try increasing `gpu_memory_utilization`")
	}
	if err := checkEnoughKVCacheMemory(b, need, s.MaxModelLen); err != nil {
		return Settings{}, err
	}
	return s, nil
}

// requestBlocks is the pool blocks one request of tokens tokens needs at most: its full-attention
// pages and each bounded group at its admission cap (Gate). It is vLLM's
// _max_memory_usage_bytes_from_groups over bytes_per_block (vllm/v1/core/kv_cache_utils.py:
// 2474-2528 at v0.31.0): every group claims its own blocks from the shared pool.
func (m *Model) requestBlocks(st sim.KVSequenceState, perBlock int64) func(tokens int) int64 {
	return func(tokens int) int64 {
		n := (m.k.SequenceVariableBytes(tokens) + perBlock - 1) / perBlock
		if st.Gate != nil {
			n += st.Gate(int64(tokens))
		}
		return n
	}
}

// checkBlocks is the pool blocks the start-up check plans against: the rank's pool less the
// null block BlockPool permanently holds back (kv_cache_utils.py:2779-2786, 1012-1023). vLLM
// compares bytes, needed = bytes_per_block x blocks <= available - bytes_per_block, which in
// whole blocks is blocks <= floor(available / bytes_per_block) - 1.
func checkBlocks(b KVBudget) int64 { return b.PerRankBlocks - 1 }

// estimateMaxModelLen is the longest window whose one request fits avail blocks, at most
// window: vLLM's binary search (_estimate_max_model_len_from_groups, kv_cache_utils.py:2531-2572
// at v0.31.0); 0 when one token does not fit.
func estimateMaxModelLen(need func(int) int64, avail int64, window int) int {
	if window <= 0 || need(1) > avail {
		return 0
	}
	lo, hi, best := 1, window, 1
	for lo <= hi {
		mid := lo + (hi-lo)/2
		if need(mid) <= avail {
			best, lo = mid, mid+1
		} else {
			hi = mid - 1
		}
	}
	return best
}

// checkEnoughKVCacheMemory refuses an engine whose KV cache cannot hold one request of
// max_model_len tokens, as vLLM v0.31.0 refuses to start it (check_enough_kv_cache_memory and
// _check_enough_kv_cache_memory, vllm/v1/core/kv_cache_utils.py:898-935, 999-1034, applied per
// worker at :2788-2800). The message is vLLM's, GiB rounded to two places as format_gib does
// (vllm/utils/mem_utils.py:30-31).
func checkEnoughKVCacheMemory(b KVBudget, need func(int) int64, maxModelLen int) error {
	avail := checkBlocks(b)
	if avail <= 0 {
		return fmt.Errorf("kernelmodel: No available memory for the cache blocks. Try increasing " +
			"`gpu_memory_utilization` when initializing the engine")
	}
	if maxModelLen <= 0 {
		return nil // no window: requireWindowFits refuses it, naming the scenario
	}
	needed := need(maxModelLen)
	if needed <= avail {
		return nil
	}
	estimated := ""
	if n := estimateMaxModelLen(need, avail, maxModelLen); n > 0 {
		estimated = fmt.Sprintf("Based on the available memory, the estimated maximum model length is %d. ", n)
	}
	gib := func(bytes int64) float64 { return math.Round(float64(bytes)/(1<<30)*100) / 100 }
	return fmt.Errorf("kernelmodel: To serve at least one request with the model's max seq len (%d), "+
		"(%v GiB KV cache is needed, which is larger than the available KV cache memory (%v GiB). %s"+
		"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing "+
		"the engine", maxModelLen, gib(needed*b.PerBlockBytes), gib(b.AllocatableBytes-b.PerBlockBytes), estimated)
}

// sequenceState is the KV store's view of the kernel's bounded per-sequence state (#1941),
// nil-valued when the kernel reports none, which keeps such models byte-identical.
//
// Hold is SequenceBoundedBytesAt(computed, inFlight) in whole pool blocks: what vLLM's
// allocate_slots holds for the scheduled tokens after remove_skipped_blocks frees what the
// window has passed (single_type_kv_cache_manager.py:178-238, 687-725, 1153-1175 at v0.31.0),
// settling at SequenceFixedBytes while decoding. Gate is vLLM's full_sequence_must_fit check
// with apply_admission_cap (kv_cache_manager.py:517-534, single_type_kv_cache_manager.py:
// 216-229): each window group at min(cdiv(tokens), max_admission_blocks_per_request), which
// for the stated rings is min(SequenceBoundedBytesAt(0, tokens), SequenceAdmissionBytes).
func (m *Model) sequenceState(b KVBudget) sim.KVSequenceState {
	if b.SequenceAdmissionBytes == 0 {
		return sim.KVSequenceState{}
	}
	k := m.k.(admissionReserver) // KVBudget refused a kernel without it
	per := b.PerBlockBytes
	blocks := func(bytes int64) int64 { return (bytes + per - 1) / per }
	return sim.KVSequenceState{
		Hold: func(computed, inFlight int64) int64 {
			return blocks(k.SequenceBoundedBytesAt(int(computed), int(inFlight)))
		},
		Gate: func(tokens int64) int64 {
			return blocks(min(k.SequenceBoundedBytesAt(0, int(tokens)), b.SequenceAdmissionBytes))
		},
	}
}

// resolvedMaxModelLen is the window vLLM v0.31.0 starts the engine with:
//
//   - a stated count is the window;
//   - unstated, the window derived from the model configuration, which the graph states as
//     its context window (vllm/config/model.py _get_and_verify_max_len);
//   - auto, that derived window shrunk to the longest one request of which a rank's KV cache
//     holds, fitTokens (estimateMaxModelLen; _auto_fit_max_model_len,
//     vllm/v1/core/kv_cache_utils.py:2575-2640 at v0.31.0). The kernel takes auto at the
//     derived cap because it cannot see the KV budget; the simulator can, so it applies the fit.
//
// A window one rank cannot hold is not refused here; checkEnoughKVCacheMemory refuses it.
func (m *Model) resolvedMaxModelLen(stated deployment.MaxModelLen, fitTokens int) (int, string) {
	if n, ok := stated.Stated(); ok {
		return n, "stated"
	}
	w := m.id.defaultWindow
	if !stated.Auto() {
		return w, "unstated: the model graph's context window"
	}
	if w > fitTokens {
		return fitTokens, "auto: fitted to one rank's KV cache"
	}
	return w, "auto: the model graph's context window, which one rank's KV cache holds"
}
