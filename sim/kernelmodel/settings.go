package kernelmodel

import "github.com/inference-sim/blis-schemas/spec/deployment"

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
	// SequenceFixedBlocks is what each resident sequence takes from the pool whatever its
	// length (KVBudget.SequenceFixedBlocks), and SequenceFixedBytes the bytes it rounds up.
	SequenceFixedBlocks int64
	SequenceFixedBytes  int64

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
		KVBlocks:              b.PerRankBlocks,
		AggregateKVBlocks:     b.TotalBlocks,
		KVBytesPerBlock:       b.PerBlockBytes,
		SequenceFixedBlocks:   b.SequenceFixedBlocks,
		SequenceFixedBytes:    b.SequenceFixedBytes,
		BlockSize:             e.BlockSize,
		MaxNumSeqs:            e.MaxNumSeqs,
		MaxNumBatchedTokens:   e.MaxNumBatchedTokens,
		DataParallel:          m.DataParallelWidth(),
		PrefixCachingDisabled: d.PrefixCachingDisabled,
	}
	if sp := e.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		s.SpeculativeTokens, s.SpeculativeMethod = sp.NumSpecTokens, sp.Method
	}
	s.MaxModelLen, s.MaxModelLenFrom = m.resolvedMaxModelLen(e.MaxModelLen, b.PerRankBlocks*int64(e.BlockSize))
	return s, nil
}

// resolvedMaxModelLen is the window vLLM v0.31.0 starts the engine with:
//
//   - a stated count is the window;
//   - unstated, the window derived from the model configuration, which the graph states as
//     its context window (vllm/config/model.py _get_and_verify_max_len);
//   - auto, that derived window shrunk to the longest one rankTokens of KV cache holds
//     (vllm/v1/core/kv_cache_utils.py). The kernel takes auto at the derived cap because it
//     cannot see the KV budget; the simulator can, so it applies the fit.
//
// A window one rank cannot hold is not refused here; the caller refuses it as vLLM does.
func (m *Model) resolvedMaxModelLen(stated deployment.MaxModelLen, rankTokens int64) (int, string) {
	if n, ok := stated.Stated(); ok {
		return n, "stated"
	}
	w := m.id.defaultWindow
	if !stated.Auto() {
		return w, "unstated: the model graph's context window"
	}
	if int64(w) > rankTokens {
		return int(rankTokens), "auto: fitted to one rank's KV cache"
	}
	return w, "auto: the model graph's context window, which one rank's KV cache holds"
}
