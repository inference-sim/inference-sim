package kernelmodel

import "fmt"

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

	BlockSize           int
	MaxNumSeqs          int
	MaxNumBatchedTokens int
	// MaxModelLen is the pool's stated context window, or 0 when it states none.
	MaxModelLen  int
	DataParallel int

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
		BlockSize:             e.BlockSize,
		MaxNumSeqs:            e.MaxNumSeqs,
		MaxNumBatchedTokens:   e.MaxNumBatchedTokens,
		MaxModelLen:           e.MaxModelLen,
		DataParallel:          m.DataParallelWidth(),
		PrefixCachingDisabled: d.PrefixCachingDisabled,
	}
	if sp := e.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		s.SpeculativeTokens, s.SpeculativeMethod = sp.NumSpecTokens, sp.Method
	}
	if s.MaxModelLen < 0 {
		return Settings{}, fmt.Errorf("kernelmodel: deployment states max_model_len %d", s.MaxModelLen)
	}
	return s, nil
}
