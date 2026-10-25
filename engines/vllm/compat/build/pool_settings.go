// Moved from inference-sim/cmd/kernel_pools.go at 5bcf93f86a60df138bef28609100f13dfe317910.

package build

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// PoolSettings is one P/D pool's engine as the cluster takes it: the per-pool settings that
// override the run's (cluster.PoolOverrides), each read from the pool's own kernel and its
// scenario pool. vLLM sets these per engine, so prefill and decode pools may differ. The host
// copies them into its per-pool overrides; the values and the rules that choose them are the
// engine's (#2021 B2, verification pass 3, gap 2).
type PoolSettings struct {
	TP                    int
	GPU                   string
	TotalKVBlocks         int64
	MaxModelLen           int64
	MaxNumSeqs            int64
	MaxNumBatchedTokens   int64
	PrefixCachingDisabled bool
	SequenceState         engine.KVSequenceState
	PrefixHit             engine.KVPrefixHit
	// LongPrefillTokenThreshold and DPLoadBalancing are the pool's own knobs (PoolEngineKnobs).
	LongPrefillTokenThreshold int64
	DPLoadBalancing           string
	// Policy is the scheduling policy the pool's engine states (EnginePolicyOf); the caller
	// applies it unless the run chose one for every pool, which it records.
	Policy EnginePolicy
	// Notes are the startup notes the pool's settings produce, in order, for the caller to log:
	// its sequence-state charge (SequenceFixedCharge).
	Notes []string
}

// PoolSettingsOf resolves a P/D pool's engine settings from its kernel model m, with
// reservedBytes (the LoRA adapter reservation) set aside from its KV budget. prefixHit is the
// pool's prefix-hit layout, which the caller resolves (and records) with the run's override. It
// refuses a pool whose settings cannot be sized, whose window does not fit, or whose engine or
// policy cannot be read.
func PoolSettingsOf(scenario string, m *kernelmodel.Model, reservedBytes int64, prefixHit engine.KVPrefixHit, f EngineKnobFlags) (PoolSettings, error) {
	what := string(m.Role()) + " pool"
	s, err := m.SettingsReserving(reservedBytes)
	if err != nil {
		return PoolSettings{}, fmt.Errorf("%s pool: %v", m.Role(), err)
	}
	if err := RequireWindowFits(scenario, what, s); err != nil {
		return PoolSettings{}, err
	}
	d := m.Deployment()
	ps := PoolSettings{
		TP: d.TP, GPU: d.Hardware, TotalKVBlocks: s.KVBlocks, MaxModelLen: int64(s.MaxModelLen),
		MaxNumSeqs: int64(s.MaxNumSeqs), MaxNumBatchedTokens: int64(s.MaxNumBatchedTokens),
		PrefixCachingDisabled: s.PrefixCachingDisabled, SequenceState: s.SequenceState,
		PrefixHit: prefixHit,
	}
	if note := SequenceFixedCharge(what, s); note != "" {
		ps.Notes = append(ps.Notes, note)
	}
	if ps.LongPrefillTokenThreshold, ps.DPLoadBalancing, err = PoolEngineKnobs(scenario, m, f); err != nil {
		return PoolSettings{}, err
	}
	if ps.Policy, err = EnginePolicyOf(scenario, m); err != nil {
		return PoolSettings{}, err
	}
	return ps, nil
}
