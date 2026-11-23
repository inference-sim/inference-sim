// Moved from inference-sim/cmd/kernel_pools.go at 5bcf93f86a60df138bef28609100f13dfe317910.

package build

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	"github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
)

// PoolSettings is one P/D pool's engine as the cluster takes it: the per-pool settings that
// override the run's (cluster.PoolOverrides), each read from the pool's own kernel and its
// scenario pool. vLLM sets these per engine, so prefill and decode pools may differ. The host
// copies them into its per-pool overrides; the values and the rules that choose them are the
// engine's (inference-sim issue 2021 B2, verification pass 3, gap 2).
type PoolSettings struct {
	TP                    int
	GPU                   string
	TotalKVBlocks         int64
	MaxModelLen           int64
	MaxNumSeqs            int64
	MaxNumBatchedTokens   int64
	PrefixCachingDisabled bool
	SequenceState         runconfig.KVSequenceState
	PrefixHit             runconfig.KVPrefixHit
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
func PoolSettingsOf(scenario string, m *kernelmodel.Model, reservedBytes int64, prefixHit runconfig.KVPrefixHit, f EngineKnobFlags) (PoolSettings, error) {
	what := string(m.Role()) + " pool"
	s, err := m.SettingsCheckedByGroupsReserving(reservedBytes)
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

// Drafting is a pool's speculative decoding as its engine resolves it: the draft length and the
// method (zero tokens when the pool drafts nothing).
type Drafting struct {
	Tokens int
	Method string
}

// Drafts reports whether the pool drafts any token.
func (d Drafting) Drafts() bool { return d.Tokens != 0 }

// PoolDrafting is pool m's speculative decoding, read from its engine's settings.
func PoolDrafting(m *kernelmodel.Model) (Drafting, error) {
	s, err := m.Settings()
	if err != nil {
		return Drafting{}, fmt.Errorf("reading the %s pool's speculative settings: %v", m.Role(), err)
	}
	return Drafting{Tokens: s.SpeculativeTokens, Method: s.SpeculativeMethod}, nil
}

// PoolsDraftDifferently reports whether the prefill and decode pools state different speculative
// blocks: no block and a block of no draft tokens both draft nothing, and any other pair draft
// alike only when the blocks are equal.
func PoolsDraftDifferently(prefill, decode *kernelmodel.Model) (bool, error) {
	blocks := make([]*engineconfig.Speculative, 0, 2)
	for _, m := range []*kernelmodel.Model{prefill, decode} {
		e, err := m.Engine()
		if err != nil {
			return false, fmt.Errorf("reading the %s pool's speculative settings: %v", m.Role(), err)
		}
		blocks = append(blocks, e.Speculative)
	}
	tokens := func(s *engineconfig.Speculative) int {
		if s == nil {
			return 0
		}
		return s.NumSpecTokens
	}
	if tokens(blocks[0]) == 0 && tokens(blocks[1]) == 0 {
		return false, nil
	}
	return !reflect.DeepEqual(blocks[0], blocks[1]), nil
}
