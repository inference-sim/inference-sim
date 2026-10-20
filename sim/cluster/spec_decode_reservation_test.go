package cluster

import (
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// #1627: an instance's KV store reserves the speculative slots its SpeculativeConfig implies,
// with no separate knob to forget: a DSpark K=5 instance holds a request's verify width and
// five lookahead slots past its computed tokens, a plain instance only its stored tokens.
func TestInstanceKVReservesTheSpeculativeConfigsSlots(t *testing.T) {
	const blockSize = 16
	usedAfterPrefill := func(spec sim.SpeculativeConfig) int64 {
		cfg := newTestSimConfig()
		cfg.KVCacheConfig = sim.NewKVCacheConfig(10000, blockSize, 0, 0, 0, 0)
		cfg.SpeculativeConfig = spec
		inst := NewInstanceSimulator("i0", cfg)
		inst.InjectRequest(&sim.Request{ID: "r", InputTokens: make([]sim.TokenID, 2*blockSize),
			OutputTokens: make([]sim.TokenID, 4), State: sim.StateQueued})
		for inst.HasPendingEvents() {
			ev := inst.ProcessNextEvent()
			if _, ok := ev.(*sim.StepEvent); ok {
				break // the first step: the prefill is scheduled and allocated
			}
		}
		return inst.FreeKVBlocks()
	}
	plain := usedAfterPrefill(sim.SpeculativeConfig{})
	dspark := usedAfterPrefill(sim.SpeculativeConfig{K: 5, Acceptance: 0.5, Method: "dspark"})
	// 32 prompt tokens + 5 lookahead slots = 37 slots: 3 blocks against the plain 2.
	if plain-dspark != 1 {
		t.Errorf("free blocks after the prefill: plain %d, DSpark K=5 %d; the lookahead should hold one more block",
			plain, dspark)
	}
}
