package cluster

import (
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// Offload tiers under P/D (#1940): each pool's engine carries its own OffloadingConnector
// tier. A prefill engine is an ordinary engine for its prompts, so its tier stores what it
// computes and serves a later turn's prefix before compute (offloading/scheduler.py:1036-1094
// at v0.31.0, reached through MultiConnector.get_num_new_matched_tokens since a prefill
// request has no remote KV for NIXL to claim, multi_connector.py:429-448).

func cpuTier() *cengine.KVOffloadConfig {
	return &cengine.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 1000 * 4096, PerBlockBytes: 4096, BlockSize: 16,
		BlocksPerChunk: 1, TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true,
	}
}

// A later turn whose prefix the prefill GPU has evicted is served from the prefill pool's
// CPU tier: the pool reports CPU-tier hits, and the turn prefills faster than with no tier
// because it computes only the new tokens.
func TestPDOffload_PrefillTierServesALaterTurnsPrefix(t *testing.T) {
	turn1 := holdTestRequest("t1", 1, 0, 160, 4)
	reqs := []*sim.Request{turn1}
	for i := 0; i < 4; i++ { // fillers that evict turn 1 from the prefill GPU
		reqs = append(reqs, holdTestRequest(string(rune('f'+i)), 10+i, int64(i+1)*5_000_000, 160, 4))
	}
	reqs = append(reqs, extend("t2", turn1, 30_000_000, 96, 4))
	run := func(tier bool) *ClusterSimulator {
		cfg := newTestDisaggDeploymentConfig(2, 1, 1)
		blocks := int64(24)
		editPrefill(&cfg, func(c *simConfig) { c.TotalKVBlocks = blocks })
		if tier {
			editPrefill(&cfg, func(c *simConfig) { c.Offload = *cpuTier() })
		}
		return runHold(t, cfg, cloneRequests(reqs)...)
	}
	with, without := run(true), run(false)
	if r := with.PDMetrics().CPUTierHitRate["prefill"]; !(r > 0) {
		t.Errorf("prefill pool CPU-tier hit rate %v, want > 0", r)
	}
	if r := without.PDMetrics().CPUTierHitRate["prefill"]; r != 0 {
		t.Errorf("with no tier the prefill pool reports CPU-tier hit rate %v", r)
	}
	prefill := func(cs *ClusterSimulator) int64 {
		p := parentsByID(cs)["t2"]
		return p.PrefillCompleteTime - p.ArrivalTime
	}
	if prefill(with) >= prefill(without) {
		t.Errorf("turn 2 prefilled in %d ticks with the prefill tier, %d without; the tier must save compute",
			prefill(with), prefill(without))
	}
}
