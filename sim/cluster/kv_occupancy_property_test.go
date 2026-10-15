package cluster

import (
	"fmt"
	"testing"

	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// After every event, every running request holds KV for every token it has computed:
// ceil(ProgressIndex / block) blocks of its own sequence, plus its lookahead slots when the
// instance speculates (INV-K4). Request mixes draw prompts from a small pool of block-aligned
// and unaligned prompts, so full prefix hits occur both at first admission (a prompt another
// request left cached) and on resume after preemption, and a tight pool forces preemption
// (#1943: a fully cached block-aligned prompt used to run holding nothing). Every request
// completes and emits exactly its tokens (INV-1), and the pool is conserved (INV-4).
func TestEveryRunningRequestHoldsKVForItsComputedTokens(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const blockSize = 4
		k := rapid.SampledFrom([]int{0, 0, 2, 5}).Draw(rt, "k")
		prompts := [][]sim.TokenID{}
		for p := 0; p < 3; p++ {
			n := rapid.IntRange(1, 6).Draw(rt, "promptBlocks") * blockSize
			if rapid.Bool().Draw(rt, "unaligned") {
				n += rapid.IntRange(1, blockSize-1).Draw(rt, "tail")
			}
			toks := make([]sim.TokenID, n)
			for i := range toks {
				toks[i] = sim.TokenID(p*1000 + i + 1)
			}
			prompts = append(prompts, toks)
		}
		n := rapid.IntRange(1, 8).Draw(rt, "requests")
		reqs := make([]*sim.Request, n)
		wantTokens, longest := 0, 0
		for i := range reqs {
			p := prompts[rapid.IntRange(0, len(prompts)-1).Draw(rt, "prompt")]
			out := rapid.IntRange(1, 24).Draw(rt, "out")
			reqs[i] = &sim.Request{ID: fmt.Sprintf("r%d", i), InputTokens: append([]sim.TokenID(nil), p...),
				OutputTokens: make([]sim.TokenID, out), ArrivalTime: int64(i * rapid.IntRange(0, 3000).Draw(rt, "gap")),
				State: sim.StateQueued}
			wantTokens += out
			longest = max(longest, len(p)+out)
		}
		spec := sim.SpeculativeConfig{}
		if k > 0 {
			spec = sim.SpeculativeConfig{K: k, Acceptance: 0.5, Method: "dspark"}
		}
		lookahead := spec.LookaheadTokens()
		floor := int64((longest+2*k+1)/blockSize + 2)
		blocks := rapid.Int64Range(floor, floor*int64(n)).Draw(rt, "kvBlocks")

		cfg := newTestSimConfig()
		cfg.KVCacheConfig = sim.NewKVCacheConfig(blocks, blockSize, 0, 0, 0, 0)
		cfg.BatchConfig = sim.NewBatchConfig(16, 128, 0)
		cfg.SpeculativeConfig = spec
		inst := NewInstanceSimulator("i0", cfg)
		store, ok := inst.sim.KVCache.(*kv.KVCacheState)
		if !ok {
			rt.Fatalf("expected the single-tier store, got %T", inst.sim.KVCache)
		}
		for _, r := range reqs {
			inst.InjectRequest(r)
		}
		for inst.HasPendingEvents() {
			inst.ProcessNextEvent()
			if inst.sim.RunningBatch == nil {
				continue
			}
			for _, r := range inst.sim.RunningBatch.Requests {
				if r.State != sim.StateRunning || r.ProgressIndex == 0 {
					continue
				}
				need := (r.ProgressIndex + blockSize - 1) / blockSize
				if k > 0 {
					need = (r.ProgressIndex + lookahead + blockSize - 1) / blockSize
				}
				have := int64(len(store.RequestMap[r.ID]) + len(store.SpareBlocks[r.ID]))
				if have < need {
					rt.Fatalf("K=%d at %d: %s computed %d tokens (prompt %d) and holds %d blocks, needs %d",
						k, inst.sim.Clock, r.ID, r.ProgressIndex, r.InputLen(), have, need)
				}
			}
		}
		inst.sim.Finalize()
		if inst.sim.Metrics.CompletedRequests != n || inst.sim.Metrics.TotalOutputTokens != wantTokens {
			rt.Fatalf("%d of %d completed, %d of %d tokens", inst.sim.Metrics.CompletedRequests, n,
				inst.sim.Metrics.TotalOutputTokens, wantTokens)
		}
		if store.UsedBlocks() != 0 {
			rt.Fatalf("%d blocks still held at the end", store.UsedBlocks())
		}
	})
}
