package sim_test

import (
	"fmt"
	"math"
	"testing"

	"pgregory.net/rapid"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kv"

	host "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil/fakelatency"
)

// checkingStore wraps a production KV store and, after every allocation it grants, checks the
// requesting sequence's hold against what INV-K3/INV-K4 state for that call -- so the check
// runs at the exact allocation boundary, whatever store (single-tier, legacy CPU tier, offload
// chain) and whatever path inside it (fresh, prefix hit, CPU reload) granted it. It forwards
// the optional reload and deferral capabilities, so batch formation drives the store as in a
// run.
type checkingStore struct {
	sim.KVStore
	gpu      *kv.KVCacheState
	check    func(req *sim.Request, start, computed, end, before int64)
	reload   sim.ReloadReportingKVStore
	deferral sim.DeferrableKVStore
}

func (c *checkingStore) AllocateKVBlocks(req *sim.Request, start, end int64, cached []int64) bool {
	before := int64(len(c.gpu.RequestMap[req.ID]) + len(c.gpu.SpareBlocks[req.ID]))
	// A same-step CPU reload computes the reloaded prefix too, so the step's computed tokens
	// start where the reload ends (vLLM's num_computed_tokens = local + external) -- when the
	// reload the pure query promised does land; the tier can fall back to the GPU prefix. A
	// reload of the whole prompt leaves its last token to compute, as batch formation caps it
	// and vLLM sets num_computed_tokens = num_tokens - 1 after such a load
	// (sched/scheduler.py:3095-3096 at v0.31.0); the window state is sized from there
	// (single_type_kv_cache_manager.py:178-245, INV-K8).
	computed := start
	if c.reload != nil {
		if e, ok := c.reload.ReloadablePrefixEnd(req, start); ok {
			computed = max(start, min(e, max(req.InputLen()-1, 0), end))
		}
	}
	ok := c.KVStore.AllocateKVBlocks(req, start, end, cached)
	if ok {
		c.check(req, start, computed, end, before)
	}
	return ok
}

func (c *checkingStore) ReloadablePrefixEnd(req *sim.Request, start int64) (int64, bool) {
	if c.reload == nil {
		return start, false
	}
	return c.reload.ReloadablePrefixEnd(req, start)
}

func (c *checkingStore) PollDeferred(now int64) []string {
	if c.deferral == nil {
		return nil
	}
	return c.deferral.PollDeferred(now)
}

func (c *checkingStore) IsDeferred(id string) bool {
	return c.deferral != nil && c.deferral.IsDeferred(id)
}

func (c *checkingStore) ClearDeferred(id string) {
	if c.deferral != nil {
		c.deferral.ClearDeferred(id)
	}
}

// gpuOf is the GPU tier of a store NewKVStore built.
func gpuOf(t *rapid.T, s sim.KVStore) *kv.KVCacheState {
	g := kv.GPUTierForTest(s)
	if g == nil {
		t.Fatalf("unexpected store %T", s)
	}
	return g
}

// Every running request holds exactly the KV its step needs, under every store and every
// charge at once (INV-K3, INV-K4, INV-4). Request mixes draw prompts from a small pool of
// block-aligned and unaligned prompts (full prefix hits at first admission and on resume
// after preemption, #1943), with a bounded per-sequence state that grows with what is in
// flight and shrinks after, speculation with DSpark's lookahead, max_model_len, a pool tight
// enough to preempt, and a single-tier, legacy-CPU-tier or offload-chain store. After every
// allocation the requesting sequence holds
//
//	token blocks + spare = max(held before, token blocks, ceil(min(end + lookahead, max_model_len) / block))
//	state blocks         = Hold(start, end - start)
//
// and after every event the pool is conserved (verifyBlockConservation). Every request
// completes with exactly its tokens, and the pool is returned whole.
func TestEveryRunningRequestHoldsKVForItsComputedTokens(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const block = 16 // the offload chain's test config pages in 16-token blocks
		k := rapid.SampledFrom([]int{0, 2, 5}).Draw(rt, "k")
		spec := sim.SpeculativeConfig{}
		if k > 0 {
			spec = sim.SpeculativeConfig{K: k, Acceptance: rapid.SampledFrom([]float64{0, 0.5, 1}).Draw(rt, "alpha"), Method: "dspark"}
		}
		var state sim.KVSequenceState
		if rapid.Bool().Draw(rt, "state") {
			window := rapid.Int64Range(1, 80).Draw(rt, "window")
			ring := func(seen, inFlight int64) int64 {
				return (min(seen, window-1+inFlight)+block-1)/block + 1
			}
			capBlocks := ring(math.MaxInt32, 64)
			state = sim.KVSequenceState{
				Hold: func(c, n int64) int64 { return ring(c+n, n) + 1 },
				Gate: func(tokens int64) int64 { return min(ring(tokens, tokens), capBlocks) + 1 },
			}
		}
		prompts := [][]sim.TokenID{}
		for p := 0; p < 3; p++ {
			n := rapid.IntRange(1, 4).Draw(rt, "promptBlocks") * block
			if rapid.Bool().Draw(rt, "unaligned") {
				n += rapid.IntRange(1, block-1).Draw(rt, "tail")
			}
			toks := make([]sim.TokenID, n)
			for i := range toks {
				toks[i] = sim.TokenID(p*1000 + i + 1)
			}
			prompts = append(prompts, toks)
		}
		n := rapid.IntRange(1, 6).Draw(rt, "requests")
		shapes := make([][2]int, n)
		wantTokens, longest := 0, 0
		for i := range shapes {
			shapes[i] = [2]int{rapid.IntRange(0, len(prompts)-1).Draw(rt, "prompt"), rapid.IntRange(1, 40).Draw(rt, "out")}
			wantTokens += shapes[i][1]
			longest = max(longest, len(prompts[shapes[i][0]])+shapes[i][1])
		}
		maxModelLen := int64(0)
		if rapid.Bool().Draw(rt, "capped") {
			maxModelLen = int64(longest + rapid.IntRange(1, 8).Draw(rt, "headroom"))
		}
		stateFloor := int64(0)
		if state.Gate != nil {
			stateFloor = state.Gate(int64(longest)) + state.Hold(0, int64(longest))
		}
		floor := int64((longest+2*k+1)/block+3) + stateFloor
		blocks := rapid.Int64Range(floor, floor*int64(n)).Draw(rt, "kvBlocks")

		kvCfg := sim.NewKVCacheConfig(blocks, block, 0, 0, 0, 0, sim.WithSequenceState(state))
		switch rapid.SampledFrom([]string{"single", "cpu", "chain"}).Draw(rt, "store") {
		case "cpu":
			kvCfg = sim.NewKVCacheConfig(blocks, block, blocks, 0.9, 100, 10, sim.WithSequenceState(state))
		case "chain":
			kvCfg = sim.NewKVCacheConfig(blocks, block, 0, 0, 0, 0, sim.WithSequenceState(state),
				sim.WithKVOffload(occupancyOffloadCfg(blocks*4096, 4096)))
		}
		kvCfg.Speculative = spec.KV()
		kvCfg.Speculative.MaxModelLen = maxModelLen
		lookahead := spec.LookaheadTokens()

		base := kv.NewKVStore(kvCfg, 1)
		store := &checkingStore{KVStore: base, gpu: gpuOf(rt, base)}
		store.reload, _ = base.(sim.ReloadReportingKVStore)
		store.deferral, _ = base.(sim.DeferrableKVStore)
		gpu := store.gpu
		store.check = func(req *sim.Request, start, computed, end, before int64) {
			tokens := int64(len(gpu.RequestMap[req.ID]))
			spare := int64(len(gpu.SpareBlocks[req.ID]))
			want := tokens
			if k > 0 {
				slots := end + lookahead
				if maxModelLen > 0 {
					slots = min(slots, max(maxModelLen, end))
				}
				// A running request's slots are never handed back while it runs: vLLM's
				// get_num_blocks_to_allocate returns max(required - held, 0)
				// (single_type_kv_cache_manager.py:223-228).
				want = max(before, tokens, (slots+block-1)/block)
			}
			if tokens+spare != want {
				rt.Fatalf("%s [%d, %d): %d token + %d spare blocks, want %d (K=%d, lookahead %d, max_model_len %d)",
					req.ID, start, end, tokens, spare, want, k, lookahead, maxModelLen)
			}
			if state.Hold != nil {
				got := int64(len(gpu.StateBlocks[req.ID]))
				if got != state.Hold(start, end-start) && got != state.Hold(computed, end-computed) {
					rt.Fatalf("%s [%d|%d, %d): %d state blocks, want %d (or %d after the reload)", req.ID, start, computed, end,
						got, state.Hold(start, end-start), state.Hold(computed, end-computed))
				}
			}
			if err := gpu.VerifyBlockConservationForTest(); err != nil {
				rt.Fatal(err)
			}
		}

		cfg := sim.SimConfig{
			Horizon: math.MaxInt64, Seed: 7, KVCacheConfig: kvCfg,
			BatchConfig:         sim.NewBatchConfig(8, 96, 0),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, "", "", 1, 1, false, maxModelLen),
			SpeculativeConfig:   spec,
		}
		s, err := host.NewSimulator(cfg, store, fakelatency.New())
		if err != nil {
			rt.Fatal(err)
		}
		for i, sh := range shapes {
			s.InjectArrival(&sim.Request{ID: fmt.Sprintf("r%d", i), InputTokens: append([]sim.TokenID(nil), prompts[sh[0]]...),
				OutputTokens: make([]sim.TokenID, sh[1]), ArrivalTime: int64(i * 300), State: sim.StateQueued})
		}
		for events := 0; s.HasPendingEvents(); events++ {
			if events > 200000 {
				rt.Fatalf("no progress after %d events at %d: %d running, %d queued, %d of %d blocks used",
					events, s.Clock, s.BatchSize(), s.QueueDepth(), gpu.UsedBlocks(), blocks)
			}
			s.ProcessNextEvent()
			if err := gpu.VerifyBlockConservationForTest(); err != nil {
				rt.Fatal(err)
			}
		}
		s.Finalize()
		if s.Metrics.CompletedRequests+s.Metrics.LengthCappedRequests < n {
			rt.Fatalf("%d completed, %d length-capped, of %d", s.Metrics.CompletedRequests, s.Metrics.LengthCappedRequests, n)
		}
		if maxModelLen == 0 && s.Metrics.TotalOutputTokens != wantTokens {
			rt.Fatalf("%d of %d output tokens", s.Metrics.TotalOutputTokens, wantTokens)
		}
		if gpu.UsedBlocks() != 0 {
			rt.Fatalf("%d blocks still held at the end", gpu.UsedBlocks())
		}
	})
}

// occupancyOffloadCfg is an enabled offload chain with no secondary tiers: a CPU tier of
// cpuBytes in perBlockBytes blocks (compat/kv's enabledOffloadCfg with no tiers).
func occupancyOffloadCfg(cpuBytes, perBlockBytes int64) sim.KVOffloadConfig {
	return sim.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: cpuBytes, PerBlockBytes: perBlockBytes,
		BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
	}
}
