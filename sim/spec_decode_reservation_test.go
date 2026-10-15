package sim

import (
	"fmt"
	"math"
	"testing"

	"pgregory.net/rapid"
)

// specReservationSim builds a simulator speculating k DSpark tokens at acceptance α, with
// the KV slot reservation on or off. NewInstanceSimulator always derives the reservation from
// the SpeculativeConfig; package-sim tests build the store themselves, so they set it here,
// and "off" is the pre-#1627 accounting the comparison is against.
func specReservationSim(t fataler, k int, alpha float64, kvBlocks, blockSize, budget int64, reserve bool) *Simulator {
	t.Helper()
	spec := SpeculativeConfig{K: k, Acceptance: alpha}
	if k > 0 {
		spec.Method = "dspark"
	}
	kvCfg := NewKVCacheConfig(kvBlocks, blockSize, 0, 0, 0, 0)
	if reserve {
		kvCfg.Speculative = spec.KV()
	}
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                7,
		KVCacheConfig:       kvCfg,
		BatchConfig:         NewBatchConfig(64, budget, 0),
		ModelHardwareConfig: NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		SpeculativeConfig:   spec,
	}
	s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), fakeLatencyFor(cfg))
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}
	return s
}

type reqShape struct{ in, out int }

// fataler is what these helpers need of a *testing.T or a *rapid.T.
type fataler interface {
	Helper()
	Fatalf(format string, args ...any)
}

// runShapes runs one request per shape, calling each (when given) after every event.
func runShapes(t fataler, s *Simulator, shapes []reqShape, each ...func()) []*Request {
	t.Helper()
	reqs := make([]*Request, len(shapes))
	for i, sh := range shapes {
		toks := make([]TokenID, sh.in)
		for j := range toks {
			toks[j] = TokenID(i*1000 + j + 1) // no shared prefixes
		}
		reqs[i] = &Request{ID: fmt.Sprintf("r%d", i), InputTokens: toks,
			OutputTokens: make([]TokenID, sh.out), ArrivalTime: int64(i), State: StateQueued}
		s.InjectArrival(reqs[i])
	}
	for s.HasPendingEvents() {
		s.ProcessNextEvent()
		for _, f := range each {
			f()
		}
	}
	s.Finalize()
	return reqs
}

// #1627: vLLM v0.31.0 schedules a speculating decode's whole verify width, 1 + K tokens, and
// debits all of it from max_num_batched_tokens, whatever is later accepted. With a budget of
// exactly one verify width, one decode runs per step and the next waits, even at zero
// acceptance (one token of progress each). Without a verify width the same two decodes,
// advancing one token each, both fit.
func TestSpecDecode_ADecodeDebitsItsVerifyWidthFromTheTokenBudget(t *testing.T) {
	const k = 5
	for _, tt := range []struct {
		name        string
		verifyWidth int64
		wantDecoded int
	}{
		{"verify width scheduled", k + 1, 1},
		{"advance only (pre-#1627 accounting)", 0, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			kv := MustNewKVCacheState(100, 4)
			running := &Batch{}
			for i := 0; i < 2; i++ {
				r := &Request{ID: fmt.Sprintf("d%d", i), InputTokens: make([]TokenID, 4),
					OutputTokens: make([]TokenID, 8), State: StateRunning}
				if !kv.AllocateKVBlocks(r, 0, 4, nil) {
					t.Fatal("setup allocation refused")
				}
				r.ProgressIndex = 4
				running.Requests = append(running.Requests, r)
			}
			res := NewBatchFormation("").FormBatch(BatchContext{
				RunningBatch: running, WaitQ: &WaitQueue{}, KVCache: kv,
				MaxNumBatchedTokens: k + 1, MaxNumSeqs: 10, ComputedTokens: map[string]int64{},
				DecodeTokensPerStep: func(*Request) int64 { return 1 }, SpecVerifyWidth: tt.verifyWidth,
			})
			decoded := 0
			for _, r := range res.RunningBatch.Requests {
				if r.NumNewTokens > 0 {
					decoded++
					if r.NumNewTokens != 1 {
						t.Errorf("%s advanced %d tokens; progress is the accepted count, 1", r.ID, r.NumNewTokens)
					}
				}
			}
			if decoded != tt.wantDecoded {
				t.Errorf("%d of 2 decodes ran under a %d-token budget, want %d", decoded, k+1, tt.wantDecoded)
			}
		})
	}
}

// #1627: under KV pressure the lookahead reservation binds. Four requests whose plain
// footprint fits the pool together run without preemption when only their stored tokens are
// charged, but holding vLLM's slots -- the verify width and K lookahead past every computed
// token -- they no longer fit, and the engine preempts. Both runs emit every token (INV-1).
func TestSpecDecode_LookaheadReservationPreemptsEarlierUnderKVPressure(t *testing.T) {
	shapes := []reqShape{{8, 9}, {8, 9}, {8, 9}, {8, 9}}
	// Each request ends at 8 + 9 - 1 = 16 computed tokens: 4 blocks of 4 stored, but
	// ceil((16 + 6 + 5) / 4) = 7 blocks of slots under DSpark K=5. 18 blocks hold four of the
	// first and two of the second.
	run := func(reserve bool) *Simulator {
		s := specReservationSim(t, 5, 0.4, 18, 4, 4096, reserve)
		runShapes(t, s, shapes)
		return s
	}
	off, on := run(false), run(true)
	for name, s := range map[string]*Simulator{"off": off, "on": on} {
		if s.Metrics.CompletedRequests != len(shapes) || s.Metrics.TotalOutputTokens != 4*9 {
			t.Errorf("reservation %s: %d completed, %d output tokens; want %d and %d", name,
				s.Metrics.CompletedRequests, s.Metrics.TotalOutputTokens, len(shapes), 4*9)
		}
		if used := s.KVCache.UsedBlocks(); used != 0 {
			t.Errorf("reservation %s: %d blocks still held after every request completed (INV-4)", name, used)
		}
	}
	if off.Metrics.PreemptionCount != 0 {
		t.Fatalf("without the reservation the pool should hold all four; %d preemptions", off.Metrics.PreemptionCount)
	}
	if on.Metrics.PreemptionCount == 0 {
		t.Errorf("holding the verify width and lookahead, four requests of 7 blocks ran in 18 blocks with no preemption")
	}
}

// Properties over random request mixes around KV saturation, with the reservation on as
// production runs it: whatever K is, every request completes and emits exactly its tokens
// (INV-1, the #1932 law that speculation changes steps, never tokens), the pool is returned
// whole (INV-4), and after every event the pool's used blocks cover every running request's
// computed tokens plus its lookahead -- the reservation is held, not just checked.
func TestSpecDecode_ReservationPropertiesAroundSaturation(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 8).Draw(rt, "requests")
		shapes := make([]reqShape, n)
		want := 0
		maxTokens := 0
		for i := range shapes {
			in := rapid.IntRange(1, 48).Draw(rt, "in")
			shapes[i] = reqShape{in, rapid.IntRange(1, 48).Draw(rt, "out")}
			want += shapes[i].out
			maxTokens = max(maxTokens, shapes[i].in+shapes[i].out)
		}
		const blockSize, k = 4, 5
		// From one request's full reservation (no smaller pool can run it) to twice the sum.
		floor := int64((maxTokens+2*k+1)/blockSize + 2)
		kvBlocks := rapid.Int64Range(floor, floor*int64(n)*2).Draw(rt, "kvBlocks")
		alpha := rapid.SampledFrom([]float64{0, 0.3, 0.7, 1}).Draw(rt, "alpha")
		for _, kk := range []int{0, 1, k} {
			s := specReservationSim(rt, kk, alpha, kvBlocks, blockSize, 256, true)
			lookahead := SpeculativeConfig{K: kk, Method: "dspark"}.LookaheadTokens()
			runShapes(rt, s, shapes, func() {
				var need int64
				if s.RunningBatch == nil {
					return
				}
				for _, r := range s.RunningBatch.Requests {
					if r.State == StateRunning && r.ProgressIndex > 0 {
						need += (r.ProgressIndex + lookahead + blockSize - 1) / blockSize
					}
				}
				if used := s.KVCache.UsedBlocks(); used < need || used > s.KVCache.TotalCapacity() {
					rt.Fatalf("K=%d at %d: %d blocks used, running requests need at least %d of %d",
						kk, s.Clock, used, need, s.KVCache.TotalCapacity())
				}
			})
			if s.Metrics.CompletedRequests != n || s.Metrics.TotalOutputTokens != want {
				rt.Fatalf("K=%d α=%v pool %d: %d of %d completed, %d of %d tokens", kk, alpha, kvBlocks,
					s.Metrics.CompletedRequests, n, s.Metrics.TotalOutputTokens, want)
			}
			if s.KVCache.UsedBlocks() != 0 {
				rt.Fatalf("K=%d: %d blocks still held at the end", kk, s.KVCache.UsedBlocks())
			}
		}
	})
}

// Metamorphic: at acceptance length 1 (α = 0) a speculating request advances one token a
// step, exactly as without speculation, so with room to spare every request takes the same
// number of decode steps at K = 5 as at K = 0. Only the price of a step may differ.
func TestSpecDecode_AcceptanceLengthOneChangesPriceNotProgress(t *testing.T) {
	shapes := []reqShape{{5, 7}, {17, 3}, {9, 12}, {30, 1}}
	steps := func(k int) []int {
		s := specReservationSim(t, k, 0, 10000, 4, 100000, true)
		var out []int
		for _, r := range runShapes(t, s, shapes) {
			out = append(out, len(r.ITL))
		}
		return out
	}
	if plain, spec := steps(0), steps(5); fmt.Sprint(plain) != fmt.Sprint(spec) {
		t.Errorf("decode steps per request: K=0 %v, K=5 at AL 1 %v", plain, spec)
	}
}

// The simulator hands batch formation its verify width: two decodes at acceptance length 1
// under a budget of one verify width take turns at K = 5, so the same work needs more steps
// than at K = 0, where both decode every step. Same tokens either way.
func TestSpecDecode_TheSimulatorSchedulesTheVerifyWidth(t *testing.T) {
	shapes := []reqShape{{3, 12}, {3, 12}}
	run := func(k int) *Simulator {
		s := specReservationSim(t, k, 0, 1000, 4, int64(5+1), true)
		runShapes(t, s, shapes)
		if s.Metrics.TotalOutputTokens != 24 {
			t.Fatalf("K=%d: %d output tokens, want 24", k, s.Metrics.TotalOutputTokens)
		}
		return s
	}
	plain, spec := run(0), run(5)
	if spec.stepCount <= plain.stepCount {
		t.Errorf("K=5 at AL 1 took %d steps, K=0 %d; one verify width per step should serialize the decodes",
			spec.stepCount, plain.stepCount)
	}
}

// A preemption returns what the victim's step debited: its whole verify width, not its
// accepted advance. Under priority preemption a decode already scheduled this step (r0, the
// least urgent) is evicted to make room for r1; the 1 + K tokens it held go back, so r2 behind
// them still schedules its full verify width.
func TestSpecDecode_APreemptedDecodeReturnsItsVerifyWidth(t *testing.T) {
	const k = 5
	kv := MustNewKVCacheState(3, 4)
	running := &Batch{}
	for i, in := range []int{3, 4, 3} { // r1's prompt fills its block; r0's and r2's do not
		r := &Request{ID: fmt.Sprintf("r%d", i), InputTokens: make([]TokenID, in),
			OutputTokens: make([]TokenID, 8), State: StateRunning, ArrivalTime: int64(i)}
		if i == 0 {
			r.Priority = 10
		}
		if !kv.AllocateKVBlocks(r, 0, int64(in), nil) {
			t.Fatal("setup allocation refused")
		}
		r.ProgressIndex = int64(in)
		running.Requests = append(running.Requests, r)
	}
	res := NewBatchFormation(string(PreemptionPriority)).FormBatch(BatchContext{
		RunningBatch: running, WaitQ: &WaitQueue{}, KVCache: kv,
		MaxNumBatchedTokens: 2 * (k + 1), MaxNumSeqs: 10, ComputedTokens: map[string]int64{},
		DecodeTokensPerStep: func(*Request) int64 { return 1 }, SpecVerifyWidth: k + 1,
	})
	if len(res.Preempted) != 1 || res.Preempted[0].Request.ID != "r0" {
		t.Fatalf("expected r0 preempted, got %v", res.Preempted)
	}
	for _, r := range res.RunningBatch.Requests {
		if r.scheduledTokens != k+1 {
			t.Errorf("%s scheduled %d tokens, want the verify width %d", r.ID, r.scheduledTokens, k+1)
		}
	}
}

// The lookahead is vLLM v0.31.0's VllmConfig.num_lookahead_tokens, method by method
// (vllm/config/vllm.py:630-654): K + 1 for DFlash, K for the hidden-state and draft-model
// drafters (MTP under any of its model-specific names, which vLLM folds into "mtp"), 0 for the
// rest; nothing when not speculating.
func TestLookaheadTokensFollowVLLMsNumLookaheadTokens(t *testing.T) {
	const k = 4
	for method, want := range map[string]int64{
		"dflash": k + 1, "dspark": k, "eagle": k, "eagle3": k, "mtp": k, "deepseek_mtp": k,
		"glm5_next_mtp": k, "draft_model": k, "ngram": 0, "ngram_gpu": 0, "suffix": 0,
		"medusa": 0, "mlp_speculator": 0, "extract_hidden_states": 0, "custom_class": 0,
	} {
		if got := (SpeculativeConfig{K: k, Method: method}).LookaheadTokens(); got != want {
			t.Errorf("%s: lookahead %d, want %d", method, got, want)
		}
	}
	if got := (SpeculativeConfig{}).KV(); got != (KVSpeculation{}) {
		t.Errorf("a non-speculating config reserves %+v; it must reserve nothing", got)
	}
}
