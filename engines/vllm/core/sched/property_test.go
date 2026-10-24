package sched

import (
	"fmt"
	"math/rand"
	"testing"
)

// The property and metamorphic tests of #2021 B8, over seeded random scenarios run through
// the Go scheduler alone (the differential tests prove agreement with vLLM; these prove laws
// a correct scheduler keeps whatever vLLM does). Step durations play no part: a step is one
// schedule() and one update_from_output(), which is the scheduler-only form the design review
// asked the metamorphic claims to be narrowed to.

type genReq struct {
	id       string
	arrival  int
	prompt   []int64
	output   int
	priority int
}

type genScenario struct {
	cfg  Config
	reqs []genReq
}

func gen(seed int64) genScenario {
	rng := rand.New(rand.NewSource(seed))
	block := []int{1, 2, 4, 8, 16}[rng.Intn(5)]
	budget := []int{8, 16, 32, 64}[rng.Intn(4)]
	maxLen := []int{64, 128, 256}[rng.Intn(3)]
	cfg := Config{
		MaxNumSeqs: min(1+rng.Intn(8), budget), MaxNumBatchedTokens: budget, BlockSize: block,
		MaxModelLen: maxLen, EnableChunkedPrefill: true, EnablePrefixCaching: rng.Intn(2) == 0,
		ReserveFullSequence: true, Policy: []Policy{FCFS, Priority}[rng.Intn(2)],
	}
	if rng.Intn(3) == 0 {
		cfg.LongPrefillTokenThreshold = 1 + rng.Intn(min(budget, maxLen))
	}
	shared := make([]int64, rng.Intn(3*block+1))
	for i := range shared {
		shared[i] = 1 + rng.Int63n(5000)
	}
	n := 1 + rng.Intn(10)
	var reqs []genReq
	biggest, total := 0, 0
	for i := 0; i < n; i++ {
		plen := 1 + rng.Intn(maxLen/2)
		var prompt []int64
		if len(shared) > 0 && rng.Intn(2) == 0 {
			prompt = append(prompt, shared[:min(len(shared), plen)]...)
		}
		for len(prompt) < plen {
			prompt = append(prompt, 1+rng.Int63n(50000))
		}
		out := 1 + rng.Intn(max(1, min(20, maxLen-plen-1)))
		reqs = append(reqs, genReq{id: fmt.Sprintf("g%02d", i), arrival: rng.Intn(10), prompt: prompt, output: out, priority: rng.Intn(4)})
		biggest = max(biggest, min(plen+out, maxLen))
		total += cdivInt(plen+out, block)
	}
	need := cdivInt(biggest, block) + 1
	cfg.NumBlocks = need + rng.Intn(max(1, total+1-need)+1)
	return genScenario{cfg: cfg, reqs: reqs}
}

func cdivInt(a, b int) int { return (a + b - 1) / b }

type obs struct {
	steps       int
	preemptions int
	finished    map[string]int // output tokens
	firstStep   map[string]int // step that produced a request's first token
}

// drive runs a scenario to completion, checking the per-step properties as it goes.
func drive(t *testing.T, sc genScenario) obs {
	t.Helper()
	s, err := New(sc.cfg)
	if err != nil {
		t.Fatal(err)
	}
	o := obs{finished: map[string]int{}, firstStep: map[string]int{}}
	byID := map[string]genReq{}
	sampled := map[string]int{}
	added := 0
	for step := 0; step < 20000; step++ {
		for _, r := range sc.reqs {
			if r.arrival == step {
				if err := s.Add(&Request{ID: r.id, Prompt: r.prompt, MaxTokens: 1 << 20, EOS: eosToken,
					Priority: r.priority, ArrivalTime: float64(r.arrival)}); err != nil {
					t.Fatal(err)
				}
				byID[r.id] = r
				added++
			}
		}
		if !s.HasUnfinished() && added == len(sc.reqs) {
			o.steps = step
			break
		}
		out := s.Schedule()
		// Budgets are never exceeded (#2021 B8 property).
		total := 0
		for _, n := range out.Scheduled {
			if n <= 0 {
				t.Fatalf("step %d: a request was scheduled %d tokens", step, n)
			}
			total += n
		}
		if total > sc.cfg.MaxNumBatchedTokens {
			t.Fatalf("step %d: %d tokens scheduled over a budget of %d", step, total, sc.cfg.MaxNumBatchedTokens)
		}
		if nr, _ := s.Counts(); nr > sc.cfg.MaxNumSeqs {
			t.Fatalf("step %d: %d running over max_num_seqs %d", step, nr, sc.cfg.MaxNumSeqs)
		}
		o.preemptions += len(out.Preempted)
		mock := map[string]int64{}
		for id := range out.Scheduled {
			if s.Request(id).IsPrefillChunk {
				continue
			}
			k := sampled[id]
			if k == 0 {
				if _, seen := o.firstStep[id]; !seen {
					o.firstStep[id] = step
				}
			}
			if k == byID[id].output-1 {
				mock[id] = eosToken
			} else {
				mock[id] = 1
			}
			sampled[id] = k + 1
		}
		for _, f := range s.Update(out, mock) {
			if _, twice := o.finished[f.ID]; twice {
				t.Fatalf("step %d: %s finished twice", step, f.ID)
			}
			o.finished[f.ID] = f.Output
		}
		// KV conservation: free + held-by-running (distinct blocks) + cached-and-free is the
		// pool; every running request's blocks hold at least one reference each.
		held := map[int]bool{}
		for _, id := range s.Running() {
			for _, b := range s.KV.Blocks(id) {
				if b.RefCnt < 1 {
					t.Fatalf("step %d: %s holds block %d with ref count %d", step, id, b.ID, b.RefCnt)
				}
				held[b.ID] = true
			}
		}
		if s.KV.Pool.NumFree()+len(held)+1 != s.KV.Pool.NumBlocks() {
			t.Fatalf("step %d: %d free + %d held + 1 null != %d blocks", step, s.KV.Pool.NumFree(), len(held), s.KV.Pool.NumBlocks())
		}
	}
	// No lost requests: every request finished, and with the output it was assigned.
	for _, r := range sc.reqs {
		got, ok := o.finished[r.id]
		if !ok {
			t.Fatalf("%s never finished", r.id)
		}
		if want := min(r.output, sc.cfg.MaxModelLen-len(r.prompt)); got != want && got != r.output {
			t.Fatalf("%s finished with %d output tokens, assigned %d", r.id, got, r.output)
		}
	}
	return o
}

func TestPropertiesHoldOnRandomScenarios(t *testing.T) {
	for seed := int64(1); seed <= 500; seed++ {
		sc := gen(seed)
		t.Run(fmt.Sprint(seed), func(t *testing.T) { drive(t, sc) })
	}
}

// The two metamorphic claims of #2021 B8 are false for vLLM itself, not only under real
// pricing as the design review noted, and the counterexamples below were checked against
// vLLM v0.32.0's own Scheduler (oracle/scenarios/meta-*.json):
//
//   - "More KV never adds preemptions": with FCFS, no prefix caching and every request at
//     step 0, 33 blocks preempt nothing and 40 blocks preempt once (meta-more-kv): the larger
//     pool admits a fourth request early, which later outgrows the pool.
//   - "A larger token budget never delays the first token": with max_num_seqs 1 and priority
//     scheduling, budget 8 gives g02 its first token at step 7 and budget 16 at step 9
//     (meta-larger-budget): the faster prefill of g00 lets g01, which arrives at step 3 with a
//     higher priority than g02, be admitted first.
//
// So what holds is narrower still, and these are the forms tested here:

// With every request at step 0 and FIFO admission of one request at a time (max_num_seqs 1),
// more KV never adds preemptions: the order requests run in cannot change, and a request that
// fits fits in a larger pool.
func TestMoreKVNeverAddsPreemptionsOneAtATime(t *testing.T) {
	for seed := int64(1); seed <= 400; seed++ {
		sc := gen(seed)
		sc.cfg.Policy, sc.cfg.EnablePrefixCaching, sc.cfg.LongPrefillTokenThreshold, sc.cfg.MaxNumSeqs = FCFS, false, 0, 1
		for i := range sc.reqs {
			sc.reqs[i].arrival = 0
		}
		base := drive(t, sc)
		more := sc
		more.cfg.NumBlocks = sc.cfg.NumBlocks + 1 + int(seed%7)
		if got := drive(t, more); got.preemptions > base.preemptions {
			t.Fatalf("seed %d: %d blocks preempt %d times, %d blocks %d times", seed,
				sc.cfg.NumBlocks, base.preemptions, more.cfg.NumBlocks, got.preemptions)
		}
	}
}

// With a single request, a larger token budget never moves its first token to a later step:
// the same prefill finishes in no more chunks.
func TestLargerBudgetNeverDelaysALoneRequestsFirstToken(t *testing.T) {
	for seed := int64(1); seed <= 400; seed++ {
		sc := gen(seed)
		sc.reqs = sc.reqs[:1]
		sc.cfg.LongPrefillTokenThreshold = 0
		base := drive(t, sc)
		more := sc
		more.cfg.MaxNumBatchedTokens = sc.cfg.MaxNumBatchedTokens * 2
		got := drive(t, more)
		for id, step := range base.firstStep {
			if got.firstStep[id] > step {
				t.Fatalf("seed %d: %s's first token at step %d with budget %d, %d with budget %d",
					seed, id, step, sc.cfg.MaxNumBatchedTokens, got.firstStep[id], more.cfg.MaxNumBatchedTokens)
			}
		}
	}
}
