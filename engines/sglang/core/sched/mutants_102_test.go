// Original to blis-sglang.

package sched

import (
	"errors"
	"slices"
	"testing"
)

// Tests that each fail on a flipped operator of scheduler.go that no other test noticed (the
// generated mutants listed in blis-sglang#102). They read what the scheduler does: the errors it
// returns, the requests it finishes, the batch it forms and the reports it makes.

func intp(v int) *int { return &v }

// A chunked prefill size at or below zero is chunked prefill off, which SGLang spells None: the
// scheduler refuses it, and accepts both an absent size and a positive one.
func TestChunkedPrefillSizeMustBeAbsentOrPositive(t *testing.T) {
	for _, tc := range []struct {
		name string
		size *int
		ok   bool
	}{{"absent", nil, true}, {"one", intp(1), true}, {"zero", intp(0), false}, {"negative", intp(-1), false}} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t)
			cfg.ChunkedPrefillSize = tc.size
			_, err := build(cfg, 13)
			if tc.ok && err != nil {
				t.Fatalf("size %v refused: %v", tc.size, err)
			}
			if !tc.ok && err == nil {
				t.Fatalf("size %d accepted", *tc.size)
			}
		})
	}
}

// A prompt token outside [0, vocab_size) is refused, whichever side it falls off; the first and
// last valid tokens are accepted.
func TestPromptTokensOutsideTheVocabularyAreRefused(t *testing.T) {
	cfg := baseConfig(t)
	for _, tc := range []struct {
		token   int64
		refused bool
	}{{0, false}, {int64(cfg.VocabSize - 1), false}, {int64(cfg.VocabSize), true}, {-1, true}} {
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		_, err = s.Step([]Arrival{{RID: "x", Prompt: []int64{5, tc.token, 7}, MaxNewTokens: 2, IgnoreEOS: true}})
		if got := errors.Is(err, ErrRefused); got != tc.refused || (!tc.refused && err != nil) {
			t.Fatalf("token %d: error %v, refused %v, want refused %v", tc.token, err, got, tc.refused)
		}
	}
}

// When the queue is full and an incoming request outranks the waiting ones, the least preferred
// waiting request is aborted; of two equal in priority and in arrival step, the first one.
func TestQueueLimitAbortsTheFirstOfEquallyLeastPreferred(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnablePriorityScheduling = true
	cfg.MaxQueuedRequests = intp(2)
	cfg.PPMaxMicroBatchSize = 0 // nothing is admitted, so the queue stays as the arrivals left it
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	arrival := func(rid string, prio int) Arrival {
		return Arrival{RID: rid, Prompt: []int64{1, 2, 3}, MaxNewTokens: 2, Priority: intp(prio)}
	}
	rec, err := s.Step([]Arrival{arrival("a", 5), arrival("b", 5), arrival("c", 9)})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(rec.Finished, []string{"a"}) {
		t.Fatalf("aborted %v, want a (the first of the two least preferred)", rec.Finished)
	}
	if !slices.Equal(rec.After.Waiting, []string{"c", "b"}) && !slices.Equal(rec.After.Waiting, []string{"b", "c"}) {
		t.Fatalf("waiting %v, want b and c", rec.After.Waiting)
	}
}

// ContainsLastPrefillChunk says whether the extend batch holds the end of every prompt it
// carries: a lone chunk of a long prompt does not, a whole short prompt does.
func TestExtendBatchSaysWhetherItHoldsTheLastPrefillChunk(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prompt int
		last   bool
	}{{"whole prompt", 10, true}, {"a middle chunk alone", 100, false}} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t) // chunked_prefill_size 64
			s, err := build(cfg, 13)
			if err != nil {
				t.Fatal(err)
			}
			p := make([]int64, tc.prompt)
			for i := range p {
				p[i] = int64(i%3 + 1)
			}
			s.handleGenerateRequest(Arrival{RID: "x", Prompt: p, MaxNewTokens: 2, IgnoreEOS: true})
			b := s.getNextBatchToRun()
			if b == nil || b.Mode != Extend || b.Size() != 1 {
				t.Fatalf("batch %+v, want one extend request", b)
			}
			if b.ContainsLastPrefillChunk != tc.last {
				t.Fatalf("ContainsLastPrefillChunk %v, want %v", b.ContainsLastPrefillChunk, tc.last)
			}
		})
	}
}

// A decode batch is reported when metrics are on or KV events are on, and not when neither is.
func TestDecodeIsReportedWithMetricsOrKVEventsAlone(t *testing.T) {
	decodeReports := func(metrics, events bool) (n int) {
		cfg := baseConfig(t)
		cfg.EnableMetrics, cfg.EnableKVEvents = metrics, events
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 6; i++ {
			var arr []Arrival
			if i == 0 {
				arr = []Arrival{{RID: "x", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 4, IgnoreEOS: true}}
			}
			rec, err := s.Step(arr)
			if err != nil {
				t.Fatal(err)
			}
			for _, r := range rec.Reports {
				if r.Decode != nil {
					n++
				}
			}
		}
		return n
	}
	if n := decodeReports(false, false); n != 0 {
		t.Fatalf("%d decode reports with neither metrics nor KV events", n)
	}
	if n := decodeReports(true, false); n != 3 {
		t.Fatalf("%d decode reports with metrics alone, want 3", n)
	}
	if n := decodeReports(false, true); n != 3 {
		t.Fatalf("%d decode reports with KV events alone, want 3", n)
	}
}

// A prefill batch is reported under the same rule.
func TestPrefillIsReportedWithMetricsOrKVEventsAlone(t *testing.T) {
	prefillReports := func(metrics, events bool) (n int) {
		cfg := baseConfig(t)
		cfg.EnableMetrics, cfg.EnableKVEvents = metrics, events
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			var arr []Arrival
			if i == 0 {
				arr = []Arrival{{RID: "x", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 4, IgnoreEOS: true}}
			}
			rec, err := s.Step(arr)
			if err != nil {
				t.Fatal(err)
			}
			for _, r := range rec.Reports {
				if r.Prefill != nil {
					n++
				}
			}
		}
		return n
	}
	for _, tc := range []struct {
		metrics, events bool
		want            int
	}{{false, false, 0}, {true, false, 1}, {false, true, 1}} {
		if n := prefillReports(tc.metrics, tc.events); n != tc.want {
			t.Fatalf("metrics %v, KV events %v: %d prefill reports, want %d", tc.metrics, tc.events, n, tc.want)
		}
	}
}

// A radix eviction policy the tree does not know is refused, and a known one other than lru is
// not silently replaced by lru.
func TestRadixEvictionPolicyIsPassedToTheTree(t *testing.T) {
	for _, tc := range []struct {
		policy string
		ok     bool
	}{{"", true}, {"lru", true}, {"lfu", true}, {"fifo", true}, {"nope", false}, {"tlru", false}} {
		cfg := baseConfig(t)
		cfg.RadixEvictionPolicy = tc.policy
		_, err := build(cfg, 13)
		if (err == nil) != tc.ok {
			t.Fatalf("eviction policy %q: error %v, want ok %v", tc.policy, err, tc.ok)
		}
	}
}

// With mixed chunked prefill under the overlap loop, a running request whose last token is in
// flight leaves the decode batch; the prefill that remains then carries no decode rows, so it is a
// plain extend batch, not a mixed one.
func TestMixedChunkWithNoDecodeRowsLeftIsAPlainExtend(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableOverlap, cfg.IsMixedChunk = true, true
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	var modes [][]Mode
	var decoding [][]string
	for i := 0; i < 6; i++ {
		var arr []Arrival
		switch i {
		case 0:
			// Three tokens, the last of them in flight when step 3 forms its batch.
			arr = []Arrival{{RID: "x", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 3, IgnoreEOS: true}}
		case 3:
			arr = []Arrival{{RID: "y", Prompt: []int64{5, 6, 7}, MaxNewTokens: 3, IgnoreEOS: true}}
		}
		rec, err := s.Step(arr)
		if err != nil {
			t.Fatal(err)
		}
		var ms []Mode
		var dec []string
		for _, f := range rec.Forwards {
			ms = append(ms, f.Mode)
			dec = append(dec, f.Decoding...)
		}
		modes = append(modes, ms)
		decoding = append(decoding, dec)
	}
	if !slices.Equal(modes[3], []Mode{Extend}) || len(decoding[3]) != 0 {
		t.Fatalf("step 3 launched %v with decode rows %v, want a plain extend of y; all steps: %v", modes[3], decoding[3], modes)
	}
}

// A request that finished while a mixed batch carrying it was in flight appears in that batch's
// result once more; the result passes it over, so it finishes once and its KV is freed once.
func TestRequestFinishedUnderAMixedBatchIsNotProcessedAgain(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableOverlap, cfg.IsMixedChunk = true, true
	eos := cfg.EOSTokenIDs[0]
	s, err := build(cfg, eos)
	if err != nil {
		t.Fatal(err)
	}
	before := s.alloc.AvailableSize()
	finished := map[string]int{}
	mixed := false
	for i := 0; i < 8; i++ {
		var arr []Arrival
		switch i {
		case 0:
			arr = []Arrival{{RID: "a", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 8}}
		case 1:
			arr = []Arrival{{RID: "b", Prompt: []int64{5, 6, 7}, MaxNewTokens: 8}}
		}
		rec, err := s.Step(arr)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		for _, f := range rec.Forwards {
			if f.Mode == Mixed && slices.Equal(f.Decoding, []string{"a"}) {
				mixed = true
			}
		}
		for _, r := range rec.Finished {
			finished[r]++
		}
	}
	if !mixed {
		t.Fatal("no mixed batch decoded a")
	}
	if finished["a"] != 1 || finished["b"] != 1 || len(finished) != 2 {
		t.Fatalf("finish counts %v, want a and b once each", finished)
	}
	if got := s.alloc.AvailableSize() + s.tree.EvictableSize(); got != before {
		t.Fatalf("pool holds %d tokens at the end, had %d", got, before)
	}
}

// The new-token ratio decays while requests decode and returns to its initial value once the
// scheduler is fully idle: nothing running, chunking or waiting, and, under the overlap loop,
// no result left to process.
func TestNewTokenRatioReturnsToInitialWhenFullyIdle(t *testing.T) {
	for _, overlap := range []bool{false, true} {
		cfg := baseConfig(t)
		cfg.EnableOverlap = overlap
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		lowest := cfg.NewTokenRatio.Init
		var last State
		for i := 0; i < 20; i++ {
			var arr []Arrival
			if i == 0 {
				arr = []Arrival{{RID: "x", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 4, IgnoreEOS: true}}
			}
			rec, err := s.Step(arr)
			if err != nil {
				t.Fatal(err)
			}
			lowest = min(lowest, rec.After.NewTokenRatio)
			last = rec.After
		}
		if lowest >= cfg.NewTokenRatio.Init {
			t.Fatalf("overlap %v: the ratio never decayed from %v", overlap, cfg.NewTokenRatio.Init)
		}
		if last.NewTokenRatio != cfg.NewTokenRatio.Init {
			t.Fatalf("overlap %v: ratio %v after the drain, want %v", overlap, last.NewTokenRatio, cfg.NewTokenRatio.Init)
		}
	}
}

// A request that cannot be admitted keeps the scheduler from being idle even with nothing
// running: the ratio is not reset, and the batch is not marked full. With the radix cache off, a
// fixed-length request is admitted only if it and the running requests could finish in order
// without running out of memory; one of 25 prompt tokens and 23 new, in a pool of 50, never fits
// alone. It arrives as the request before it finishes, so the ratio has decayed by then.
func TestAWaitingRequestThatDoesNotFitKeepsTheSchedulerBusy(t *testing.T) {
	for _, tc := range []struct {
		overlap bool
		arrives int
	}{{false, 6}, {true, 7}} {
		cfg := baseConfig(t)
		cfg.EnableOverlap = tc.overlap
		cfg.Disable = true
		cfg.MaxTotalNumTokens, cfg.MaxReqLen, cfg.MaxReqInputLen = 50, 49, 44
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		p := make([]int64, 25)
		for i := range p {
			p[i] = int64(i%3 + 1)
		}
		decayed := cfg.NewTokenRatio.Init
		for i := 0; i < 30; i++ {
			var arr []Arrival
			switch i {
			case 0:
				arr = []Arrival{{RID: "x", Prompt: []int64{1, 2, 3, 4, 5}, MaxNewTokens: 6, IgnoreEOS: true}}
			case tc.arrives:
				arr = []Arrival{{RID: "y", Prompt: p, MaxNewTokens: 23, IgnoreEOS: true}}
			}
			rec, err := s.Step(arr)
			if err != nil {
				t.Fatal(err)
			}
			if i == tc.arrives {
				decayed = rec.After.NewTokenRatio
				if len(rec.After.Running) != 0 || !slices.Equal(rec.After.Waiting, []string{"y"}) {
					t.Fatalf("overlap %v: step %d holds running %v waiting %v, want only y waiting", tc.overlap, i, rec.After.Running, rec.After.Waiting)
				}
				if decayed >= cfg.NewTokenRatio.Init {
					t.Fatalf("overlap %v: ratio %v at the arrival, want it decayed below %v by x's decode steps", tc.overlap, decayed, cfg.NewTokenRatio.Init)
				}
			}
			if i > tc.arrives && (rec.After.NewTokenRatio != decayed || rec.After.BatchIsFull || len(rec.Forwards) != 0 || len(rec.After.Waiting) != 1) {
				t.Fatalf("overlap %v, step %d: ratio %v (was %v), batch full %v, forwards %d, waiting %v; want the stuck scheduler unchanged",
					tc.overlap, i, rec.After.NewTokenRatio, decayed, rec.After.BatchIsFull, len(rec.Forwards), rec.After.Waiting)
			}
		}
	}
}
