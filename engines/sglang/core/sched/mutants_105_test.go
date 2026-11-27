// Original to blis-sglang.

package sched

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// Tests that fail on the generated mutants of batch.go, budget.go, overlap.go and report.go that
// the first sweep left standing (blis-sglang#105). Each reads what the scheduler does: the passes
// it launches, the requests it keeps, the figures its budget gives, the reports it makes.

// repeat is n copies of the token v: a prompt whose requests differ by v.
func repeat(n int, v int64) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = v
	}
	return p
}

// buildWithEvents is build with the tree cache queueing its KV events.
func buildWithEvents(cfg Config, sampled int64) (*Scheduler, error) {
	alloc := cache.NewTokenAllocator(cfg.MaxTotalNumTokens, nil, cfg.NeedSort)
	pool := cache.NewRequestPool(cfg.ReqPoolSize, cfg.MaxContextLen)
	tree, err := cache.NewRadixCache(cache.RadixConfig{PageSize: cfg.PageSize, Disable: cfg.Disable,
		EvictionPolicy: cfg.radixEvictionPolicy(), KVCacheEvents: true}, alloc, pool)
	if err != nil {
		return nil, err
	}
	return New(cfg, tree, alloc, pool, sampled)
}

func mustStep(t *testing.T, s *Scheduler, arr ...Arrival) StepRecord {
	t.Helper()
	rec, err := s.Step(arr)
	if err != nil {
		t.Fatal(err)
	}
	return rec
}

func forwardRIDs(rec StepRecord) [][]string {
	var out [][]string
	for _, f := range rec.Forwards {
		out = append(out, f.RIDs)
	}
	return out
}

// ---- budget.go ------------------------------------------------------------------------------

// A budget has capacity only while both what remains in total and what remains right now are
// positive: with nothing left in either, no request is admitted.
func TestBudgetHasNoCapacityWhenEitherRemainderIsSpent(t *testing.T) {
	s, err := build(baseConfig(t), 13)
	if err != nil {
		t.Fatal(err)
	}
	room := s.alloc.AvailableSize() + s.tree.EvictableSize()

	roomy := newBudget(s.alloc, s.tree, 0)
	if !roomy.hasCapacity() {
		t.Fatal("an untouched budget has no capacity")
	}

	// The expected decode tokens use up all the room, while the tokens needed right now do not.
	totalSpent := newBudget(s.alloc, s.tree, 0)
	totalSpent.reserve(10, float64(room-11), 0, 0, 0, false)
	if totalSpent.remainingTotal() != 0 || totalSpent.remainingCurrent() <= 0 {
		t.Fatalf("setup: remaining total %v, current %v", totalSpent.remainingTotal(), totalSpent.remainingCurrent())
	}
	if totalSpent.hasCapacity() {
		t.Fatal("a budget whose total is spent still has capacity")
	}

	// After a preemption gives back running requests' expected tokens, the total offset falls
	// below the current one; the current offset can then be the one that is spent.
	currentSpent := newBudget(s.alloc, s.tree, 0)
	currentSpent.reserve(10, 0, 0, 0, 0, false)
	currentSpent.totalOffset = 1
	currentSpent.currentOffset = float64(room)
	if currentSpent.remainingCurrent() != 0 || currentSpent.remainingTotal() <= 0 {
		t.Fatalf("setup: remaining total %v, current %v", currentSpent.remainingTotal(), currentSpent.remainingCurrent())
	}
	if currentSpent.hasCapacity() {
		t.Fatal("a budget whose current room is spent still has capacity")
	}
}

// A chunk is cut to what remains in total; a continuing chunk with nothing left parks (no chunk is
// offered) rather than take the headroom the running decodes were promised.
func TestChunkTokensAreCutToWhatRemainsAndParkWhenNothingDoes(t *testing.T) {
	s, err := build(baseConfig(t), 13)
	if err != nil {
		t.Fatal(err)
	}
	room := s.alloc.AvailableSize() + s.tree.EvictableSize()
	for _, c := range []struct{ remaining, limit, want int }{
		{room, 7, 7}, // plenty left: the limit
		{3, 7, 3},    // less left than the limit: what is left
		{0, 7, 0},    // nothing left: parks
	} {
		b := newBudget(s.alloc, s.tree, room-c.remaining)
		got, ok := b.availableChunkTokens(c.limit)
		if ok != (c.want > 0) || got != c.want {
			t.Errorf("limit %d with %d tokens remaining: %d chunk tokens, want %d", c.limit, c.remaining, got, c.want)
		}
	}
}

// ---- report.go ------------------------------------------------------------------------------

func metricsConfig(t *testing.T, interval int) Config {
	cfg := baseConfig(t)
	cfg.EnableMetrics = true
	cfg.DecodeLogInterval = interval
	return cfg
}

// With metrics on, the batch of a prefill pass is reported when its result is processed.
func TestPrefillPassIsReported(t *testing.T) {
	s, err := build(metricsConfig(t, 1), 13)
	if err != nil {
		t.Fatal(err)
	}
	rec := mustStep(t, s, Arrival{RID: "a", Prompt: repeat(6, 1), MaxNewTokens: 4})
	if len(rec.Reports) != 1 {
		t.Fatalf("the prefill step made %d reports, want 1", len(rec.Reports))
	}
	r := rec.Reports[0]
	if r.Prefill == nil || r.Prefill.LogInputTokens != 6 || !r.Logged {
		t.Fatalf("report %+v, want a logged prefill report of 6 input tokens", r)
	}
}

// Every decode batch is reported, but only every decode_log_interval-th one reaches the logging
// part of the reporter.
func TestDecodeReportsAreLoggedEveryIntervalthBatch(t *testing.T) {
	loggedOf := func(interval int) []bool {
		s, err := build(metricsConfig(t, interval), 13)
		if err != nil {
			t.Fatal(err)
		}
		var logged []bool
		for i := 0; i < 12; i++ {
			var arr []Arrival
			if i == 0 {
				arr = []Arrival{{RID: "a", Prompt: repeat(4, 1), MaxNewTokens: 8}}
			}
			rec := mustStep(t, s, arr...)
			for _, r := range rec.Reports {
				if r.Decode != nil {
					logged = append(logged, r.Logged)
				}
			}
		}
		return logged
	}
	// A request of 8 tokens has one prefill and 7 decodes.
	for _, c := range []struct {
		interval int
		want     []bool
	}{
		{3, []bool{false, false, true, false, false, true, false}},
		{1, []bool{true, true, true, true, true, true, true}},
		{0, []bool{true, true, true, true, true, true, true}}, // an interval of 0 logs every batch
	} {
		if got := loggedOf(c.interval); !reflect.DeepEqual(got, c.want) {
			t.Errorf("decode_log_interval %d: logged %v, want %v", c.interval, got, c.want)
		}
	}
}

// The tree's KV events leave with the prefill report that is made after they were queued; and,
// when the scheduler goes idle with events still queued, in a report of their own.
func TestKVEventsLeaveWithAReportOrOnIdle(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableKVEvents = true
	// No decode report is logged, so the events of the finish wait for the idle step.
	cfg.DecodeLogInterval = 1000
	s, err := buildWithEvents(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	rec := mustStep(t, s, Arrival{RID: "a", Prompt: repeat(6, 1), MaxNewTokens: 4})
	if len(rec.Reports) != 1 || rec.Reports[0].Idle || len(rec.Reports[0].Events) == 0 {
		t.Fatalf("the prefill step made reports %+v, want one prefill report carrying the prompt's events", rec.Reports)
	}
	var idle []Report
	for i := 0; i < 12; i++ {
		for _, r := range mustStep(t, s).Reports {
			if r.Decode != nil && (r.Logged || len(r.Events) != 0) {
				t.Fatalf("an unlogged decode report carries %d events", len(r.Events))
			}
			if r.Idle {
				idle = append(idle, r)
			}
		}
	}
	if len(idle) != 1 || len(idle[0].Events) == 0 || idle[0].Prefill != nil || idle[0].Decode != nil {
		t.Fatalf("idle reports %+v, want one carrying the events the finish queued", idle)
	}
}

// ---- overlap.go -----------------------------------------------------------------------------

// With the switch on, the result of an extend batch, a mixed batch being one, is processed before the
// next batch is formed; a decode batch or none leaves it to overlap, and with the switch off no
// batch is processed early.
func TestOverlapAfterPrefillSwitchAppliesToExtendAndMixedOnly(t *testing.T) {
	for _, on := range []bool{true, false} {
		cfg := baseConfig(t)
		cfg.EnableOverlap = true
		cfg.DisableOverlapAfterPrefill = on
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		b := func(m Mode) *Batch { return &Batch{Mode: m} }
		for _, c := range []struct {
			last *Batch
			want bool
		}{
			{b(Extend), on},
			{b(Mixed), on},
			{b(Decode), false},
			{nil, false},
		} {
			if got := s.shouldProcessLastBatchEarly(c.last); got != c.want {
				name := Mode("none")
				if c.last != nil {
					name = c.last.Mode
				}
				t.Errorf("switch %v, last batch %s: processed early = %v, want %v", on, name, got, c.want)
			}
		}
	}
}

// A small pool and a request that ends by length are enough to leave the overlap loop with a
// running request and no queued result: the pool is empty, the last decode of request a is in
// flight, so b's next decode waits for a's result to free a token; the step launches nothing, and
// the one after decodes b with nothing in the result queue.
func TestOverlapDecodesAfterAStepThatLaunchedNothing(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableOverlap = true
	cfg.Disable = true
	cfg.MaxTotalNumTokens = 10
	cfg.ChunkedPrefillSize = nil
	// The expected decode tokens of running requests count for next to nothing, so both are admitted.
	cfg.NewTokenRatio = NewTokenRatioConfig{Init: 0.01, Min: 0.01}
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("the loop panicked: %v", p)
		}
	}()
	type step struct {
		forwards [][]string
		finished []string
	}
	var got []step
	for i := 0; i < 10; i++ {
		var arr []Arrival
		switch i {
		case 0:
			arr = []Arrival{{RID: "a", Prompt: repeat(1, 1), MaxNewTokens: 5}}
		case 1:
			arr = []Arrival{{RID: "b", Prompt: repeat(1, 2), MaxNewTokens: 6}}
		}
		rec := mustStep(t, s, arr...)
		if len(rec.After.Waiting) != 0 {
			t.Fatalf("step %d: %v waiting; no request should be retracted", i, rec.After.Waiting)
		}
		got = append(got, step{forwardRIDs(rec), rec.Finished})
	}
	want := []step{
		{[][]string{{"a"}}, nil},
		{[][]string{{"b"}}, nil},
		{[][]string{{"a", "b"}}, nil},
		{[][]string{{"a", "b"}}, nil},
		{[][]string{{"a", "b"}}, nil},
		{[][]string{{"a", "b"}}, nil},
		{nil, []string{"a"}}, // the pool is empty: b's decode waits for a's result
		{[][]string{{"b"}}, nil},
		{nil, []string{"b"}},
		{nil, nil},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("steps %+v\nwant %+v", got, want)
	}
}

// new_token_ratio is reset to its initial value when the overlap loop finds the scheduler fully
// idle: the step after the last result is processed has no batch, no last batch and nothing queued.
func TestOverlapResetsNewTokenRatioWhenIdle(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableOverlap = true
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	lowest := cfg.NewTokenRatio.Init
	var last float64
	for i := 0; i < 14; i++ {
		var arr []Arrival
		if i == 0 {
			arr = []Arrival{{RID: "a", Prompt: repeat(3, 1), MaxNewTokens: 5}}
		}
		last = mustStep(t, s, arr...).After.NewTokenRatio
		lowest = min(lowest, last)
	}
	if lowest >= cfg.NewTokenRatio.Init {
		t.Fatalf("setup: the ratio never fell below %v while a decoded", cfg.NewTokenRatio.Init)
	}
	if last != cfg.NewTokenRatio.Init {
		t.Fatalf("new_token_ratio is %v after the scheduler went idle, want %v", last, cfg.NewTokenRatio.Init)
	}
}

// A request that cannot be admitted keeps the scheduler from being idle, so the ratio is kept.
// Here b needs the whole pool (its prompt takes 15 of 16 tokens, and the page that goes with it
// the last), which is never more than the pool holds, so the strict fit fails and b waits for
// good, even after a has finished and nothing is left in flight.
func TestOverlapKeepsNewTokenRatioWhileARequestWaits(t *testing.T) {
	cfg := baseConfig(t)
	cfg.EnableOverlap = true
	cfg.MaxTotalNumTokens = 16
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	var ratios []float64
	var rec StepRecord
	for i := 0; i < 20; i++ {
		var arr []Arrival
		if i == 0 {
			arr = []Arrival{
				{RID: "a", Prompt: repeat(3, 1), MaxNewTokens: 5},
				{RID: "b", Prompt: repeat(15, 2), MaxNewTokens: 5},
			}
		}
		rec = mustStep(t, s, arr...)
		ratios = append(ratios, rec.After.NewTokenRatio)
	}
	if !reflect.DeepEqual(rec.After.Waiting, []string{"b"}) || len(rec.After.Running) != 0 {
		t.Fatalf("setup: waiting %v, running %v; want only b waiting", rec.After.Waiting, rec.After.Running)
	}
	if !slices.Contains(rec.After.Waiting, "b") || rec.After.AvailableTokens+rec.After.Evictable != 16 {
		t.Fatalf("setup: pool holds %d free and %d evictable of 16", rec.After.AvailableTokens, rec.After.Evictable)
	}
	if got, init := ratios[len(ratios)-1], cfg.NewTokenRatio.Init; !(got < init) {
		t.Fatalf("new_token_ratio is %v with b still waiting, want it kept below %v", got, init)
	}
}

// ---- batch.go -------------------------------------------------------------------------------

// A request that matches a prefix in the tree reads that prefix's KV: its request row begins with
// the slots of the request that put the prefix there.
func TestNewRequestRowStartsWithTheMatchedPrefixSlots(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	rec := mustStep(t, s, Arrival{RID: "a", Prompt: []int64{1, 2, 3, 4, 5, 6}, MaxNewTokens: 4})
	aRow := slices.Clone(s.pool.Row(rec.Forwards[0].ReqPoolIndices[0])[:6])
	if slices.Contains(aRow, 0) {
		t.Fatalf("setup: a's row %v holds slot 0", aRow)
	}
	rec = mustStep(t, s, Arrival{RID: "b", Prompt: []int64{1, 2, 3, 4, 5, 9}, MaxNewTokens: 4})
	f := rec.Forwards[0]
	if f.Mode != Extend || f.RIDs[0] != "b" || f.ExtendPrefixLen[0] != 5 {
		t.Fatalf("setup: forward %+v, want b's prefill after a 5-token hit", f)
	}
	if got := s.pool.Row(f.ReqPoolIndices[0])[:5]; !slices.Equal(got, aRow[:5]) {
		t.Fatalf("b's row begins %v, want the slots a's prompt holds, %v", got, aRow[:5])
	}
}

// A decode step that does not fit retracts requests until it does: here two at once, the two
// that came last, and the pass decodes the six that remain.
func TestRetractionTakesAsManyRequestsAsTheStepNeeds(t *testing.T) {
	cfg := baseConfig(t)
	cfg.PageSize = 4
	cfg.Allocator = "PagedTokenToKVPoolAllocator"
	cfg.MaxTotalNumTokens = 160
	cfg.ChunkedPrefillSize = nil
	// Few expected decode tokens per running request, so that all eight are admitted.
	cfg.NewTokenRatio = NewTokenRatioConfig{Init: 0.01, Min: 0.01}
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 60; i++ {
		var arr []Arrival
		if i < 8 {
			arr = []Arrival{{RID: fmt.Sprintf("r%d", i), Prompt: repeat(1, int64(i+1)), MaxNewTokens: 100}}
		}
		rec := mustStep(t, s, arr...)
		if len(rec.After.Waiting) == 0 {
			continue
		}
		if want := []string{"r7", "r6"}; !reflect.DeepEqual(rec.After.Waiting, want) {
			t.Fatalf("step %d: waiting %v, want %v retracted together", i, rec.After.Waiting, want)
		}
		if want := [][]string{{"r0", "r1", "r2", "r3", "r4", "r5"}}; !reflect.DeepEqual(forwardRIDs(rec), want) {
			t.Fatalf("step %d: decoded %v, want %v", i, forwardRIDs(rec), want)
		}
		return
	}
	t.Fatal("no request was retracted")
}

// When the one request left cannot decode either, the step is refused by name: SGLang aborts that
// request and the port does not model the abort.
func TestRetractionRefusesToAbortTheLastRequest(t *testing.T) {
	cfg := baseConfig(t)
	cfg.Disable = true
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	mustStep(t, s, Arrival{RID: "a", Prompt: repeat(5, 1), MaxNewTokens: 20})
	// Something else takes every free token, so a's next token has no room.
	if _, ok := s.alloc.Alloc(s.alloc.AvailableSize()); !ok {
		t.Fatal("setup: cannot take the free tokens")
	}
	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("the step did not return the refusal: %v", p)
		}
	}()
	_, err = s.Step(nil)
	if !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), RetractionRefusal) {
		t.Fatalf("step returned %v, want a refusal citing %s", err, RetractionRefusal)
	}
}
