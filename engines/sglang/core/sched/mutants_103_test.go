// Original to blis-sglang.

package sched

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// These tests pin what the schedule policies of policy.go decide: the order a policy gives the
// waiting queue, the requests the in-batch prefix check defers, the chunk a shortest-prefill-first
// policy leaves, and the prefixes a policy touches to keep them from eviction. They drive the
// scheduler's own policy code over a tree whose matches the test states.

// policyTree is the scheduler's tree with the answers a test needs to state: match gives the
// prefix a lookup finds (the real tree's when nil), fast and noShare change
// what the tree says it supports, and touched records the prefixes the scheduler refreshed.
type policyTree struct {
	cache.PrefixCache
	match   func(tokens []int64) int
	fast    bool
	noShare bool
	touched [][]int64
}

func (p *policyTree) MatchPrefix(q cache.MatchPrefixParams) cache.MatchResult {
	if p.match == nil {
		return p.PrefixCache.MatchPrefix(q)
	}
	return cache.MatchResult{DevicePrefixLen: p.match(q.Key.Tokens)}
}

func (p *policyTree) TouchPrefix(k cache.Key) { p.touched = append(p.touched, k.Tokens) }
func (p *policyTree) SupportsFastMatchPrefix() bool {
	return p.fast
}
func (p *policyTree) SupportsPrefixSharing() bool { return !p.noShare }

// firstTokenMatch finds a prefix as long as the key's first token says.
func firstTokenMatch(tokens []int64) int { return int(tokens[0]) }

func policySched(t *testing.T, policy string, edit func(*Config)) (*Scheduler, *policyTree) {
	t.Helper()
	cfg := baseConfig(t)
	cfg.Policy = policy
	cfg.PageSize = 1
	if edit != nil {
		edit(&cfg)
	}
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	pt := &policyTree{PrefixCache: s.tree}
	s.tree = pt
	return s, pt
}

// prompt is a ten-token prompt whose first token is first and whose rest is filler of one kind.
func prompt(first int64, kind int64) []int64 {
	p := []int64{first}
	for i := int64(1); i < 10; i++ {
		p = append(p, 1000*kind+i)
	}
	return p
}

func rids(queue []*Req) []string {
	var out []string
	for _, r := range queue {
		out = append(out, r.RID())
	}
	return out
}

func newReq(id string, p []int64) *Req {
	return &Req{C: cache.Req{ID: id, OriginInputIDs: p}, MaxNewTokens: 8}
}

// matchedQueue is n requests whose matches cycle through 0..9, so that sorting by match reorders it.
func matchedQueue(n int) []*Req {
	var q []*Req
	for i := 0; i < n; i++ {
		q = append(q, newReq(fmt.Sprintf("r%03d", i), prompt(int64((i*7)%10), int64(i+1))))
	}
	return q
}

func matchOf(r *Req) int { return int(r.C.OriginInputIDs[0]) }

func nonIncreasing(queue []*Req) bool {
	for i := 1; i < len(queue); i++ {
		if matchOf(queue[i]) > matchOf(queue[i-1]) {
			return false
		}
	}
	return true
}

// lpm and hrrn sort the waiting queue by its prefix matches, but SGLang turns the matching and the
// sort off for a queue longer than 128: the queue is then left in arrival order.
func TestLargeWaitingQueueIsLeftInArrivalOrder(t *testing.T) {
	for _, policy := range []string{"lpm", "hrrn"} {
		t.Run(policy, func(t *testing.T) {
			sorted := func(n int) ([]*Req, []string) {
				s, pt := policySched(t, policy, nil)
				pt.match = firstTokenMatch
				s.processedTokens = 100
				q := matchedQueue(n)
				arrival := rids(q)
				s.calcPriority(q, &Batch{})
				return q, arrival
			}
			q, _ := sorted(waitingQueuePrefixMatchMax)
			if !nonIncreasing(q) {
				t.Fatalf("%s with %d waiting requests did not put the longest match first: %v", policy, waitingQueuePrefixMatchMax, rids(q))
			}
			q, arrival := sorted(waitingQueuePrefixMatchMax + 1)
			if !reflect.DeepEqual(rids(q), arrival) {
				t.Fatalf("%s with %d waiting requests reordered the queue; SGLang leaves it in arrival order", policy, waitingQueuePrefixMatchMax+1)
			}
		})
	}
}

// A request that ties with another on every key keeps its place in the queue, whatever policy sorts
// the queue; hrrn breaks a tie of its key by request id.
func TestTiedRequestsKeepTheirOrder(t *testing.T) {
	t.Run("hrrn orders equal keys by request id", func(t *testing.T) {
		s, _ := policySched(t, "hrrn", nil)
		s.processedTokens = 50
		q := []*Req{newReq("c", prompt(0, 1)), newReq("a", prompt(0, 2)), newReq("b", prompt(0, 3))}
		s.calcPriority(q, &Batch{})
		if got, want := rids(q), []string{"a", "b", "c"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("hrrn order %v, want %v", got, want)
		}
	})
	t.Run("hrrn keeps requests that share an id and every key", func(t *testing.T) {
		s, _ := policySched(t, "hrrn", nil)
		first, second := newReq("x", prompt(0, 1)), newReq("x", prompt(0, 2))
		q := []*Req{first, second}
		s.calcPriority(q, &Batch{})
		if q[0] != first || q[1] != second {
			t.Fatal("hrrn swapped two requests that tie on every key")
		}
	})
	t.Run("shortest-prefill-first keeps requests that tie", func(t *testing.T) {
		s, _ := policySched(t, "shortest-prefill-first", nil)
		q := []*Req{newReq("b", prompt(0, 1)), newReq("a", prompt(0, 2))}
		s.calcPriority(q, &Batch{})
		if got, want := rids(q), []string{"b", "a"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("shortest-prefill-first order %v, want arrival order %v", got, want)
		}
	})
}

// A request that waited longer, or entered the queue earlier, goes first among equal work.
func TestShortestPrefillFirstBreaksTiesByQueueEntry(t *testing.T) {
	s, _ := policySched(t, "shortest-prefill-first", nil)
	late, early := newReq("late", prompt(0, 1)), newReq("early", prompt(0, 2))
	late.WaitQueueEntryStep, early.WaitQueueEntryStep = 5, 2
	q := []*Req{late, early}
	s.calcPriority(q, &Batch{})
	if got, want := rids(q), []string{"early", "late"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order %v, want %v", got, want)
	}
}

// A request whose whole prompt is cached has no prefill left; hrrn runs it before any request that
// has, even one that has waited a long time.
func TestHRRNRunsAFullyCachedRequestFirst(t *testing.T) {
	s, pt := policySched(t, "hrrn", nil)
	pt.match = firstTokenMatch
	s.processedTokens = 100
	waiting := newReq("waiting", prompt(0, 1))
	cached := newReq("cached", prompt(12, 2))
	cached.C.OutputIDs = []int64{1, 2, 3}
	cached.ArrivalProcessedTokens = 100
	q := []*Req{waiting, cached}
	s.calcPriority(q, &Batch{})
	if got, want := rids(q), []string{"cached", "waiting"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("hrrn order %v, want %v", got, want)
	}
}

// The in-batch prefix check covers a request whose match is exactly the check threshold: a request
// behind it that shares a long prefix with it is deferred.
func TestInBatchPrefixCheckIncludesTheThreshold(t *testing.T) {
	s, pt := policySched(t, "lpm", func(c *Config) {
		c.InBatchPrefixCheckThreshold = 3
		c.InBatchPrefixDeprioritizeThreshold = 4
	})
	pt.match = func([]int64) int { return 3 }
	first, second := newReq("first", prompt(0, 1)), newReq("second", prompt(0, 1))
	got := s.computePrefixMatches([]*Req{first, second})
	if got["first"] || !got["second"] {
		t.Fatalf("deprioritized %v, want the second request alone", got)
	}
	// One token past the threshold, the check is skipped.
	pt.match = func([]int64) int { return 4 }
	if got := s.computePrefixMatches([]*Req{first, second}); len(got) != 0 {
		t.Fatalf("deprioritized %v with a match above the check threshold, want none", got)
	}
}

// The waiting-queue tree finds the longest prefix a stored key shares with the lookup, and does not
// run past the end of either.
func TestWaitingTreeMatchesTheLongestSharedPrefix(t *testing.T) {
	var tr simTree
	tr.insert([]int64{1, 2, 3, 4})
	tr.insert([]int64{1, 2, 9})
	for _, tc := range []struct {
		ids  []int64
		want int
	}{
		{[]int64{1, 2, 3, 4, 5}, 4}, // the key ends before the lookup
		{[]int64{1, 2, 3}, 3},       // the lookup ends before the key
		{[]int64{1, 2, 3, 4}, 4},
		{[]int64{1, 2, 9, 9}, 3},
		{[]int64{7}, 0},
		{nil, 0},
	} {
		if got := tr.match(tc.ids); got != tc.want {
			t.Errorf("match(%v) = %d, want %d", tc.ids, got, tc.want)
		}
	}
}

// chunkLimit asks for the chunk limit with a chunked request of remaining tokens left and waiting
// requests of the given prefill work.
func chunkLimit(t *testing.T, budget, remaining int, work ...int) *int {
	t.Helper()
	s, _ := policySched(t, "shortest-prefill-first", nil)
	chunked := &Req{C: cache.Req{FillIDs: make([]int64, remaining)}}
	var q []*Req
	for i, w := range work {
		q = append(q, newReq(fmt.Sprintf("w%d", i), make([]int64, w)))
	}
	return s.shortestPrefillChunkLimit(chunked, q, budget)
}

func show(p *int) string {
	if p == nil {
		return "none"
	}
	return fmt.Sprint(*p)
}

// The chunked request is capped so a shorter waiting request keeps some of the budget, at the
// edges of each test: the budget is just two pages, a waiting request is as long as what the chunked
// one has left, and the reserved tokens fill the budget less a page exactly.
func TestShortestPrefillChunkLimitAtItsEdges(t *testing.T) {
	cases := []struct {
		name              string
		budget, remaining int
		work              []int
		want              string
	}{
		{"budget of two pages still caps", 2, 50, []int{1}, "1"},
		{"budget under two pages does not", 1, 50, []int{1}, "none"},
		{"a request as long as the chunk's rest is not shorter", 100, 5, []int{5}, "none"},
		{"a shorter request is kept room", 100, 6, []int{5}, "95"},
		{"reserved tokens may fill the budget less a page", 10, 20, []int{9}, "1"},
		{"reserved tokens may not pass it", 10, 20, []int{10}, "none"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := show(chunkLimit(t, tc.budget, tc.remaining, tc.work...)); got != tc.want {
				t.Fatalf("chunk limit %s, want %s", got, tc.want)
			}
		})
	}
}

func catchPanic(f func()) (p any) {
	defer func() { p = recover() }()
	f()
	return nil
}

// A supplied shuffle must return each queued request once: an id twice, or an id not queued, is
// refused rather than put in the queue.
func TestSuppliedShuffleMustBeAPermutation(t *testing.T) {
	queue := func() []*Req { return []*Req{newReq("a", prompt(0, 1)), newReq("b", prompt(0, 2))} }
	run := func(order []string) (q []*Req, p any) {
		s, _ := policySched(t, "random", nil)
		s.Shuffle = func([]string) []string { return order }
		q = queue()
		p = catchPanic(func() { s.shuffleQueue(q) })
		return
	}
	if q, p := run([]string{"b", "a"}); p != nil || !reflect.DeepEqual(rids(q), []string{"b", "a"}) {
		t.Fatalf("a permutation gave order %v and panic %v, want b then a", rids(q), p)
	}
	if _, p := run([]string{"a", "a"}); p == nil {
		t.Fatal("a shuffle that returned one request twice was accepted")
	}
	if _, p := run([]string{"a", "zzz"}); p == nil {
		t.Fatal("a shuffle that returned a request that is not queued was accepted")
	}
}

// The routing-key policy puts requests whose key a running request holds first, the more frequent
// key first and equally frequent keys alphabetically; the rest follow in key order.
func TestRoutingKeyOrdersByFrequencyThenKey(t *testing.T) {
	keyed := func(keys ...string) []*Req {
		var q []*Req
		for _, k := range keys {
			r := newReq("req-"+k, prompt(0, 1))
			r.RoutingKey = k
			q = append(q, r)
		}
		return q
	}
	keys := func(q []*Req) []string {
		var out []string
		for _, r := range q {
			out = append(out, r.RoutingKey)
		}
		return out
	}
	t.Run("more frequent key first", func(t *testing.T) {
		q := keyed("b", "m", "c", "")
		sortByRoutingKey(q, &Batch{Reqs: keyed("m", "m", "b")})
		if got, want := keys(q), []string{"m", "b", "", "c"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
	})
	t.Run("equal counts go alphabetically", func(t *testing.T) {
		q := keyed("q", "d")
		sortByRoutingKey(q, &Batch{Reqs: keyed("q", "d")})
		if got, want := keys(q), []string{"d", "q"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
	})
}

// A policy that does not match the queue touches the waiting requests' prefixes on the first pass,
// the head of the queue last; every other case leaves the tree alone.
func TestWaitingPrefixesAreTouchedHeadLast(t *testing.T) {
	queue := func() []*Req {
		return []*Req{newReq("q0", prompt(0, 1)), newReq("q1", prompt(0, 2)), newReq("q2", prompt(0, 3))}
	}
	t.Run("the head of the queue is touched last", func(t *testing.T) {
		s, pt := policySched(t, "fcfs", nil)
		q := queue()
		s.touchWaitingPrefixes(FCFS, q)
		want := [][]int64{q[2].allIDs(), q[1].allIDs(), q[0].allIDs()}
		if !reflect.DeepEqual(pt.touched, want) {
			t.Fatalf("touched %v, want the queue from its tail to its head %v", pt.touched, want)
		}
	})
	t.Run("a policy that matches the queue touches nothing", func(t *testing.T) {
		s, pt := policySched(t, "lpm", nil)
		s.touchWaitingPrefixes(LPM, queue())
		if len(pt.touched) != 0 {
			t.Fatalf("touched %v under lpm", pt.touched)
		}
	})
	t.Run("an empty queue does not use up the refresh", func(t *testing.T) {
		s, pt := policySched(t, "fcfs", nil)
		s.touchWaitingPrefixes(FCFS, nil)
		s.touchWaitingPrefixes(FCFS, queue())
		if len(pt.touched) != 3 {
			t.Fatalf("touched %d prefixes on the first non-empty pass, want 3", len(pt.touched))
		}
	})
	t.Run("only lru eviction needs the refresh", func(t *testing.T) {
		s, pt := policySched(t, "fcfs", nil)
		s.cfg.RadixEvictionPolicy = "lfu"
		s.touchWaitingPrefixes(FCFS, queue())
		if len(pt.touched) != 0 {
			t.Fatalf("touched %v under lfu eviction", pt.touched)
		}
	})
	t.Run("a tree that matches fast needs no refresh", func(t *testing.T) {
		s, pt := policySched(t, "fcfs", nil)
		pt.fast = true
		s.touchWaitingPrefixes(FCFS, queue())
		if len(pt.touched) != 0 {
			t.Fatalf("touched %v on a tree with fast matching", pt.touched)
		}
	})
	t.Run("a tree that shares no prefixes needs no refresh", func(t *testing.T) {
		s, pt := policySched(t, "fcfs", nil)
		pt.noShare = true
		s.touchWaitingPrefixes(FCFS, queue())
		if len(pt.touched) != 0 {
			t.Fatalf("touched %v on a tree that shares no prefixes", pt.touched)
		}
	})
}
