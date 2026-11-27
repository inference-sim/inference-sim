// Original to blis-sglang.

package sched

import (
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// What PrefillAdder admits, refuses and preempts, read through the adder's own results and lists
// (blis-sglang#104). Each test builds a small scheduler, hands the adder requests by hand and
// reads which of them it admits, the verdict it returns and the chunk it cuts; the numbers are
// chosen so that the verdict sits exactly on a boundary.

// adderRig is a scheduler with a roomy pool, no chunking and fcfs, for the adder tests to edit.
func adderRig(t *testing.T, edit func(*Config)) *Scheduler {
	t.Helper()
	cfg := baseConfig(t)
	cfg.PageSize = 1
	cfg.Policy = "fcfs"
	cfg.ChunkedPrefillSize = nil
	cfg.MaxTotalNumTokens = 10000
	cfg.MaxPrefillTokens = 10000
	cfg.MaxReqLen = 10000
	cfg.MaxReqInputLen = 10000
	cfg.MaxContextLen = 10000
	cfg.ReqPoolSize = 16
	cfg.PPMaxMicroBatchSize = 16
	cfg.MaxRunningRequests = 16
	cfg.EnablePriorityPreemption = false
	cfg.PrefillMaxRequests = nil
	if edit != nil {
		edit(&cfg)
	}
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// tokens returns n distinct token ids that no other seed shares, so no two requests share a prefix.
func tokens(seed, n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(seed*1000 + i + 1)
	}
	return out
}

// queued takes a request through the front of the scheduler, as the loop does before it offers
// the request to the adder: it is queued and has matched the tree.
func queued(s *Scheduler, seed, promptLen, maxNew int) *Req {
	rid := string(rune('a' + seed))
	s.handleGenerateRequest(Arrival{RID: rid, Prompt: tokens(seed, promptLen), MaxNewTokens: maxNew})
	r := s.waiting[len(s.waiting)-1]
	s.initNextRoundInput(r, true)
	return r
}

// bare is a request made by hand, with no queue and no tree match: what the fixed-length
// admission reads of it is its tokens and its limits.
func bare(seed, promptLen, maxNew int, ignoreEOS bool) *Req {
	p := tokens(seed, promptLen)
	return &Req{
		C:            cache.Req{ID: string(rune('a' + seed)), OriginInputIDs: p, FillIDs: append([]int64(nil), p...)},
		MaxNewTokens: maxNew, IgnoreEOS: ignoreEOS,
	}
}

func admittedIDs(a *adder) []string {
	var ids []string
	for _, r := range a.canRunList {
		ids = append(ids, r.RID())
	}
	return ids
}

func mustPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: no refusal", what)
		}
	}()
	f()
}

// A chunk limit of zero is refused where SGLang asserts it positive; it does not cut a chunk of nothing.
func TestAdderRefusesAZeroChunkLimit(t *testing.T) {
	s := adderRig(t, func(c *Config) { n := 64; c.ChunkedPrefillSize = &n })
	r := queued(s, 0, 100, 4)
	a := s.newAdder(s.running, 0)
	zero := 0
	a.chunkedReqLimit = &zero
	mustPanic(t, "chunked_req_limit 0", func() { a.addChunkedReq(r) })
	if len(a.canRunList) != 0 {
		t.Fatalf("a chunk was admitted under a zero limit: %v", admittedIDs(a))
	}
}

// --prefill-max-requests N admits N requests in a pass and stops the (N+1)th with Other.
func TestPrefillMaxRequestsStopsAtTheLimit(t *testing.T) {
	two := 2
	s := adderRig(t, func(c *Config) { c.PrefillMaxRequests = &two })
	a := s.newAdder(s.running, 0)
	var res []AddResult
	for i := 0; i < 3; i++ {
		res = append(res, a.addOneReq(queued(s, i, 10, 4), false))
	}
	if got := admittedIDs(a); len(got) != 2 {
		t.Fatalf("admitted %v, want exactly the first two", got)
	}
	if res[0] != Continue || res[1] != Continue || res[2] != Other {
		t.Fatalf("verdicts %v, want Continue, Continue, Other", res)
	}
}

// Without chunking the first request of a pass is admitted whatever the input cap, because
// nothing else would ever run it.
func TestFirstRequestMayExceedTheInputCapWithoutChunking(t *testing.T) {
	s := adderRig(t, func(c *Config) { c.MaxPrefillTokens = 10 })
	a := s.newAdder(s.running, 0)
	r := queued(s, 0, 30, 4)
	a.addOneReq(r, false)
	if got := admittedIDs(a); len(got) != 1 || r.ExtendEnd != 30 {
		t.Fatalf("admitted %v with extend end %d, want the 30-token request whole", got, r.ExtendEnd)
	}
}

// Without chunking, a later request whose pages reach what is left of the input cap is refused,
// at the cap exactly as much as above it.
func TestLaterRequestAtOrOverTheInputCapIsRefusedWithoutChunking(t *testing.T) {
	for _, n := range []int{30, 40} {
		s := adderRig(t, func(c *Config) { c.MaxPrefillTokens = 50 })
		a := s.newAdder(s.running, 0)
		a.addOneReq(queued(s, 0, 20, 4), false)
		res := a.addOneReq(queued(s, 1, n, 4), false)
		if res != Other || len(admittedIDs(a)) != 1 {
			t.Fatalf("a %d-token request after 20 of 50: verdict %v, admitted %v, want Other and the first alone", n, res, admittedIDs(a))
		}
	}
	// One token under the cap goes in.
	s := adderRig(t, func(c *Config) { c.MaxPrefillTokens = 50 })
	a := s.newAdder(s.running, 0)
	a.addOneReq(queued(s, 0, 20, 4), false)
	a.addOneReq(queued(s, 1, 29, 4), false)
	if len(admittedIDs(a)) != 2 {
		t.Fatalf("a 29-token request after 20 of 50 was refused: admitted %v", admittedIDs(a))
	}
}

// With chunking on, the input cap does not stop a later request: the chunk budget cuts it.
func TestChunkedPassAdmitsALaterRequestPastTheInputCap(t *testing.T) {
	s := adderRig(t, func(c *Config) {
		n := 100
		c.ChunkedPrefillSize = &n
		c.MaxPrefillTokens = 50
	})
	a := s.newAdder(s.running, 0)
	a.addOneReq(queued(s, 0, 20, 4), false)
	r := queued(s, 1, 40, 4)
	a.addOneReq(r, false)
	if got := admittedIDs(a); len(got) != 2 || r.ExtendEnd != 40 {
		t.Fatalf("admitted %v with extend end %d, want both and the second whole", got, r.ExtendEnd)
	}
}

// A pass that has spent its input cap exactly says Other to the next candidate.
func TestSpentInputCapEndsThePass(t *testing.T) {
	s := adderRig(t, func(c *Config) { c.MaxPrefillTokens = 5 })
	a := s.newAdder(s.running, 0)
	res := a.addOneReq(queued(s, 0, 5, 4), false)
	if len(admittedIDs(a)) != 1 || res != Other {
		t.Fatalf("admitted %v with verdict %v, want the request and Other (the cap is spent)", admittedIDs(a), res)
	}
	// The same through the mixed-decode tokens: a cap of 5 less 5 decode rows is spent at the start.
	if v := s.newAdder(s.running, 5).budgetState(); v != Other {
		t.Fatalf("a pass with no input cap left says %v, want Other", v)
	}
}

// A request longer than the chunk budget is cut into a chunk, unless a chunked request is already
// unfinished: only one can be tracked, under every policy (schedule_policy.py:1111-1113@c4af57ff;
// before it was refused only under shortest-prefill-first).
func TestSecondChunkedRequestIsRefusedUnderEveryPolicy(t *testing.T) {
	for _, policy := range []string{"fcfs", "shortest-prefill-first"} {
		for _, hasChunked := range []bool{false, true} {
			s := adderRig(t, func(c *Config) {
				n := 16
				c.ChunkedPrefillSize = &n
				c.Policy = policy
			})
			a := s.newAdder(s.running, 0)
			r := queued(s, 0, 40, 4)
			res := a.addOneReq(r, hasChunked)
			if !hasChunked {
				if len(admittedIDs(a)) != 1 || r.ExtendEnd != 16 || a.newChunkedReq != r {
					t.Fatalf("%s: admitted %v, extend end %d, chunked %v; want a 16-token chunk", policy, admittedIDs(a), r.ExtendEnd, a.newChunkedReq)
				}
			} else if res != Other || len(admittedIDs(a)) != 0 || a.newChunkedReq != nil {
				t.Fatalf("%s: verdict %v, admitted %v; want Other and nothing", policy, res, admittedIDs(a))
			}
		}
	}
}

// The fixed-length admission works on a pool with no radix cache.
func ignoreRig(t *testing.T, total int, edit func(*Config)) *Scheduler {
	return adderRig(t, func(c *Config) {
		c.Disable = true
		c.MaxTotalNumTokens = total
		if edit != nil {
			edit(c)
		}
	})
}

// The request is admitted only if the room left after it exceeds its remaining decode tokens by
// more than one: a room of exactly one more is refused.
func TestFixedLengthAdmissionNeedsMoreThanTheReserve(t *testing.T) {
	// 10 prompt tokens and 5 to decode: 11 pages now, so a pool of 17 leaves 6 = 5 + 1.
	for _, tc := range []struct {
		total int
		ok    bool
	}{{17, false}, {18, true}} {
		s := ignoreRig(t, tc.total, nil)
		a := s.newAdder(s.running, 0)
		res := a.addOneReqIgnoreEOS(bare(0, 10, 5, true))
		if got := len(admittedIDs(a)) == 1; got != tc.ok {
			t.Fatalf("pool %d: admitted %v (verdict %v), want %v", tc.total, got, res, tc.ok)
		}
		if !tc.ok && res != NoToken {
			t.Fatalf("pool %d: verdict %v, want NoToken", tc.total, res)
		}
	}
}

// A request whose pages equal the smaller of the two remainders still gets past the first test,
// and the second test then decides on the larger remainder. After a request with a long decode
// the total remainder is the smaller one.
func TestFixedLengthAdmissionAcceptsPagesEqualToTheTotalRemainder(t *testing.T) {
	s := ignoreRig(t, 120, nil)
	a := s.newAdder(s.running, 0)
	a.addOneReqIgnoreEOS(bare(0, 3, 100, true))
	// Now 116 tokens remain now and 16 in total; the next request needs 15+1 = 16 pages.
	if r, c := a.budget.remainingCurrent(), a.budget.remainingTotal(); r != 116 || c != 16 {
		t.Fatalf("setup: remaining %v now and %v in total, want 116 and 16", r, c)
	}
	a.addOneReqIgnoreEOS(bare(1, 15, 1, true))
	if got := admittedIDs(a); len(got) != 2 {
		t.Fatalf("admitted %v, want both: 16 pages fit in the 16 left", got)
	}
}

// A request that is not shorter than the ones before goes in front of the last entry, as in
// SGLang (the loop variable keeps its last value), so a long newcomer is checked first.
func TestFixedLengthLongNewcomerIsCheckedBeforeTheLastEntry(t *testing.T) {
	s := ignoreRig(t, 24, nil)
	a := s.newAdder(s.running, 0)
	a.addOneReqIgnoreEOS(bare(0, 3, 2, true))
	// 15 tokens remain after the 4 pages; the newcomer needs 5 pages and decodes 10, and being
	// checked first it must clear 2*10 + 2 with two requests still to finish.
	res := a.addOneReqIgnoreEOS(bare(1, 4, 10, true))
	if res != NoToken || len(admittedIDs(a)) != 1 {
		t.Fatalf("verdict %v, admitted %v; want NoToken and the first request alone", res, admittedIDs(a))
	}
}

// A request that has decoded all it will is not among the requests whose memory is released
// later, so it does not make the pool look tighter.
func TestFixedLengthIgnoresARunningRequestWithNothingLeftToDecode(t *testing.T) {
	s := ignoreRig(t, 8, nil)
	s.newTokenRatio.current = 0.1
	done := bare(5, 4, 3, true)
	done.C.OutputIDs = []int64{1, 2, 3}
	s.running.Reqs = []*Req{done}
	a := s.newAdder(s.running, 0)
	// 5 prompt tokens leave 2 after the 6 pages; one decode token at 0.1 is 0.1 left.
	res := a.addOneReqIgnoreEOS(bare(0, 5, 1, false))
	if len(admittedIDs(a)) != 1 {
		t.Fatalf("the request was refused (verdict %v), want it admitted", res)
	}
}

// With no chunk budget left, a request that does not fit in one piece is not given a chunk of
// nothing.
func TestFixedLengthAdmitsNoEmptyChunk(t *testing.T) {
	s := ignoreRig(t, 1000, func(c *Config) { n := 64; c.ChunkedPrefillSize = &n })
	a := s.newAdder(s.running, 64)
	if *a.remChunkTokens != 0 {
		t.Fatalf("setup: chunk budget %d, want 0", *a.remChunkTokens)
	}
	res := a.addOneReqIgnoreEOS(bare(0, 10, 3, true))
	if res != Other || len(admittedIDs(a)) != 0 || a.newChunkedReq != nil {
		t.Fatalf("verdict %v, admitted %v, chunked %v; want Other and nothing", res, admittedIDs(a), a.newChunkedReq)
	}
}

// A running request that finished in the last result is still in the batch but is not a victim.
func TestPreemptionSkipsAFinishedRunningRequest(t *testing.T) {
	s := adderRig(t, func(c *Config) { c.MaxTotalNumTokens = 100 })
	y := bare(1, 4, 10, true)
	y.FinishedReason = "length"
	s.running.Reqs = []*Req{y}
	a := s.newAdder(s.running, 0)
	hi := bare(2, 5, 5, false)
	hi.Priority = 100
	if a.preemptToSchedule(hi) {
		t.Fatalf("preempted %v, want no victim: the only running request has finished", a.preemptList)
	}
	if len(a.preemptList) != 0 || len(s.running.Reqs) != 1 {
		t.Fatalf("preempt list %d, running %d; want 0 and 1", len(a.preemptList), len(s.running.Reqs))
	}
}

// preemptRig is two running requests of equal priority that each hold 10 decode tokens in a pool
// of 100, and a better request that is short of exactly 10 tokens.
func preemptRig(t *testing.T) (*adder, *Req, *Req, *Req) {
	s := adderRig(t, func(c *Config) { c.MaxTotalNumTokens = 100 })
	y1, y2 := bare(1, 4, 10, true), bare(2, 4, 10, true)
	s.running.Reqs = []*Req{y1, y2}
	a := s.newAdder(s.running, 0)
	hi := bare(3, 85, 5, false)
	hi.Priority = 100
	return a, y1, y2, hi
}

// Victims are taken until the shortfall is covered; a victim that covers it exactly is the last.
func TestPreemptionStopsWhenTheShortfallIsCoveredExactly(t *testing.T) {
	a, y1, y2, hi := preemptRig(t)
	if !a.preemptToSchedule(hi) {
		t.Fatal("not preempted: one victim frees exactly the 10 tokens that are short")
	}
	if len(a.preemptList) != 1 || len(a.running.Reqs) != 1 || a.running.Reqs[0] != y2 || a.preemptList[0] != y1 {
		t.Fatalf("preempted %d, running %d; want only the first of the two taken", len(a.preemptList), len(a.running.Reqs))
	}
}

// Of running requests of equal priority and equal queue entry, the first in the batch is the
// victim: the order is stable.
func TestPreemptionTakesTheFirstOfEqualRequests(t *testing.T) {
	a, y1, _, hi := preemptRig(t)
	a.preemptToSchedule(hi)
	if len(a.preemptList) == 0 || a.preemptList[0] != y1 {
		t.Fatalf("victim %v, want the request first in the batch", a.preemptList)
	}
}
