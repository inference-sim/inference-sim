// Moved from inference-sim/sim/cluster/dp_replica_test.go at 6e9ef2fb1a370717bcdc34e13998ea7a5a83cd46.

package dplb

import "testing"

type rank struct {
	c        Counts
	inflight int
}

func (r *rank) Counts() Counts { return r.c }
func (r *rank) InFlight() int  { return r.inflight }

// Pick reads the coordinator's counts as published at the start of a 100 ms period, bumping
// the chosen rank's waiting count between publishes; the next period's first assignment
// re-samples the ranks. Waiting requests on a rank with KV usage above one half are penalised,
// and the front-end's exact in-flight count floors a rank's load.
func TestPickStalenessAndScore(t *testing.T) {
	r0, r1 := &rank{}, &rank{}
	ranks := []Rank{r0, r1}
	b := New(2)
	pick := func(now int64) int { return b.Pick(ranks, now) }

	// A stale publish says rank 0 is queued; the live ranks are idle.
	b.SetPublished(0, 0, []Counts{{Waiting: 3, KVUsage: 0.9}, {}})
	if got := []int{pick(10_000), pick(20_000)}; got[0] != 1 || got[1] != 1 {
		t.Errorf("within the published period the stale queue must steer both picks to rank 1, got %v", got)
	}
	// The next period re-samples (both idle) and the rotated scan starts at rank 0.
	if got := pick(StatsIntervalUs + 1); got != 0 {
		t.Errorf("after the next publish the idle ranks tie and the scan starts at rank 0, got %d", got)
	}

	// The local bump: rank 0 starts lighter, takes the first request, and is then counted as
	// waiting it, so the second goes to rank 1 though no publish has happened.
	b.SetPublished(3, 0, []Counts{{}, {Waiting: 1}})
	if got := []int{pick(3*StatsIntervalUs + 1), pick(3*StatsIntervalUs + 2)}; got[0] != 0 || got[1] != 1 {
		t.Errorf("between publishes the chosen rank's waiting count must be bumped: got %v", got)
	}

	// KV pressure: 2 waiting at 95% usage (2 + 2*6*0.45 = 7.4) loses to 3 waiting at 0%.
	b.SetPublished(5, 0, []Counts{{Waiting: 2, KVUsage: 0.95}, {Waiting: 3}})
	if got := pick(5*StatsIntervalUs + 1); got != 1 {
		t.Errorf("the KV-pressure penalty must steer away from rank 0, got %d", got)
	}

	// In-flight floor: published counts are empty, but rank 0 holds five unfinished requests.
	b.SetPublished(6, 0, []Counts{{}, {}})
	r0.inflight = 5
	if got := pick(6*StatsIntervalUs + 1); got != 1 {
		t.Errorf("the in-flight floor must steer away from rank 0, got %d", got)
	}
}

// A new period samples every rank's live counts: a rank that became loaded since the last
// publish is avoided on the first pick of the next period.
func TestPickSamplesAtANewPeriod(t *testing.T) {
	r0, r1 := &rank{}, &rank{}
	b := New(2)
	if got := b.Pick([]Rank{r0, r1}, 0); got != 0 {
		t.Fatalf("idle ranks: got %d, want the scan start, rank 0", got)
	}
	r0.c, r1.c = Counts{Running: 1}, Counts{Waiting: 4, Running: 4}
	if got := b.Pick([]Rank{r0, r1}, StatsIntervalUs); got != 0 {
		t.Errorf("after a publish showing rank 1 loaded: got %d, want rank 0", got)
	}
}
