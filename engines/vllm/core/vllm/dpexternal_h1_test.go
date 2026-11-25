// Original to this repository: H1 of the DP external load-balancing RFC (inference-sim#2699).
// These exercise the per-rank surface a router-visible endpoint needs — each rank publishing its
// own KV events, each rank's KV staying on that rank, and a step being due when any rank has work.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// runExternalToIdle drives an external-balancing group to idle, calling onStep after each
// non-empty Complete with the step's result. It is the leaderless loop (serve, specialised to
// capture results), used where a test needs each step's KV events or per-rank state.
func runExternalToIdle(t *testing.T, g *DPGroup, onStep func(res engine.Result)) {
	t.Helper()
	now := engine.Time(0)
	for range 2000 {
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if w, ok := g.NextWake(now); ok {
				now = max(w, now+1)
				continue
			}
			return
		}
		now += g.StepTime(b)
		res, err := g.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		if onStep != nil {
			onStep(res)
		}
	}
	t.Fatal("the group never went idle")
}

// BC-H1-4 (enablement): under external balancing every rank is its own endpoint, so EnableKVEvents
// must turn on every rank's events — a prompt that lands only on a non-zero rank still publishes.
// This fails if EnableKVEvents enables rank 0 alone (the internal-balancer rule, #2677).
func TestExternalBalancingEnablesEveryRanksKVEvents(t *testing.T) {
	g := group(t, "data_parallel_external_lb: true\n")
	g.EnableKVEvents()
	// A unique prompt sent only to endpoint 3: any stored block it publishes is rank 3's.
	if err := g.AddToEndpoint(3, req(strings.Repeat("z", 90), 90, 4), 0); err != nil {
		t.Fatal(err)
	}
	stored := map[string]bool{}
	runExternalToIdle(t, g, func(res engine.Result) {
		for _, e := range res.KVEvents {
			if !e.Removed && !e.Cleared {
				stored[e.Hash] = true
			}
		}
	})
	if len(stored) == 0 {
		t.Fatal("rank 3's endpoint published no KV store events; EnableKVEvents must enable every rank under external balancing")
	}
}

// BC-H1-1: a request's KV lands only on the rank that holds it. Sending one prompt to endpoint 1
// and never driving a request onto any other rank, only rank 1's KV cache is ever used. This fails
// if a request's blocks were allocated on, or leaked to, another rank.
func TestARequestsKVLandsOnlyOnItsRank(t *testing.T) {
	g := group(t, "data_parallel_external_lb: true\n")
	const held = 1
	if err := g.AddToEndpoint(held, req(strings.Repeat("q", 90), 90, 4), 0); err != nil {
		t.Fatal(err)
	}
	maxUsed := make([]int64, g.Ranks())
	record := func() {
		for r := 0; r < g.Ranks(); r++ {
			if u := g.RankSnapshot(r).KVBlocksUsed; u > maxUsed[r] {
				maxUsed[r] = u
			}
		}
	}
	record()
	runExternalToIdle(t, g, func(engine.Result) { record() })
	if maxUsed[held] == 0 {
		t.Fatalf("rank %d served the request but its KV cache was never used", held)
	}
	for r := 0; r < g.Ranks(); r++ {
		if r != held && maxUsed[r] != 0 {
			t.Errorf("rank %d used %d KV blocks for a request held by rank %d; a request's KV must stay on its own rank", r, maxUsed[r], held)
		}
	}
}

// BC-H1-4 (tagging): a KV event carries the rank that produced it, so the leader can route it to
// that rank's endpoint under external balancing — including a Removed or Cleared event, which
// names no request and so cannot be placed from the batch split. LastKVEventsByRank groups each
// Complete's events by producing rank. Driving distinct prompts onto ranks 1 and 2 (and nothing
// onto 0 or 3), and forcing evictions on rank 1: rank 1's stores and removals appear only under
// index 1, rank 2's stores only under index 2, the two stores are disjoint, and no event of any
// kind (store, removal or clear) lands on an idle rank. This fails if events are not grouped by
// their producing rank.
func TestExternalKVEventsCarryTheirRanksIndex(t *testing.T) {
	g := group(t, "data_parallel_external_lb: true\n")
	g.EnableKVEvents()
	// req seeds a prompt from len(id), so ids of distinct lengths give disjoint hash chains. Give
	// rank 1 many distinct prompts (enough to store and then evict) and rank 2 a few of its own, at
	// disjoint lengths from rank 1's.
	for i := 0; i < 12; i++ {
		if err := g.AddToEndpoint(1, req(strings.Repeat("a", 5+i), 90, 4), 0); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 3; i++ {
		if err := g.AddToEndpoint(2, req(strings.Repeat("b", 60+i), 90, 4), 0); err != nil {
			t.Fatal(err)
		}
	}
	store := make([]map[string]bool, g.Ranks())
	removed := make([]int, g.Ranks())
	cleared := make([]int, g.Ranks())
	for r := range store {
		store[r] = map[string]bool{}
	}
	runExternalToIdle(t, g, func(engine.Result) {
		for r, evs := range g.LastKVEventsByRank() {
			for _, e := range evs {
				switch {
				case e.Cleared:
					cleared[r]++
				case e.Removed:
					removed[r]++
				default:
					store[r][e.Hash] = true
				}
			}
		}
	})
	if len(store[1]) == 0 || len(store[2]) == 0 {
		t.Fatalf("rank 1 stored %d blocks, rank 2 stored %d; both endpoints must publish their own stores", len(store[1]), len(store[2]))
	}
	for h := range store[1] {
		if store[2][h] {
			t.Errorf("block %s is published under both rank 1 and rank 2; each endpoint's events must carry only its own rank", h)
		}
	}
	if removed[1] == 0 {
		t.Error("rank 1 evicted blocks but published no removal under its own index")
	}
	for r := 0; r < g.Ranks(); r++ {
		if r == 1 || r == 2 {
			continue
		}
		if len(store[r]) != 0 || removed[r] != 0 || cleared[r] != 0 {
			t.Errorf("idle rank %d published events (store %d, removed %d, cleared %d); no event may land on a rank that served no request", r, len(store[r]), removed[r], cleared[r])
		}
	}
}

// BC-H1-3: StepDue is true whenever any rank has work, not only rank 0. A request on a non-zero
// endpoint must make the group's step due, or the leader (which steps only when StepDue holds)
// would never step a group whose work sits on ranks 1..dp-1. This fails if StepDue consulted rank
// 0 alone.
func TestStepDueTrueWhenOnlyANonZeroRankHasWork(t *testing.T) {
	g := group(t, "data_parallel_external_lb: true\n")
	if g.StepDue(0) {
		t.Fatal("an empty group reports a step due")
	}
	if err := g.AddToEndpoint(g.Ranks()-1, req(strings.Repeat("w", 40), 40, 2), 0); err != nil {
		t.Fatal(err)
	}
	if !g.StepDue(0) {
		t.Fatalf("StepDue is false with work on rank %d; it must be true when any rank has work", g.Ranks()-1)
	}
	runExternalToIdle(t, g, nil)
	if g.StepDue(0) {
		t.Fatal("StepDue is true after the group drained")
	}
}
