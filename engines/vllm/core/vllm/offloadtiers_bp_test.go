package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// bpTier is fsTier with store backpressure on the fs tier: one completed store is a sample (warmup 1),
// and a tier whose stores average more than high seconds a MiB is under pressure until they fall under
// half of it. DropStorePolicy drops every store of a tier under pressure.
func bpTier(high string) string {
	return strings.Replace(fsTier, "secondary_tiers:", "backpressure: {backpressure_cls: EMABackpressureDetector, "+
		"policy_cls: DropStorePolicy, warmup_completions: 1, high_water_s: "+high+", low_water_s: 0.000001}, secondary_tiers:", 1)
}

// settle runs e from now until it is idle, each transfer ending 5 microseconds after it starts, and
// returns the transfers it started and the instant it went idle.
func settle(t *testing.T, e *Engine, now engine.Time) ([]engine.Transfer, engine.Time) {
	t.Helper()
	type due struct {
		at engine.Time
		id engine.TransferID
	}
	var started []engine.Transfer
	var transfers []due
	for steps := 0; steps < 10000; steps++ {
		for len(transfers) > 0 && transfers[0].at <= now {
			if err := e.TransferDone(transfers[0].id, true, now); err != nil {
				t.Fatal(err)
			}
			transfers = transfers[1:]
		}
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			started = append(started, tr)
			transfers = append(transfers, due{now + 5, tr.ID})
		}
		if b.Empty() {
			if !e.s.HasUnfinished() && len(transfers) == 0 {
				return started, now
			}
			next := now + 1000
			if len(transfers) > 0 {
				next = min(next, transfers[0].at)
			}
			now = max(next, now+1)
			continue
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("no progress")
	return nil, 0
}

// tierStores counts the stores to the fs tier among transfers.
func tierStores(ts []engine.Transfer) int {
	n := 0
	for _, tr := range ts {
		if tr.Tier == "1:fs" && tr.Kind == engine.TransferStore {
			n++
		}
	}
	return n
}

// prompts runs n requests of fresh 64-token prompts one after another, the i-th arriving gap
// microseconds after the engine went idle with the last, and returns each request's tier stores.
func prompts(t *testing.T, e *Engine, from, n int, gap engine.Time, now engine.Time) ([]int, engine.Time) {
	t.Helper()
	var per []int
	for i := from; i < from+n; i++ {
		r := req("p"+string(rune('a'+i)), 0, 2)
		for k := range r.Prompt {
			r.Prompt[k] = int32(100000*(i+1) + k)
		}
		r.Prompt = r.Prompt[:0]
		for k := 0; k < 64; k++ {
			r.Prompt = append(r.Prompt, int32(100000*(i+1)+k))
		}
		now += gap
		if err := e.Add(r, now); err != nil {
			t.Fatal(err)
		}
		var started []engine.Transfer
		started, now = settle(t, e, now)
		per = append(per, tierStores(started))
	}
	return per, now
}

func sum(xs []int) int {
	n := 0
	for _, x := range xs {
		n += x
	}
	return n
}

// A tier whose stores take longer than the high water mark is under pressure, and the engine drops
// the cascades that would add to it: the same workload makes fewer tier stores than with a high water
// mark no store reaches, or with no backpressure.
func TestSlowStoresPutATierUnderPressureAndItsCascadesAreDropped(t *testing.T) {
	run := func(extra string) []int {
		per, _ := prompts(t, mustEngine(t, extra), 0, 8, 1000, 0)
		return per
	}
	none, healthy, pressed := run(fsTier), run(bpTier("1000")), run(bpTier("0.00001"))
	if sum(none) == 0 || sum(healthy) != sum(none) {
		t.Fatalf("tier stores: none %v, healthy %v; want a tier that never reaches its high water mark to store as much as one with no backpressure", none, healthy)
	}
	if sum(pressed) >= sum(none) || sum(pressed) == 0 {
		t.Fatalf("tier stores: none %v, pressed %v; want the first sample to put the tier under pressure and drop the rest, not all", none, pressed)
	}
}

// The detector's moving average and cooldown run on the engine's virtual time: a tier under pressure
// keeps dropping while requests follow within the half-life, and stores again once the clock has run
// on past its decay and cooldown.
func TestPressureClearsAsVirtualTimePasses(t *testing.T) {
	e := mustEngine(t, bpTier("0.00001"))
	first, now := prompts(t, e, 0, 4, 1000, 0)
	soon, now := prompts(t, e, 4, 2, 1000, now)
	if sum(soon) != 0 {
		t.Fatalf("stores %v a millisecond after the tier was under pressure (%v), want it still dropping", soon, first)
	}
	later, _ := prompts(t, e, 6, 2, 200_000_000, now)
	if later[0] == 0 {
		t.Fatalf("stores %v 200 virtual seconds on, want the pressure cleared by decay and cooldown", later)
	}
}
