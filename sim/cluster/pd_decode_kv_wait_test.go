package cluster

import (
	"container/heap"
	"math"
	"sort"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// A handoff whose decode instance has no free KV waits in the decode engine's queue, as vLLM's
// waiting queue does, rather than being dropped: with 3 decode blocks and four 2-block prompts,
// every request completes, none is dropped, some handoff starts after its prefill finished, and
// the waiters' pulls start in the order their prefills finished (FCFS).
func TestPDDecodeKVWait_FullDecodeKVWaitsRatherThanDrops(t *testing.T) {
	config := newTestDisaggDeploymentConfig(3, 2, 1)
	config.KVCacheConfig = sim.NewKVCacheConfig(3, 16, 0, 0, 0, 0)
	config.PDPrefillKVHold = true
	cs := runHold(t, config, newShortRequests(4)...)
	if cs.droppedAtDecodeKV != 0 {
		t.Fatalf("%d handoffs dropped; a prompt the decode KV can hold must wait", cs.droppedAtDecodeKV)
	}
	if l := newClusterLedger(cs); l.accounted() != 4 || cs.AggregatedMetrics().CompletedRequests != 4 {
		t.Fatalf("want all 4 completed; ledger %s", l)
	}
	if cs.DecodeKVWaiting() != 0 || cs.PrefillKVHeld() != 0 || cs.transfersInitiated != cs.transfersCompleted {
		t.Fatalf("after the run: %d waiting, %d held, initiated %d completed %d",
			cs.DecodeKVWaiting(), cs.PrefillKVHeld(), cs.transfersInitiated, cs.transfersCompleted)
	}
	ps := cs.ParentRequests()
	sort.Slice(ps, func(i, j int) bool {
		if ps[i].PrefillCompleteTime != ps[j].PrefillCompleteTime {
			return ps[i].PrefillCompleteTime < ps[j].PrefillCompleteTime
		}
		return ps[i].ID < ps[j].ID
	})
	waited := false
	for i, p := range ps {
		waited = waited || p.TransferStartTime > p.PrefillCompleteTime
		if i > 0 && p.TransferStartTime < ps[i-1].TransferStartTime {
			t.Errorf("%s's pull started at %d, before %s's at %d though its prefill finished later (FCFS)",
				p.ID, p.TransferStartTime, ps[i-1].ID, ps[i-1].TransferStartTime)
		}
	}
	if !waited {
		t.Fatal("fixture: no handoff waited for decode KV")
	}
}

// A decode instance that is unroutable when the prefill finishes never received the request,
// so it sends no abort notice and no heartbeat: the prefill side frees its KV only when the
// lease that began at prefill completion runs out (nixl/base_worker.py:3056-3082).
func TestPDDecodeKVWait_DecodeThatNeverHadTheRequestLeavesTheLease(t *testing.T) {
	const lease = 3_000_000
	run := func(horizon int64) *ClusterSimulator {
		cfg := holdConfig(true, 1000)
		cfg.PDKVLeaseUs = lease
		cfg.Horizon = horizon
		cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{holdTestRequest("r", 1, 0, 160, 4)}), nil)
		heap.Push(&cs.clusterEvents, clusterEventEntry{event: &pdDrainAt{time: 1, inst: cs.Instances()[1].ID()}, seqID: cs.nextSeqID()})
		mustRun(t, cs)
		return cs
	}
	full := run(math.MaxInt64)
	p := full.ParentRequests()[0]
	if p.DecodeSubReq != nil || p.TransferStartTime == 0 || p.TransferCompleteTime != 0 {
		t.Fatalf("fixture: the handoff must be dropped at transfer start")
	}
	expiry := p.PrefillCompleteTime + lease
	if before := run(expiry - 1); usedBlocks(before.Instances()[0]) != 10 {
		t.Errorf("before the lease runs out the prefill KV must stay held: %d blocks used", usedBlocks(before.Instances()[0]))
	}
	if after := run(expiry); usedBlocks(after.Instances()[0]) != 0 || after.PrefillKVHeld() != 0 {
		t.Errorf("at prefill end + lease the prefill KV must be freed: %d blocks used", usedBlocks(after.Instances()[0]))
	}
}

// A waiter whose decode instance drains while it waits is dropped; the decode engine had the
// request and heartbeated, so its prefill KV is freed at lease expiry, and every request is
// still accounted for (INV-1, INV-PD-3).
func TestPDDecodeKVWait_DrainedDecodeDropsItsWaiters(t *testing.T) {
	config := holdConfig(true, 1000)
	// 40 decode blocks: "long" reserves 10 and grows to 35 over its 400-token decode, so the
	// waiter's 31-block prompt never fits beside it.
	blocks := int64(40)
	config.DecodeOverrides.TotalKVBlocks = &blocks
	long := holdTestRequest("long", 1, 0, 160, 400)
	waiter := holdTestRequest("waiter", 2, 1000, 496, 4)
	run := func(horizon int64) *ClusterSimulator {
		cfg := config
		cfg.Horizon = horizon
		cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{long, waiter}), nil)
		heap.Push(&cs.clusterEvents, clusterEventEntry{event: &pdDrainAt{time: 300_000, inst: cs.Instances()[1].ID()}, seqID: cs.nextSeqID()})
		mustRun(t, cs)
		return cs
	}
	cs := run(math.MaxInt64)
	ps := parentsByID(cs)
	if w := ps["waiter"]; w.DecodeSubReq != nil || cs.droppedAtDecodeKV != 1 || w.PrefillCompleteTime >= 300_000 {
		t.Fatalf("fixture: the waiter must finish prefill before the drain and be dropped while waiting (dropped %d)", cs.droppedAtDecodeKV)
	}
	// Dropped while it still waited, never admitted: no pull ran, so the drop is at the tick the
	// drained engine next looked at its waiters, with no transfer between start and end. A
	// drained engine that admitted it anyway would start a pull and drop it only on landing.
	if w := ps["waiter"]; w.TransferStartTime != w.CompletionTime || w.TransferCompleteTime != 0 {
		t.Errorf("the waiter was dropped at %d after a pull from %d to %d; a drained decode instance must drop its waiters without admitting them",
			w.CompletionTime, w.TransferStartTime, w.TransferCompleteTime)
	}
	// The decode engine had the request, so the lease, renewed until the drop, holds the
	// waiter's prefill KV past it: still held one tick after the drop.
	if after := run(ps["waiter"].CompletionTime + 1); usedBlocks(after.Instances()[0]) != 31 {
		t.Errorf("one tick after the waiter was dropped its 31 prefill blocks must still be held (lease), used %d",
			usedBlocks(after.Instances()[0]))
	}
	if l := newClusterLedger(cs); l.accounted() != 2 {
		t.Fatalf("INV-1: accounted %d of 2 [%s]", l.accounted(), l)
	}
	if cs.transfersInitiated != cs.transfersCompleted || cs.PrefillKVHeld() != 0 || cs.DecodeKVWaiting() != 0 {
		t.Fatalf("initiated %d completed %d, held %d, waiting %d", cs.transfersInitiated, cs.transfersCompleted,
			cs.PrefillKVHeld(), cs.DecodeKVWaiting())
	}
}

// The decode engine's queue is FCFS with head-of-line blocking, as vLLM's waiting queue stops at
// the first request it cannot allocate: a small handoff that would fit does not start ahead of
// an earlier, larger one that does not.
func TestPDDecodeKVWait_HeadOfLineBlocks(t *testing.T) {
	config := holdConfig(false, 1000)
	blocks := int64(40)
	config.DecodeOverrides.TotalKVBlocks = &blocks
	long := holdTestRequest("long", 1, 0, 160, 400)      // 10 blocks, grows to 35
	big := holdTestRequest("big", 2, 1000, 496, 4)       // 31 blocks: waits for long
	small := holdTestRequest("small", 3, 200_000, 32, 4) // 2 blocks: would fit beside long
	ps := parentsByID(runHold(t, config, long, big, small))
	if ps["small"].PrefillCompleteTime >= ps["big"].TransferStartTime {
		t.Fatalf("fixture: small must finish prefill while big still waits")
	}
	if ps["small"].TransferStartTime < ps["big"].TransferStartTime {
		t.Errorf("small's pull started at %d, ahead of the earlier waiter big's at %d (FCFS head-of-line)",
			ps["small"].TransferStartTime, ps["big"].TransferStartTime)
	}
}

// The cluster starts exactly the handoffs the decode engine admits, and moves the KV the engine
// says each pull must move (sim.EngineCore.AdmitRemoteKV). While long's decode holds the decode
// pool, a and b queue behind it; when long finishes, both fit at once, so the engine admits
// both at that one schedule and both pulls start at the same tick, each priced for its whole
// prompt. A cluster that started fewer than the engine admitted would leave b for a later
// schedule; a dropped pulled count would price a shorter pull.
func TestPDDecodeKVWait_StartsEveryHandoffTheEngineAdmits(t *testing.T) {
	config := holdConfig(false, 1000)
	config.PDTransferTime = func(tokens int64, _, _ InstanceID) int64 { return 1 + tokens }
	blocks := int64(40)
	config.DecodeOverrides.TotalKVBlocks = &blocks
	long := holdTestRequest("long", 1, 0, 160, 400) // 10 blocks, grows to 35 while it decodes
	a := holdTestRequest("a", 2, 100_000, 96, 4)    // 6 blocks: does not fit beside long's 35
	b := holdTestRequest("b", 3, 100_000, 96, 4)
	ps := parentsByID(runHold(t, config, long, a, b))
	pa, pb := ps["a"], ps["b"]
	if pa.TransferStartTime <= pa.PrefillCompleteTime || pb.TransferStartTime <= pb.PrefillCompleteTime {
		t.Fatalf("fixture: a and b must both wait for decode KV (a %d after %d, b %d after %d)",
			pa.TransferStartTime, pa.PrefillCompleteTime, pb.TransferStartTime, pb.PrefillCompleteTime)
	}
	if pa.TransferStartTime != pb.TransferStartTime {
		t.Errorf("a's pull started at %d and b's at %d; both fit once long freed its blocks, so both start together",
			pa.TransferStartTime, pb.TransferStartTime)
	}
	for _, p := range []*ParentRequest{pa, pb} {
		if p.PulledKVBlocks != 6 || p.TransferAloneTicks != 1+6*16 {
			t.Errorf("%s pulled %d blocks priced at %d ticks, want its whole 6-block prompt at %d",
				p.ID, p.PulledKVBlocks, p.TransferAloneTicks, 1+6*16)
		}
	}
}
