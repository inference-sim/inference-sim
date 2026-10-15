package cluster

import (
	"container/heap"
	"math"
	"reflect"
	"testing"

	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/sim"
)

// INV-PD-7 (#1934 point 3): a NIXL prefill engine keeps a finished request's KV blocks until
// the decode side has pulled them (nixl/pull_scheduler.py:247-266 at v0.31.0). These tests pin
// when the hold begins and ends, that it costs prefill KV capacity, and that it never leaks.

// holdTestRequest is a request whose prompt is unique to id, so two of them share no prefix
// block and every block they allocate is their own.
func holdTestRequest(id string, seed int, arrival int64, inputLen, outputLen int) *sim.Request {
	in := make([]sim.TokenID, inputLen)
	for i := range in {
		in[i] = sim.TokenID(seed*100_003 + i + 1)
	}
	return &sim.Request{
		ID: id, InputTokens: in, OutputTokens: make([]sim.TokenID, outputLen),
		MaxOutputLen: outputLen, ArrivalTime: arrival, State: sim.StateQueued,
	}
}

// holdConfig is a 1P1D cluster whose every handoff takes transfer ticks.
func holdConfig(hold bool, transfer int64) DeploymentConfig {
	cfg := newTestDisaggDeploymentConfig(2, 1, 1)
	cfg.PDPrefillKVHold = hold
	cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return transfer }
	return cfg
}

func usedBlocks(inst *InstanceSimulator) int64 { return inst.TotalKVBlocks() - inst.FreeKVBlocks() }

func runHold(t interface {
	Helper()
	Fatalf(string, ...any)
}, cfg DeploymentConfig, reqs ...*sim.Request) *ClusterSimulator {
	t.Helper()
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	return cs
}

// Between the prefill finishing and the pull completing, the prefill instance still holds the
// request's ten blocks; without the hold it holds none. Once the pull is done, nothing is held.
func TestPrefillKVHold_HeldFromPrefillEndUntilThePullCompletes(t *testing.T) {
	const transfer = 1_000_000
	full := runHold(t, holdConfig(true, transfer), holdTestRequest("r", 1, 0, 160, 4))
	p := full.ParentRequests()[0]
	if p.TransferCompleteTime-p.TransferStartTime != transfer {
		t.Fatalf("transfer took %d ticks, want %d", p.TransferCompleteTime-p.TransferStartTime, transfer)
	}
	if got := full.PrefillKVHoldPeakTokens(); got != 160 {
		t.Errorf("peak held = %d tokens, want the request's 160", got)
	}
	if got := usedBlocks(full.Instances()[0]); got != 0 || full.PrefillKVHeld() != 0 {
		t.Errorf("after the pull the prefill instance uses %d blocks and holds %d requests, want 0 and 0",
			got, full.PrefillKVHeld())
	}

	mid := p.TransferStartTime + transfer/2
	for _, tc := range []struct {
		hold bool
		want int64
	}{{true, 10}, {false, 0}} {
		cfg := holdConfig(tc.hold, transfer)
		cfg.Horizon = mid
		cs := runHold(t, cfg, holdTestRequest("r", 1, 0, 160, 4))
		if got := usedBlocks(cs.Instances()[0]); got != tc.want {
			t.Errorf("hold=%v: mid-pull the prefill instance uses %d blocks, want %d", tc.hold, got, tc.want)
		}
		if tc.hold && cs.HeldPrefillKVTokens(string(cs.Instances()[0].ID())) != 160 {
			t.Errorf("hold=%v: mid-pull %d tokens are held, want 160", tc.hold,
				cs.HeldPrefillKVTokens(string(cs.Instances()[0].ID())))
		}
	}
}

// The hold costs prefill capacity: with room for one request's blocks and a slow pull, a
// second prefill cannot finish until the first pull has completed and freed them. Without the
// hold the second prefill finishes long before.
func TestPrefillKVHold_HeldBlocksGateTheNextPrefill(t *testing.T) {
	const transfer = 1_000_000
	for _, hold := range []bool{true, false} {
		cfg := holdConfig(hold, transfer)
		blocks := int64(15) // one 10-block request fits, two do not
		cfg.PrefillOverrides.TotalKVBlocks = &blocks
		cs := runHold(t, cfg, holdTestRequest("a", 1, 0, 160, 4), holdTestRequest("b", 2, 1, 160, 4))
		byID := map[string]*ParentRequest{}
		for _, p := range cs.ParentRequests() {
			byID[p.ID] = p
		}
		a, b := byID["a"], byID["b"]
		if a == nil || b == nil || a.TransferCompleteTime == 0 || b.TransferCompleteTime == 0 {
			t.Fatalf("hold=%v: both requests must complete their handoff", hold)
		}
		gated := b.PrefillCompleteTime >= a.TransferCompleteTime
		if gated != hold {
			t.Errorf("hold=%v: b's prefill ended at %d, a's pull at %d; gated=%v, want %v",
				hold, b.PrefillCompleteTime, a.TransferCompleteTime, gated, hold)
		}
	}
}

// A decode side that cannot take the request aborts it before pulling, which notifies the
// prefill side to free at once (pull_scheduler.py:219-232): nothing waits for the lease.
func TestPrefillKVHold_DropAtTransferStartFreesAtOnce(t *testing.T) {
	cfg := holdConfig(true, 1000)
	tiny := int64(2)
	cfg.DecodeOverrides.TotalKVBlocks = &tiny
	full := runHold(t, cfg, holdTestRequest("r", 1, 0, 160, 4))
	p := full.ParentRequests()[0]
	if p.DecodeSubReq != nil || p.TransferStartTime == 0 {
		t.Fatalf("the decode reservation must fail at transfer start (DecodeSubReq=%v)", p.DecodeSubReq)
	}
	cfg.Horizon = p.TransferStartTime + 1
	cs := runHold(t, cfg, holdTestRequest("r", 1, 0, 160, 4))
	if got := usedBlocks(cs.Instances()[0]); got != 0 || cs.PrefillKVHeld() != 0 {
		t.Errorf("one tick after the drop the prefill instance uses %d blocks and holds %d, want 0 and 0",
			got, cs.PrefillKVHeld())
	}
}

// drainAt is a test event that starts draining an instance.
type drainAt struct {
	time int64
	inst InstanceID
}

func (e *drainAt) Timestamp() int64 { return e.time }
func (e *drainAt) Priority() int    { return 5 }
func (e *drainAt) Execute(cs *ClusterSimulator) {
	cs.instanceByID(string(e.inst)).TransitionTo(sim.InstanceStateDraining)
}

// A decode side that vanishes mid-pull sends neither a completion notice nor more heartbeats,
// so the prefill side frees only at lease expiry: the later of prefill end + lease and the
// vanish time + 2/3 lease (nixl/base_worker.py:563-567, 3056-3096).
func TestPrefillKVHold_VanishedDecodeFreesAtLeaseExpiry(t *testing.T) {
	const lease = 3_000_000
	// A short pull vanishes before the decode side's heartbeats outlast the first deadline; a
	// long one after, so each case pins one term of the expiry.
	for _, transfer := range []int64{600_000, 3_000_000} {
		run := func(horizon int64) *ClusterSimulator {
			cfg := holdConfig(true, transfer)
			cfg.PDKVLeaseUs = lease
			cfg.Horizon = horizon
			cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{holdTestRequest("r", 1, 0, 160, 4)}), nil)
			heap.Push(&cs.clusterEvents, clusterEventEntry{event: &drainAt{time: 500_000, inst: cs.Instances()[1].ID()}, seqID: cs.nextSeqID()})
			mustRun(t, cs)
			return cs
		}
		full := run(math.MaxInt64)
		p := full.ParentRequests()[0]
		if p.TransferCompleteTime == 0 || p.DecodeSubReq != nil || p.TransferStartTime >= 500_000 {
			t.Fatalf("transfer %d: the decode instance must vanish mid-pull (late drop); got start=%d complete=%d decode=%v",
				transfer, p.TransferStartTime, p.TransferCompleteTime, p.DecodeSubReq)
		}
		expiry := max(p.PrefillCompleteTime+lease, p.TransferCompleteTime+lease*2/3)
		if before := run(expiry - 1); usedBlocks(before.Instances()[0]) != 10 {
			t.Errorf("transfer %d: one tick before lease expiry %d the prefill instance uses %d blocks, want the 10 held",
				transfer, expiry, usedBlocks(before.Instances()[0]))
		}
		if after := run(expiry); usedBlocks(after.Instances()[0]) != 0 || after.PrefillKVHeld() != 0 {
			t.Errorf("transfer %d: at lease expiry the prefill instance uses %d blocks, want 0", transfer, usedBlocks(after.Instances()[0]))
		}
	}
}

// Metamorphic: when the prefill pool's KV never runs short, holding blocks changes nothing a
// client sees. The hold only removes capacity; it does not move a step or a prefix hit, since a
// released block keeps its hash on the free list just as a held one keeps it in use.
func TestPrefillKVHold_AmpleCapacityChangesNoLatency(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 10).Draw(rt, "n")
		transfer := rapid.Int64Range(1, 200_000).Draw(rt, "transfer")
		reqs := func() []*sim.Request {
			out := make([]*sim.Request, n)
			for i := range out {
				// A shared head makes some prefix hits non-trivial.
				r := holdTestRequest(string(rune('a'+i)), i, int64(i)*rapid.Int64Range(0, 50_000).Draw(rt, "gap"), 64+16*(i%4), 1+i%5)
				copy(r.InputTokens[:32], make([]sim.TokenID, 32))
				out[i] = r
			}
			return out
		}()
		clone := func() []*sim.Request {
			out := make([]*sim.Request, n)
			for i, r := range reqs {
				c := *r
				out[i] = &c
			}
			return out
		}
		on := runHold(rt, holdConfig(true, transfer), clone()...).AggregatedMetrics()
		off := runHold(rt, holdConfig(false, transfer), clone()...).AggregatedMetrics()
		for _, m := range []struct {
			name   string
			on, of any
		}{{"TTFT", on.RequestTTFTs, off.RequestTTFTs}, {"E2E", on.RequestE2Es, off.RequestE2Es},
			{"ITL", on.AllITLs, off.AllITLs}, {"completed", on.CompletedRequests, off.CompletedRequests}} {
			if !reflect.DeepEqual(m.on, m.of) {
				rt.Fatalf("%s differs with ample capacity:\nhold %v\nfree %v", m.name, m.on, m.of)
			}
		}
	})
}

// Property: under any load and any prefill capacity, the held KV is a subset of what the
// prefill instance has allocated (equal to it once nothing else is queued or running there),
// never exceeds its capacity, and is all released by the end of an unbounded run, leaving
// every request conserved (INV-1) and no prefill block in use (INV-4).
func TestPrefillKVHold_NeverLeaksAndNeverExceedsCapacity(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 12).Draw(rt, "n")
		cfg := holdConfig(true, rapid.Int64Range(1, 400_000).Draw(rt, "transfer"))
		blocks := rapid.Int64Range(13, 60).Draw(rt, "prefillBlocks")
		cfg.PrefillOverrides.TotalKVBlocks = &blocks
		reqs := make([]*sim.Request, n)
		var at int64
		for i := range reqs {
			at += rapid.Int64Range(0, 100_000).Draw(rt, "gap")
			reqs[i] = holdTestRequest(string(rune('a'+i)), i%3, at, rapid.IntRange(1, 200).Draw(rt, "in"), rapid.IntRange(1, 6).Draw(rt, "out"))
		}
		clone := func() []*sim.Request {
			out := make([]*sim.Request, n)
			for i, r := range reqs {
				c := *r
				out[i] = &c
			}
			return out
		}
		capTokens := blocks * cfg.BlockSizeTokens

		cut := cfg
		cut.Horizon = rapid.Int64Range(1, at+2_000_000).Draw(rt, "horizon")
		cs := runHold(rt, cut, clone()...)
		pre := cs.Instances()[0]
		held := cs.HeldPrefillKVTokens(string(pre.ID()))
		used := usedBlocks(pre) * cfg.BlockSizeTokens
		if held > used || used > capTokens {
			rt.Fatalf("at horizon %d: held %d tokens, allocated %d, capacity %d", cut.Horizon, held, used, capTokens)
		}
		if pre.QueueDepth() == 0 && pre.BatchSize() == 0 && held != used {
			rt.Fatalf("idle prefill instance allocates %d tokens but holds %d for pulls", used, held)
		}

		full := runHold(rt, cfg, clone()...)
		if full.PrefillKVHeld() != 0 || usedBlocks(full.Instances()[0]) != 0 {
			rt.Fatalf("after the run %d holds remain and the prefill instance uses %d blocks",
				full.PrefillKVHeld(), usedBlocks(full.Instances()[0]))
		}
		if peak := full.PrefillKVHoldPeakTokens(); peak > capTokens || peak <= 0 {
			rt.Fatalf("peak held %d tokens, capacity %d", peak, capTokens)
		}
		if l := newClusterLedger(full); l.accounted() != n {
			rt.Fatalf("INV-1: injected %d != accounted %d [%s]", n, l.accounted(), l)
		}
	})
}
