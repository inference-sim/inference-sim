package cluster

import (
	"fmt"
	"testing"

	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/sim"
)

// observedLatency prices steps with the default fake and calls check before each step, so a
// test can assert a property of the whole cluster at every step any instance takes.
type observedLatency struct {
	sim.LatencyModel
	check  func()
	priced func(ticks int64)
}

func (m observedLatency) StepTime(batch []*sim.Request) int64 {
	if m.check != nil {
		m.check()
	}
	d := m.LatencyModel.StepTime(batch)
	if m.priced != nil {
		m.priced(d)
	}
	return d
}

// saturatedPD runs n requests of in prompt and out output tokens, all arriving at once,
// through 1P1D whose decode pool holds decodeBlocks 16-token blocks. Before every step it
// checks INV-PD-9's state laws:
//   - no instance uses more blocks than it has;
//   - every handoff whose pull is in flight still holds its whole prompt on the decode side
//     (a transfer-holding request is never preempted).
func saturatedPD(t interface {
	Helper()
	Fatalf(string, ...any)
}, prefills, n, in, out int, decodeBlocks int64) *ClusterSimulator {
	t.Helper()
	cfg := newTestDisaggDeploymentConfig(prefills+1, prefills, 1)
	cfg.PDPrefillKVHold = true
	cfg.DecodeOverrides.TotalKVBlocks = &decodeBlocks
	var cs *ClusterSimulator
	var decodeSteps [][2]int64
	check := func() {
		for _, inst := range cs.instances {
			if u := usedBlocks(inst); u > inst.TotalKVBlocks() {
				t.Fatalf("%s uses %d blocks of %d", inst.ID(), u, inst.TotalKVBlocks())
			}
		}
		for _, p := range cs.parentRequests {
			if p.DecodeSubReq == nil || p.TransferStartTime == 0 || p.TransferCompleteTime != 0 {
				continue
			}
			want := (p.DecodeSubReq.InputLen() + 15) / 16
			for _, inst := range cs.instances {
				if inst.ID() == p.DecodeInstanceID {
					if got := int64(len(inst.KVBlocksOf(p.DecodeSubReq))); got != want {
						t.Fatalf("%s holds %d of its %d blocks mid-pull: a transfer-holding request lost its KV",
							p.DecodeSubReq.ID, got, want)
					}
				}
			}
		}
	}
	cfg.LatencyModel = observedLatency{LatencyModel: testFakeLatency(), check: check}
	cfg.DecodeOverrides.LatencyModel = observedLatency{LatencyModel: testFakeLatency(), check: check, priced: func(d int64) {
		for _, inst := range cs.instances {
			if cs.poolMembership[string(inst.ID())] == PoolRoleDecode {
				decodeSteps = append(decodeSteps, [2]int64{inst.sim.Clock, inst.sim.Clock + d})
			}
		}
	}}
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = &sim.Request{ID: fmt.Sprintf("r%02d", i), InputTokens: make([]sim.TokenID, in),
			OutputTokens: make([]sim.TokenID, out), State: sim.StateQueued}
		for k := range reqs[i].InputTokens {
			reqs[i].InputTokens[k] = sim.TokenID(i*100000 + k) // distinct prompts: no prefix sharing
		}
	}
	cs = NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	// A pull starts at a decode schedule (a step's start) or on an idle decode engine, never
	// while a decode step is executing.
	for _, p := range cs.parentRequests {
		for _, s := range decodeSteps {
			if at := p.TransferStartTime; p.DecodeSubReq != nil && s[0] < at && at < s[1] {
				t.Fatalf("%s's pull started at %d, inside the decode step [%d, %d)", p.ID, at, s[0], s[1])
			}
		}
	}
	return cs
}

func decodePreemptions(cs *ClusterSimulator) int64 {
	var n int64
	for _, inst := range cs.instances {
		if cs.poolMembership[string(inst.ID())] == PoolRoleDecode {
			n += inst.Metrics().PreemptionCount
		}
	}
	return n
}

func queuedHandoffs(cs *ClusterSimulator) int {
	n := 0
	for _, p := range cs.parentRequests {
		if p.TransferStartTime > p.PrefillCompleteTime {
			n++
		}
	}
	return n
}

// INV-PD-9 behavior: a saturated P/D decode pool queues its handoffs rather than thrashing.
// With many prefills feeding one small decode pool, most handoffs wait for decode KV, every
// request completes with none dropped, and decode preemptions stay bounded (at most one per
// two requests). Admitting pulls outside the decode schedule, or without vLLM's reservation,
// preempts many times more or deadlocks.
func TestPDDecodeAdmission_ASaturatedPoolQueuesRatherThanThrashes(t *testing.T) {
	for _, c := range []struct {
		prefills, n, in, out int
		blocks               int64
	}{
		{1, 40, 160, 40, 64}, {3, 120, 640, 64, 200},
	} {
		cs := saturatedPD(t, c.prefills, c.n, c.in, c.out, c.blocks)
		if got := cs.AggregatedMetrics().CompletedRequests; got != c.n || cs.droppedAtDecodeKV != 0 {
			t.Errorf("%+v: completed %d of %d, %d dropped at decode KV", c, got, c.n, cs.droppedAtDecodeKV)
		}
		if q := queuedHandoffs(cs); q < c.n/2 {
			t.Errorf("%+v: only %d handoffs waited for decode KV; the fixture must saturate the pool", c, q)
		}
		if p := decodePreemptions(cs); p > int64(c.n/2) {
			t.Errorf("%+v: %d decode preemptions for %d requests; a saturated pool must queue, not thrash", c, p, c.n)
		}
	}
}

// INV-PD-9 laws over random saturated 1P1D and 3P1D runs (checked at every step by
// saturatedPD): no instance exceeds its blocks, no transfer-holding request loses its KV, and
// the run completes every request (no admission deadlock).
func TestPDDecodeAdmission_LawsHoldUnderRandomSaturation(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		prefills := rapid.IntRange(1, 3).Draw(rt, "prefills")
		n := rapid.IntRange(4, 40).Draw(rt, "requests")
		in := rapid.IntRange(16, 400).Draw(rt, "in")
		out := rapid.IntRange(2, 60).Draw(rt, "out")
		need := int64((in+out+15)/16) + 2
		blocks := rapid.Int64Range(need, 4*need).Draw(rt, "decodeBlocks")
		cs := saturatedPD(rt, prefills, n, in, out, blocks)
		if got := cs.AggregatedMetrics().CompletedRequests; got != n {
			rt.Fatalf("completed %d of %d", got, n)
		}
	})
}
