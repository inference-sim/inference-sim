// Original to blis-vllm.

package contract

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

// tokenLatency prices a step by its tokens, so ranks with different batches take different times.
type tokenLatency struct{}

func (tokenLatency) StepTime(batch []*cengine.Request) int64 {
	var n int64
	for _, r := range batch {
		n += int64(r.NumNewTokens)
	}
	return 100 + n
}
func (tokenLatency) QueueingTime(*cengine.Request) int64 { return 0 }
func (tokenLatency) OutputTokenProcessingTime() int64    { return 0 }
func (tokenLatency) PostDecodeFixedOverhead() int64      { return 0 }

func newGroup(t *testing.T, dp int) *DPGroup {
	t.Helper()
	cfg := cengine.SimConfig{
		Horizon:       1 << 40,
		KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:   cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel:  tokenLatency{},
	}
	g, err := NewDPGroup("instance_0", dp, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// The group keeps the contract's laws, its balancer picking each request's rank.
func TestTheDPGroupKeepsTheContractsLaws(t *testing.T) {
	build := func() (engine.Engine, error) { return newGroup(t, 2), nil }
	tr := enginetest.TestEngine(t, build, []enginetest.Arrival{
		{At: 0, Req: req("a", 40, 6)}, {At: 0, Req: req("bb", 200, 3)}, {At: 2500, Req: req("ccc", 64, 10)},
	}, enginetest.Options{})
	if len(tr.Finished) != 3 {
		t.Errorf("finished %d of 3 requests", len(tr.Finished))
	}
}

// Every rank steps together: one step runs both ranks' requests and lasts the slowest rank's
// duration (vLLM's all-reduce over the MoE dispatch, sim.LockstepGroup). The balancer spreads
// requests over the ranks.
func TestADPGroupStepsItsRanksTogetherAtTheSlowestRanksPace(t *testing.T) {
	g := newGroup(t, 2)
	if err := g.Add(req("a", 40, 4), 0); err != nil {
		t.Fatal(err)
	}
	if err := g.Add(req("bb", 400, 4), 0); err != nil {
		t.Fatal(err)
	}
	ka, _ := g.RankOf("a")
	kb, _ := g.RankOf("bb")
	if ka == kb {
		t.Fatalf("the balancer put both requests on rank %d", ka)
	}
	b, err := g.Schedule(0)
	if err != nil || len(b.Reqs) != 2 {
		t.Fatalf("the step ran %d requests (%v), want both ranks' 2", len(b.Reqs), err)
	}
	if got, want := g.StepTime(b), int64(100+400); got != want {
		t.Errorf("step time %d, want the slowest rank's %d", got, want)
	}
	if _, err := g.Complete(b.ID, engine.Outcome{}, 500); err != nil {
		t.Fatal(err)
	}
	if at, ok := g.NextWake(0); !ok || at != 500 {
		t.Errorf("next wake %d, %v: want the next lockstep step at the step's end, 500", at, ok)
	}
}

// A wave runs on past its last request until the ranks' next all-reduce (DPSyncInterval), its
// steps then running dummy batches, and ends there.
func TestADPGroupsWaveRunsDummyStepsUntilItsNextSync(t *testing.T) {
	g := newGroup(t, 2)
	if err := g.AddToRank(1, req("a", 16, 2), 0); err != nil {
		t.Fatal(err)
	}
	now := int64(0)
	for i := 0; i < 64; i++ {
		at, ok := g.NextWake(now)
		if i > 0 && !ok {
			break
		}
		if i > 0 {
			now = at
		}
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if !b.Empty() {
			end := now + g.StepTime(b)
			if _, err := g.Complete(b.ID, engine.Outcome{}, end); err != nil {
				t.Fatal(err)
			}
		}
	}
	steps, dummy := g.LockstepSteps()
	if steps != cengine.DPSyncInterval || dummy == 0 {
		t.Errorf("the wave ran %d steps, %d of them dummy: want %d, ending at the sync after the last request",
			steps, dummy, cengine.DPSyncInterval)
	}
	if _, ok := g.NextWake(now); ok {
		t.Error("the wave did not end at its sync")
	}
	// The sync step that ended the wave still holds the GPUs until it ends (100 ticks, a dummy
	// batch): a request arriving during it starts the next wave at that step's end.
	if err := g.AddToRank(0, req("late", 16, 2), now+50); err != nil {
		t.Fatal(err)
	}
	if at, ok := g.NextWake(now + 50); !ok || at != now+100 {
		t.Errorf("a request arriving during the last step wakes the group at %d, %v: want that step's end %d", at, ok, now+100)
	}
}

// A rank's view runs the requests added through it on its rank, and rank 0's view steps the whole
// group: its batch carries the other ranks' requests, and the other views schedule nothing.
func TestARankViewAddsToItsRankAndRankZeroStepsTheGroup(t *testing.T) {
	g := newGroup(t, 2)
	v0, v1 := g.View(0), g.View(1)
	if err := v1.Add(req("a", 40, 3), 0); err != nil {
		t.Fatal(err)
	}
	if k, _ := g.RankOf("a"); k != 1 {
		t.Fatalf("a request added through rank 1's view runs on rank %d", k)
	}
	if b, _ := v1.Schedule(0); !b.Empty() {
		t.Error("rank 1's view scheduled a batch; only rank 0's steps the group")
	}
	if _, ok := v1.NextWake(0); ok {
		t.Error("rank 1's view reported a wake; the group's wakes are rank 0's view's")
	}
	b, err := v0.Schedule(0)
	if err != nil || len(b.Reqs) != 1 || b.Reqs[0].ID != "a" {
		t.Fatalf("rank 0's view ran %v (%v), want rank 1's request a", b.Reqs, err)
	}
	if s := v1.Snapshot(); s.Running != 1 {
		t.Errorf("rank 1's snapshot runs %d requests, want 1", s.Running)
	}
}
