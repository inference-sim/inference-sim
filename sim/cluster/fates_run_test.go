package cluster

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
)

// fixedRequests are n requests of in prompt and out output tokens, gapUs apart.
func fixedRequests(n, in, out int, gapUs int64) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		r := &sim.Request{ID: fmt.Sprintf("request_%d", i), InputTokens: make([]sim.TokenID, in),
			OutputTokens: make([]sim.TokenID, out), State: sim.StateQueued, ArrivalTime: int64(i) * gapUs}
		for j := range r.InputTokens {
			r.InputTokens[j] = sim.TokenID(i*10_000 + j + 1)
		}
		reqs[i] = r
	}
	return reqs
}

// runFates runs a cluster and returns it with the fate tally of its ledger.
func runFates(t *testing.T, cfg DeploymentConfig, reqs []*sim.Request, gw Gateway) (*ClusterSimulator, sim.Tally) {
	t.Helper()
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	if gw != nil {
		cs.SetGateway(gw)
	}
	mustRun(t, cs)
	return cs, cs.Fates().Tally()
}

// DEV-9's known-bad scenario for each fate (X18): a run built so that its requests end in that
// fate, with the ledger's arrival-count check and the INV-1 agreement passing (Run returns no
// error) and the fate's count equal to what the fixture forces.
func TestFates_KnownBadScenarioPerFate(t *testing.T) {
	type tc struct {
		fate  sim.Fate
		drop  string
		build func(t *testing.T) (DeploymentConfig, []*sim.Request, Gateway)
		want  int
	}
	cases := []tc{
		{sim.FateCompleted, "", func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			return newTestDeploymentConfig(2), fixedRequests(6, 32, 4, 1000), nil
		}, 6},
		{sim.FateDropped, sim.DropAdmission, func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			return newTestDeploymentConfig(1), fixedRequests(3, 32, 4, 1000),
				&fakeGateway{reject: func(*sim.Request) bool { return true }}
		}, 3},
		{sim.FateDropped, sim.DropRouting, func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			return newTestDeploymentConfig(1), fixedRequests(2, 32, 4, 1000),
				&fakeGateway{reject: func(*sim.Request) bool { return true }, rejectAs: RejectedRouting}
		}, 2},
		{sim.FateDropped, sim.DropUnservable, func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			// A prompt larger than the whole KV cache: the engine refuses it at Add.
			cfg := newTestDeploymentConfig(1)
			editEngine(&cfg, func(c *simConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(4, 16, 0, 0, 0, 0) })
			return cfg, fixedRequests(2, 200, 4, 1000), nil
		}, 2},
		{sim.FateDropped, sim.DropGatewayEvicted, func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			// A disaggregated request evicted while it decodes (the fake gateway evicts on P/D).
			return newTestDisaggDeploymentConfig(4, 2, 2), fixedRequests(2, 64, 20_000, 1000),
				&fakeGateway{disaggregate: true, evictAfterUs: 100_000}
		}, 2},
		{sim.FateTimedOut, "", func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			reqs := fixedRequests(3, 64, 20_000, 1000)
			for _, r := range reqs {
				r.Deadline = r.ArrivalTime + 100_000
			}
			return newTestDeploymentConfig(1), reqs, nil
		}, 3},
		{sim.FateInFlightAtHorizon, "", func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			// Long decodes, cut by a horizon after their first steps.
			cfg := newTestDeploymentConfig(1)
			cfg.Horizon = 20_000
			editEngine(&cfg, func(c *simConfig) { c.Horizon = cfg.Horizon })
			return cfg, fixedRequests(2, 16, 5000, 10), nil
		}, 2},
		{sim.FateQueuedAtHorizon, "", func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			// The gateway holds every request past the horizon.
			cfg := newTestDeploymentConfig(1)
			cfg.Horizon = 20_000
			editEngine(&cfg, func(c *simConfig) { c.Horizon = cfg.Horizon })
			return cfg, fixedRequests(3, 16, 4, 10), &fakeGateway{holdUs: 1_000_000}
		}, 3},
		{sim.FateNotStarted, "", func(*testing.T) (DeploymentConfig, []*sim.Request, Gateway) {
			// Two arrivals after the horizon (#1917: they must not vanish from the ledger).
			cfg := newTestDeploymentConfig(1)
			cfg.Horizon = 1_000_000
			editEngine(&cfg, func(c *simConfig) { c.Horizon = cfg.Horizon })
			reqs := fixedRequests(4, 16, 4, 600_000) // arrivals at 0, 0.6 s, 1.2 s, 1.8 s
			return cfg, reqs, nil
		}, 2},
	}
	for _, c := range cases {
		name := string(c.fate)
		if c.drop != "" {
			name += "/" + c.drop
		}
		t.Run(name, func(t *testing.T) {
			cfg, reqs, gw := c.build(t)
			cs, tally := runFates(t, cfg, reqs, gw)
			got := tally.ByFate[c.fate]
			if c.drop != "" {
				got = tally.Drops[c.drop]
			}
			if got != c.want {
				t.Errorf("%d requests ended %s, want %d (tally %+v)", got, name, c.want, tally)
			}
			if tally.Created != len(reqs) || tally.Total() != len(reqs) {
				t.Errorf("created %d, fates %d, want %d of each", tally.Created, tally.Total(), len(reqs))
			}
			for _, r := range reqs {
				if _, ok := cs.Fates().Fate(r.ID); !ok {
					t.Errorf("request %s has no fate", r.ID)
				}
			}
		})
	}
}

// Every fate of §8.7 has a known-bad scenario above.
func TestFates_EveryFateHasAKnownBadScenario(t *testing.T) {
	covered := map[sim.Fate]bool{sim.FateCompleted: true, sim.FateDropped: true, sim.FateTimedOut: true,
		sim.FateInFlightAtHorizon: true, sim.FateQueuedAtHorizon: true, sim.FateNotStarted: true}
	for _, f := range sim.Fates() {
		if !covered[f] {
			t.Errorf("fate %s has no known-bad scenario in TestFates_KnownBadScenarioPerFate", f)
		}
	}
}

// DEV-9's arrival-count check is independent of INV-1's buckets: a request written into a
// bucket by the old counters and never given a fate fails the run (the known-bad input of the
// check), naming the request.
func TestFates_ArrivalCountCheckFailsOnAMissingFate(t *testing.T) {
	cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(fixedRequests(2, 16, 4, 10)), nil)
	mustRun(t, cs)
	// A request created and never ended, as a code path that forgot to write its fate leaves it.
	if err := cs.fates.Create("lost"); err != nil {
		t.Fatal(err)
	}
	err := cs.checkFates()
	if err == nil || !strings.Contains(err.Error(), "lost") {
		t.Fatalf("checkFates: %v, want a conservation error naming the request without a fate", err)
	}
}

// The fates must agree with INV-1's buckets, which other code counts at other sites: a fate
// written into the wrong bucket fails the run.
func TestFates_DisagreementWithINV1BucketsFailsTheRun(t *testing.T) {
	cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(fixedRequests(2, 16, 4, 10)), nil)
	mustRun(t, cs)
	cs.aggregatedMetrics.CompletedRequests-- // one completion lost from the bucket
	if err := cs.checkFates(); err == nil || !strings.Contains(err.Error(), "completed") {
		t.Fatalf("checkFates: %v, want a disagreement on completed", err)
	}
}

// exactFitHandoff reports a P/D prompt of in tokens that fills a decode pool of blocks 16-token
// blocks exactly. The cluster's NIXLDecodeAccepts takes it, but the engine reserves a block
// for the first decode step of a block-aligned prompt (FirstDecodeStepBlocks), so the pull is
// never admitted.
func exactFitHandoff(in, blocks int) bool { return in%16 == 0 && in/16 == blocks }

// neverFinishesOnDecodePool reports a P/D request the cluster hands to a decode pool of blocks
// 16-token blocks that the pool can never finish: an exact-fit prompt, or a prompt that fits
// with a prompt plus output that does not (with one block for the next decode step). A prompt
// larger than the pool is not one: the cluster drops it at decode KV.
func neverFinishesOnDecodePool(in, out, blocks int) bool {
	if (in+15)/16 > blocks {
		return false
	}
	return exactFitHandoff(in, blocks) || (in+out+15)/16+1 > blocks
}

// SIM-7: a P/D handoff whose prompt fills the decode pool exactly is accepted and never
// admitted; with no other event to move it, the run used to end early and report the request
// nowhere. It now fails with the watchdog's error, naming the request. One token less fits
// and completes (SIM-9's pair).
func TestWatchdog_ExactFitHandoffFailsTheRun(t *testing.T) {
	run := func(in int) error {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.Horizon = 5_000_000
		editEngine(&cfg, func(c *simConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(13, 16, 0, 0, 0, 0) })
		editPrefill(&cfg, func(c *simConfig) { c.TotalKVBlocks = 100 })
		return newTestCluster(cfg, NewSliceRequestSource(fixedRequests(1, in, 1, 0)), nil).Run()
	}
	if !exactFitHandoff(208, 13) {
		t.Fatal("premise: 208 tokens fill 13 blocks of 16")
	}
	err := run(208)
	if !errors.Is(err, sim.ErrNoProgress) || !strings.Contains(err.Error(), "request_0") {
		t.Fatalf("exact-fit handoff: Run = %v, want ErrNoProgress naming request_0", err)
	}
	if err := run(207); err != nil {
		t.Fatalf("one token under the exact fit: Run = %v, want nil", err)
	}
}

// SIM-7: a P/D request whose prompt fits the decode pool and whose prompt plus output does not
// lands, decodes until the pool is full, is preempted, and is never admitted again. With no
// max_model_len a deployment can state this (a valid one cannot, INV-K10); the run used to end
// early and count the request still queued. It now fails with the watchdog's error. An output
// that fits completes (SIM-9's pair).
func TestWatchdog_DecodeThatOutgrowsThePoolFailsTheRun(t *testing.T) {
	run := func(out int) error {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.Horizon = 5_000_000
		editEngine(&cfg, func(c *simConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(10, 16, 0, 0, 0, 0) })
		editPrefill(&cfg, func(c *simConfig) { c.TotalKVBlocks = 100 })
		return newTestCluster(cfg, NewSliceRequestSource(fixedRequests(1, 136, out, 0)), nil).Run()
	}
	if !neverFinishesOnDecodePool(136, 26, 10) || neverFinishesOnDecodePool(136, 4, 10) {
		t.Fatal("premise: 136+26 tokens outgrow 10 blocks of 16, 136+4 do not")
	}
	if err := run(26); !errors.Is(err, sim.ErrNoProgress) || !strings.Contains(err.Error(), "request_0") {
		t.Fatalf("a decode that outgrows its pool: Run = %v, want ErrNoProgress naming request_0", err)
	}
	if err := run(4); err != nil {
		t.Fatalf("a decode that fits: Run = %v, want nil", err)
	}
}
