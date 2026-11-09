package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// A session's next round arrives think_time after the client has the previous round's last
// chunk, on the colocated path and the P/D path alike (SIM-8, #1913). The client sees a round
// complete only after the post-decode overhead, which E2E includes; releasing the follow-up at
// the engine's last step made every round arrive PostDecodeFixedOverhead early.
func TestSessionRelease_NextRoundFollowsClientVisibleCompletion(t *testing.T) {
	const overhead, think = int64(5000), int64(1000)
	for _, tc := range []struct {
		name string
		cfg  func() DeploymentConfig
		done func(cs *ClusterSimulator, id string) (int64, bool)
	}{
		{"colocated", func() DeploymentConfig {
			cfg := newTestDeploymentConfig(1)
			coeffs := testutil.DefaultFakeLatency()
			coeffs.PostDecodeOverheadTicks = overhead
			editEngine(&cfg, func(c *simConfig) { c.LatencyModel = fakeLatencyWithCoeffs(coeffs) })
			return cfg
		}, func(cs *ClusterSimulator, id string) (int64, bool) {
			t, ok := cs.AggregatedMetrics().RequestCompletionTimes[id]
			return int64(t), ok
		}},
		{"pd", func() DeploymentConfig { return newTestDisaggDeploymentConfigWithOverhead(float64(overhead)) },
			func(cs *ClusterSimulator, id string) (int64, bool) {
				p := cs.parentRequests[id]
				return p.CompletionTime, p != nil && p.CompletionTime > 0
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sampler := func(v float64) workload.LengthSampler {
				s, err := workload.NewLengthSampler(workload.DistSpec{Type: "constant", Params: map[string]float64{"value": v}})
				if err != nil {
					t.Fatal(err)
				}
				return s
			}
			const sess = "rel_sess"
			bp := workload.SessionBlueprint{SessionID: sess, ClientID: "c", MaxRounds: 4, ContextGrowth: "accumulate",
				ThinkTimeUs: think, Horizon: math.MaxInt64, InputSampler: sampler(20), OutputSampler: sampler(8),
				Streams: workload.SessionStreams(1913, sess), Model: "test-model"}
			seed := &sim.Request{ID: sess + "_r0", InputTokens: make([]sim.TokenID, 20), OutputTokens: make([]sim.TokenID, 8),
				MaxOutputLen: 8, State: sim.StateQueued, SessionID: sess}
			sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})
			arrival := map[string]int64{}
			var order []string
			onDone := func(req *sim.Request, tick int64) []*sim.Request {
				if req.SessionID == sess {
					arrival[req.ID] = req.ArrivalTime
					order = append(order, req.ID)
				}
				next := sm.OnComplete(req, tick)
				for _, n := range next {
					arrival[n.ID] = n.ArrivalTime
				}
				return next
			}
			cs := newTestCluster(tc.cfg(), NewSliceRequestSource([]*sim.Request{seed}), onDone)
			mustRun(t, cs)
			if len(order) != 4 {
				t.Fatalf("%d rounds completed, want 4", len(order))
			}
			for i := 1; i < len(order); i++ {
				seen, ok := tc.done(cs, order[i-1])
				if !ok {
					t.Fatalf("round %s has no client-visible completion", order[i-1])
				}
				if got := arrival[order[i]] - seen; got != think {
					t.Errorf("round %d arrives %d µs after round %d's last chunk, want think time %d (post-decode overhead %d)",
						i, got, i-1, think, overhead)
				}
			}
		})
	}
}
