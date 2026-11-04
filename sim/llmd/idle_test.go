package llmd

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestFastForwardIdle_IsExact runs workloads with long idle gaps between bursts twice, once with
// the idle fast-forward and once ticking every router loop through every gap, and requires the
// same routing, dispatch times and latencies.
func TestFastForwardIdle_IsExact(t *testing.T) {
	burstsWithGaps := func() []*sim.Request {
		var reqs []*sim.Request
		for b, start := range []int64{0, 30_000_000, 95_000_000} { // bursts 30 s and 65 s apart
			for _, r := range testRequests(40, int64(3+b), 400, 32) {
				r.ID = fmt.Sprintf("%s_b%d", r.ID, b)
				r.ArrivalTime += start
				reqs = append(reqs, r)
			}
		}
		return reqs
	}
	for name, cfg := range map[string]string{"default": "", "flow-control": globalStrictConfig} {
		t.Run(name, func(t *testing.T) {
			run := func(neverSkip bool) (map[string]string, map[string]float64, map[string]float64) {
				reqs := burstsWithGaps()
				dep := testDeployment(3)
				editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
				cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
				flags := DefaultEPPFlags()
				if cfg != "" {
					flags = append(flags, "--allow-experimental-plugins", "--config-text="+cfg)
				}
				gw := New(Options{Model: "test-model", EPPFlags: flags})
				gw.neverSkipIdle = neverSkip
				cs.SetGateway(gw)
				if err := cs.Run(); err != nil {
					t.Fatalf("Run: %v", err)
				}
				placed := map[string]string{}
				for _, r := range reqs {
					placed[r.ID] = fmt.Sprintf("%s@%d", r.AssignedInstance, r.GatewayDispatchTime)
				}
				m := cs.AggregatedMetrics()
				return placed, m.RequestTTFTs, m.RequestE2Es
			}
			p1, ttft1, e2e1 := run(false)
			p2, ttft2, e2e2 := run(true)
			if !reflect.DeepEqual(p1, p2) || !reflect.DeepEqual(ttft1, ttft2) || !reflect.DeepEqual(e2e1, e2e2) {
				t.Fatal("the idle fast-forward changed routing or latency")
			}
			if len(ttft1) != 120 {
				t.Fatalf("%d of 120 requests completed", len(ttft1))
			}
		})
	}
}

// TestFastForwardIdle_IsExactWithStatefulUsageLimit: soft-reflective-ceiling-policy keeps a tick
// counter that advances on dispatch cycles while some band is in its alternating regime. With two
// priority bands and saturating bursts separated by long idle gaps, the fast-forward must still
// give the routing, dispatch times and latencies of ticking through every gap: an idle pool is
// below the alternating regime, so the skipped cycles leave the counter unchanged.
func TestFastForwardIdle_IsExactWithStatefulUsageLimit(t *testing.T) {
	for name, extra := range map[string][]string{
		"fresh metrics": nil,
		// Metrics older than the utilization detector's 200 ms staleness count as saturated, so
		// with staggered scrapes an idle pool sits in the policy's alternating regime (review
		// 6200849987).
		"stale while idle": {"--refresh-metrics-interval=500ms"},
	} {
		t.Run(name, func(t *testing.T) { checkStatefulUsageLimitExact(t, extra) })
	}
}

func checkStatefulUsageLimitExact(t *testing.T, extraFlags []string) {
	run := func(neverSkip bool) (map[string]string, map[string]float64) {
		var reqs []*sim.Request
		for b, start := range []int64{0, 40_000_000, 130_000_000} {
			for i, r := range testRequests(300, int64(7+b), 1000, 32) { // 300 ms bursts, across several scrapes
				r.ID = fmt.Sprintf("%s_b%d", r.ID, b)
				r.ArrivalTime += start
				r.SLOClass = map[bool]string{true: "critical", false: "sheddable"}[i%2 == 0]
				reqs = append(reqs, r)
			}
		}
		dep := testDeployment(2)
		editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{
			Model:        "test-model",
			EPPFlags:     append([]string{"--pool-name=vllm", "--allow-experimental-plugins", "--config-text=" + softCeilingConfig}, extraFlags...),
			Manifests:    []byte(poolManifests),
			PodLabels:    func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} },
			ScrapeJitter: 400 * time.Millisecond,
		})
		gw.neverSkipIdle = neverSkip
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if gw.skipIdle {
			t.Fatal("the idle fast-forward stayed on with soft-reflective-ceiling-policy")
		}
		placed := map[string]string{}
		held := 0
		for _, r := range reqs {
			placed[r.ID] = fmt.Sprintf("%s@%d", r.AssignedInstance, r.GatewayDispatchTime)
			if r.GatewayDispatchTime > r.ArrivalTime {
				held++
			}
		}
		if held == 0 {
			t.Fatal("no request waited in flow control; the bursts do not exercise the usage limit")
		}
		return placed, cs.AggregatedMetrics().RequestTTFTs
	}
	p1, ttft1 := run(false)
	p2, ttft2 := run(true)
	if !reflect.DeepEqual(p1, p2) || !reflect.DeepEqual(ttft1, ttft2) {
		t.Fatal("the idle fast-forward changed routing or latency with a stateful usage-limit policy and two bands")
	}
}

// TestFastForwardIdle_DeadTimeoutsDoNotBlockIt (#2072): every request carries a client deadline
// far past the run, so each completed request leaves a dead timeout in its engine's queue. The
// idle fast-forward must still skip the gaps between bursts, and give exactly the routing,
// dispatch times, metrics and router event streams (KV events, dispatches, usage) of ticking
// through every gap. It runs colocated, with lockstep data-parallel ranks, disaggregated, and
// with AgentX-style sessions whose follow-up rounds arrive after a think time.
func TestFastForwardIdle_DeadTimeoutsDoNotBlockIt(t *testing.T) {
	type outcome struct {
		placed  map[string]string
		metrics sim.Metrics
		events  []string // KV events, dispatches and usage, in the router's order
		scrapes int
	}
	burstsWithDeadlines := func() []*sim.Request {
		var reqs []*sim.Request
		for b, start := range []int64{0, 30_000_000, 95_000_000} { // bursts 30 s and 65 s apart
			for i, r := range testRequests(40, int64(3+b), 400, 32) {
				r.ID = fmt.Sprintf("%s_b%d", r.ID, b)
				r.ArrivalTime += start
				r.Deadline = r.ArrivalTime + 200_000_000 // far past the run, as aiperf's 6 h client timeout is
				if b == 2 && i%10 == 0 {
					r.Deadline = r.ArrivalTime + 2_000 // a few live timeouts that must still fire
				}
				reqs = append(reqs, r)
			}
		}
		return reqs
	}
	sessions := false
	for name, setup := range map[string]func(*cluster.DeploymentConfig, *Options){
		"colocated": func(*cluster.DeploymentConfig, *Options) {},
		"sessions":  func(*cluster.DeploymentConfig, *Options) { sessions = true },
		"dp-lockstep": func(d *cluster.DeploymentConfig, _ *Options) {
			*d = testDeployment(4)
			d.DPLockstepRanks = 2
		},
		"disaggregated": func(d *cluster.DeploymentConfig, o *Options) {
			*d = testDeployment(4)
			d.PrefillInstances, d.DecodeInstances = 2, 2
			d.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
			o.PodLabels = roleLabels
			o.EPPFlags = append(o.EPPFlags, "--config-text="+pdConfig)
		},
	} {
		t.Run(name, func(t *testing.T) {
			run := func(neverSkip bool) outcome {
				reqs := burstsWithDeadlines()
				dep := testDeployment(3)
				opts := Options{Model: "test-model", EPPFlags: DefaultEPPFlags()}
				sessions = false
				setup(&dep, &opts)
				editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
				var onDone func(*sim.Request, int64) []*sim.Request
				if sessions {
					// Each completed first round sends one follow-up after a 7 s think time, with
					// the whole context as its prompt and the same far-off deadline.
					onDone = func(r *sim.Request, now int64) []*sim.Request {
						if r.State != sim.StateCompleted || strings.HasSuffix(r.ID, "_r2") {
							return nil
						}
						next := &sim.Request{ID: r.ID + "_r2", ArrivalTime: now + 7_000_000,
							InputTokens:  append(append([]sim.TokenID{}, r.InputTokens...), r.OutputTokens...),
							OutputTokens: r.OutputTokens[:16], State: sim.StateQueued}
						next.Deadline = next.ArrivalTime + 200_000_000
						reqs = append(reqs, next)
						return []*sim.Request{next}
					}
				}
				cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), onDone)
				gw := New(opts)
				gw.neverSkipIdle = neverSkip
				var o outcome
				gw.observe = func(kind, detail string) {
					switch kind {
					case "scrape":
						o.scrapes++
					case "kvdeliver", "kvevent", "dispatch", "usage":
						o.events = append(o.events, kind+" "+detail)
					}
				}
				cs.SetGateway(gw)
				if err := cs.Run(); err != nil {
					t.Fatalf("Run: %v", err)
				}
				o.placed = map[string]string{}
				for _, r := range reqs {
					o.placed[r.ID] = fmt.Sprintf("%s@%d", r.AssignedInstance, r.GatewayDispatchTime)
				}
				o.metrics = *cs.AggregatedMetrics()
				return o
			}
			ff, tick := run(false), run(true)
			if !reflect.DeepEqual(ff.placed, tick.placed) || !reflect.DeepEqual(ff.events, tick.events) {
				t.Fatal("the idle fast-forward changed routing, dispatch times or the router's event stream")
			}
			if !reflect.DeepEqual(ff.metrics, tick.metrics) {
				t.Fatal("the idle fast-forward changed the run's metrics")
			}
			want := len(ff.placed) // 120 first rounds, plus the follow-ups of a session run
			if sessions && want <= 120 {
				t.Fatal("no session sent a follow-up round")
			}
			if done, timed := ff.metrics.CompletedRequests, ff.metrics.TimedOutRequests; done+timed != want || timed == 0 {
				t.Fatalf("completed %d, timed out %d of %d; want all accounted and some live timeouts", done, timed, want)
			}
			// Ticking scrapes every 50 ms through about 95 s of gaps; the fast-forward skips them.
			if ff.scrapes*4 > tick.scrapes {
				t.Fatalf("scrapes: %d with the fast-forward, %d ticking; dead timeouts still block it", ff.scrapes, tick.scrapes)
			}
			t.Logf("scrapes: %d with the fast-forward, %d ticking", ff.scrapes, tick.scrapes)
		})
	}
}
