package conformance

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	k8stypes "k8s.io/apimachinery/pkg/types"
	testclock "k8s.io/utils/clock/testing"
	"pgregory.net/rapid"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The flow controller's queueing differential: llm-d-router v0.10.0's own FlowController
// (registry, FCFS ordering, global-strict / round-robin fairness, static usage-limit
// ceiling, capacity checks, HoL-blocking dispatch cycle) and BLIS's gateway queue, built
// from the same flowControl block through BLIS's EPP ingestion, see one random sequence
// of arrivals and dispatch cycles at chosen pool saturations. Every arrival must be
// queued or rejected alike and every cycle must dispatch the same request, or none.
//
// llm-d's controller runs goroutines; the harness steps them deterministically: the
// processor's 1 ms dispatch ticker runs on a FakeClock, a cycle is observed through the
// saturation detector's call, and arrivals are held off dispatch (saturation above every
// ceiling) until the request is visibly queued or rejected. TTL expiry runs on real-time
// context deadlines in llm-d (controller.go createRequestContext) and is not driven here.

// fcBands are the request priorities the harness uses, with the BLIS SLO class whose
// default priority each is (sim.DefaultSLOPriorityMap).
var fcBands = []struct {
	priority int
	class    string
}{{4, "critical"}, {3, "standard"}, {-1, "batch"}}

var fcFlows = []string{"a", "b", "c"}

type fcDetector struct {
	sat   atomic.Value
	calls atomic.Int64
}

func (d *fcDetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "harness-detector", Name: "harness-detector"}
}

func (d *fcDetector) Saturation(context.Context, []fwkdl.Endpoint) float64 {
	v := d.sat.Load().(float64)
	d.calls.Add(1)
	return v
}

type fcBand struct {
	priority, maxRequests int
	maxBytes              int64
	roundRobin            bool
}

type fcCase struct {
	threshold                 float64
	maxRequests               int
	maxBytes                  int64
	defaultReq                int
	defaultBytes              int64
	bands                     []fcBand
	ops                       []fcOp
	arrivals                  int
	hasDefaultBand, hasGlobal bool
}

type fcOp struct {
	arrive     bool
	id, flow   string
	band       int // index into fcBands
	tokens     int
	saturation float64
}

func drawFlowControlCase(t *rapid.T) fcCase {
	c := fcCase{threshold: rapid.SampledFrom([]float64{1.0, 0.7}).Draw(t, "threshold")}
	if rapid.Bool().Draw(t, "global") {
		c.hasGlobal = true
		c.maxRequests = rapid.IntRange(0, 8).Draw(t, "maxRequests")
		c.maxBytes = int64(rapid.SampledFrom([]int{0, 200, 600}).Draw(t, "maxBytes"))
	}
	if rapid.Bool().Draw(t, "defaultBand") {
		c.hasDefaultBand = true
		c.defaultReq = rapid.IntRange(0, 6).Draw(t, "defaultReq")
		c.defaultBytes = int64(rapid.SampledFrom([]int{0, 300}).Draw(t, "defaultBytes"))
	}
	for i, b := range fcBands {
		if !rapid.Bool().Draw(t, fmt.Sprint("band", i)) {
			continue
		}
		c.bands = append(c.bands, fcBand{
			priority:    b.priority,
			maxRequests: rapid.IntRange(0, 5).Draw(t, fmt.Sprint("bandReq", i)),
			maxBytes:    int64(rapid.SampledFrom([]int{0, 150, 400}).Draw(t, fmt.Sprint("bandBytes", i))),
			roundRobin:  rapid.Bool().Draw(t, fmt.Sprint("bandRR", i)),
		})
	}
	n := rapid.IntRange(1, 30).Draw(t, "ops")
	for i := 0; i < n; i++ {
		if rapid.IntRange(0, 9).Draw(t, fmt.Sprint("kind", i)) < 6 {
			c.arrivals++
			c.ops = append(c.ops, fcOp{arrive: true, id: fmt.Sprint("r", i),
				flow:   rapid.SampledFrom(fcFlows).Draw(t, fmt.Sprint("flow", i)),
				band:   rapid.IntRange(0, len(fcBands)-1).Draw(t, fmt.Sprint("prio", i)),
				tokens: rapid.IntRange(1, 60).Draw(t, fmt.Sprint("tokens", i))})
			continue
		}
		c.ops = append(c.ops, fcOp{saturation: rapid.SampledFrom([]float64{0, 0.5, 0.69, 0.7, 0.99, 1.0, 1.3}).Draw(t, fmt.Sprint("sat", i))})
	}
	return c
}

// eppYAML is the flowControl block both sides read: BLIS through its EPP ingestion.
func (c fcCase) eppYAML() string {
	var b strings.Builder
	b.WriteString(`apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates: [flowControl]
plugins:
- type: concurrency-detector
- type: inflight-load-producer
- type: token-load-scorer
- type: round-robin-fairness-policy
- type: static-usage-limit-policy
`)
	fmt.Fprintf(&b, "  parameters: {threshold: %v}\n", c.threshold)
	b.WriteString(`schedulingProfiles:
- name: default
  plugins: [{pluginRef: token-load-scorer}]
flowControl:
  saturationDetector: {pluginRef: concurrency-detector}
  defaultRequestTTL: 1h
`)
	if c.hasGlobal {
		fmt.Fprintf(&b, "  maxRequests: %d\n  maxBytes: %d\n", c.maxRequests, c.maxBytes)
	}
	if c.hasDefaultBand {
		fmt.Fprintf(&b, "  defaultPriorityBand: {maxRequests: %d, maxBytes: %d}\n", c.defaultReq, c.defaultBytes)
	}
	if len(c.bands) > 0 {
		b.WriteString("  priorityBands:\n")
		for _, band := range c.bands {
			fmt.Fprintf(&b, "  - {priority: %d, maxRequests: %d, maxBytes: %d", band.priority, band.maxRequests, band.maxBytes)
			if band.roundRobin {
				b.WriteString(", fairnessPolicyRef: round-robin-fairness-policy")
			}
			b.WriteString("}\n")
		}
	}
	return b.String()
}

func blisGatewayQueue(t *rapid.T, c fcCase) *cluster.GatewayQueue {
	in, err := sim.ParseEndpointPickerConfig([]byte(c.eppYAML()), 64, version)
	if err != nil {
		t.Fatalf("BLIS ingestion: %v\n%s", err, c.eppYAML())
	}
	var cfg cluster.DeploymentConfig
	cfg.ApplyEPPIngest(in)
	return cluster.NewConformanceGatewayQueue(cfg)
}

// llmdFlowController builds v0.10.0's controller as its config loader would for the
// same block (registry.NewConfig defaults: FCFS ordering, global-strict fairness).
func llmdFlowController(t *rapid.T, ctx context.Context, c fcCase, det *fcDetector, clk *testclock.FakeClock) (*controller.FlowController, *registry.FlowRegistry) {
	plug := func(p fwkplugin.Plugin, err error) fwkplugin.Plugin {
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	order := plug(fcfs.FCFSOrderingPolicyFactory("fcfs-ordering-policy", nil, nil)).(fwkfc.OrderingPolicy)
	strict := plug(globalstrict.GlobalStrictFairnessPolicyFactory("global-strict-fairness-policy", nil, nil)).(fwkfc.FairnessPolicy)
	rr := plug(roundrobin.RoundRobinFairnessPolicyFactory("round-robin-fairness-policy", nil, nil)).(fwkfc.FairnessPolicy)
	defaults := registry.PriorityBandPolicyDefaults{OrderingPolicy: order, FairnessPolicy: strict}
	var opts []registry.ConfigOption
	if c.hasGlobal {
		opts = append(opts, registry.WithMaxRequests(uint64(c.maxRequests)), registry.WithMaxBytes(uint64(c.maxBytes)))
	}
	if c.hasDefaultBand {
		db, err := registry.NewPriorityBandConfig(0, defaults,
			registry.WithBandMaxRequests(uint64(c.defaultReq)), registry.WithBandMaxBytes(uint64(c.defaultBytes)))
		if err != nil {
			t.Fatal(err)
		}
		opts = append(opts, registry.WithDefaultPriorityBand(db))
	}
	for _, band := range c.bands {
		bopts := []registry.PriorityBandConfigOption{
			registry.WithBandMaxRequests(uint64(band.maxRequests)), registry.WithBandMaxBytes(uint64(band.maxBytes))}
		if band.roundRobin {
			bopts = append(bopts, registry.WithFairnessPolicy(rr))
		}
		pb, err := registry.NewPriorityBandConfig(band.priority, defaults, bopts...)
		if err != nil {
			t.Fatal(err)
		}
		opts = append(opts, registry.WithPriorityBand(pb))
	}
	// A priority without a configured band is provisioned from the default band template,
	// as the control plane does for an InferenceObjective's priority.
	for _, b := range fcBands {
		configured := false
		for _, band := range c.bands {
			configured = configured || band.priority == b.priority
		}
		if configured {
			continue
		}
		var bopts []registry.PriorityBandConfigOption
		if c.hasDefaultBand {
			bopts = append(bopts, registry.WithBandMaxRequests(uint64(c.defaultReq)), registry.WithBandMaxBytes(uint64(c.defaultBytes)))
		}
		pb, err := registry.NewPriorityBandConfig(b.priority, defaults, bopts...)
		if err != nil {
			t.Fatal(err)
		}
		opts = append(opts, registry.WithPriorityBand(pb))
	}
	rcfg, err := registry.NewConfig(defaults, opts...)
	if err != nil {
		t.Fatal(err)
	}
	reg := registry.NewFlowRegistry(rcfg, logr.Discard())
	ccfg, err := controller.NewConfig(controller.WithDefaultRequestTTL(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: "pod0"}}, &fwkdl.Metrics{})
	fc := controller.NewFlowController(ctx, "pool", ccfg, controller.Deps{
		Registry:           reg,
		SaturationDetector: det,
		EndpointCandidates: &mocks.MockEndpointCandidates{Candidates: []fwkdl.Endpoint{ep}},
		UsageLimitPolicy:   usagelimits.NewConstPolicy("static-usage-limit-policy", c.threshold),
		Clock:              clk,
	})
	if !controllerRunning(clk) {
		t.Fatal("llm-d flow controller did not start")
	}
	return fc, reg
}

func controllerRunning(clk *testclock.FakeClock) bool {
	deadline := time.Now().Add(5 * time.Second)
	for !clk.HasWaiters() {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(10 * time.Microsecond)
	}
	return true
}

func waitFor(t *rapid.T, what string, cond func() bool) {
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("llm-d flow controller: timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Microsecond)
	}
}

type fcOutcome struct {
	id      string
	outcome types.QueueOutcome
}

func TestConformance_FlowControlQueueing(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawFlowControlCase(rt)
		blis := blisGatewayQueue(rt, c)

		ctx, cancel := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		defer func() { cancel(); wg.Wait() }()
		det := &fcDetector{}
		const held = 10.0 // above every ceiling: the cycle holds
		det.sat.Store(held)
		clk := testclock.NewFakeClock(time.Now())
		fc, reg := llmdFlowController(rt, ctx, c, det, clk)
		results := make(chan fcOutcome, c.arrivals)
		done := map[string]types.QueueOutcome{}
		drain := func() {
			for {
				select {
				case r := <-results:
					done[r.id] = r.outcome
				default:
					return
				}
			}
		}

		var seq int64
		for i, op := range c.ops {
			if op.arrive {
				b := fcBands[op.band]
				before := reg.Stats().TotalLen
				// The processor runs a dispatch cycle after every enqueue, accepted or rejected
				// (processor.go Run); the arrival is settled only once that cycle has run.
				calls := det.calls.Load()
				req := &fwkfcmocks.MockFlowControlRequest{IDV: op.id, ByteSizeV: uint64(4 * op.tokens),
					FlowKeyV: fwkfc.FlowKey{ID: op.flow, Priority: b.priority}}
				wg.Add(1)
				go func() {
					defer wg.Done()
					o, _ := fc.EnqueueAndWait(ctx, req)
					results <- fcOutcome{op.id, o}
				}()
				var rejected bool
				waitFor(rt, "arrival "+op.id, func() bool {
					drain()
					if o, ok := done[op.id]; ok {
						rejected = o != types.QueueOutcomeDispatched
						return det.calls.Load() > calls
					}
					return reg.Stats().TotalLen == before+1 && det.calls.Load() > calls
				})
				seq++
				outcome, _ := blis.Enqueue(&sim.Request{ID: op.id, InputTokens: make([]sim.TokenID, op.tokens),
					TenantID: op.flow, SLOClass: b.class}, seq)
				if blisRejected := outcome == cluster.Rejected; blisRejected != rejected {
					rt.Fatalf("op %d: arrival %s (flow %s, priority %d, %d bytes): llm-d rejected=%v, BLIS %v\n%s",
						i, op.id, op.flow, b.priority, 4*op.tokens, rejected, outcome, c.eppYAML())
				}
				clk.Step(time.Microsecond) // distinct enqueue times for FCFS
				continue
			}
			before := reg.Stats().TotalLen
			det.sat.Store(op.saturation)
			c0 := det.calls.Load()
			clk.Step(time.Millisecond)
			waitFor(rt, "dispatch cycle", func() bool { return det.calls.Load() > c0 })
			det.sat.Store(held)
			c1 := det.calls.Load()
			clk.Step(time.Millisecond) // barrier: the processor is single-threaded
			waitFor(rt, "barrier cycle", func() bool { return det.calls.Load() > c1 })
			got := ""
			if reg.Stats().TotalLen == before-1 {
				waitFor(rt, "dispatched request", func() bool {
					drain()
					for id, o := range done {
						if o == types.QueueOutcomeDispatched {
							got = id
							delete(done, id)
							return true
						}
					}
					return false
				})
			} else if reg.Stats().TotalLen != before {
				rt.Fatalf("op %d: llm-d queue length %d -> %d in one cycle", i, before, reg.Stats().TotalLen)
			}
			want := ""
			if r := blis.DequeueGated(op.saturation); r != nil {
				want = r.ID
			}
			if got != want {
				rt.Fatalf("op %d: cycle at saturation %v: llm-d dispatched %q, BLIS %q\n%s", i, op.saturation, got, want, c.eppYAML())
			}
		}
	})
}
