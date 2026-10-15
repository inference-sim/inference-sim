package cluster

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// eppTestConfig ingests a fixture and applies it to a test deployment exactly as
// blis run/replay do (DeploymentConfig.ApplyEPPIngest).
func eppTestConfig(t *testing.T, fixture string, v sim.EPPVersion, base DeploymentConfig) DeploymentConfig {
	t.Helper()
	in, err := sim.LoadEndpointPickerConfig("../testdata/epp/"+fixture, int(base.BlockSizeTokens), v)
	if err != nil {
		t.Fatalf("ingest %s: %v", fixture, err)
	}
	base.ApplyEPPIngest(in)
	return base
}

func eppRun(t *testing.T, cfg DeploymentConfig, n int) *ClusterSimulator {
	t.Helper()
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(n)), nil)
	mustRun(t, cs)
	return cs
}

func perInstanceCompleted(cs *ClusterSimulator) []int {
	var out []int
	for _, m := range cs.PerInstanceMetrics() {
		out = append(out, m.CompletedRequests)
	}
	return out
}

// The curvebender routers run end to end under both llm-d versions: every request
// is accounted for (INV-1), the in-flight token ledger drains to zero once all
// requests finish, and two runs with one seed are identical (INV-6).
func TestEPP_CurvebenderRouters_RunConserveAndDrain(t *testing.T) {
	for _, fixture := range []string{"optimized-baseline-cpu-tier.yaml", "precise-gpu.yaml"} {
		for _, v := range []sim.EPPVersion{sim.EPPVersionMain, sim.EPPVersionV0_10_0} {
			t.Run(fixture+"/"+string(v), func(t *testing.T) {
				base := newTestDeploymentConfig(4)
				base.BlockSizeTokens = 64
				cfg := eppTestConfig(t, fixture, v, base)
				const n = 200
				cs := eppRun(t, cfg, n)
				assertClusterINV1Conservation(t, cs, n, cs.RejectedRequests(), fixture)
				if cs.routingPipeline == nil || len(cs.inFlightLoadTrackers) == 0 {
					t.Fatal("EPP pipeline not built")
				}
				for _, tr := range cs.inFlightLoadTrackers {
					for _, inst := range cs.instances {
						if got := tr.Tokens(string(inst.ID())); got != 0 {
							t.Errorf("tracker %s: %d in-flight tokens left on %s after all requests finished", tr.Name(), got, inst.ID())
						}
					}
				}
				again := eppRun(t, cfg, n)
				if a, b := perInstanceCompleted(cs), perInstanceCompleted(again); !reflect.DeepEqual(a, b) {
					t.Errorf("INV-6: per-instance completions differ across identical runs: %v vs %v", a, b)
				}
			})
		}
	}
}

// curvebender's P/D router (always-disagg-pd-decider, decode-first) runs on BLIS's
// P/D machinery: every request is disaggregated, prefill lands only on prefill
// pods, and the in-flight ledger drains.
func TestEPP_CurvebenderPD_RunsDisaggregated(t *testing.T) {
	cases := []struct {
		fixture string
		v       sim.EPPVersion
		prefill int
	}{
		{"pd-1p1d.yaml", sim.EPPVersionMain, 3},
		{"pd-1p1d.yaml", sim.EPPVersionV0_10_0, 3},
		{"llm-d-guides/pd-disaggregation.yaml", sim.EPPVersionMain, 1},
	}
	for _, tc := range cases {
		fixture, v := tc.fixture, tc.v
		t.Run(fixture+"/"+string(v), func(t *testing.T) {
			base := newTestDisaggDeploymentConfig(4, 2, 2)
			base.BlockSizeTokens = 64
			base.PDDecider = ""
			cfg := eppTestConfig(t, fixture, v, base)
			if cfg.PDDecider != "always" || len(cfg.PrefillScorerConfigs) != tc.prefill || len(cfg.DecodeScorerConfigs) != 1 {
				t.Fatalf("P/D mapping: decider=%q prefill=%v decode=%v", cfg.PDDecider, cfg.PrefillScorerConfigs, cfg.DecodeScorerConfigs)
			}
			const n = 60
			cs := eppRun(t, cfg, n)
			if got := len(cs.ParentRequests()); got != n {
				t.Errorf("always-disagg: %d parents, want %d", got, n)
			}
			for _, p := range cs.ParentRequests() {
				if role := cs.poolMembership[string(p.PrefillInstanceID)]; !role.Has(PoolRolePrefill) {
					t.Errorf("parent %s: prefill on %s (role %v)", p.ID, p.PrefillInstanceID, role)
				}
			}
			if tc.prefill == 1 && len(cs.inFlightLoadTrackers) == 0 {
				t.Fatal("pd-disaggregation guide: expected an inflight-load-producer")
			}
			for _, tr := range cs.inFlightLoadTrackers {
				for _, inst := range cs.instances {
					if got := tr.Tokens(string(inst.ID())); got != 0 {
						t.Errorf("tracker %s: %d tokens left on %s", tr.Name(), got, inst.ID())
					}
				}
			}
		})
	}
}

// With addEstimatedOutputTokens the decode charge outlives the first token and is
// released only when the client stream (the decode sub-request) completes
// (main:requestcontrol/dataproducer/inflightload/producer.go:569-640); the ledger
// must still drain, and must hold tokens while requests are in flight.
func TestEPP_PD_OutputEstimateChargeReleasedAtDecodeCompletion(t *testing.T) {
	const cfgYAML = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters: {deciders: {prefill: always-disagg-pd-decider}}
- type: prefill-filter
- type: decode-filter
- type: inflight-load-producer
  parameters: {addEstimatedOutputTokens: true}
- type: token-load-scorer
schedulingProfiles:
- name: prefill
  plugins: [{pluginRef: prefill-filter}, {pluginRef: token-load-scorer}]
- name: decode
  plugins: [{pluginRef: decode-filter}, {pluginRef: token-load-scorer}]
`
	for _, v := range []sim.EPPVersion{sim.EPPVersionMain, sim.EPPVersionV0_10_0} {
		t.Run(string(v), func(t *testing.T) {
			in, err := sim.ParseEndpointPickerConfig([]byte(cfgYAML), 64, v)
			if err != nil {
				t.Fatal(err)
			}
			base := newTestDisaggDeploymentConfig(4, 2, 2)
			base.BlockSizeTokens = 64
			base.ApplyEPPIngest(in)
			cs := NewClusterSimulator(base, NewSliceRequestSource(newTestRequests(40)), nil)
			hook := &peakTokensHook{cs: cs}
			cs.SetProgressHook(hook, 1000)
			mustRun(t, cs)
			if hook.peak == 0 {
				t.Error("no in-flight tokens were ever charged")
			}
			for _, inst := range cs.instances {
				if got := cs.inFlightLoadTrackers[0].Tokens(string(inst.ID())); got != 0 {
					t.Errorf("%d tokens left on %s", got, inst.ID())
				}
			}
		})
	}
}

// A P/D request whose prefill sub-request times out ends there; its decode and
// prefill charges are released (llm-d frees every entry on stream end), so the
// ledger drains and every request is accounted for (INV-1).
func TestEPP_PD_PrefillTimeoutDrainsLedger(t *testing.T) {
	const cfgYAML = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters: {deciders: {prefill: always-disagg-pd-decider}}
- type: prefill-filter
- type: decode-filter
- type: inflight-load-producer
  parameters: {addEstimatedOutputTokens: true}
- type: token-load-scorer
schedulingProfiles:
- name: prefill
  plugins: [{pluginRef: prefill-filter}, {pluginRef: token-load-scorer}]
- name: decode
  plugins: [{pluginRef: decode-filter}, {pluginRef: token-load-scorer}]
`
	in, err := sim.ParseEndpointPickerConfig([]byte(cfgYAML), 64, sim.EPPVersionMain)
	if err != nil {
		t.Fatal(err)
	}
	base := newTestDisaggDeploymentConfig(4, 2, 2)
	base.BlockSizeTokens = 64
	base.ApplyEPPIngest(in)
	reqs := newTestRequests(60)
	for _, r := range reqs {
		r.Deadline = r.ArrivalTime + 1 // every prefill times out before it can finish
	}
	cs := NewClusterSimulator(base, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	assertClusterINV1Conservation(t, cs, len(reqs), cs.RejectedRequests(), "prefill timeouts")
	for _, inst := range cs.instances {
		if got := cs.inFlightLoadTrackers[0].Tokens(string(inst.ID())); got != 0 {
			t.Errorf("%d tokens left on %s", got, inst.ID())
		}
	}
}

type peakTokensHook struct {
	cs   *ClusterSimulator
	peak int64
}

func (h *peakTokensHook) OnProgress(sim.ProgressSnapshot) {
	for _, inst := range h.cs.instances {
		if tok := h.cs.inFlightLoadTrackers[0].Tokens(string(inst.ID())); tok > h.peak {
			h.peak = tok
		}
	}
}

// Byte and per-band caps (llm-d flowControl maxBytes / priorityBands): a request
// that would exceed a cap is rejected, the byte counters return to zero as
// requests leave by any path, and the length counters stay consistent (INV-16).
func TestGatewayQueue_ByteAndBandCaps(t *testing.T) {
	q := NewGatewayQueue("fifo", 0, nil)
	q.SetByteCapacity(4*250, 0)   // global: 250 prompt tokens' worth
	q.SetBandLimits(3, 2, 0, nil) // the "standard" band (priority 3) holds 2
	mk := func(id string, tokens int) *sim.Request {
		return &sim.Request{ID: id, InputTokens: make([]sim.TokenID, tokens), SLOClass: "standard"}
	}
	if out, _ := q.Enqueue(mk("a", 100), 1); out != Enqueued {
		t.Fatalf("a: %v", out)
	}
	if out, _ := q.Enqueue(mk("b", 200), 2); out != Rejected {
		t.Errorf("b exceeds the byte cap: %v", out)
	}
	if out, _ := q.Enqueue(mk("c", 150), 3); out != Enqueued {
		t.Errorf("c fits exactly: %v", out)
	}
	if out, _ := q.Enqueue(mk("d", 1), 4); out != Rejected {
		t.Errorf("d exceeds the band's 2-request cap: %v", out)
	}
	assertINV16(t, q, "enqueue")
	q.RemoveByRequestID("a")
	if q.DequeueGated(0) == nil || q.totalBytes != 0 {
		t.Errorf("bytes after draining: %d", q.totalBytes)
	}
	assertINV16(t, q, "drain")
}

// --routing-scorers token-load:1,prefix-cache:1 builds the llm-d data layer with
// llm-d's default producers and keeps the in-flight ledger consistent.
func TestEPP_FlagScorersBuildDefaultPipeline(t *testing.T) {
	scorers, err := sim.ParseScorerConfigs("token-load:1,prefix-cache:1")
	if err != nil {
		t.Fatal(err)
	}
	cfg := newTestDeploymentConfig(3)
	cfg.RoutingPolicy, cfg.RoutingScorerConfigs = "weighted", scorers
	cs := eppRun(t, cfg, 80)
	assertClusterINV1Conservation(t, cs, 80, cs.RejectedRequests(), "flag scorers")
	if cs.routingPipeline == nil || len(cs.inFlightLoadTrackers) != 1 {
		t.Fatalf("default pipeline not built: %v", cs.routingPipeline)
	}
	for _, inst := range cs.instances {
		if got := cs.inFlightLoadTrackers[0].Tokens(string(inst.ID())); got != 0 {
			t.Errorf("%d tokens left on %s", got, inst.ID())
		}
	}
}

// v0.10.0's concurrency guard does not fail open: with no flow-control gate and a
// tiny maxConcurrency, saturated endpoints are filtered out and requests are
// rejected at routing, as llm-d fails them. Every request is still accounted for
// (INV-1), the in-flight ledger drains, and each rejection reaches the unserved
// hook (recordUnserved), as the agentx branch's other refusals do. main fails open and rejects nothing.
func TestEPP_ConcurrencyGuardRejections(t *testing.T) {
	const cfgYAML = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: concurrency-detector
  parameters: {maxConcurrency: 1}
- type: inflight-load-producer
- type: token-load-scorer
schedulingProfiles:
- name: default
  plugins: [{pluginRef: concurrency-detector}, {pluginRef: token-load-scorer}]
`
	for _, tc := range []struct {
		v         sim.EPPVersion
		rejecting bool
	}{{sim.EPPVersionV0_10_0, true}, {sim.EPPVersionMain, false}} {
		t.Run(string(tc.v), func(t *testing.T) {
			in, err := sim.ParseEndpointPickerConfig([]byte(cfgYAML), 16, tc.v)
			if err != nil {
				t.Fatal(err)
			}
			cfg := newTestDeploymentConfig(2)
			cfg.ApplyEPPIngest(in)
			reqs := newTestRequests(120)
			for i, r := range reqs {
				r.ArrivalTime = int64(i) // a burst: far more arrivals than 2 × maxConcurrency
			}
			unserved := 0
			cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
			cs.SetUnservedHook(func(*sim.Request, int64) []*sim.Request { unserved++; return nil })
			mustRun(t, cs)
			assertClusterINV1Conservation(t, cs, len(reqs), cs.RejectedRequests(), string(tc.v))
			if got := cs.RoutingRejections() > 0; got != tc.rejecting {
				t.Errorf("routing rejections %d, rejecting=%v", cs.RoutingRejections(), tc.rejecting)
			}
			if unserved != cs.RoutingRejections() {
				t.Errorf("unserved hook told of %d requests, %d rejected at routing", unserved, cs.RoutingRejections())
			}
			for _, inst := range cs.instances {
				if got := cs.inFlightLoadTrackers[0].Tokens(string(inst.ID())); got != 0 {
					t.Errorf("%d tokens left on %s", got, inst.ID())
				}
			}
		})
	}
}
