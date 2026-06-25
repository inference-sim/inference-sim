package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// newTestEPDDeploymentConfig is an E/P/D test harness layered on top of
// newTestDisaggDeploymentConfig. It adds an encode pool of `encode` instances
// and an encode decider. Pool layout (via BuildPoolMembershipFromIndices):
//   [0, prefill)                       — prefill-only
//   [prefill, prefill+decode)          — decode-only
//   [prefill+decode, +shared)          — shared-role (zero here)
//   [+shared, +encode)                 — encode-only
//
// Tracing is enabled so tests can inspect EncodeRoutingRecord emission.
func newTestEPDDeploymentConfig(prefill, decode, encode int, encodeDecider string) DeploymentConfig {
	total := prefill + decode + encode
	cfg := newTestDisaggDeploymentConfig(total, prefill, decode)
	cfg.EncodeInstances = encode
	cfg.EncodeDecider = encodeDecider
	cfg.TraceLevel = "decisions"
	return cfg
}

// multimodalRequests returns n requests, alternating multimodal (image-bearing)
// and text-only, with IDs "mm_<i>" / "txt_<i>". Uses the existing
// test-workload generator then mutates per-modality token counts so
// IsMultimodal() is deterministic.
func multimodalRequests(n int) []*sim.Request {
	reqs := newTestRequests(n)
	for i, r := range reqs {
		if i%2 == 0 {
			r.ID = "mm_" + r.ID
			r.ImageTokenCount = 32
		} else {
			r.ID = "txt_" + r.ID
		}
	}
	return reqs
}

// allTextRequests returns n text-only requests (never multimodal).
func allTextRequests(n int) []*sim.Request {
	reqs := newTestRequests(n)
	for _, r := range reqs {
		r.ImageTokenCount = 0
		r.AudioTokenCount = 0
		r.VideoTokenCount = 0
	}
	return reqs
}

// TestEPD_EncodeNever_IsPDUnchanged (BC-EPD-1 guard): configuring the encode
// pool but using decider="never" leaves PD behavior structurally unchanged —
// no EncodeRoutingRecords are emitted regardless of request modality.
func TestEPD_EncodeNever_IsPDUnchanged(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "never")
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(multimodalRequests(6)), nil)
	mustRun(t, cs)

	if cs.trace == nil {
		t.Fatalf("trace must be enabled")
	}
	if got := len(cs.trace.EncodeRoutings); got != 0 {
		t.Errorf("EncodeRoutings count = %d, want 0 with decider=never", got)
	}
	if cs.EncodeRoutingRejections() != 0 {
		t.Errorf("EncodeRoutingRejections = %d, want 0", cs.EncodeRoutingRejections())
	}
}

// TestEPD_EncodeFires_WithDisagg (BC-EPD-3): with decider="always" and a
// disaggregating PD config, every request produces exactly one encode record,
// and encode clock is <= subsequent prefill clock (encode before prefill).
// Parent.EncodeInstanceID is populated.
func TestEPD_EncodeFires_WithDisagg(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "always")
	cfg.PDDecider = "always" // force disaggregation
	reqs := newTestRequests(3)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.trace == nil {
		t.Fatalf("trace must be enabled")
	}
	if got := len(cs.trace.EncodeRoutings); got != len(reqs) {
		t.Errorf("EncodeRoutings count = %d, want %d", got, len(reqs))
	}
	if got := len(cs.parentRequests); got != len(reqs) {
		t.Fatalf("parentRequests count = %d, want %d", got, len(reqs))
	}
	for _, parent := range cs.parentRequests {
		if parent.EncodeInstanceID == "" {
			t.Errorf("parent %s: EncodeInstanceID empty, want non-empty", parent.ID)
			continue
		}
		// Encode instance must be an encode-pool member.
		role, ok := cs.poolMembership[string(parent.EncodeInstanceID)]
		if !ok || !role.Has(PoolRoleEncode) {
			t.Errorf("parent %s: EncodeInstanceID=%q has role=%v, want PoolRoleEncode",
				parent.ID, parent.EncodeInstanceID, role)
		}
	}

	// Encode record clock <= Prefill record clock for each parent (option A:
	// both recorded at the same simulation tick, so the relation is <=, not <).
	encodeByReq := make(map[string]trace.EncodeRoutingRecord, len(cs.trace.EncodeRoutings))
	for _, r := range cs.trace.EncodeRoutings {
		encodeByReq[r.ParentRequestID] = r
	}
	for _, pref := range cs.trace.PrefillRoutings {
		enc, ok := encodeByReq[pref.ParentRequestID]
		if !ok {
			t.Errorf("no encode record for parent %q that has a prefill record", pref.ParentRequestID)
			continue
		}
		if enc.Clock > pref.Clock {
			t.Errorf("encode.Clock=%d > prefill.Clock=%d for parent %q — encode must not follow prefill",
				enc.Clock, pref.Clock, pref.ParentRequestID)
		}
	}
}

// TestEPD_EncodeFires_WithoutDisagg (BC-EPD-4): when encode fires but
// disaggregation is off (decider="never"), the request is injected directly
// to the pre-selected decode pod, with an EncodeRoutingRecord recorded and no
// PrefillRoutingRecord.
func TestEPD_EncodeFires_WithoutDisagg(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "always")
	cfg.PDDecider = "never" // disagg decider says "don't disaggregate"
	reqs := newTestRequests(3)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.trace == nil {
		t.Fatalf("trace must be enabled")
	}
	if got := len(cs.trace.EncodeRoutings); got != len(reqs) {
		t.Errorf("EncodeRoutings count = %d, want %d (encode must fire on non-disagg path)", got, len(reqs))
	}
	if got := len(cs.trace.PrefillRoutings); got != 0 {
		t.Errorf("PrefillRoutings count = %d, want 0 (disagg off)", got)
	}
	if got := len(cs.parentRequests); got != 0 {
		t.Errorf("parentRequests count = %d, want 0 (no disagg, no parent created)", got)
	}
	// Every request must have been routed to a decode-pool member.
	for _, r := range cs.trace.Routings {
		role, ok := cs.poolMembership[r.ChosenInstance]
		if !ok || !role.Has(PoolRoleDecode) {
			t.Errorf("non-disagg routing chose %q (role=%v), want a PoolRoleDecode member",
				r.ChosenInstance, role)
		}
	}
}

// TestEPD_EmptyEncodePool_RoutingRejection (BC-EPD-6 stress): force every
// encode instance to be non-routable by marking it Terminating. Encode
// routing must reject each request and increment encodeRoutingRejections;
// INV-1 ledger (including the new encode term) balances.
func TestEPD_EmptyEncodePool_RoutingRejection(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(1, 1, 1, "always")
	cfg.PDDecider = "never"
	reqs := newTestRequests(3)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)

	// Take the only encode instance offline before Run so the encode pool is empty.
	// Draining state makes IsRoutable() return false.
	for _, inst := range cs.instances {
		if cs.poolMembership[string(inst.ID())].Has(PoolRoleEncode) {
			inst.TransitionTo(sim.InstanceStateDraining)
		}
	}
	mustRun(t, cs)

	if got := cs.EncodeRoutingRejections(); got != len(reqs) {
		t.Errorf("EncodeRoutingRejections = %d, want %d", got, len(reqs))
	}
	// Aggregate INV-1 check: injected == completed + queued + running + dropped + timedout
	// + routingRejections + encodeRoutingRejections + gwEvicted (+ gw terms, all zero here).
	m := cs.AggregatedMetrics()
	injected := len(reqs) - cs.RejectedRequests()
	accounted := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable +
		m.TimedOutRequests + cs.RoutingRejections() + cs.EncodeRoutingRejections() +
		cs.GatewayQueueDepth() + cs.GatewayQueueShed() + cs.GatewayQueueRejected() +
		cs.GatewayEvicted()
	if injected != accounted {
		t.Errorf("INV-1: injected=%d != accounted=%d (completed=%d timedout=%d dropped=%d encRej=%d routingRej=%d)",
			injected, accounted, m.CompletedRequests, m.TimedOutRequests,
			m.DroppedUnservable, cs.EncodeRoutingRejections(), cs.RoutingRejections())
	}
}

// TestEPD_DeciderReadsDecodeInstanceID (llm-d parity): the encode decider is
// called with the pre-selected decode instance ID (matching
// decodeRes.TargetEndpoints[0] in llm-d's Permalink 2). The decoded instance
// must be a PoolRoleDecode member (not PoolRolePrefill or PoolRoleEncode).
func TestEPD_DeciderReadsDecodeInstanceID(t *testing.T) {
	// Spy decider: captures every decodeInstanceID it sees, then returns true
	// so the downstream encode-routing path runs normally.
	spy := &spyEncodeDecider{}
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "always")
	cfg.PDDecider = "always"
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(4)), nil)
	// Inject the spy after construction (the factory can't return the spy).
	cs.encodeDecider = spy

	mustRun(t, cs)

	if len(spy.seen) == 0 {
		t.Fatal("spy decider was never called")
	}
	for _, id := range spy.seen {
		role, ok := cs.poolMembership[id]
		if !ok {
			t.Errorf("spy saw decodeInstanceID %q that is not in pool membership", id)
			continue
		}
		if !role.Has(PoolRoleDecode) {
			t.Errorf("spy saw decodeInstanceID %q with role=%v, want a PoolRoleDecode member",
				id, role)
		}
	}
}

// TestEPD_TextOnly_MultimodalDecider_Skips (BC-EPD-2 negative on the full
// pipeline): multimodal decider does not fire for text-only requests even
// when the encode pool is configured.
func TestEPD_TextOnly_MultimodalDecider_Skips(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "multimodal")
	cfg.PDDecider = "always"
	reqs := allTextRequests(4)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if got := len(cs.trace.EncodeRoutings); got != 0 {
		t.Errorf("EncodeRoutings = %d, want 0 for all-text workload with multimodal decider", got)
	}
}

// TestEPD_MultimodalSubsetEncoded (BC-EPD-2 positive on the full pipeline):
// multimodal decider fires exactly for requests that carry any non-text modality.
func TestEPD_MultimodalSubsetEncoded(t *testing.T) {
	cfg := newTestEPDDeploymentConfig(2, 2, 1, "multimodal")
	cfg.PDDecider = "always"
	// Use a fixed split — half multimodal, half text.
	reqs := multimodalRequests(6)
	wantMM := 0
	for _, r := range reqs {
		if r.IsMultimodal() {
			wantMM++
		}
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if got := len(cs.trace.EncodeRoutings); got != wantMM {
		t.Errorf("EncodeRoutings = %d, want %d (one per multimodal request)", got, wantMM)
	}
}

// TestEPD_Determinism (BC-EPD-8): two runs with the same seed + same workload +
// encode pool enabled produce identical encode-record slices.
func TestEPD_Determinism(t *testing.T) {
	runOnce := func() []trace.EncodeRoutingRecord {
		cfg := newTestEPDDeploymentConfig(2, 2, 1, "always")
		cfg.PDDecider = "always"
		// Build a stable request list from the same seed (multimodalRequests
		// is deterministic given newTestRequests' seed=42).
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(multimodalRequests(5)), nil)
		mustRun(t, cs)
		return cs.trace.EncodeRoutings
	}
	a := runOnce()
	b := runOnce()
	if len(a) != len(b) {
		t.Fatalf("determinism: len(a)=%d len(b)=%d", len(a), len(b))
	}
	for i := range a {
		if a[i].ParentRequestID != b[i].ParentRequestID || a[i].ChosenInstance != b[i].ChosenInstance || a[i].Clock != b[i].Clock {
			t.Errorf("determinism diff at [%d]: a=%+v b=%+v", i, a[i], b[i])
		}
	}
}

// TestValidatePoolTopology_EncodeEdgeCases covers BC-EPD-5 config-time
// rejection at the validation boundary (encode-without-decode).
func TestValidatePoolTopology_EncodeEdgeCases(t *testing.T) {
	// encode-only with no PD pool must be rejected.
	if err := ValidatePoolTopology(0, 0, 0, 2, 2); err == nil {
		t.Error("ValidatePoolTopology(encode-only) accepted; want error")
	}
	// encode with shared-role only: allowed.
	if err := ValidatePoolTopology(0, 0, 2, 1, 3); err != nil {
		t.Errorf("ValidatePoolTopology(shared+encode) rejected: %v", err)
	}
}

// TestEPDDisabled_ZeroInstances_NoEncodeActivity (BC-EPD-1 direct guard):
// with --encode-instances == 0 there is no encode decider, no encode snapshots,
// no encode routing records, and EncodeRoutingRejections is always zero — for
// any mix of multimodal and text-only requests. This is the dedicated
// zero-instances test the plan's "TestEPDDisabled_PDUnchanged" named;
// TestEPD_EncodeNever_IsPDUnchanged exercises the decider=never path with
// a pool configured, which is subtly different.
func TestEPDDisabled_ZeroInstances_NoEncodeActivity(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2) // standard PD, no encode pool
	cfg.PDDecider = "always"
	cfg.TraceLevel = "decisions"
	// EncodeInstances defaulted to 0 by newTestDisaggDeploymentConfig.
	reqs := multimodalRequests(6) // half multimodal, half text — worst case
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.encodeDecider != nil {
		t.Error("encodeDecider must be nil when EncodeInstances == 0")
	}
	if cs.EncodeRoutingRejections() != 0 {
		t.Errorf("EncodeRoutingRejections = %d, want 0", cs.EncodeRoutingRejections())
	}
	if cs.trace == nil {
		t.Fatalf("trace must be enabled")
	}
	if got := len(cs.trace.EncodeRoutings); got != 0 {
		t.Errorf("EncodeRoutings = %d, want 0 with EncodeInstances == 0", got)
	}
}

// --- Helpers ---

// spyEncodeDecider records the decodeInstanceID arguments it receives.
// Always approves encoding so the full pipeline runs.
type spyEncodeDecider struct {
	seen []string
}

func (s *spyEncodeDecider) ShouldEncode(_ *sim.Request, decodeInstanceID string) bool {
	s.seen = append(s.seen, decodeInstanceID)
	return true
}

// compile-time assertion
var _ sim.EncodeDecider = (*spyEncodeDecider)(nil)
